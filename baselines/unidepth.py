"""UniDepth adapter, preserving the benchmark's camera/padding conventions."""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import numpy as np

from contextdepth.baseline import DepthBaseline
from contextdepth.model_compat import (
    add_source, camera_on_canvas, compensate_unidepth_camera,
    lazy_training_imports, matched_constraints, shape_scope,
    unidepth_geometry, unidepth_padding_scope,
)
from contextdepth.types import CameraEstimate, Prediction
from ._common import _array, input_depth


class UniDepth(DepthBaseline):
    """UniDepth V2.

    Known camera: the supplied K goes to the official ``infer`` unchanged. The
    option ``compensate_camera: true`` restores the earlier behaviour that first
    corrected K for the model's internal resize rounding; the paper's reference
    calibration was computed that way, so the evaluation config keeps it for the
    reference bank only.

    Unknown camera: an outpainted candidate may reuse the camera estimated on
    the unexpanded query (``supports_query_camera``). The query's public-API
    intrinsics keep their focal lengths; only the principal point moves by the
    query's offset on the canvas. Preprocessing stays the unknown-camera one.
    """

    source_files = ("unidepth/models/unidepthv2/unidepthv2.py",)
    matmul_tf32 = (False, True)
    supports_batch = True
    supports_setting_reuse = True
    supports_query_camera = True

    def __init__(self, name, paths, device):
        add_source(paths["source"], "unidepth")
        super().__init__(name, paths, device)
        self.compensate_camera = bool(self.options.get("compensate_camera", False))
        with lazy_training_imports():
            from unidepth.models import UniDepthV2
        from unidepth.utils.camera import Pinhole
        self.Pinhole = Pinhole
        self.implementation = importlib.import_module("unidepth.models.unidepthv2.unidepthv2")
        weights = Path(paths["weights"])
        config_path = Path(paths.get("config", weights / "config.json"))
        checkpoint = weights / "pytorch_model.bin" if weights.is_dir() else weights
        config = json.loads(config_path.read_text())
        self.model = UniDepthV2(config)
        state = self.torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.metadata.update(precision="official FP16 autocast", native_shape_constraints=dict(self.model.shape_constraints))
        self._describe()

    def _describe(self):
        known_budget = "native bounds with " + ("rounding-compensated K" if self.compensate_camera else "K supplied unchanged")
        self.metadata.update(
            budget=known_budget if self.known else "matched ROI pixels",
            padding="ceil outside ratio bounds" if self.known else "historical floor; negative in-range padding repaired")

    def configure(self, name, paths):
        super().configure(name, paths)
        self._describe()
        return self


    def batch_key(self, request):
        known, g = self._validate(request)
        return known, *g.as_dict().values()

    def preprocess(self, requests):
        if not requests:
            raise ValueError("UniDepth requires at least one input")
        key = self.batch_key(requests[0])
        if any(self.batch_key(request) != key for request in requests[1:]):
            raise ValueError("UniDepth batches require matching geometry and camera settings")
        known, g = self._validate(requests[0])
        native = self.model.shape_constraints
        active = native if known else matched_constraints(native, g)
        # Preserve the historical CHW transfer for a singleton. Native infer
        # adds its batch axis; pre-stacking changes that tensor's strides.
        if len(requests) == 1:
            tensor = self.torch.from_numpy(np.array(requests[0].rgb, copy=True)).permute(2, 0, 1).to(self.device)
        else:
            rgb = np.stack([request.rgb for request in requests])
            tensor = self.torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self.device)
        with unidepth_padding_scope(self.implementation, known=known), shape_scope(self.model, active, remove_level=not known):
            prep = unidepth_geometry(self.implementation, self.model, g)
        cameras = [camera_on_canvas(request.camera, g) if known else None for request in requests]
        calibrated = [None if not known else compensate_unidepth_camera(k, prep) if self.compensate_camera
                      else np.asarray(k, dtype=np.float32).copy() for k in cameras]
        camera = transferred = None
        if requests[0].query_camera is not None:
            # One image per call (enforced by DepthBaseline.predict). Focal lengths of the
            # query's estimated camera are unchanged; its principal point moves with the query.
            state = requests[0].query_camera
            if tuple(state["wh"]) != (g.roi_width, g.roi_height):
                raise ValueError("The reused camera was estimated on a different query")
            transferred = np.array(state["K"], dtype=np.float32, copy=True)
            transferred[0, 2] += g.roi_x
            transferred[1, 2] += g.roi_y
            camera = self.torch.as_tensor(transferred.copy(), device=self.device,
                                          dtype=self.torch.float32).unsqueeze(0)
        if known:
            if len(requests) == 1:
                # Keep the original singleton tensor shape and conversion.
                camera = self.torch.from_numpy(calibrated[0]).to(self.device)
            else:
                matrices = self.torch.from_numpy(np.stack(calibrated)).to(self.device)
                # Native get_rays builds a batch-1 grid. Separate camera
                # objects preserve the per-image K during unprojection.
                camera = self.torch.cat([self.Pinhole(K=matrices[i:i + 1]) for i in range(len(requests))])
        return dict(requests=requests, geometry=g, known=known, active=active, transferred=transferred,
                    tensor=tensor, prep=prep, cameras=cameras, calibrated=calibrated, camera=camera)

    def infer_model(self, prepared):
        with unidepth_padding_scope(self.implementation, known=prepared["known"]), shape_scope(
            self.model, prepared["active"], remove_level=not prepared["known"]
        ):
            return self.model.infer(prepared["tensor"], camera=prepared["camera"])

    def postprocess(self, output, prepared):
        g, known = prepared["geometry"], prepared["known"]
        active, prep = prepared["active"], prepared["prep"]
        depth, intrinsics = _array(output["depth"]), _array(output["intrinsics"])
        count = len(prepared["requests"])
        if depth.shape != (count, 1, g.canvas_height, g.canvas_width) or intrinsics.shape != (count, 3, 3):
            raise ValueError("UniDepth must return one native input-grid depth and camera per image")
        predictions = []
        for i in range(count):
            k, calibrated = prepared["cameras"][i], prepared["calibrated"][i]
            reported = intrinsics[i]
            focal = None
            if not known:
                # Preserve the historical correction's exact numeric order;
                # ROI normalization belongs to the selection pipeline.
                nominal = prep["nominal_scale"]
                ph, pw = prep["padded_hw"]
                ih = math.ceil(int(ph * nominal) / 14) * 14
                iw = math.ceil(int(pw * nominal) / 14) * 14
                fx = float(reported[0, 0]) * (nominal / (iw / pw))
                fy = float(reported[1, 1]) * (nominal / (ih / ph))
                focal = np.array([fx, fy], dtype=np.float64)
            camera = CameraEstimate(focal_px=focal, intrinsics=k,
                                    source_intrinsics=reported)
            transferred = prepared["transferred"]
            # What an unknown-camera call needs to reuse this input's camera on a larger canvas.
            state = None
            if not known and transferred is None:
                state = dict(K=reported.astype(np.float32).copy(), wh=(g.canvas_width, g.canvas_height))
            predictions.append(Prediction(
                input_depth(depth[i, 0], g), camera=camera, camera_state=state,
                metadata=dict(preprocessing=prep, active_shape_constraints=active,
                              intrinsics_api=reported.tolist(),
                              K_input=None if k is None else k.tolist(),
                              K_supplied=None if calibrated is None else calibrated.tolist(),
                              camera_rounding_compensation=bool(known and self.compensate_camera),
                              query_camera_K=None if transferred is None else transferred.tolist(),
                              inference_batch_size=count, batch_rng_policy="first_request")))
        return predictions
