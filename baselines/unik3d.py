"""UniK3D adapter using its native Pinhole input and camera-Z output.

Unknown camera: an outpainted candidate may reuse the camera that the model
estimated on the unexpanded query (``supports_query_camera``). UniK3D predicts
a camera as angular parameters plus spherical-harmonic coefficients. That same
continuous camera function is evaluated at the candidate's pixel centres,
mapped through the official preprocessing of both inputs, and handed to the
decoder as its ray field. No pinhole approximation and no ground truth enter.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from contextdepth.baseline import DepthBaseline, set_seed
from contextdepth.model_compat import (
    add_source, camera_on_canvas,
    lazy_training_imports, matched_constraints, shape_scope,
)
from contextdepth.types import CameraEstimate, Prediction
from ._common import _array, input_depth, stable_seed


def preprocessing_geometry(constraints, g):
    """Padding and resize that the official ``infer`` applies to a canvas of this size."""
    h, w = g.canvas_height, g.canvas_width
    ratio = w / h
    low, high = constraints["ratio_bounds"]
    target_ratio = min(high, max(low, ratio))
    left = right = top = bottom = 0
    if ratio > target_ratio:
        padded_h, padded_w = round(w / target_ratio), w
        top = (padded_h - h) // 2
        bottom = padded_h - h - top
    elif ratio < target_ratio:
        padded_h, padded_w = h, round(h * target_ratio)
        left = (padded_w - w) // 2
        right = padded_w - w - left
    else:
        padded_h, padded_w = h, w
    pixels = padded_w * padded_h
    target_pixels = min(constraints["pixels_max"], max(constraints["pixels_min"], pixels))
    factor = (target_pixels / pixels) ** 0.5
    new_h = math.ceil(round(padded_h * factor) / 14) * 14
    new_w = math.ceil(round(padded_w * factor) / 14) * 14
    return dict(paddings_lrtb=[left, right, top, bottom], internal_hw=[new_h, new_w],
                actual_scale_xy=[new_w / padded_w, new_h / padded_h])


def map_centers(x, y, source, target, roi_x, roi_y):
    """Map pixel centres of the candidate's internal grid onto the query's internal grid."""
    sx, sy = source["actual_scale_xy"]
    tx, ty = target["actual_scale_xy"]
    sl, _, st, _ = source["paddings_lrtb"]
    tl, _, tt, _ = target["paddings_lrtb"]
    if min(sx, sy, tx, ty) <= 0:
        raise ValueError("Nonpositive internal scale")
    if source == target and roi_x == 0 and roi_y == 0:
        return x, y
    return ((x / tx - tl - roi_x + sl) * sx,
            (y / ty - tt - roi_y + st) * sy)


def native_rays(torch, angles, coefficients, coordinates, source_hw):
    """Evaluate the model's own camera function (angular + degree-3 SH) at continuous centres."""
    from unik3d.utils.sht import rsh_cart_3

    h, w = source_hw
    if (angles.ndim != 2 or angles.shape[1] != 4 or tuple(coefficients.shape) != (angles.shape[0], 15, 3)
            or coordinates.ndim != 4 or coordinates.shape[:2] != (angles.shape[0], 2)):
        raise ValueError("Unexpected UniK3D camera parameters")
    if not bool(torch.isfinite(angles).all() and torch.isfinite(coefficients).all()):
        raise ValueError("Invalid predicted camera")
    longitude = (coordinates[:, 0] - angles[:, 2].view(-1, 1, 1)) / w * angles[:, 0].view(-1, 1, 1)
    latitude = (coordinates[:, 1] - angles[:, 3].view(-1, 1, 1)) / h * angles[:, 1].view(-1, 1, 1)
    x = torch.cos(latitude) * torch.sin(longitude)
    z = torch.cos(latitude) * torch.cos(longitude)
    y = -torch.sin(latitude)
    sphere = torch.stack([x, y, z], dim=-1)
    sphere = sphere / torch.norm(sphere, dim=-1, keepdim=True).clip(min=1e-5)
    harmonics = rsh_cart_3(sphere)[..., 1:]
    rays = torch.einsum("bhwc,bcd->bhwd", harmonics, coefficients)
    rays = rays / torch.norm(rays, dim=-1, keepdim=True).clip(min=1e-5)
    return rays.permute(0, 3, 1, 2)


class UniK3D(DepthBaseline):
    source_files = ("unik3d/models/unik3d.py", "unik3d/utils/camera.py")
    reset_seed_per_input = False
    supports_batch = True
    supports_setting_reuse = True
    supports_query_camera = True

    def __init__(self, name, paths, device):
        add_source(paths["source"], "unik3d")
        super().__init__(name, paths, device)
        with lazy_training_imports():
            from unik3d.models.unik3d import UniK3D
        from unik3d.utils.camera import Pinhole
        from safetensors.torch import load_file
        self.Pinhole = Pinhole
        weights = Path(paths["weights"])
        config = json.loads(Path(paths.get("config", weights / "config.json")).read_text())
        encoder = config["model"]["pixel_encoder"]
        if encoder["name"] != "dinov2_vitl14" or encoder.get("pretrained") is not None:
            raise ValueError("Expected the complete UniK3D ViT-L config without a secondary checkpoint")
        self.model = UniK3D(config)
        self.model.load_state_dict(load_file(str(weights / "model.safetensors"), device="cpu"), strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.metadata.update(precision="official FP16 autocast", budget="matched ROI pixels in both settings",
                             native_shape_constraints=dict(self.model.shape_constraints))


    def batch_key(self, request):
        known, g = self._validate(request)
        return known, *g.as_dict().values()

    def preprocess(self, requests):
        if not requests:
            raise ValueError("UniK3D requires at least one input")
        key = self.batch_key(requests[0])
        if any(self.batch_key(request) != key for request in requests[1:]):
            raise ValueError("UniK3D batches require matching geometry and camera settings")
        known, g = self._validate(requests[0])
        setting = "known" if known else "unknown"
        seeds = [stable_seed(f"unik3d/{r.uid}/{'direct' if r.arm == 'reference' else r.arm}/{setting}") for r in requests]
        # Keep the historical singleton seed timing and CHW transfer. Adding
        # the batch axis before the transfer changes the native tensor strides.
        set_seed(seeds[0], self.torch)
        if len(requests) == 1:
            tensor = self.torch.from_numpy(np.array(requests[0].rgb, copy=True)).permute(2, 0, 1).to(self.device)
        else:
            rgb = np.stack([request.rgb for request in requests])
            tensor = self.torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self.device)
        cameras = [camera_on_canvas(request.camera, g) if known else None for request in requests]
        camera = None
        if known:
            if len(requests) == 1:
                camera = self.Pinhole(K=self.torch.from_numpy(cameras[0]).unsqueeze(0).to(self.device))
            else:
                matrices = self.torch.from_numpy(np.stack(cameras)).to(self.device)
                camera = self.torch.cat([self.Pinhole(K=matrices[i:i + 1]) for i in range(len(requests))])
        active = matched_constraints(self.model.shape_constraints, g)
        # Query-camera reuse: a lone unknown-camera query records its camera function;
        # a candidate carrying that record gets its rays from it.
        state = requests[0].query_camera
        if state is not None and tuple(state["wh"]) != (g.roi_width, g.roi_height):
            raise ValueError("The reused camera was estimated on a different query")
        capture = not known and state is None and len(requests) == 1 and requests[0].baseline
        prep = preprocessing_geometry(active, g) if (capture or state is not None) else None
        return dict(requests=requests, geometry=g, known=known, seeds=seeds, tensor=tensor, cameras=cameras,
                    camera=camera, active=active, state=state, capture=capture, prep=prep, observed={})

    def _capture_hooks(self, prepared):
        decoder, observed = self.model.pixel_decoder, prepared["observed"]

        def angular(_module, _inputs, output):
            observed.setdefault("camera", []).append(tuple(item.detach().clone() for item in output))

        def decoded(_module, inputs, output):
            observed.setdefault("decoder", []).append(
                (list(inputs[0]["image"].shape[-2:]), output["rays"].detach().clone()))

        return [decoder.angular_module.register_forward_hook(angular), decoder.register_forward_hook(decoded)]

    def _transfer_hooks(self, prepared):
        from unik3d.utils.coordinate import coords_grid
        torch, decoder = self.torch, self.model.pixel_decoder
        state, prep, g, observed = prepared["state"], prepared["prep"], prepared["geometry"], prepared["observed"]
        if decoder.camera_gt is not True:
            raise RuntimeError("This UniK3D checkpoint does not accept an external ray field")

        def inject(_module, inputs):
            value = inputs[0]
            if value.get("camera") is not None or value.get("rays") is not None:
                raise RuntimeError("Unexpected external camera in an unknown-camera call")
            if list(value["image"].shape) != [1, 3, *prep["internal_hw"]]:
                raise RuntimeError("UniK3D preprocessing differs from the replayed geometry")
            coords = coords_grid(1, *prep["internal_hw"], device=value["image"].device)
            x, y = map_centers(coords[:, 0], coords[:, 1], state["prep"], prep, g.roi_x, g.roi_y)
            rays = native_rays(torch, state["angles"], state["coefficients"], torch.stack([x, y], dim=1),
                               state["prep"]["internal_hw"])
            norms = torch.linalg.vector_norm(rays.float(), dim=1)
            if not bool(torch.isfinite(rays).all()) or float((norms - 1).abs().max()) >= 2e-3:
                raise RuntimeError("The transferred query camera produced invalid rays")
            value["rays"] = rays
            observed.setdefault("supplied", []).append(rays)

        def verify(_module, inputs, output):
            supplied = observed.get("supplied", [])
            if len(supplied) != 1 or not torch.equal(output["rays"], supplied[0].flatten(2).transpose(1, 2)):
                raise RuntimeError("The decoder did not use the transferred query camera")
            observed["used"] = True

        return [decoder.register_forward_pre_hook(inject), decoder.register_forward_hook(verify)]

    def infer_model(self, prepared):
        active, tensor, camera = prepared["active"], prepared["tensor"], prepared["camera"]
        handles = []
        try:
            if prepared["state"] is not None:
                handles = self._transfer_hooks(prepared)
            elif prepared["capture"]:
                handles = self._capture_hooks(prepared)
            with shape_scope(self.model, active, interpolation="bilinear"):
                output = self.model.infer(tensor, camera=camera, normalize=True)
        finally:
            for handle in handles:
                handle.remove()
        return output

    def _camera_state(self, prepared):
        """The query's own camera function, checked against the rays the decoder produced."""
        from unik3d.utils.coordinate import coords_grid
        torch, observed, prep, g = self.torch, prepared["observed"], prepared["prep"], prepared["geometry"]
        if len(observed.get("camera", [])) != 1 or len(observed.get("decoder", [])) != 1:
            raise RuntimeError("Expected exactly one UniK3D camera prediction for the query")
        angles, coefficients = observed["camera"][0]
        hw, actual = observed["decoder"][0]
        if hw != prep["internal_hw"]:
            raise RuntimeError("UniK3D preprocessing differs from the replayed geometry")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            coordinates = coords_grid(1, *hw, device=angles.device)
            rebuilt = native_rays(torch, angles, coefficients, coordinates, hw).flatten(2).transpose(1, 2)
        if not torch.allclose(rebuilt, actual, rtol=2e-5, atol=2e-5):
            raise RuntimeError("Could not reproduce UniK3D's query rays from its camera parameters")
        return dict(angles=angles, coefficients=coefficients, prep=prep, wh=(g.canvas_width, g.canvas_height))

    def postprocess(self, output, prepared):
        g, active = prepared["geometry"], prepared["active"]
        count = len(prepared["requests"])
        depth = _array(output["depth"])
        points = _array(output["points"])
        rays = _array(output["rays"])
        if (depth.shape != (count, 1, g.canvas_height, g.canvas_width)
                or points.shape != (count, 3, g.canvas_height, g.canvas_width)
                or rays.shape != points.shape
                or not np.array_equal(depth, points[:, 2:3], equal_nan=True)):
            raise ValueError("UniK3D depth must be native input-grid camera-space Z")
        transferred = prepared["state"] is not None
        if transferred and not prepared["observed"].get("used"):
            raise RuntimeError("The decoder did not use the transferred query camera")
        state = self._camera_state(prepared) if prepared["capture"] else None
        predictions = []
        for i in range(count):
            k = prepared["cameras"][i]
            camera = CameraEstimate(intrinsics=k, rays=rays[i])
            predictions.append(Prediction(
                input_depth(depth[i, 0], g), camera=camera, camera_state=state,
                metadata=dict(inference_seed=prepared["seeds"][0],
                              individual_inference_seed=prepared["seeds"][i],
                              active_shape_constraints=active,
                              K_input=None if k is None else k.tolist(),
                              query_camera_reused=transferred,
                              inference_batch_size=count, batch_rng_policy="first_request")))
        return predictions
