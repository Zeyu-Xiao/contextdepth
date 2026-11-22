"""Depth Pro 2 (H+) with its separate focal-length estimator.

``weights`` is a folder holding the two checkpoints as published:
``DepthPro2-H/{config.json,model.safetensors}`` and
``DepthPro2-focal/{config.json,model.safetensors}``.

The network predicts inverse depth normalised by focal length, so the camera
only enters the final conversion to metres:
``depth = focal_px / (width * inverse_depth)``. Unknown camera uses the
model's own focal estimate; known camera uses the supplied horizontal focal
length with the same inverse-depth field. Sky can come out as +inf depth; it
is returned as is.
"""
from __future__ import annotations

from functools import partial
import json
import math
from pathlib import Path

import numpy as np

from contextdepth.baseline import DepthBaseline
from contextdepth.model_compat import add_source, camera_on_canvas
from contextdepth.types import CameraEstimate, Prediction


class DepthPro2(DepthBaseline):
    source_files = ("depth_pro_2/model.py",)
    supports_setting_reuse = True

    def __init__(self, name, paths, device):
        add_source(paths["source"], "depth_pro_2")
        super().__init__(name, paths, device)
        from depth_pro_2.model import DepthPro2 as Network, to_normalized_tensor
        from depth_pro_2.network.fov import FOVEstimator
        from safetensors.torch import load_file
        torch = self.torch
        root = Path(paths["weights"])
        depth_dir, focal_dir = root / "DepthPro2-H", root / "DepthPro2-focal"
        depth_config = json.loads((depth_dir / "config.json").read_text())
        focal_config = json.loads((focal_dir / "config.json").read_text())
        if depth_config["backbone_preset"] != "vit_huge_plus_patch16_dinov3" or depth_config["fov"]:
            raise ValueError(f"Expected the H+ depth checkpoint without a built-in focal head under {depth_dir}")
        precision = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        with torch.device(self.device):
            model = Network(**depth_config, precision=precision)
            focal = FOVEstimator(**focal_config, precision=precision)
        model.load_state_dict(load_file(str(depth_dir / "model.safetensors")), strict=True)
        focal.load_state_dict(load_file(str(focal_dir / "model.safetensors")), strict=True)
        model.fov = focal
        self.model = model.eval().requires_grad_(False)
        self.transform = partial(to_normalized_tensor, device=self.device)
        self.metadata.update(weights_dtype=str(precision).split(".")[-1],
                             budget="official preprocessing at the input's own resolution",
                             known_camera="supplied focal length applied to the same inverse depth")

    def preprocess(self, requests):
        if len(requests) != 1:
            raise ValueError("Depth Pro 2 runs one image per call")
        request = requests[0]
        known, g = self._validate(request)
        camera = focal = None
        if known:
            camera = camera_on_canvas(request.camera, g)
            # Horizontal focal length in pixels of this input, in float64. Moving the
            # principal point onto the canvas leaves it unchanged.
            focal = float(np.asarray(request.camera, np.float64)[0, 0]) * g.roi_width / float(g.fixed_w)
            if not math.isfinite(focal) or focal <= 0:
                raise ValueError("Known-camera inference requires a positive focal length")
        tensor = self.transform(np.array(request.rgb, copy=True))
        return dict(requests=requests, geometry=g, known=known, camera=camera, focal=focal, tensor=tensor)

    def infer_model(self, prepared):
        return self.model.infer(prepared["tensor"], f_px=None)

    def postprocess(self, output, prepared):
        torch, g, tensor = self.torch, prepared["geometry"], prepared["tensor"]
        estimated = float(output["focallength_px"].item())
        if not math.isfinite(estimated) or estimated <= 0:
            raise ValueError(f"Depth Pro 2 estimated an invalid focal length: {estimated}")
        depth = output["depth"]
        if prepared["known"]:
            focal = torch.as_tensor(prepared["focal"], dtype=torch.float32, device=tensor.device)
            depth = focal / (tensor.shape[-1] * output["inverse_depth"])
        depth = np.ascontiguousarray(depth.float().cpu().numpy(), dtype=np.float32)
        if depth.shape != (g.canvas_height, g.canvas_width):
            raise ValueError("Depth Pro 2 must return depth on the input grid")
        camera = CameraEstimate(focal_px=np.array([estimated], dtype=np.float64), intrinsics=prepared["camera"])
        return [Prediction(depth, camera=camera, metadata=dict(
            focallength_px=estimated, focal_px_supplied=prepared["focal"],
            infinite_depth_pixels=int(np.isposinf(depth).sum())))]
