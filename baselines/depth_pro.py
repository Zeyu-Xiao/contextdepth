"""Batched Depth Pro inference with native input-grid metric depth."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from contextdepth.baseline import DepthBaseline
from contextdepth.model_compat import add_source, camera_on_canvas
from contextdepth.types import CameraEstimate, Prediction
from ._common import _array, input_depth


class DepthPro(DepthBaseline):
    source_files = ("depth_pro/depth_pro.py",)
    matmul_tf32 = (False, True)
    supports_batch = True
    supports_setting_reuse = True

    def __init__(self, name, paths, device):
        add_source(paths["source"], "depth_pro")
        super().__init__(name, paths, device)
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(paths["weights"]))
        precision = self.torch.float16 if self.device.type == "cuda" else self.torch.float32
        self.model, self.transform = depth_pro.create_model_and_transforms(config=config, device=self.device, precision=precision)
        self.model.eval().requires_grad_(False)
        self.metadata.update(weights_dtype=str(precision).split(".")[-1], internal_hw=[1536, 1536],
                             budget="official fixed square; ROI pixel matching unsupported")


    def batch_key(self, request):
        known, g = self._validate(request)
        return known, *g.as_dict().values()

    def preprocess(self, requests):
        if not requests:
            raise ValueError("Depth Pro requires at least one input")
        key = self.batch_key(requests[0])
        if any(self.batch_key(request) != key for request in requests[1:]):
            raise ValueError("Depth Pro batches require matching geometry and camera settings")
        known, g = self._validate(requests[0])
        cameras = [camera_on_canvas(request.camera, g) if known else None for request in requests]
        # The pinned upstream transform is ToTensor -> device -> Normalize ->
        # ConvertImageDtype. Keep that operation order but transfer the stacked
        # CPU tensors once, then normalize/convert the entire BCHW tensor.
        transforms = self.transform.transforms
        tensor = self.torch.stack([transforms[0](np.array(request.rgb, copy=True)) for request in requests]).to(self.device)
        for transform in transforms[2:]:
            tensor = transform(tensor)
        focal = None
        if known:
            focal = self.torch.tensor([float(k[0, 0]) for k in cameras], dtype=self.torch.float32).to(self.device)
            # Native inverse depth is Bx1xHxW. A plain B-vector would silently
            # broadcast along W, mixing image focal lengths with pixel columns.
            focal = focal.reshape(-1, 1, 1, 1) if len(requests) > 1 else focal[0]
        return dict(requests=requests, geometry=g, cameras=cameras, tensor=tensor, focal=focal)

    def infer_model(self, prepared):
        return self.model.infer(prepared["tensor"], f_px=prepared["focal"])

    def postprocess(self, output, prepared):
        g, count = prepared["geometry"], len(prepared["requests"])
        depth = _array(output["depth"])
        focals = _array(output["focallength_px"]).reshape(-1)
        # Upstream infer squeezes singleton batch and channel dimensions.
        if depth.size != count * g.canvas_height * g.canvas_width or focals.size != count:
            raise ValueError("Depth Pro must return one depth and focal length per image")
        depth = depth.reshape(count, g.canvas_height, g.canvas_width)
        predictions = []
        for i in range(count):
            reported = float(focals[i])
            camera = CameraEstimate(focal_px=np.array([reported], dtype=np.float64),
                                    intrinsics=prepared["cameras"][i])
            predictions.append(Prediction(
                input_depth(depth[i], g), camera=camera,
                metadata=dict(focallength_px=reported, inference_batch_size=count,
                              batch_rng_policy="first_request")))
        return predictions
