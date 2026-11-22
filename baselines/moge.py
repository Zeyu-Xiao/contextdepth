"""MoGe public inference adapters and the retained full-K replay protocol.

MoGe2 uses the public horizontal-FOV input. MoGe3's known-camera paper preset
retains its historical full-K recovery; this is an explicit experiment rule,
not a requirement of ordinary MoGe inference.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from contextdepth.baseline import DepthBaseline, set_seed
from contextdepth.model_compat import add_source, camera_on_canvas
from contextdepth.types import CameraEstimate, Prediction
from ._common import _array, _float_images, _fov, input_depth, stable_seed, token_budget


class _MoGe(DepthBaseline):
    supports_batch = True
    supports_setting_reuse = True

    def __init__(self, name, paths, device):
        add_source(paths["source"], "moge")
        if self.v3:
            add_source(paths["flex_gemm_source"], "flex_gemm")
            add_source(paths["utils3d_source"], "utils3d_moge")
            os.environ["FLEX_GEMM_AUTOTUNE_MODE"] = "never"
            if paths.get("cache_dir"):
                cache = Path(paths["cache_dir"]).expanduser().resolve()
                for key, suffix in (("TRITON_CACHE_DIR", "triton"),
                                    ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                                    ("FLEX_GEMM_AUTOTUNE_CACHE_PATH", "flex_gemm/autotune_cache.json")):
                    path = cache / suffix
                    (path.parent if path.suffix else path).mkdir(parents=True, exist_ok=True)
                    os.environ[key] = str(path)
            if paths.get("ptxas"):
                compiler = Path(paths["ptxas"]).expanduser().resolve(strict=True)
                os.environ["TRITON_PTXAS_PATH"] = str(compiler)
        super().__init__(name, paths, device)
        if self.v3:
            if self.device.type != "cuda":
                raise ValueError("MoGe3 sparse refinement requires CUDA")
            from moge.model.v3 import MoGeModel
            checkpoint = self.torch.load(paths["weights"], map_location="cpu", weights_only=True, mmap=True)
            config = checkpoint["model_config"]
            if "vitg" not in config["encoder"]["backbone"] or not config.get("refiner"):
                raise ValueError("Expected MoGe3 ViT-G with the sparse refiner")
            self.model = MoGeModel(**config)
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.metadata.update(refine_steps=3, force_projection=True,
                                 known_recovery="fullk_whole, one fitted z shift; fixed raw metric scale")
        else:
            from moge.model import import_model_class_by_version
            self.model = import_model_class_by_version("v2").from_pretrained(paths["weights"])
        self.model.to(self.device).eval().requires_grad_(False)
        self.metadata.update(precision="official use_fp16=True", base_roi_tokens=2500,
                             unknown_token_ceiling=12000, apply_mask=False)


    def batch_key(self, request):
        g = request.geometry.direct() if request.baseline else request.geometry
        known = request.setting == "known"
        return (tuple(request.rgb.shape), request.setting, token_budget(g, known))

    def preprocess(self, requests):
        if not requests:
            raise ValueError("MoGe inference requires at least one image")
        key = self.batch_key(requests[0])
        if any(self.batch_key(request) != key for request in requests[1:]):
            raise ValueError("MoGe batches require matching image shapes, camera settings and token budgets")
        items = []
        for request in requests:
            g = request.geometry.direct() if request.baseline else request.geometry
            known = request.setting == "known"
            k = camera_on_canvas(request.camera, g, full_block=self.v3) if known else None
            seed = None
            if self.v3:
                arm = "reference_calibrated_direct" if request.arm == "reference" else request.arm
                seed = stable_seed(f"moge3/{request.uid}/{arm}")
            items.append(dict(request=request, geometry=g, known=known, camera=k, seed=seed))
        tokens = token_budget(items[0]["geometry"], items[0]["known"])
        args = dict(num_tokens=tokens, apply_mask=False, use_fp16=True, fov_x=None)
        if not self.v3 and items[0]["known"]:
            fovs = [_fov(item["camera"], item["geometry"].canvas_width) for item in items]
            # Preserve the scalar API for batch=1. Native MoGe accepts a B-vector
            # for distinct per-image FOVs in a real image batch.
            args["fov_x"] = fovs[0] if len(items) == 1 else self.torch.as_tensor(
                fovs, dtype=self.torch.float32, device=self.device)
        if self.v3:
            args.update(force_projection=True, refine_steps=3, return_per_step=False)
        images = _float_images([request.rgb for request in requests], self.device, self.torch)
        return dict(items=items, images=images, args=args, tokens=tokens)

    def infer_model(self, prepared):
        items = prepared["items"]
        raw = {}
        handles = []
        original = None
        seed = items[0]["seed"]
        if self.v3:
            # A tensor batch has one RNG stream. Keep the historical first-item
            # seed and report this explicitly instead of claiming per-item replay.
            set_seed(seed, self.torch)
            calls = []
            handles.append(self.model.refiner.register_forward_hook(lambda *a: calls.append(1)))
            if items[0]["known"]:
                original = self.model.forward
                def captured(*a, **kw):
                    value = original(*a, **kw)
                    if raw:
                        raise ValueError("Expected a single MoGe3 raw forward")
                    raw["output"] = value
                    return value
                self.model.forward = captured
        try:
            output = self.model.infer(prepared["images"], **prepared["args"])
        finally:
            for handle in handles:
                handle.remove()
            if original is not None:
                self.model.forward = original
        if self.v3 and len(calls) != 3:
            raise ValueError(f"Expected 3 actual MoGe3 refiner calls, got {len(calls)}")
        return dict(output=output, raw=raw, seed=seed)

    def postprocess(self, result, prepared):
        items = prepared["items"]
        output, raw, seed = result["output"], result["raw"], result["seed"]
        full_k = self.v3 and items[0]["known"]
        depths = None if full_k else _array(output["depth"])
        intrinsics = None if items[0]["known"] else _array(output["intrinsics"])
        predictions = []
        for index, item in enumerate(items):
            g, known, k = item["geometry"], item["known"], item["camera"]
            meta = dict(num_tokens=prepared["tokens"], inference_seed=seed)
            if len(items) > 1:
                meta.update(batch_size=len(items), batch_index=index,
                            batching="independent images in one native forward")
                if self.v3:
                    meta.update(batch_rng_policy="first_request_seed", requested_inference_seed=item["seed"])
            camera = None
            if full_k:
                # Full-K geometry recovery remains per image; the expensive
                # encoder and all three refinement steps ran once for the batch.
                per_item = {name: value[index:index + 1] if isinstance(value, self.torch.Tensor) else value
                            for name, value in raw["output"].items()}
                depth, fit = self._full_k_depth(per_item, k, g)
                meta.update(fit, GT_camera_in_neural_forward=False)
            else:
                depth = input_depth(depths[index], g)
            if not known:
                native_k = intrinsics[index]
                focal = float(native_k[0, 0]) * g.canvas_width
                if not math.isfinite(focal) or focal <= 0:
                    raise ValueError("MoGe returned an invalid focal")
                pixel_k = native_k.astype(np.float64, copy=True)
                pixel_k[0] *= g.canvas_width
                pixel_k[1] *= g.canvas_height
                camera = CameraEstimate(focal_px=np.asarray([focal, focal], np.float64),
                                        intrinsics=pixel_k, source_intrinsics=native_k.copy())
                meta["intrinsics_normalized"] = native_k.tolist()
            predictions.append(Prediction(depth=depth, camera=camera, metadata=meta))
        return predictions

    def _full_k_depth(self, raw, k, g):
        import torch.nn.functional as F
        from moge.utils.geometry_numpy import solve_optimal_shift
        from moge.utils.geometry_torch import normalized_view_plane_uv
        points = raw["points"].float()
        if tuple(points.shape) != (1, g.canvas_height, g.canvas_width, 3):
            raise ValueError("MoGe3 raw points do not use the input canvas")
        uv = normalized_view_plane_uv(g.canvas_width, g.canvas_height, dtype=points.dtype, device=points.device)
        points64 = _array(F.interpolate(points.permute(0, 3, 1, 2), (64, 64), mode="nearest").permute(0, 2, 3, 1))[0]
        uv64 = _array(F.interpolate(uv[None].permute(0, 3, 1, 2), (64, 64), mode="nearest").permute(0, 2, 3, 1))[0]
        binary = raw.get("mask")
        mask = np.ones((64, 64), bool) if binary is None else (
            F.interpolate((binary > .5).float()[:, None], (64, 64), mode="nearest")[0, 0] > 0).cpu().numpy()
        if mask.sum() < 2:
            raise ValueError("MoGe3 full-K recovery needs at least two model-supported samples")
        aspect = g.canvas_width / g.canvas_height
        fov_tensor = self.torch.as_tensor(_fov(k, g.canvas_width), device=self.device, dtype=self.torch.float32)
        focal = np.float32(float((aspect / (1 + aspect**2)**.5 /
                                 self.torch.tan(self.torch.deg2rad(fov_tensor / 2))).cpu()))
        kd = k.astype(np.float64)
        diagonal = math.hypot(g.canvas_width, g.canvas_height)
        offset = np.asarray([2 * (kd[0, 2] - g.canvas_width / 2) / diagonal,
                             2 * (kd[1, 2] - g.canvas_height / 2) / diagonal], np.float32)
        matrix = np.asarray(2 * kd[:2, :2] / diagonal / float(focal), np.float32)
        target = np.asarray(uv64 - offset, np.float32)
        transformed = points64.copy()
        if k[0, 1] == 0 and k[1, 0] == 0:
            transformed[..., :2] *= np.diag(matrix)
        else:
            transformed[..., :2] = points64[..., :2] @ matrix.T
        if not np.isfinite(target[mask]).all() or not np.isfinite(transformed[mask]).all():
            raise ValueError("Nonfinite MoGe3 full-K fit support")
        shift = np.asarray(solve_optimal_shift(target[mask], transformed[mask], focal), dtype=np.float32)
        if shift.size != 1 or not np.isfinite(shift).all():
            raise ValueError("MoGe3 full-K z-shift recovery failed")
        shift = np.float32(shift.reshape(()))
        xyz = transformed[mask].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            error = float(focal) * xyz[:, :2] / (xyz[:, 2] + float(shift))[:, None] - target[mask].astype(np.float64)
        if not np.isfinite(error).all():
            raise ValueError("Nonfinite MoGe3 projection at recovered z shift")
        scale = np.float32(float(raw["metric_scale"].float().cpu()[0]))
        z = _array(points)[0, :, :, 2]
        depth = (z + shift) * scale
        return np.ascontiguousarray(depth, dtype=np.float32), dict(
            method="fullk_whole", z_shift=float(shift), metric_scale=float(scale),
            focal_fixed=float(focal), K_input=k.tolist(), fit_support_count=int(mask.sum()))



class MoGe2(_MoGe):
    """MoGe2: optional GT horizontal FOV and native metric-scale inference."""

    v3 = False
    source_files = ("moge/model/v2.py",)
    matmul_tf32 = (False, True)


class MoGe3(_MoGe):
    """MoGe3 paper settings, including its explicit full-K known-camera rule."""

    v3 = True
    source_files = ("moge/model/v3.py",)
    reset_seed_per_input = False
