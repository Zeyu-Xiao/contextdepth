"""Inference-only adapters reproducing the ContextDepth model settings.

The public boundary contains RGB, pixel geometry and optional calibration, but
never depth labels or evaluation masks. Upstream code and weights are explicit
configuration inputs. Different MoGe versions need separate worker processes.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import types

import numpy as np

from .model_compat import (
    add_source, camera_on_canvas, compensate_unidepth_camera,
    da3_resize_transform, equivalent_ray_camera, lazy_training_imports,
    matched_constraints, restore_rng_state, shape_scope, unidepth_geometry, unidepth_padding_scope,
)
from .types import Geometry, InferenceInput, Prediction


MODEL_NAMES = (
    "da3nested", "da3metric_gt_focal", "moge2", "moge2_gt_fov",
    "moge3_unknown", "moge3_known", "depth_pro", "depth_pro_gt_focal",
    "unidepth_v2_vitl14", "unidepth_gt_k", "unik3d_unknown", "unik3d_known",
)
KNOWN_NAMES = frozenset(("da3metric_gt_focal", "moge2_gt_fov", "moge3_known",
                         "depth_pro_gt_focal", "unidepth_gt_k", "unik3d_known",
                         "da3nested_reference"))


def _source_identity(name, paths):
    if name.startswith("da3"):
        files = ("depth_anything_3/api.py", "depth_anything_3/model/da3.py", "depth_anything_3/utils/io/input_processor.py")
    elif name.startswith("moge"):
        files = (f"moge/model/{'v3' if name.startswith('moge3') else 'v2'}.py",)
    elif name.startswith("depth_pro"):
        files = ("depth_pro/depth_pro.py",)
    elif name.startswith("unidepth"):
        files = ("unidepth/models/unidepthv2/unidepthv2.py",)
    else:
        files = ("unik3d/models/unik3d.py", "unik3d/utils/camera.py")
    actual = {name: hashlib.sha256((Path(paths["source"]) / name).read_bytes()).hexdigest() for name in files}
    for relative, expected in paths.get("source_sha256", {}).items():
        observed = hashlib.sha256((Path(paths["source"]) / relative).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Pinned model source differs: {relative}")
        actual[relative] = observed
    return actual


def role_seed(model: str, uid: str, arm: str, setting: str | None = None) -> int:
    if arm == "reference":
        arm = "reference_calibrated_direct" if model == "moge3" else "direct"
    if model == "da3nested_reference":
        text = f"nested-camera-pilot-v1:{uid}:{arm}"
    else:
        text = f"{model}/{uid}/{arm}"
        if model == "unik3d":
            text += f"/{setting}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2**31)


def _set_seed(value, torch):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def token_budget(g: Geometry, known: bool) -> int:
    if known:
        return int(round(2500.0 * g.canvas_width * g.canvas_height / float(g.roi_width * g.roi_height)))
    fraction = max(float((g.roi_width * g.roi_height) / max(float(g.canvas_height * g.canvas_width), 1.0)), 1e-6)
    return max(2500, min(12000, int(round(2500.0 / fraction))))


def _fov(k, width):
    return float(np.rad2deg(2.0 * np.arctan(.5 * width / float(k[0, 0]))))


def _float_image(rgb, device, torch):
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def _array(tensor):
    return tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)


def fixed_depth(raw, g: Geometry, *, known: bool, full_always=False, fixed_always=False):
    """Preserve both original OpenCV interpolation stages and their conditions."""
    import cv2
    depth = np.asarray(raw, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a two-dimensional depth map, got {depth.shape}")
    if full_always or depth.shape != (g.canvas_height, g.canvas_width):
        depth = cv2.resize(depth, (g.canvas_width, g.canvas_height), interpolation=cv2.INTER_LINEAR)
    roi = depth[g.roi_slices]
    if fixed_always or not known or roi.shape != (g.fixed_h, g.fixed_w):
        roi = cv2.resize(roi, (g.fixed_w, g.fixed_h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(roi, dtype=np.float32)


class _Adapter:
    # Official APIs accepting BCHW do not establish same-seed, independent-image
    # numerical parity. The public runner can use separate workers instead.
    max_batch_size = 1

    def __init__(self, name, paths, device):
        import torch
        self.name, self.paths, self.torch = name, dict(paths), torch
        self.device = torch.device(device)
        self.known = name in KNOWN_NAMES
        self._busy = False
        # These switches differ across the historical wrappers. The additional
        # MoGe3/UniK3D experiments explicitly required torch's False/True TF32
        # defaults, whereas old known-camera wrappers and both DA3 routes set
        # CUDA matmul TF32 on. Do not unify them across model settings.
        matmul_tf32 = name.startswith("da3") or name in (
            "moge2_gt_fov", "depth_pro_gt_focal", "unidepth_gt_k")
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        # Do not serialize configured filesystem locations into prediction
        # records. Revision and source hashes below identify the implementation
        # without exposing local filesystem paths.
        self.metadata = dict(model=name, batching="sequential independent images",
                             weights_dtype="float32", depth_semantics="camera Z",
                             query_GT_depth_used=False, output_mask_applied=False,
                             input_resized_before_official_preprocessing=False)
        self.metadata["numerical_flags"] = dict(cuda_matmul_tf32=matmul_tf32, cudnn_tf32=True,
                                                cudnn_benchmark=False, cudnn_deterministic=False,
                                                deterministic_algorithms=False)
        try:
            revision = subprocess.check_output(["git", "-C", str(paths["source"]), "rev-parse", "HEAD"],
                                               text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        self.metadata["source_git_revision"] = revision
        self.metadata["source_files_sha256"] = _source_identity(name, paths)
        self.rng_policy = paths.get("rng_policy", "per_input")
        if self.rng_policy not in ("per_input", "caller"):
            raise ValueError("rng_policy must be per_input or caller")
        self.base_seed = int(paths.get("seed", os.environ.get("DEPTH_EVAL_SEED", "20260621")))

    def _validate(self, request):
        known = self.known or (self.name == "da3nested" and request.setting == "known")
        if request.setting not in ("unknown", "known"):
            raise ValueError("setting must be unknown or known")
        if request.setting != ("known" if known else "unknown"):
            raise ValueError(f"{self.name} requires setting={'known' if known else 'unknown'}")
        if known != (request.camera is not None):
            raise ValueError("Only known-camera inference receives a camera, and requires it")
        g = request.geometry.direct() if request.baseline else request.geometry
        values = g.as_dict()
        if (any(isinstance(v, (bool, np.bool_)) or int(v) != v for v in values.values())
                or min(g.fixed_w, g.fixed_h, g.roi_width, g.roi_height, g.canvas_width, g.canvas_height) < 1
                or min(g.roi_x, g.roi_y) < 0
                or g.roi_x + g.roi_width > g.canvas_width or g.roi_y + g.roi_height > g.canvas_height):
            raise ValueError("Invalid visible-ROI geometry")
        rgb = np.asarray(request.rgb)
        if rgb.dtype != np.uint8 or rgb.shape != (g.canvas_height, g.canvas_width, 3):
            raise ValueError("RGB must be uint8 HWC matching the declared input canvas")
        return known, g

    def predict(self, requests: list[InferenceInput]) -> list[Prediction]:
        if self._busy:
            raise RuntimeError("A model worker is sequential and cannot accept overlapping calls")
        self._busy = True
        try:
            # A worker may also own its model's calibrated reference adapter.
            # Restore this setting's process-global switches on every call so
            # constructing/calling that reference cannot contaminate queries.
            flags = self.metadata["numerical_flags"]
            self.torch.backends.cuda.matmul.allow_tf32 = flags["cuda_matmul_tf32"]
            self.torch.backends.cudnn.allow_tf32 = flags["cudnn_tf32"]
            self.torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]
            self.torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
            self.torch.use_deterministic_algorithms(flags["deterministic_algorithms"])
            results = []
            for request in requests:
                known, g = self._validate(request)
                supplied_rng = getattr(request, "rng_state", None)
                if supplied_rng is not None:
                    restore_rng_state(supplied_rng, self.torch, self.device)
                elif not self.name.startswith(("unik3d", "moge3")) and self.rng_policy == "per_input":
                    _set_seed(self.base_seed, self.torch)
                with self.torch.inference_mode():
                    result = self._predict_one(request, g, known)
                if result.depth.dtype != np.float32 or result.depth.shape != (g.fixed_h, g.fixed_w):
                    raise ValueError("Adapter output violates the fixed-grid float32 depth contract")
                result.metadata = dict(self.metadata, geometry=g.as_dict(),
                                       camera_setting="known" if known else "unknown",
                                       **result.metadata)
                result.metadata["legacy_rng_policy"] = "supplied_before_input_state" if supplied_rng is not None else self.rng_policy
                result.metadata["legacy_base_seed"] = self.base_seed
                results.append(result)
            return results
        finally:
            self._busy = False

    def close(self):
        pass


class _MoGe(_Adapter):
    def __init__(self, name, paths, device):
        self.v3 = name.startswith("moge3")
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

    def _predict_one(self, request, g, known):
        k = camera_on_canvas(request.camera, g, full_block=self.v3) if known else None
        tokens = token_budget(g, known)
        args = dict(num_tokens=tokens, apply_mask=False, use_fp16=True,
                    fov_x=None if self.v3 or not known else _fov(k, g.canvas_width))
        raw = {}
        handles = []
        original = None
        seed = None
        if self.v3:
            seed = role_seed("moge3", request.uid, request.arm)
            _set_seed(seed, self.torch)
            args.update(force_projection=True, refine_steps=3, return_per_step=False)
            calls = []
            handles.append(self.model.refiner.register_forward_hook(lambda *a: calls.append(1)))
            if known:
                original = self.model.forward
                def captured(*a, **kw):
                    value = original(*a, **kw)
                    if raw:
                        raise ValueError("Expected a single MoGe3 raw forward")
                    raw["output"] = value
                    return value
                self.model.forward = captured
        try:
            output = self.model.infer(_float_image(request.rgb, self.device, self.torch), **args)
        finally:
            for handle in handles:
                handle.remove()
            if original is not None:
                self.model.forward = original
        if self.v3 and len(calls) != 3:
            raise ValueError(f"Expected 3 actual MoGe3 refiner calls, got {len(calls)}")
        meta = dict(num_tokens=tokens, inference_seed=seed)
        camera = None
        if self.v3 and known:
            depth, fit = self._full_k_depth(raw.pop("output"), k, g)
            meta.update(fit, GT_camera_in_neural_forward=False)
        else:
            depth = fixed_depth(_array(output["depth"])[0], g, known=known, full_always=not known)
        if not known:
            native_k = _array(output["intrinsics"])[0]
            focal = float(native_k[0, 0]) * g.canvas_width
            if not math.isfinite(focal) or focal <= 0:
                raise ValueError("MoGe returned an invalid focal")
            camera = np.array([math.log(focal / g.roi_width), math.log(focal / g.roi_height)])
            meta["intrinsics_normalized"] = native_k.tolist()
        return Prediction(depth, camera, meta)

    def _full_k_depth(self, raw, k, g):
        import cv2
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
        z = _array(points)[0, :, :, 2][g.roi_slices]
        depth = (z + shift) * scale
        if depth.shape != (g.fixed_h, g.fixed_w):
            depth = cv2.resize(depth, (g.fixed_w, g.fixed_h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(depth, dtype=np.float32), dict(
            method="fullk_whole", z_shift=float(shift), metric_scale=float(scale),
            focal_fixed=float(focal), K_input=k.tolist(), fit_support_count=int(mask.sum()))


class _DepthPro(_Adapter):
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

    def _predict_one(self, request, g, known):
        k = camera_on_canvas(request.camera, g) if known else None
        focal = None if k is None else self.torch.as_tensor(float(k[0, 0]), dtype=self.torch.float32, device=self.device)
        # Input is already canonical uint8 RGB. The historical temporary PNG
        # carried no EXIF/ICC, so official transform receives these exact pixels.
        output = self.model.infer(self.transform(np.array(request.rgb, copy=True)), f_px=focal)
        raw = np.squeeze(_array(output["depth"]))
        reported = float(output["focallength_px"].detach().float().cpu().item())
        camera = None if known else np.array([math.log(reported / g.roi_width)])
        return Prediction(fixed_depth(raw, g, known=known), camera, dict(focallength_px=reported))


class _UniDepth(_Adapter):
    def __init__(self, name, paths, device):
        add_source(paths["source"], "unidepth")
        super().__init__(name, paths, device)
        with lazy_training_imports():
            from unidepth.models import UniDepthV2
        self.implementation = importlib.import_module("unidepth.models.unidepthv2.unidepthv2")
        weights = Path(paths["weights"])
        config_path = Path(paths.get("config", weights / "config.json"))
        checkpoint = weights / "pytorch_model.bin" if weights.is_dir() else weights
        config = json.loads(config_path.read_text())
        self.model = UniDepthV2(config)
        state = self.torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.metadata.update(precision="official FP16 autocast", native_shape_constraints=dict(self.model.shape_constraints),
                             budget="native bounds with compensated K" if self.known else "matched ROI pixels",
                             padding="ceil outside ratio bounds" if self.known else "historical floor; negative in-range padding repaired")

    def _predict_one(self, request, g, known):
        native = self.model.shape_constraints
        active = native if known else matched_constraints(native, g)
        tensor = self.torch.from_numpy(np.array(request.rgb, copy=True)).permute(2, 0, 1).to(self.device)
        with unidepth_padding_scope(self.implementation, known=known), shape_scope(self.model, active, remove_level=not known):
            prep = unidepth_geometry(self.implementation, self.model, g)
            k = camera_on_canvas(request.camera, g) if known else None
            calibrated = compensate_unidepth_camera(k, prep) if known else None
            camera = None if calibrated is None else self.torch.from_numpy(calibrated).to(self.device)
            output = self.model.infer(tensor, camera=camera)
        vector = None
        reported = _array(output["intrinsics"])[0]
        if not known:
            # These are the exact old API-to-actual integer-grid focal factors.
            nominal = prep["nominal_scale"]
            ph, pw = prep["padded_hw"]
            ih = math.ceil(int(ph * nominal) / 14) * 14
            iw = math.ceil(int(pw * nominal) / 14) * 14
            fx = float(reported[0, 0]) * (nominal / (iw / pw))
            fy = float(reported[1, 1]) * (nominal / (ih / ph))
            vector = np.array([math.log(fx / g.roi_width), math.log(fy / g.roi_height)])
        return Prediction(fixed_depth(np.squeeze(_array(output["depth"])), g, known=known), vector,
                          dict(preprocessing=prep, active_shape_constraints=active,
                               intrinsics_api=reported.tolist(),
                               K_input=None if k is None else k.tolist(),
                               compensated_K=None if calibrated is None else calibrated.tolist()))


class _UniK3D(_Adapter):
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

    def _predict_one(self, request, g, known):
        seed = role_seed("unik3d", request.uid, request.arm, "known" if known else "unknown")
        _set_seed(seed, self.torch)
        tensor = self.torch.from_numpy(np.array(request.rgb, copy=True)).permute(2, 0, 1).to(self.device)
        k = camera_on_canvas(request.camera, g) if known else None
        camera = None if k is None else self.Pinhole(K=self.torch.from_numpy(k).unsqueeze(0).to(self.device))
        active = matched_constraints(self.model.shape_constraints, g)
        with shape_scope(self.model, active, interpolation="bilinear"):
            output = self.model.infer(tensor, camera=camera, normalize=True)
        depth = _array(output["depth"])
        points = _array(output["points"])
        if depth.shape != (1, 1, g.canvas_height, g.canvas_width) or not np.array_equal(depth, points[:, 2:3], equal_nan=True):
            raise ValueError("UniK3D depth must be native input-grid camera-space Z")
        vector, fit = (None, None) if known else equivalent_ray_camera(_array(output["rays"])[0], g)
        return Prediction(fixed_depth(depth[0, 0], g, known=known), vector,
                          dict(inference_seed=seed, active_shape_constraints=active,
                               equivalent_camera=fit, K_input=None if k is None else k.tolist()))


@contextmanager
def _nested_camera_scope(nested, g, resolution, known):
    attribute = "_apply_metric_scaling"
    original = getattr(nested, attribute)
    had_override = attribute in nested.__dict__
    override = nested.__dict__.get(attribute)
    captured = {}
    def intercept(this, output, metric_output):
        if captured:
            raise ValueError("Expected exactly one native Nested metric-recovery call")
        k = output["intrinsics"]
        matrix = np.array(k.detach().cpu().float().numpy(), copy=True).reshape(3, 3)
        raw_hw = tuple(output["depth"].shape[-2:])
        captured.update(K_pred_raw=matrix, raw_hw=raw_hw)
        if known is not None:
            # Same native focal scaling, confidence/LS alignment and sky recovery
            # as the original reference replay, with unchanged neural inputs.
            target = da3_resize_transform(g, resolution, raw_hw, camera=known)
            target_mean = float((target[0, 0] + target[1, 1]) / 2)
            native_mean = float(((k[..., 0, 0] + k[..., 1, 1]) / 2).item())
            alpha = target_mean / native_mean
            if not math.isfinite(alpha) or alpha <= 0:
                raise ValueError("Invalid reference mean-focal calibration")
            if alpha != 1.0:
                k[..., 0, 0] *= alpha
                k[..., 1, 1] *= alpha
            captured.update(reference_focal_raw_px=target_mean, alpha=alpha)
        return original(output, metric_output)
    setattr(nested, attribute, types.MethodType(intercept, nested))
    try:
        yield captured
        if not captured:
            raise ValueError("Native Nested recovery exposed no predicted camera")
    finally:
        if had_override:
            setattr(nested, attribute, override)
        else:
            delattr(nested, attribute)


class _DA3(_Adapter):
    def __init__(self, name, paths, device):
        add_source(paths["source"], "depth_anything_3")
        super().__init__(name, paths, device)
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(str(paths["weights"]), local_files_only=True).to(self.device).eval()
        self.model.requires_grad_(False)
        self.nested = name != "da3metric_gt_focal"
        self.metadata.update(precision="official BF16 when supported, otherwise FP16",
                             base_process_res=504, process_res_method="upper_bound_resize",
                             neural_camera_input=False,
                             checkpoint_family="DA3Nested Giant-Large" if self.nested else "DA3Metric Large")

    def _predict_one(self, request, g, known):
        resolution = 504 if request.baseline else int(round(504 * max(g.canvas_width, g.canvas_height) / max(g.roi_width, g.roi_height)))
        camera, seed, meta = None, None, dict(process_res=resolution)
        if self.nested:
            if known:
                if not request.baseline:
                    raise ValueError("DA3Nested known-camera recovery is reference calibration on Direct RGB only")
                seed = role_seed("da3nested_reference", request.uid, request.arm)
                _set_seed(seed, self.torch)
            with _nested_camera_scope(self.model.model, g, resolution, request.camera if known else None) as capture:
                images_cpu, ex, intr = self.model._preprocess_inputs([request.rgb], None, None, resolution, "upper_bound_resize")
                images, ex_t, in_t = self.model._prepare_model_inputs(images_cpu, ex, intr)
                ex_norm = self.model._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
                raw_output = self.model._run_model_forward(images, ex_norm, in_t, [], False, False, "saddle_balanced")
                result = self.model._convert_to_prediction(raw_output)
            raw = np.asarray(result.depth[0], dtype=np.float32)
            if not known:
                transform = da3_resize_transform(g, resolution, raw.shape)
                fixed_k = np.linalg.solve(transform.astype(np.float64), capture["K_pred_raw"].astype(np.float64))
                camera = np.array([math.log(float(fixed_k[0, 0]) / g.fixed_w), math.log(float(fixed_k[1, 1]) / g.fixed_h)])
                meta.update(K_pred_raw=capture["K_pred_raw"].tolist(), K_pred_fixed=fixed_k.tolist())
            else:
                meta.update(reference_focal_raw_px=capture["reference_focal_raw_px"], alpha=capture["alpha"],
                            reference_recovery="native focal scaling, native confidence/LS, native sky")
        else:
            result = self.model.inference([request.rgb], process_res=resolution, export_dir=None)
            raw = np.asarray(result.depth[0], dtype=np.float32)
            k = np.asarray(request.camera, dtype=np.float32)
            fx_roi = float(k[0, 0]) * g.roi_width / g.fixed_w
            fy_roi = float(k[1, 1]) * g.roi_height / g.fixed_h
            focal_raw = .5 * (fx_roi * raw.shape[1] / g.canvas_width + fy_roi * raw.shape[0] / g.canvas_height)
            raw = raw * (focal_raw / 300.0)
            meta["focal_raw_px"] = focal_raw
        meta.update(raw_hw=list(raw.shape), inference_seed=seed)
        return Prediction(fixed_depth(raw, g, known=known, full_always=True, fixed_always=True), camera, meta)


def create_model(name: str, paths: dict, device: str = "cuda:0"):
    """Construct one configured model setting, without downloads or environment changes.

    ``source`` is an import root and ``weights`` is a local checkpoint/directory.
    MoGe3 also requires ``flex_gemm_source`` and ``utils3d_source``. Its optional
    ``ptxas`` and ``cache_dir`` affect this worker process only. UniDepth/UniK3D
    accept an optional explicit ``config`` JSON path.
    """
    if name not in (*MODEL_NAMES, "da3nested_reference"):
        raise ValueError(f"Unknown model: {name}; expected one of {MODEL_NAMES}")
    paths = dict(paths)
    for key in ("source", "weights", "config", "flex_gemm_source", "utils3d_source", "ptxas", "cache_dir"):
        if paths.get(key):
            paths[key] = str(Path(paths[key]).expanduser().resolve())
    for key in ("source", "weights"):
        if not paths.get(key):
            raise ValueError(f"Model {name} requires a local {key} path")
        Path(paths[key]).expanduser().resolve(strict=True)
    if name.startswith("moge"):
        return _MoGe(name, paths, device)
    if name.startswith("depth_pro"):
        return _DepthPro(name, paths, device)
    if name.startswith("unidepth"):
        return _UniDepth(name, paths, device)
    if name.startswith("unik3d"):
        return _UniK3D(name, paths, device)
    return _DA3(name, paths, device)
