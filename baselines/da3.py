"""DA3 adapters preserving native scaling and the reference-camera replay.

Independent images occupy the batch dimension with one view each. Neural
inference is batched; DA3's native camera calibration, alignment and sky
recovery run separately for each image with its own RNG state.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import math
import types
import numpy as np

from contextdepth.baseline import DepthBaseline, set_seed
from contextdepth.model_compat import add_source, da3_resize_transform
from contextdepth.types import CameraEstimate, Prediction
from ._common import input_depth, stable_seed


@contextmanager
def _method_scope(owner, attribute, replacement):
    """Override one worker-owned method, preserving any previous override."""
    had_override = attribute in owner.__dict__
    override = owner.__dict__.get(attribute)
    setattr(owner, attribute, types.MethodType(replacement, owner))
    try:
        yield
    finally:
        if had_override:
            setattr(owner, attribute, override)
        else:
            delattr(owner, attribute)


@contextmanager
def _nested_camera_scope(nested, g, resolution, known):
    original = nested._apply_metric_scaling
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
    with _method_scope(nested, "_apply_metric_scaling", intercept):
        yield captured
        if not captured:
            raise ValueError("Native Nested recovery exposed no predicted camera")


@contextmanager
def _defer_recovery(network, nested):
    """Run the original neural forward while deferring cross-sample reductions."""
    captured = {}

    def identity(this, output, *unused):
        return output

    def capture_metric(this, output, metric_output):
        if "metric" in captured:
            raise ValueError("Expected one Nested metric branch per batch")
        captured["metric"] = metric_output
        return output

    with ExitStack() as scopes:
        branches = (network.da3, network.da3_metric) if nested else (network,)
        for branch in branches:
            scopes.enter_context(_method_scope(branch, "_process_mono_sky_estimation", identity))
        if nested:
            scopes.enter_context(_method_scope(network, "_apply_metric_scaling", capture_metric))
            scopes.enter_context(_method_scope(network, "_apply_depth_alignment", identity))
            scopes.enter_context(_method_scope(network, "_handle_sky_regions", identity))
        yield captured
        if nested and "metric" not in captured:
            raise ValueError("Nested forward did not expose its metric branch")


def _slice_batch(value, index, count, torch):
    """Keep the B=1 dimension expected by native recovery and output conversion."""
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.clone()
        if value.shape[0] != count:
            raise ValueError(f"Unexpected DA3 output batch shape: {tuple(value.shape)}")
        return value[index:index + 1].clone()
    if isinstance(value, dict):
        return type(value)({key: _slice_batch(item, index, count, torch) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return type(value)(_slice_batch(item, index, count, torch) for item in value)
    return value



class _DA3(DepthBaseline):
    source_files = ("depth_anything_3/api.py", "depth_anything_3/model/da3.py",
                    "depth_anything_3/utils/io/input_processor.py")
    matmul_tf32 = (True, True)
    supports_batch = True
    supports_setting_reuse = True

    def __init__(self, name, paths, device):
        add_source(paths["source"], "depth_anything_3")
        super().__init__(name, paths, device)
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(str(paths["weights"]), local_files_only=True).to(self.device).eval()
        self.model.requires_grad_(False)
        self.metadata.update(precision="official BF16 when supported, otherwise FP16",
                             base_process_res=504, process_res_method="upper_bound_resize",
                             neural_camera_input=False,
                             checkpoint_family="DA3Nested Giant-Large" if self.nested else "DA3Metric Large")

    def _item(self, request):
        known, g = self._validate(request)
        resolution = 504 if request.baseline else int(round(504 * max(g.canvas_width, g.canvas_height) / max(g.roi_width, g.roi_height)))
        if self.nested and known and not request.baseline:
            raise ValueError("DA3Nested known-camera recovery is reference calibration on Direct RGB only")
        return dict(request=request, geometry=g, known=known, resolution=resolution,
                    camera=request.camera if known else None)

    def batch_key(self, request):
        item = self._item(request)
        return (item["known"], item["resolution"], *item["geometry"].as_dict().values())

    def preprocess(self, requests):
        if not requests:
            raise ValueError("DA3 preprocessing requires at least one image")
        keys = [self.batch_key(request) for request in requests]
        if any(key != keys[0] for key in keys[1:]):
            raise ValueError("DA3 batches must share camera setting, geometry and processing resolution")
        items = [self._item(request) for request in requests]
        images_cpu, ex, intr = self.model._preprocess_inputs(
            [request.rgb for request in requests], None, None, items[0]["resolution"], "upper_bound_resize")
        images, ex_t, in_t = self.model._prepare_model_inputs(images_cpu, ex, intr)
        # The convenience API creates one scene with N views. Move independent
        # images to B instead: attention never joins different batch elements.
        if images.ndim != 5 or images.shape[:2] != (1, len(items)):
            raise ValueError(f"Unexpected DA3 preprocessing shape: {tuple(images.shape)}")
        if len(items) > 1:
            images = images.transpose(0, 1).contiguous()
        ex_norm = self.model._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
        return dict(items=items, images=images, extrinsics=ex_norm, intrinsics=in_t)

    def _restore_item_rng(self, item):
        request = item["request"]
        self.restore_request_rng(request)
        seed = None
        if self.nested and item["known"]:
            arm = "direct" if request.arm == "reference" else request.arm
            seed = stable_seed(f"nested-camera-pilot-v1:{request.uid}:{arm}")
            set_seed(seed, self.torch)
        return seed

    def infer_model(self, prepared):
        items = prepared["items"]
        seed = self._restore_item_rng(items[0])

        def forward():
            return self.model._run_model_forward(
                prepared["images"], prepared["extrinsics"], prepared["intrinsics"],
                [], False, False, "saddle_balanced")

        if len(items) == 1:
            # Preserve the complete native singleton recovery sequence.
            capture = {}
            if self.nested:
                item = items[0]
                with _nested_camera_scope(self.model.model, item["geometry"], item["resolution"], item["camera"]) as capture:
                    output = forward()
            else:
                output = forward()
            return dict(output=output, capture=capture, seed=seed, batched=False)
        with _defer_recovery(self.model.model, self.nested) as deferred:
            output = forward()
        return dict(output=output, metric=deferred.get("metric"), batched=True)

    def _recover_one(self, output, metric, item):
        network = self.model.model
        capture = {}
        seed = self._restore_item_rng(item)
        # The native forward runs these operations inside this same autocast
        # context, after its explicitly-float32 prediction heads.
        dtype = self.torch.bfloat16 if self.torch.cuda.is_bf16_supported() else self.torch.float16
        with self.torch.autocast(device_type=self.device.type, dtype=dtype):
            if self.nested:
                output = network.da3._process_mono_sky_estimation(output)
                metric = network.da3_metric._process_mono_sky_estimation(metric)
                with _nested_camera_scope(network, item["geometry"], item["resolution"], item["camera"]) as capture:
                    output = network._apply_metric_scaling(output, metric)
                    output = network._apply_depth_alignment(output, metric)
                    output = network._handle_sky_regions(output, metric)
            else:
                output = network._process_mono_sky_estimation(output)
        return output, capture, seed

    def postprocess(self, raw_output, prepared):
        items = prepared["items"]
        results = []
        for index, item in enumerate(items):
            if raw_output["batched"]:
                output = _slice_batch(raw_output["output"], index, len(items), self.torch)
                metric = _slice_batch(raw_output["metric"], index, len(items), self.torch) if self.nested else None
                output, capture, seed = self._recover_one(output, metric, item)
            else:
                output, capture, seed = raw_output["output"], raw_output["capture"], raw_output["seed"]
            result = self.model._convert_to_prediction(output)
            results.append(self._format_one(result, capture, seed, item))
        return results

    def _format_one(self, result, capture, seed, item):
        g, known, resolution = item["geometry"], item["known"], item["resolution"]
        camera, meta = None, dict(process_res=resolution)
        raw = np.asarray(result.depth[0], dtype=np.float32)
        if self.nested:
            if not known:
                # Validate the official two-stage integer resize. Keep its
                # native K and each scale separately, so callers can preserve
                # operation order when mapping into a legacy evaluation ROI.
                da3_resize_transform(g, resolution, raw.shape)
                scale = resolution / float(max(g.canvas_width, g.canvas_height))
                mw = max(1, int(round(g.canvas_width * scale)))
                mh = max(1, int(round(g.canvas_height * scale)))
                stages = ((mw / float(g.canvas_width), mh / float(g.canvas_height)),
                          (raw.shape[1] / float(mw), raw.shape[0] / float(mh)))
                transform = np.eye(3, dtype=np.float32)
                for sx, sy in stages:
                    transform[:1] *= sx
                    transform[1:2] *= sy
                native_k = capture["K_pred_raw"]
                input_k = np.linalg.solve(transform.astype(np.float64), native_k.astype(np.float64))
                camera = CameraEstimate(intrinsics=input_k, source_intrinsics=native_k,
                                        resize_factors=stages)
                meta["K_pred_raw"] = native_k.tolist()
            else:
                meta.update(reference_focal_raw_px=capture["reference_focal_raw_px"], alpha=capture["alpha"],
                            reference_recovery="native focal scaling, native confidence/LS, native sky")
        else:
            k = np.asarray(item["camera"], dtype=np.float32)
            fx_roi = float(k[0, 0]) * g.roi_width / g.fixed_w
            fy_roi = float(k[1, 1]) * g.roi_height / g.fixed_h
            focal_raw = .5 * (fx_roi * raw.shape[1] / g.canvas_width + fy_roi * raw.shape[0] / g.canvas_height)
            raw = raw * (focal_raw / 300.0)
            meta["focal_raw_px"] = focal_raw
        meta.update(raw_hw=list(raw.shape), inference_seed=seed)
        return Prediction(input_depth(raw, g), camera=camera, metadata=meta)


class DA3Nested(_DA3):
    nested = True
    accepts_reference_camera = True


class DA3Metric(_DA3):
    nested = False
    allowed_settings = ("known",)

    def __init__(self, name, paths, device):
        if paths["setting"] != "known":
            raise ValueError("DA3 metric recovery requires a known camera")
        super().__init__(name, paths, device)
