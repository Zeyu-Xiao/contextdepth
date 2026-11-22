"""The common depth contract, independent of model names and upstream APIs.

Adapters prepare their camera input, call their model, and format a metric
camera-Z prediction on the complete RGB input grid. Model implementations
own preprocessing and any explicitly retained experiment-specific recovery.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import random
import subprocess

import numpy as np

from .config import model_load_config
from .model_compat import restore_rng_state
from .types import CameraEstimate, InferenceInput, Prediction


def set_seed(value, torch):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


class DepthBaseline:
    """Sequential process-local adapter with three explicit inference stages.

    ``paths.setting`` selects known/unknown camera inference. ``name`` is only
    a result label. RGB is uint8 HWC; input K uses the fixed evaluation grid.
    Output depth is float32 camera-Z in metres on the complete input RGB grid. A
    camera estimate is optional here; selectors may impose a stricter contract.
    """

    source_files = ()
    matmul_tf32 = (False, False)  # unknown, known
    reset_seed_per_input = True
    accepts_reference_camera = False
    # True when unknown-camera inference on an outpainted candidate can reuse
    # the camera this adapter estimated on the unexpanded query
    # (InferenceInput.query_camera, produced as Prediction.camera_state).
    supports_query_camera = False
    supports_batch = False
    # Adapters opt in after ensuring setting-dependent state is refreshed by
    # configure(). Unmodified third-party subclasses retain separate instances.
    supports_setting_reuse = False
    allowed_settings = ("unknown", "known")
    max_batch_size = 1

    def __init__(self, name, paths, device):
        import torch
        self.name, self.paths, self.torch = name, deepcopy(dict(paths)), torch
        self._load_config = deepcopy(model_load_config(paths))
        self.options = deepcopy(paths.get("options", {}))
        self.max_batch_size = int(paths.get("batch_size", 1))
        if self.max_batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_batch_size > 1 and not self.supports_batch:
            raise ValueError(f"{type(self).__name__} has not implemented native batch inference")
        self.device = torch.device(device)
        self.setting = paths["setting"]
        if self.setting not in self.allowed_settings:
            raise ValueError(f"setting must be one of {self.allowed_settings}")
        self.known = self.setting == "known"
        self._busy = False
        self.metadata = dict(model=name, camera_setting=self.setting,
                             batching="independent images", requested_batch_size=self.max_batch_size,
                             weights_dtype="float32", depth_semantics="camera Z",
                             query_GT_depth_used=False, output_mask_applied=False,
                             input_resized_before_official_preprocessing=False)
        self.metadata["numerical_flags"] = self._numerical_flags()
        self._apply_numerical_flags()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        revision = None
        if paths.get("source"):
            try:
                revision = subprocess.check_output(
                    ["git", "-C", str(paths["source"]), "rev-parse", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                pass
        self.metadata["source_git_revision"] = revision
        hashes = {}
        for relative in self.source_files:
            hashes[relative] = hashlib.sha256((Path(paths["source"]) / relative).read_bytes()).hexdigest()
        for relative, expected in paths.get("source_sha256", {}).items():
            observed = hashlib.sha256((Path(paths["source"]) / relative).read_bytes()).hexdigest()
            if observed != expected:
                raise ValueError(f"Pinned model source differs: {relative}")
            hashes[relative] = observed
        self.metadata["source_files_sha256"] = hashes
        self.rng_policy = paths.get("rng_policy", "per_input")
        if self.rng_policy not in ("per_input", "caller"):
            raise ValueError("rng_policy must be per_input or caller")
        self.base_seed = int(paths.get("seed", os.environ.get("DEPTH_EVAL_SEED", "20260621")))

    def configure(self, name, paths):
        """Switch experiment identity and camera setting without loading weights.

        Every loading option except ``setting`` must match construction. This
        does not migrate devices, change precision/options or replace resources.
        """
        if self._busy:
            raise RuntimeError("Cannot configure a model while inference is running")
        setting = paths.get("setting")
        if setting not in self.allowed_settings:
            raise ValueError(f"setting must be one of {self.allowed_settings}")
        if model_load_config(paths) != self._load_config:
            raise ValueError("Model loading configuration changed; create a new model instance")
        if setting != self.setting and not self.supports_setting_reuse:
            raise RuntimeError(f"{type(self).__name__} does not support setting reuse")
        self.name, self.paths = name, deepcopy(dict(paths))
        self.setting, self.known = setting, setting == "known"
        # Replace nested flags instead of mutating metadata already attached
        # to predictions produced under the preceding experiment setting.
        self.metadata = dict(self.metadata, model=name, camera_setting=setting,
                             numerical_flags=self._numerical_flags())
        self._apply_numerical_flags()
        return self

    def _numerical_flags(self):
        return dict(
            cuda_matmul_tf32=self.options.get("matmul_tf32", self.matmul_tf32[int(self.known)]),
            cudnn_tf32=True, cudnn_benchmark=False, cudnn_deterministic=False,
            deterministic_algorithms=False)

    def _apply_numerical_flags(self):
        flags = self.metadata["numerical_flags"]
        self.torch.backends.cuda.matmul.allow_tf32 = flags["cuda_matmul_tf32"]
        self.torch.backends.cudnn.allow_tf32 = flags["cudnn_tf32"]
        self.torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]
        self.torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
        self.torch.use_deterministic_algorithms(flags["deterministic_algorithms"])

    def _validate(self, request):
        known = self.known or (self.accepts_reference_camera and request.setting == "known")
        if request.setting not in ("unknown", "known"):
            raise ValueError("setting must be unknown or known")
        if request.setting != ("known" if known else "unknown"):
            raise ValueError(f"{self.name} requires setting={'known' if known else 'unknown'}")
        if known != (request.camera is not None):
            raise ValueError("Only known-camera inference receives a camera, and requires it")
        if request.query_camera is not None:
            if not self.supports_query_camera:
                raise ValueError(f"{type(self).__name__} cannot reuse a query-estimated camera")
            if known or request.baseline:
                raise ValueError("A query-estimated camera applies to unknown-camera candidates only")
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

    def batch_key(self, request):
        """Adapters may add native resolution/precision constraints to this key."""
        return request.setting, tuple(request.rgb.shape)

    def preprocess(self, requests: list[InferenceInput]):
        """Prepare a compatible batch of RGB and optional camera inputs."""
        raise NotImplementedError

    def infer_model(self, prepared):
        """Run the model once for the prepared independent-image batch."""
        raise NotImplementedError

    def postprocess(self, output, prepared) -> list[Prediction]:
        """Return one complete input-grid prediction per request, in order."""
        raise NotImplementedError

    def restore_request_rng(self, request):
        if request.rng_state is not None:
            restore_rng_state(request.rng_state, self.torch, self.device)
        elif self.reset_seed_per_input and self.rng_policy == "per_input":
            set_seed(self.base_seed, self.torch)

    def _check_output(self, request, result):
        if (not isinstance(result, Prediction) or not isinstance(result.depth, np.ndarray)
                or result.depth.dtype != np.float32 or result.depth.shape != request.rgb.shape[:2]):
            raise ValueError("Adapter must return full-input-grid float32 depth; evaluation crops belong to the pipeline")
        camera = result.camera
        if camera is not None:
            if not isinstance(camera, CameraEstimate):
                raise ValueError("Camera output must be a CameraEstimate")
            if camera.focal_px is not None:
                focal = np.asarray(camera.focal_px)
                if focal.ndim != 1 or focal.size not in (1, 2) or not np.isfinite(focal).all() or (focal <= 0).any():
                    raise ValueError("Camera focal_px must contain one or two finite positive pixel focal lengths")
            for k in (camera.intrinsics, camera.source_intrinsics):
                if k is not None and (np.shape(k) != (3, 3) or not np.isfinite(k).all()):
                    raise ValueError("Camera intrinsics must be a finite 3x3 matrix")
            if camera.rays is not None and np.shape(camera.rays) != (3, *request.rgb.shape[:2]):
                raise ValueError("Camera rays must cover the full input RGB")
        return result

    def predict(self, requests: list[InferenceInput]) -> list[Prediction]:
        if self._busy:
            raise RuntimeError("A model worker cannot accept overlapping calls")
        if not requests:
            return []
        self._busy = True
        try:
            self._apply_numerical_flags()
            for request in requests:
                self._validate(request)
            # A reused query camera belongs to one image: never batch those calls.
            if self.max_batch_size == 1 or any(r.query_camera is not None for r in requests):
                batches = [[i] for i in range(len(requests))]
            else:
                groups = {}
                for i, request in enumerate(requests):
                    groups.setdefault(self.batch_key(request), []).append(i)
                batches = [indices[start:start + self.max_batch_size]
                           for indices in groups.values()
                           for start in range(0, len(indices), self.max_batch_size)]
            results = [None] * len(requests)
            for indices in batches:
                batch = [requests[i] for i in indices]
                self.restore_request_rng(batch[0])
                with self.torch.inference_mode():
                    prepared = self.preprocess(batch)
                    raw = self.infer_model(prepared)
                    predictions = list(self.postprocess(raw, prepared))
                if len(predictions) != len(batch):
                    raise ValueError("Adapter returned an incomplete batch")
                for index, request, prediction in zip(indices, batch, predictions):
                    self._check_output(request, prediction)
                    metadata = dict(self.metadata)
                    metadata.update(prediction.metadata)
                    metadata.update(geometry=request.geometry.as_dict(), camera_setting=request.setting,
                                    execution_batch_size=len(batch), output_grid="full RGB input",
                                    legacy_rng_policy="supplied_before_input_state" if request.rng_state is not None else self.rng_policy,
                                    legacy_base_seed=self.base_seed)
                    prediction.metadata = metadata
                    results[index] = prediction
            return results
        finally:
            self._busy = False

    def infer(self, request: InferenceInput | np.ndarray, camera=None) -> Prediction:
        """Infer a request or ordinary uint8 RGB; return depth for that full RGB."""
        if isinstance(request, np.ndarray):
            request = InferenceInput("image", "direct", request, camera=camera, setting=self.setting)
        elif camera is not None:
            raise ValueError("Supply camera inside InferenceInput, or alongside an RGB array")
        return self.predict([request])[0]

    def close(self):
        pass
