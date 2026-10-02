"""Small, process-local compatibility helpers for the pinned model releases.

No helper installs packages or edits an upstream checkout. A worker owns its
model and its imports; importing this module does not import torch or use CUDA.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import math
from pathlib import Path
import sys
import types
import random

import numpy as np

from .types import Geometry


def restore_rng_state(state, torch, device):
    """Restore portable benchmark RNG metadata captured before this input.

    This accepts arrays and a JSON scalar, not pickle or an executable object.
    The original four rng_* field names are retained to make exported benchmark
    states auditable. An AFTER-query legacy cache must first be associated with
    the next query in the original worker stream by the benchmark exporter.
    """
    required = {"rng_torch_cpu", "rng_torch_cuda", "rng_numpy_keys", "rng_json"}
    if not required <= set(state):
        raise ValueError(f"RNG state is missing {sorted(required - set(state))}")
    metadata = json.loads(str(np.asarray(state["rng_json"]).reshape(())))
    for key in ("rng_torch_cpu", "rng_torch_cuda"):
        if np.asarray(state[key]).dtype != np.uint8 or np.asarray(state[key]).ndim != 1:
            raise ValueError(f"{key} must be a one-dimensional uint8 array")
    torch.set_rng_state(torch.from_numpy(np.array(state["rng_torch_cpu"], copy=True)))
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(torch.from_numpy(np.array(state["rng_torch_cuda"], copy=True)), device)
    np.random.set_state((metadata["numpy_name"], np.asarray(state["rng_numpy_keys"], dtype=np.uint32),
                         metadata["numpy_position"], metadata["numpy_has_gauss"], metadata["numpy_cached_gaussian"]))
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value
    random.setstate(tuples(metadata["python"]))


def add_source(source: str, package: str) -> Path:
    """Use a configured source without replacing an already imported package."""
    root = Path(source).expanduser().resolve(strict=True)
    if not (root / package).is_dir():
        raise ValueError(f"{root} does not contain {package}; supply its import root")
    imported = sys.modules.get(package)
    if imported is not None:
        origin = getattr(imported, "__file__", None)
        locations = ([Path(origin).resolve()] if origin is not None else
                     [Path(p).resolve() for p in getattr(imported, "__path__", ())])
        if not locations or not all(path.is_relative_to(root) for path in locations):
            raise RuntimeError(f"{package} is already imported from another checkout; use a fresh worker")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class _LazyTrainingModule(types.ModuleType):
    """Resolve optional logging only when a logging function is actually used."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        module = importlib.import_module(self.__name__)
        if module is self:
            raise RuntimeError("Optional training dependency was used while importing an inference model")
        return getattr(module, name)


@contextmanager
def lazy_training_imports():
    """Keep old UniDepth logging imports out of inference, then restore imports.

    Upstream visualization functions keep a lazy module reference. There is no
    no-op logger: calling one of them imports the real dependency and may fail
    normally. The temporary sys.modules entry never survives model import.
    """
    missing = "wandb" not in sys.modules
    if missing:
        sys.modules["wandb"] = _LazyTrainingModule("wandb")
    try:
        yield
    finally:
        if missing:
            sys.modules.pop("wandb", None)


def camera_on_canvas(camera: np.ndarray, g: Geometry, *, full_block=False) -> np.ndarray:
    k = np.array(camera, dtype=np.float32, copy=True)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0
            or not np.array_equal(k[2], np.array([0, 0, 1], np.float32))):
        raise ValueError("Expected finite, positive-focal fixed-grid camera K")
    sx, sy = g.roi_width / float(g.fixed_w), g.roi_height / float(g.fixed_h)
    k[0, 0] *= sx
    k[1, 1] *= sy
    if full_block:
        k[0, 1] *= sx
        k[1, 0] *= sy
        if np.linalg.det(k[:2, :2].astype(np.float64)) <= 0:
            raise ValueError("Full K must have a positive-orientation, nonsingular 2x2 block")
    k[0, 2] = k[0, 2] * sx + g.roi_x
    k[1, 2] = k[1, 2] * sy + g.roi_y
    return k


def matched_constraints(native: dict, g: Geometry) -> dict:
    active = dict(native)
    area = float(g.roi_height * g.roi_width)
    target = min(float(native["pixels_max"]), max(float(native["pixels_min"]), area))
    scale = math.sqrt(target / max(area, 1.0))
    # Keep the original operation order, including the two multiplications.
    desired = float(g.canvas_height * g.canvas_width) * scale * scale
    active["pixels_min"] = active["pixels_max"] = desired
    return active


def unidepth_paddings(shape, ratios, *, known=False):
    """Original floor padding, with the setting-specific accepted repairs."""
    h, w = shape
    ratio = w / h
    low, high = map(float, ratios)
    target = min(high, max(low, ratio))
    if known and low <= ratio <= high:
        return (0, 0, 0, 0), (h, w)
    rounding = math.ceil if known else int
    if ratio > target:
        ph, pw = rounding(w / target), w
        top = (ph - h) // 2
        pads = (0, 0, top, ph - h - top)
    else:
        ph, pw = h, rounding(h * target)
        left = (pw - w) // 2
        pads = (left, pw - w - left, 0, 0)
    if min(pads) < 0:
        if not low <= ratio <= high:
            raise ValueError("Unexpected negative padding outside the in-range roundoff case")
        return (0, 0, 0, 0), (h, w)
    return pads, (ph, pw)


@contextmanager
def unidepth_padding_scope(module, *, known):
    original = module.get_paddings
    module.get_paddings = lambda shape, ratios: unidepth_paddings(shape, ratios, known=known)
    try:
        yield
    finally:
        module.get_paddings = original


@contextmanager
def shape_scope(model, constraints, *, remove_level=True, interpolation=None):
    sentinel = object()
    previous = model.shape_constraints
    level = getattr(model, "resolution_level", sentinel)
    mode = getattr(model, "interpolation_mode", sentinel)
    try:
        model.shape_constraints = constraints
        if remove_level and level is not sentinel:
            delattr(model, "resolution_level")
        if interpolation is not None:
            model.interpolation_mode = interpolation
        yield
    finally:
        model.shape_constraints = previous
        if level is not sentinel:
            model.resolution_level = level
        elif hasattr(model, "resolution_level"):
            delattr(model, "resolution_level")
        if mode is not sentinel:
            model.interpolation_mode = mode
        elif hasattr(model, "interpolation_mode"):
            delattr(model, "interpolation_mode")


def unidepth_geometry(module, model, g):
    bounds = [float(model.shape_constraints[k]) for k in ("pixels_min", "pixels_max")]
    if hasattr(model, "resolution_level"):
        level = float(model.resolution_level)
        if not 0 <= level < 10:
            raise ValueError("Invalid UniDepth resolution_level")
        interval = (bounds[1] - bounds[0]) / 10.0
        bounds = [level * interval + bounds[0], (level + 1.0) * interval + bounds[0]]
    pads, (ph, pw) = module.get_paddings((g.canvas_height, g.canvas_width), model.shape_constraints["ratio_bounds"])
    scale, (ih, iw) = module.get_resize_factor((ph, pw), bounds)
    return dict(paddings=list(pads), padded_hw=[ph, pw], internal_hw=[ih, iw],
                nominal_scale=float(scale), actual_scale_xy=[iw / pw, ih / ph])


def compensate_unidepth_camera(camera, prep):
    k = np.asarray(camera, dtype=np.float64).copy()
    nominal = prep["nominal_scale"]
    sx, sy = prep["actual_scale_xy"]
    left, _, top, _ = prep["paddings"]
    k[0, 0] = float(camera[0, 0]) * sx / nominal
    k[1, 1] = float(camera[1, 1]) * sy / nominal
    k[0, 2] = (float(camera[0, 2]) + left) * sx / nominal - left
    k[1, 2] = (float(camera[1, 2]) + top) * sy / nominal - top
    return k.astype(np.float32)


def da3_resize_transform(g, resolution, raw_hw, camera=None):
    w, h = g.canvas_width, g.canvas_height
    scale = resolution / float(max(w, h))
    mw, mh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    def patch(value):
        down = value // 14 * 14
        up = down + 14
        return max(1, up if abs(up - value) <= abs(value - down) else down)
    rh, rw = raw_hw
    if (rw, rh) != (patch(mw), patch(mh)):
        raise ValueError("DA3 raw grid differs from upper_bound_resize with 14-pixel patches")
    k = camera_on_canvas(np.eye(3, dtype=np.float32) if camera is None else camera, g)
    k[:1] *= mw / float(w)
    k[1:2] *= mh / float(h)
    k[:1] *= rw / float(mw)
    k[1:2] *= rh / float(mh)
    return k


def equivalent_ray_camera(rays, g):
    rays = np.asarray(rays, dtype=np.float64)
    if rays.shape != (3, g.canvas_height, g.canvas_width):
        raise ValueError("Expected native CHW rays on the entire input canvas")
    roi = rays[(slice(None), *g.roi_slices)]
    if not np.isfinite(roi).all() or np.any(roi[2] <= 1e-6):
        raise ValueError("The complete visible ROI must have finite, forward-facing rays")
    q = (roi[0] / roi[2], roi[1] / roi[2])
    u = np.broadcast_to(np.arange(g.roi_x, g.roi_x + g.roi_width, dtype=np.float64)[None] + .5, roi.shape[1:])
    v = np.broadcast_to(np.arange(g.roi_y, g.roi_y + g.roi_height, dtype=np.float64)[:, None] + .5, roi.shape[1:])
    focals, centers = [], []
    for axis, coordinates in zip(q, (u, v)):
        mean = np.mean(axis, dtype=np.float64)
        cm = np.mean(coordinates, dtype=np.float64)
        centered = axis - mean
        variance = np.mean(centered * centered, dtype=np.float64)
        floor = np.finfo(np.float64).eps * max(1.0, float(np.mean(axis * axis, dtype=np.float64)))
        if not np.isfinite(variance) or variance <= floor:
            raise ValueError("Degenerate UniK3D equivalent-camera axis")
        focal = np.mean(centered * (coordinates - cm), dtype=np.float64) / variance
        center = cm - focal * mean
        if not np.isfinite(focal) or not np.isfinite(center) or focal <= 0:
            raise ValueError("UniK3D equivalent camera has invalid focal or center")
        focals.append(float(focal))
        centers.append(float(center))
    return np.log([focals[0] / g.roi_width, focals[1] / g.roi_height]), dict(
        support_count=g.roi_width * g.roi_height, dropped_pixels=0,
        focal_input_pixels=focals, principal_point_input_pixels=centers,
        fit="independent-axis float64 OLS on every ROI ray and pixel center")
