"""Public RGB outpainting contract, independent of the paper's canvas layout.

An adapter implements ``generate_images`` and receives the requested canvas,
query bounding box and independent seeds. The common layer validates outputs
and restores the original RGB pixels exactly. Loading a custom adapter does
not require a registry: pass a module/file entrypoint, class or ready instance.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from numbers import Integral
from typing import Mapping

import numpy as np


DEFAULT_IMPLEMENTATION = "contextdepth.outpainting:OneRewardBaseline"
SIDES = ("top", "bottom", "left", "right")


def _rgb(value, name):
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"{name} must be a nonempty uint8 RGB image with shape (H, W, 3)")
    return image


def _bbox(value, width, height):
    if len(value) != 4 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in value):
        raise ValueError("bbox must contain four integer coordinates (x0, y0, x1, y1)")
    x0, y0, x1, y1 = map(int, value)
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("bbox must enclose the complete query inside the generated RGB")
    return x0, y0, x1, y1


def expansion_geometry(width, height, expansion=0.1):
    """Return ``(canvas_size, bbox)`` without resizing the query.

    A scalar expands every side by that fraction of its input axis. A mapping
    must specify top/bottom/left/right; a four-item sequence uses that order.
    Pixel margins use decimal half-up rounding. No model alignment is included.
    """
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 1 for n in (width, height)):
        raise ValueError("width and height must be positive integers")
    if isinstance(expansion, Mapping):
        if set(expansion) != set(SIDES):
            raise ValueError("expansion must specify top, bottom, left and right")
        values = [expansion[side] for side in SIDES]
    elif isinstance(expansion, (tuple, list)):
        if len(expansion) != 4:
            raise ValueError("expansion sequence must contain top, bottom, left and right")
        values = expansion
    else:
        values = [expansion] * 4
    margins = []
    for value, size in zip(values, (height, height, width, width)):
        try:
            ratio = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as error:
            raise ValueError("Expansion fractions must be finite nonnegative numbers") from error
        if isinstance(value, (bool, np.bool_)) or not ratio.is_finite() or ratio < 0:
            raise ValueError("Expansion fractions must be finite nonnegative numbers")
        margins.append(int((ratio * int(size)).to_integral_value(rounding=ROUND_HALF_UP)))
    top, bottom, left, right = margins
    return (int(width) + left + right, int(height) + top + bottom), (left, top, left + int(width), top + int(height))


def _seeds(values):
    values = tuple(values)
    if not values or any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral)
                         or not 0 <= seed < 2**63 for seed in values):
        raise ValueError("Seeds must be nonempty integers between 0 and 2**63 - 1")
    if len(set(values)) != len(values):
        raise ValueError("Each outpainting candidate must have a different seed")
    return tuple(map(int, values))


@dataclass
class Outpainting:
    """Complete generated RGB and the intact input's exclusive-end pixel bbox."""

    rgb: np.ndarray
    bbox: tuple[int, int, int, int]
    seed: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rgb = _rgb(self.rgb, "Outpainting")
        self.bbox = _bbox(self.bbox, self.rgb.shape[1], self.rgb.shape[0])
        self.seed = _seeds((self.seed,))[0]

    def map_intrinsics(self, camera):
        """Translate input-image pixel K onto this canvas; focal lengths stay fixed."""
        camera = np.asarray(camera, dtype=np.float64)
        if (camera.shape != (3, 3) or not np.isfinite(camera).all()
                or not np.array_equal(camera[2], [0, 0, 1]) or camera[0, 0] <= 0 or camera[1, 1] <= 0):
            raise ValueError("Camera must be a finite pixel 3x3 K with positive focal lengths and last row [0, 0, 1]")
        x0, y0, _, _ = self.bbox
        return np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64) @ camera


class OutpaintingBaseline:
    """Sequential adapter returning one complete, center-locked RGB per seed.

    Override ``generate_images(rgb, *, bbox, canvas_size, seeds)``. Any internal
    padding/resizing is the adapter's responsibility: returned arrays must use
    exactly ``canvas_size``. No paper UID, evaluation ROI or depth enters here.
    """

    def __init__(self, paths=None, device="cuda:0"):
        self.paths = deepcopy(dict(paths or {}))
        self.device = device
        self.metadata = {}
        self._busy = self._closed = False

    def generate(self, rgb, *, expansion=0.1, seeds=(0, 1, 2)) -> list[Outpainting]:
        if self._closed:
            raise RuntimeError("Outpainting model is closed")
        if self._busy:
            raise RuntimeError("One outpainting instance must be called sequentially")
        query = np.array(_rgb(rgb, "Input"), copy=True, order="C")
        seeds = _seeds(seeds)
        canvas_size, bbox = expansion_geometry(query.shape[1], query.shape[0], expansion)
        self._busy = True
        try:
            images = list(self.generate_images(query.copy(), bbox=bbox, canvas_size=canvas_size, seeds=seeds))
            if len(images) != len(seeds):
                raise ValueError("Outpainting model must return exactly one image per seed")
            x0, y0, x1, y1 = bbox
            outputs = []
            for image, seed in zip(images, seeds):
                image = _rgb(image, "Generated image")
                if image.shape[:2] != (canvas_size[1], canvas_size[0]):
                    raise ValueError("Generated image must use the requested canvas size")
                locked = np.array(image, copy=True, order="C")
                locked[y0:y1, x0:x1] = query
                outputs.append(Outpainting(locked, bbox, seed, deepcopy(self.metadata)))
            return outputs
        finally:
            self._busy = False

    def generate_images(self, rgb, *, bbox, canvas_size, seeds):
        raise NotImplementedError

    def close(self):
        """Release model resources; adapters owning weights should extend this hook."""
        if self._busy:
            raise RuntimeError("Cannot close an outpainting model while generating")
        self._closed = True


def create_outpainter(paths=None, device="cuda:0", *, implementation=None):
    """Load a module:Class/file.py:Class, adapter class, or ready adapter object.

    A class is constructed with ``(paths, device=device)``. An injected object
    is already initialized and must implement ``generate(rgb, expansion=...,
    seeds=...)`` returning ``Outpainting`` objects. The caller owns its lifetime.
    """
    import inspect
    from .models import load_implementation

    paths = dict(paths or {})
    implementation = implementation if implementation is not None else paths.get("implementation", DEFAULT_IMPLEMENTATION)
    if isinstance(implementation, str):
        implementation = load_implementation(implementation)
    if inspect.isclass(implementation):
        implementation = implementation(paths, device=device)
    if not callable(getattr(implementation, "generate", None)):
        raise TypeError("Outpainting implementation must provide generate(rgb, *, expansion, seeds)")
    return implementation
