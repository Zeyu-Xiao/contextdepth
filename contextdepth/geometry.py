"""Pixel geometry for the unchanged query and its generated context."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from numbers import Integral

import numpy as np

from .types import Geometry


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_geometry(geometry: Geometry) -> None:
    for name in ("fixed_w", "fixed_h", "roi_width", "roi_height", "canvas_width", "canvas_height"):
        _positive_integer(getattr(geometry, name), name)
    for name in ("roi_x", "roi_y"):
        value = getattr(geometry, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if (geometry.roi_x + geometry.roi_width > geometry.canvas_width
            or geometry.roi_y + geometry.roi_height > geometry.canvas_height):
        raise ValueError("The complete query must fit inside the canvas")


def _rgb(rgb: np.ndarray, width: int, height: int, name: str) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
        raise ValueError(f"{name} must be uint8 RGB with shape {(height, width, 3)}")
    return rgb


def canvas_geometry(width: int, height: int, fixed_width: int | None = None,
                    fixed_height: int | None = None) -> Geometry:
    """Center the intact query in the historical 1.5x canvas, rounded up to 16.

    This describes generation at approximately 25% per side. Alignment can
    make a border slightly larger; it does not resize the supplied query.
    """
    width, height = _positive_integer(width, "width"), _positive_integer(height, "height")
    fixed_width = width if fixed_width is None else _positive_integer(fixed_width, "fixed_width")
    fixed_height = height if fixed_height is None else _positive_integer(fixed_height, "fixed_height")
    canvas_width = ((3 * width + 31) // 32) * 16
    canvas_height = ((3 * height + 31) // 32) * 16
    return Geometry(fixed_width, fixed_height, width, height,
                    (canvas_width - width) // 2, (canvas_height - height) // 2,
                    canvas_width, canvas_height)


def crop_context(rgb: np.ndarray, geometry25: Geometry, fraction: float) -> tuple[np.ndarray, Geometry]:
    """Crop existing context to a strict per-side fraction of the query axes.

    Margins use decimal half-up rounding. The query pixels and fixed camera
    grid stay unchanged. In particular 10% is cropped from a 25% generation;
    no new generation or interpolation occurs. Calling this at 25% requests
    strict 25% margins, which can remove the original alignment border.
    """
    _validate_geometry(geometry25)
    rgb = _rgb(rgb, geometry25.canvas_width, geometry25.canvas_height, "Context")
    try:
        ratio = Decimal(str(fraction))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise ValueError("fraction must be between 0 and 0.25") from error
    if isinstance(fraction, (bool, np.bool_)) or not ratio.is_finite() or not 0 <= ratio <= Decimal("0.25"):
        raise ValueError("fraction must be between 0 and 0.25")
    mx = int((ratio * geometry25.roi_width).to_integral_value(rounding=ROUND_HALF_UP))
    my = int((ratio * geometry25.roi_height).to_integral_value(rounding=ROUND_HALF_UP))
    x0, y0 = geometry25.roi_x - mx, geometry25.roi_y - my
    x1 = geometry25.roi_x + geometry25.roi_width + mx
    y1 = geometry25.roi_y + geometry25.roi_height + my
    if x0 < 0 or y0 < 0 or x1 > geometry25.canvas_width or y1 > geometry25.canvas_height:
        raise ValueError("Requested context exceeds the saved canvas; borders are never silently clipped")
    geometry = Geometry(geometry25.fixed_w, geometry25.fixed_h,
                        geometry25.roi_width, geometry25.roi_height, mx, my, x1 - x0, y1 - y0)
    return np.array(rgb[y0:y1, x0:x1], copy=True, order="C"), geometry


def restore_query_pixels(result: np.ndarray, query: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Paste the exact input query into the generated image, without resizing."""
    _validate_geometry(geometry)
    result = _rgb(result, geometry.canvas_width, geometry.canvas_height, "Generated image")
    query = _rgb(query, geometry.roi_width, geometry.roi_height, "Query")
    locked = np.array(result, copy=True, order="C")
    locked[geometry.roi_slices] = query
    return locked


def map_intrinsics(Kfixed: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Map pixel K from the fixed query grid into this canvas exactly once."""
    _validate_geometry(geometry)
    camera = np.asarray(Kfixed, dtype=np.float64)
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise ValueError("Camera must be a finite 3x3 pixel intrinsic matrix")
    if not np.array_equal(camera[2], [0, 0, 1]) or camera[0, 0] <= 0 or camera[1, 1] <= 0:
        raise ValueError("Camera must have positive focal lengths and final row [0, 0, 1]")
    transform = np.array([[geometry.roi_width / geometry.fixed_w, 0, geometry.roi_x],
                          [0, geometry.roi_height / geometry.fixed_h, geometry.roi_y],
                          [0, 0, 1]], dtype=np.float64)
    return transform @ camera
