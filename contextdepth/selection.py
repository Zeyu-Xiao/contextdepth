"""Reference-guided candidate selection, independent of models and query GT.

Candidate order is preserved. The paper uses three candidates; the same rules
also accept any nonempty candidate list. Known-camera fallback has index N
(the query-only prediction), whereas unknown-camera fallback is its anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Sequence

import numpy as np


DEPTH_EPS = 1e-6


@dataclass(frozen=True)
class Selection:
    """One complete decision; tuples keep the saved evidence immutable.

    ``responses`` are signed candidate-minus-Direct responses when camera is
    known, and candidate-minus-anchor responses when camera is unknown.
    ``camera_distances`` is absent for the known-camera branch.
    """

    choice: int
    anchor: int | None
    eligible: tuple[bool, ...]
    fallback: bool
    responses: tuple[float, ...]
    camera_distances: tuple[float, ...] | None

    @property
    def is_direct(self) -> bool:
        return self.choice == len(self.eligible)

    @property
    def selected_arm(self) -> str:
        return "direct" if self.is_direct else f"candidate{self.choice}"


def _array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    return array.astype(np.float64, copy=False)


def _finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")


def _residual(value) -> float:
    array = _array(value, "reference_residual")
    if array.ndim != 0:
        raise ValueError("reference_residual must be one scalar")
    _finite(array, "reference_residual")
    return float(array)


def select_known(shifts: Sequence[float], reference_residual: float) -> Selection:
    """Select the largest strict response opposing the reference depth bias.

    ``shifts[i] = median(log(candidate_i) - log(Direct))`` on the paired
    prediction-only support. A positive reference bias requests a negative
    response; a nonpositive bias requests a positive response. Zero responses
    are ineligible. Empty eligibility returns Direct at index ``len(shifts)``.
    Ties select the first candidate in the supplied order.
    """
    values = _array(shifts, "shifts")
    if values.ndim != 1 or not len(values):
        raise ValueError("shifts must have shape [N] with N >= 1")
    _finite(values, "shifts")
    residual = _residual(reference_residual)
    signed = (-1.0 if residual > 0 else 1.0) * values
    eligible = signed > 0
    fallback = not bool(eligible.any())
    choice = len(values) if fallback else int(signed.argmax())
    return Selection(
        choice=choice,
        anchor=None,
        eligible=tuple(bool(value) for value in eligible),
        fallback=fallback,
        responses=tuple(float(value) for value in signed),
        camera_distances=None,
    )


def select_unknown(
    candidate_cameras: Sequence[Sequence[float]],
    pairwise: Sequence[Sequence[float]],
    reference_camera: Sequence[float],
    reference_residual: float,
) -> Selection:
    """Anchor by camera distance, require opposing depth change, rank by camera.

    Camera coordinates are normalized log focal lengths on the same query ROI
    convention. ``pairwise[j, a]`` must be the directly measured median log
    difference of candidate j and candidate a, not a difference of medians.
    A zero reference bias or empty eligible set returns the nearest-camera
    anchor. This branch never returns Direct. All ties use the first index.
    """
    cameras = _array(candidate_cameras, "candidate_cameras")
    responses = _array(pairwise, "pairwise")
    reference = _array(reference_camera, "reference_camera")
    if cameras.ndim != 2 or not all(cameras.shape):
        raise ValueError("candidate_cameras must have shape [N, D] with N, D >= 1")
    count, dimensions = cameras.shape
    if responses.shape != (count, count):
        raise ValueError("pairwise must have shape [N, N]")
    if reference.shape != (dimensions,):
        raise ValueError("reference_camera must have shape [D]")
    for name, array in (("candidate_cameras", cameras), ("pairwise", responses),
                        ("reference_camera", reference)):
        _finite(array, name)
    residual = _residual(reference_residual)
    with np.errstate(over="ignore", invalid="ignore"):
        distances = ((cameras - reference) ** 2).mean(axis=1)
    _finite(distances, "camera distances")
    anchor = int(distances.argmin())
    changes = responses[:, anchor]
    eligible = np.sign(changes) * np.sign(residual) < 0
    fallback = not bool(eligible.any())
    choice = anchor if fallback else int(np.where(eligible, distances, np.inf).argmin())
    return Selection(
        choice=choice,
        anchor=anchor,
        eligible=tuple(bool(value) for value in eligible),
        fallback=fallback,
        responses=tuple(float(value) for value in changes),
        camera_distances=tuple(float(value) for value in distances),
    )


def pairwise_log_ratios(depths: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Return float64 ``P[i,j] = median(log(D_i) - log(D_j))``.

    Inputs have shape [N, H, W]. Each pair uses its own intersection of finite
    predictions strictly above 1e-6. No GT or evaluation mask is an input.
    Empty pair support is an error. For [Direct, candidate0, ...] input order,
    known shifts are ``P[1:, 0]`` and unknown responses are ``P[1:, 1:]``.
    """
    maps = _array(depths, "depths")
    if maps.ndim != 3 or not all(maps.shape):
        raise ValueError("depths must have shape [N, H, W] with nonempty dimensions")
    count = len(maps)
    result = np.empty((count, count), dtype=np.float64)
    for i, left in enumerate(maps):
        for j in range(i, count):
            right = maps[j]
            support = (np.isfinite(left) & np.isfinite(right)
                       & (left > DEPTH_EPS) & (right > DEPTH_EPS))
            if not support.any():
                raise ValueError(f"Empty prediction-only support for depth pair ({i}, {j})")
            value = float(np.median(np.log(left[support]) - np.log(right[support])))
            result[i, j] = value
            result[j, i] = -value if i != j else value
    return result


def reference_bias(depth: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Calibrate a reference: median(log(predicted depth) - log(reference GT)).

    Use the same model's camera-conditioned reference Direct prediction. The
    support is every finite GT pixel strictly above 1e-6. An optional saved
    mask must equal that support. Invalid predictions on the support raise;
    pixels are never silently removed and the bias is not clipped.
    """
    prediction, target = _array(depth, "depth"), _array(gt, "gt")
    if target.ndim != 2 or prediction.shape != target.shape or not all(target.shape):
        raise ValueError("Reference prediction and GT must have the same nonempty [H, W] shape")
    support = np.isfinite(target) & (target > DEPTH_EPS)
    if mask is not None:
        supplied = np.asarray(mask)
        if supplied.dtype != np.bool_ or supplied.shape != target.shape:
            raise ValueError("Reference mask must be boolean and match the GT shape")
        if not np.array_equal(supplied, support):
            raise ValueError("Reference mask differs from the full finite-positive GT support")
    if not support.any():
        raise ValueError("Empty reference GT support")
    if np.any(support & (~np.isfinite(prediction) | (prediction <= DEPTH_EPS))):
        raise ValueError("Invalid reference prediction on fixed GT support; no pixels dropped")
    return float(np.median(np.log(prediction[support]) - np.log(target[support])))


def reference_camera(K: np.ndarray, fixed_w: int, fixed_h: int, dimensions: int = 2) -> np.ndarray:
    """Reference-only GT camera as log(fx/fixed_w), log(fy/fixed_h).

    Depth Pro uses dimensions=1; the other paper models use dimensions=2.
    This helper must not substitute query GT for an unknown query camera.
    """
    camera = _array(K, "K")
    if camera.shape != (3, 3):
        raise ValueError("K must have shape [3, 3]")
    _finite(camera, "K")
    sizes = []
    for value in (fixed_w, fixed_h, dimensions):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Camera dimensions must be positive integers")
        try:
            sizes.append(operator.index(value))
        except TypeError as error:
            raise ValueError("Camera dimensions must be positive integers") from error
    width, height, axes = sizes
    if min(width, height) <= 0 or axes not in (1, 2):
        raise ValueError("Expected positive fixed dimensions and one or two focal axes")
    if min(camera[0, 0], camera[1, 1]) <= 0:
        raise ValueError("Reference focal lengths must be positive")
    return np.asarray([math.log(float(camera[i, i]) / size)
                       for i, size in enumerate((width, height)[:axes])], dtype=np.float64)
