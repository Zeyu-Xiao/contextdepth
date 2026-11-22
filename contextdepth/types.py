"""Small, explicit interfaces between image preparation and depth models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Geometry:
    """Experimental ROI and evaluation grid, plus legacy inference budgets.

    fixed_w/h describe the query's original evaluation grid. The ROI may have
    been resized for the historical benchmark. For ordinary user inputs these
    equal roi_width/height. Intrinsics are supplied on the fixed grid. Adapters
    always return the complete input grid; only evaluation applies this ROI.
    """

    fixed_w: int
    fixed_h: int
    roi_width: int
    roi_height: int
    roi_x: int
    roi_y: int
    canvas_width: int
    canvas_height: int

    @classmethod
    def from_dict(cls, row: dict) -> "Geometry":
        return cls(**{name: int(row[name]) for name in cls.__dataclass_fields__})

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def direct(self) -> "Geometry":
        return Geometry(self.fixed_w, self.fixed_h, self.roi_width, self.roi_height,
                        0, 0, self.roi_width, self.roi_height)

    @property
    def roi_slices(self) -> tuple[slice, slice]:
        return (slice(self.roi_y, self.roi_y + self.roi_height),
                slice(self.roi_x, self.roi_x + self.roi_width))


@dataclass
class InferenceInput:
    """A model call contains RGB, geometry, identity and optional calibration.

    camera is pixel K on fixed_w x fixed_h, only set for known-camera calls.
    query_camera is the opaque ``Prediction.camera_state`` that the same
    adapter returned for the unexpanded query; an unknown-camera call on an
    outpainted candidate may carry it to reuse the query's estimated camera.
    No ground-truth depth or evaluation mask enters this interface.
    """

    uid: str
    arm: str
    rgb: np.ndarray
    geometry: Geometry | None = None
    camera: np.ndarray | None = None
    setting: str = "unknown"
    rng_state: dict[str, Any] | None = None
    query_camera: Any = None

    def __post_init__(self):
        if self.geometry is None:
            h, w = self.rgb.shape[:2]
            self.geometry = Geometry(w, h, w, h, 0, 0, w, h)

    @property
    def baseline(self) -> bool:
        return self.arm in ("direct", "reference")


@dataclass
class CameraEstimate:
    """Camera of the complete input RGB, independent of evaluation crops.

    Focal lengths use input-image pixels (one or two axes). A non-pinhole
    model may provide its complete CHW ray field instead. Native intrinsics
    and the image resize stages optionally retain coordinate-transform
    provenance for exact projection onto historical evaluation grids.
    """

    focal_px: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    rays: np.ndarray | None = None
    source_intrinsics: np.ndarray | None = None
    resize_factors: tuple[tuple[float, float], ...] = ()


@dataclass
class Prediction:
    """Metric camera-Z depth on the complete RGB input grid, without a crop.

    camera_state is optional and adapter-defined: whatever the adapter needs
    to apply this input's estimated camera to a larger canvas later on. It is
    never serialized and never interpreted outside the adapter.
    """

    depth: np.ndarray
    camera: CameraEstimate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    camera_state: Any = None
