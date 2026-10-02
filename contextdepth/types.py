"""Small, explicit interfaces between image preparation and depth models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Geometry:
    """ROI in the model input, and the resolution at which depth is returned.

    fixed_w/h describe the query's original evaluation grid. The ROI may have
    been resized for the historical benchmark. For ordinary user inputs these
    equal roi_width/height. Intrinsics are always supplied on the fixed grid.
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
    No ground-truth depth or evaluation mask enters this interface.
    """

    uid: str
    arm: str
    rgb: np.ndarray
    geometry: Geometry
    camera: np.ndarray | None = None
    setting: str = "unknown"
    rng_state: dict[str, Any] | None = None

    @property
    def baseline(self) -> bool:
        return self.arm in ("direct", "reference")


@dataclass
class Prediction:
    depth: np.ndarray
    camera_log_focal: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
