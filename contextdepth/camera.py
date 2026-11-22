"""Camera coordinates for ordinary RGB inputs and generated context."""
from __future__ import annotations

import numpy as np

from .types import CameraEstimate, Geometry, Prediction


def pixel_intrinsics(camera):
    value = np.asarray(camera, dtype=np.float64)
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or not np.array_equal(value[2], [0, 0, 1])
            or min(value[0, 0], value[1, 1]) <= 0):
        raise ValueError("Camera must be a finite pixel-space 3x3 K with positive focal lengths")
    return value.copy()


def crop_prediction(prediction: Prediction, bbox) -> Prediction:
    """Express a generated-image prediction on the original RGB extent.

    This only removes generated borders. It does not resize or apply a dataset
    ROI. Depth adapters themselves always return their complete input image.
    """
    x0, y0, x1, y1 = bbox
    height, width = prediction.depth.shape
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("Query bbox must be inside the predicted RGB grid")
    camera = prediction.camera
    if camera is not None:
        intrinsics = None
        if camera.intrinsics is not None:
            intrinsics = np.asarray(camera.intrinsics).copy()
            intrinsics[0, 2] -= x0
            intrinsics[1, 2] -= y0
        camera = CameraEstimate(
            focal_px=None if camera.focal_px is None else np.array(camera.focal_px, copy=True),
            intrinsics=intrinsics,
            rays=None if camera.rays is None else np.ascontiguousarray(camera.rays[:, y0:y1, x0:x1]))
    return Prediction(np.ascontiguousarray(prediction.depth[y0:y1, x0:x1]), camera,
                      dict(prediction.metadata, query_bbox=list(bbox), output_grid="query RGB input"))


def camera_features(camera: CameraEstimate | None, width: int, height: int):
    """Normalized log focal coordinates for reference-guided selection."""
    if camera is None:
        raise ValueError("Reference selection requires a predicted camera for every candidate")
    if camera.focal_px is not None:
        focal = np.asarray(camera.focal_px, np.float64)
    elif camera.intrinsics is not None:
        k = pixel_intrinsics(camera.intrinsics)
        focal = np.array([k[0, 0], k[1, 1]])
    elif camera.rays is not None:
        from .model_compat import equivalent_ray_camera
        geometry = Geometry(width, height, width, height, 0, 0, width, height)
        values, _ = equivalent_ray_camera(camera.rays, geometry)
        return np.asarray(values, np.float64)
    else:
        raise ValueError("Camera must provide focal lengths, intrinsics, or complete rays")
    if focal.ndim != 1 or focal.size not in (1, 2) or not np.isfinite(focal).all() or (focal <= 0).any():
        raise ValueError("Predicted focal lengths must be finite positive pixel values")
    return np.log(focal / np.asarray((width, height)[:len(focal)]))
