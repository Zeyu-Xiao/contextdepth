"""Project full-image model outputs onto the experiment's evaluation ROI.

This is deliberately outside the model adapters. Ordinary adapter inference
always returns a prediction for the complete RGB that was supplied.
"""
from dataclasses import dataclass
import math

import numpy as np

from contextdepth.model_compat import camera_on_canvas, equivalent_ray_camera
from contextdepth.types import CameraEstimate, Geometry, Prediction


@dataclass
class EvaluationPrediction:
    depth: np.ndarray
    camera_log_focal: np.ndarray | None
    metadata: dict


def camera_observation(camera: CameraEstimate | None, g: Geometry):
    if camera is None:
        return None, {}
    if camera.focal_px is not None:
        focal = np.asarray(camera.focal_px, dtype=np.float64)
        dimensions = (g.roi_width, g.roi_height)[:len(focal)]
        return np.array([math.log(float(f) / size) for f, size in zip(focal, dimensions)]), {}
    if camera.rays is not None:
        vector, fit = equivalent_ray_camera(camera.rays, g)
        return vector, {"equivalent_camera": fit}
    if camera.source_intrinsics is not None and camera.resize_factors:
        # Preserve the recorded float32 sequence of image-coordinate transforms.
        # In particular, solving on the fixed grid avoids an extra round trip
        # through input-grid K when auditing historical camera observations.
        transform = camera_on_canvas(np.eye(3, dtype=np.float32), g)
        for sx, sy in camera.resize_factors:
            transform[:1] *= sx
            transform[1:2] *= sy
        fixed = np.linalg.solve(transform.astype(np.float64), np.asarray(camera.source_intrinsics, np.float64))
        return np.array([math.log(float(fixed[0, 0]) / g.fixed_w),
                         math.log(float(fixed[1, 1]) / g.fixed_h)]), {"K_pred_fixed": fixed.tolist()}
    if camera.intrinsics is not None:
        k = np.asarray(camera.intrinsics, np.float64)
        return np.array([math.log(float(k[0, 0]) / g.roi_width),
                         math.log(float(k[1, 1]) / g.roi_height)]), {}
    return None, {}


def project_prediction(prediction: Prediction, geometry: Geometry) -> EvaluationPrediction:
    import cv2
    if prediction.depth.shape != (geometry.canvas_height, geometry.canvas_width):
        raise ValueError("Model depth must cover the entire input RGB before evaluation cropping")
    roi = prediction.depth[geometry.roi_slices]
    size = (geometry.fixed_w, geometry.fixed_h)
    far = np.isposinf(roi)
    if far.any():
        # +inf is a legitimate "infinitely far" prediction (sky). Keep it +inf wherever it has
        # positive interpolation weight, instead of letting inf * 0 turn neighbours into NaN.
        depth = cv2.resize(np.where(far, np.float32(0), roi), size, interpolation=cv2.INTER_LINEAR)
        depth[cv2.resize(far.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > 0] = np.inf
    else:
        depth = cv2.resize(roi, size, interpolation=cv2.INTER_LINEAR)
    camera, metadata = camera_observation(prediction.camera, geometry)
    return EvaluationPrediction(np.ascontiguousarray(depth, dtype=np.float32), camera,
                                dict(prediction.metadata, **metadata))
