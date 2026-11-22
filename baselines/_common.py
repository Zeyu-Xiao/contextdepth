"""Array and geometry helpers shared by the model adapters."""
from __future__ import annotations

import hashlib
import numpy as np

from contextdepth.types import Geometry


def stable_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2**31)


def token_budget(g: Geometry, known: bool) -> int:
    if known:
        return int(round(2500.0 * g.canvas_width * g.canvas_height / float(g.roi_width * g.roi_height)))
    fraction = max(float((g.roi_width * g.roi_height) / max(float(g.canvas_height * g.canvas_width), 1.0)), 1e-6)
    return max(2500, min(12000, int(round(2500.0 / fraction))))



def _fov(k, width):
    return float(np.rad2deg(2.0 * np.arctan(.5 * width / float(k[0, 0]))))



def _float_image(rgb, device, torch):
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def _float_images(rgbs, device, torch):
    if len(rgbs) == 1:
        return _float_image(rgbs[0], device, torch)
    images = np.stack([np.asarray(rgb, dtype=np.float32) / 255.0 for rgb in rgbs])
    return torch.from_numpy(images).permute(0, 3, 1, 2).to(device)



def _array(tensor):
    return tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)



def input_depth(raw, g: Geometry):
    """Restore the model's depth to the complete RGB grid; never crop an ROI."""
    import cv2
    depth = np.asarray(raw, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a two-dimensional depth map, got {depth.shape}")
    depth = cv2.resize(depth, (g.canvas_width, g.canvas_height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(depth, dtype=np.float32)
