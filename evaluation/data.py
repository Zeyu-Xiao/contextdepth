"""Prepare the paper's fixed ROI cohort from the MoGe evaluation release."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from contextdepth.geometry import canvas_geometry
from contextdepth.io import read_json, read_jsonl, write_json, write_npz
from contextdepth.types import Geometry


def paper_roi_geometry(width, height):
    """Compute the centered paper ROI and its 640-wide generation canvas.

    Remove one sixth from each side, rounding ties to even as in the original
    preparation. GT and pixel K stay on the cropped, unscaled evaluation grid.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Source image dimensions must be positive")

    def margin(size):
        quotient, remainder = divmod(size, 6)
        return quotient + int(remainder > 3 or (remainder == 3 and quotient % 2))

    x, y = margin(width), margin(height)
    fixed_w, fixed_h = width - 2 * x, height - 2 * y
    roi_w = 640
    roi_h = max(128, round(fixed_h * roi_w / fixed_w))
    return (x, y, fixed_w, fixed_h), canvas_geometry(roi_w, roi_h, fixed_w, fixed_h)


def _geometry_and_camera(spec, image_path, width, height):
    source = spec["source"]
    explicit = ("crop_xywh" in source, "geometry" in spec, "camera" in spec)
    if any(explicit):
        # Preserve complete, explicitly supplied custom ROI specifications.
        if not all(explicit):
            raise ValueError("Explicit ROI specifications require crop_xywh, geometry and camera together")
        crop = tuple(source["crop_xywh"])
        geometry = Geometry(**spec["geometry"])
        camera = np.asarray(spec["camera"], dtype=np.float64).copy()
    else:
        crop, geometry = paper_roi_geometry(width, height)
        meta_path = image_path.with_name("meta.json")
        camera = np.asarray(read_json(meta_path)["intrinsics"], dtype=np.float64).copy()
        if camera.shape != (3, 3) or not np.isfinite(camera).all():
            raise ValueError(f"Invalid source intrinsics: {meta_path}")
        camera[0, :] *= width
        camera[1, :] *= height
        camera[0, 2] -= crop[0]
        camera[1, 2] -= crop[1]

    x, y, w, h = crop
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError(f"ROI outside image for {spec['uid']}")
    if (geometry.fixed_w, geometry.fixed_h) != (w, h):
        raise ValueError(f"ROI and evaluation grid differ for {spec['uid']}")
    if (camera.shape != (3, 3) or not np.isfinite(camera).all()
            or camera[0, 0] <= 0 or camera[1, 1] <= 0 or not np.array_equal(camera[2], [0, 0, 1])):
        raise ValueError(f"Invalid camera for {spec['uid']}")
    return crop, geometry, camera


def read_moge_depth(path):
    """Decode MoGe's near/far PNG format, including its legacy unit field."""
    with Image.open(path) as image:
        near, far = float(image.info["near"]), float(image.info["far"])
        codes = np.asarray(image)
        t = (codes.astype(np.float32) - 1) / 65533
        depth = near ** (1 - t) * far ** t
        if "unit" in image.info:
            depth = depth * float(image.info["unit"])
    depth[codes == 0] = np.nan
    depth[codes == 65535] = np.inf
    return depth


def subset_specs(specs, subset_path):
    """Keep the queries listed in a subset file, plus every reference image they are bound to."""
    queries = read_json(subset_path)["queries"]
    by_uid = {spec["uid"]: spec for spec in specs}
    missing = [uid for uid in queries if uid not in by_uid]
    if missing or len(set(queries)) != len(queries):
        raise ValueError(f"Subset queries must be distinct members of the cohort: {missing}")
    needed = dict.fromkeys(queries)
    for uid in queries:
        needed.update(dict.fromkeys(by_uid[uid]["paper_reference_ids"].values()))
    return [by_uid[uid] for uid in needed], queries


def prepare_benchmark(spec_path, eval_root, output, *, workers=8, subset=None):
    """Compute paper ROIs and cameras from official images and metadata.

    ``subset`` names a file next to the spec (``<subset>.json``) that lists a few
    queries. Only those queries and their reference images are prepared, so only
    the datasets they come from need to be present under ``eval_root``.
    """
    specs = read_jsonl(spec_path)
    queries = None
    if subset is not None:
        specs, queries = subset_specs(specs, Path(spec_path).parent / f"{subset}.json")
    eval_root, output = Path(eval_root), Path(output)
    for name in ("images", "retrieval", "gt"):
        (output / name).mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    def one(spec):
        source = spec["source"]
        path = eval_root / source["image"]
        image_bytes = path.read_bytes()
        raw = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"Cannot decode {path}")
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        crop, geometry, camera = _geometry_and_camera(spec, path, rgb.shape[1], rgb.shape[0])
        x, y, w, h = crop
        fixed = np.ascontiguousarray(rgb[y:y + h, x:x + w])
        if fixed.shape != (h, w, 3):
            raise ValueError(f"ROI outside image for {spec['uid']}")
        retrieval_hash = hashlib.sha256(fixed.tobytes()).hexdigest()
        if retrieval_hash != source["retrieval_rgb_sha256"]:
            raise ValueError(f"Source RGB differs from paper for {spec['uid']}")
        size = (geometry.roi_width, geometry.roi_height)
        decoder = source.get("query_decoder", "opencv")
        query = Image.fromarray(fixed).resize(size, Image.Resampling.LANCZOS)
        # The historical generation banks used both JPEG decoders. Retrieval
        # consistently used OpenCV. Preserve those distinct RGB inputs rather
        # than silently changing a few edge pixels in the generated query.
        if decoder == "pillow" or hashlib.sha256(query.tobytes()).hexdigest() != spec["query_rgb_sha256"]:
            import io
            with Image.open(io.BytesIO(image_bytes)) as image:
                query = image.convert("RGB").crop((x, y, x + w, y + h)).resize(size, Image.Resampling.LANCZOS)
            decoder = "pillow"
        if hashlib.sha256(query.tobytes()).hexdigest() != spec["query_rgb_sha256"]:
            raise ValueError(f"Prepared query pixels differ from paper for {spec['uid']}")
        depth = read_moge_depth(eval_root / source["depth"])
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"Source depth and RGB dimensions differ for {spec['uid']}")
        depth = depth[y:y + h, x:x + w]
        query.save(output / spec["image"])
        Image.fromarray(fixed).save(output / spec["retrieval_image"])
        write_npz(output / spec["depth"], depth=depth, mask=np.isfinite(depth) & (depth > 1e-6))
        return {k: v for k, v in spec.items() if k != "source"} | {
            "geometry": geometry.as_dict(), "camera": camera.tolist(), "query_decoder": decoder}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, specs))
    manifest = {"data_root": ".", "protocol": "paper-roi", "samples": rows}
    if queries is not None:
        wanted = set(queries)
        manifest.update(cohort=subset, samples=[row for row in rows if row["uid"] in wanted], references=rows)
    availability = Path(spec_path).parent / "reference_availability.json"
    if availability.is_file() and queries is None:
        manifest["reference_availability"] = read_json(availability)["eligible_uids"]
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"
