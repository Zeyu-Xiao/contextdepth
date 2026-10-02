"""Prepare the paper's fixed ROI cohort from the MoGe evaluation release."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .io import read_json, read_jsonl, write_json, write_npz


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


def prepare_benchmark(spec_path, eval_root, output, *, workers=8):
    """Exact saved ROI boxes/UIDs/seeds, rather than approximate center crops."""
    specs = read_jsonl(spec_path)
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
        x, y, w, h = source["crop_xywh"]
        fixed = np.ascontiguousarray(rgb[y:y + h, x:x + w])
        if fixed.shape != (h, w, 3):
            raise ValueError(f"ROI outside image for {spec['uid']}")
        retrieval_hash = hashlib.sha256(fixed.tobytes()).hexdigest()
        if retrieval_hash != source["retrieval_rgb_sha256"]:
            raise ValueError(f"Source RGB differs from paper for {spec['uid']}")
        size = (spec["geometry"]["roi_width"], spec["geometry"]["roi_height"])
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
        query.save(output / spec["image"])
        Image.fromarray(fixed).save(output / spec["retrieval_image"])
        depth = read_moge_depth(eval_root / source["depth"])[y:y + h, x:x + w]
        write_npz(output / spec["depth"], depth=depth, mask=np.isfinite(depth) & (depth > 1e-6))
        return {k: v for k, v in spec.items() if k != "source"} | {"query_decoder": decoder}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, specs))
    manifest = {"data_root": ".", "protocol": "paper-roi", "samples": rows}
    availability = Path(spec_path).parent / "reference_availability.json"
    if availability.is_file():
        manifest["reference_availability"] = read_json(availability)["eligible_uids"]
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"
