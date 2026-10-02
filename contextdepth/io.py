"""Portable manifests and atomic result writes; no experiment-specific paths."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".partial", delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(temporary, path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as f:
        temporary = Path(f.name)
        np.savez(f, **arrays)
    os.replace(temporary, path)


def load_rgb(path):
    if str(path).endswith(".npy"):
        rgb = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected uint8 RGB: {path}")
    return np.ascontiguousarray(rgb)


def resolve(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root) / path


def load_manifest(path, data_root=None):
    """JSON: {samples: [...], references: [...]} or JSONL query rows.

    A reference bank may reuse the samples list. That does not expose query GT
    to inference: only the separately selected reference is calibrated.
    """
    path = Path(path)
    manifest = {"samples": read_jsonl(path)} if ".jsonl" in path.name else read_json(path)
    # A single-image run naturally needs only a reference bank. Accept the
    # concise {"references": [...]} form as well as a full dataset manifest.
    if "samples" not in manifest and "references" in manifest:
        manifest["samples"] = []
    if "samples" not in manifest:
        raise ValueError("Manifest must contain samples or references")
    samples = manifest["samples"]
    refs = manifest.get("references", samples)
    for rows in (samples, refs):
        ids = [r["uid"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate image identifiers in manifest")
        for uid in ids:
            if not uid or Path(uid).name != uid or uid in (".", ".."):
                raise ValueError(f"Image id must be a file name: {uid!r}")
    root = Path(data_root) if data_root else resolve(path.parent, manifest.get("data_root", "."))
    queries = {r["uid"]: r for r in samples}
    for ref in refs:
        query = queries.get(ref["uid"])
        if query is not None:
            for field in ("image", "retrieval_image", "geometry", "camera"):
                q, r = query.get(field), ref.get(field)
                if field.endswith("image") and q is not None and r is not None:
                    q, r = resolve(root, q).resolve(), resolve(root, r).resolve()
                if q != r:
                    raise ValueError(f"One UID describes different query/reference inputs: {ref['uid']}/{field}")
    return manifest, root
