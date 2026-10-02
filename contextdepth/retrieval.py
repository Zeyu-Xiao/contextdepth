"""Batched SigLIP2 image encoding and scene-disjoint cosine retrieval."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .io import load_rgb, resolve


def encode_images(samples, root, checkpoint, *, device="cuda:0", batch_size=16, workers=4):
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel

    torch.set_num_threads(2)
    torch.manual_seed(20260915)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = SiglipVisionModel.from_pretrained(checkpoint, torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
    processor = SiglipImageProcessor.from_pretrained(checkpoint)

    def prepare(row):
        rgb = load_rgb(resolve(root, row.get("retrieval_image", row["image"])))
        return processor(images=Image.fromarray(rgb), return_tensors="pt")["pixel_values"][0]

    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    result = []
    # Only two batches are materialized ahead of the GPU; large banks stay bounded.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, torch.inference_mode():
        pending = {}
        for i in range(min(2, len(batches))):
            pending[i] = [pool.submit(prepare, row) for row in batches[i]]
        for i in range(len(batches)):
            x = torch.stack([f.result() for f in pending.pop(i)])
            if i + 2 < len(batches):
                pending[i + 2] = [pool.submit(prepare, row) for row in batches[i + 2]]
            if str(device).startswith("cuda"):
                x = x.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
            with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16,
                                enabled=str(device).startswith("cuda")):
                features = model(pixel_values=x).pooler_output
            features = torch.nn.functional.normalize(features.float(), dim=1)
            result.append(features.cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def retrieve_top1(query_rows, reference_rows, query_features, reference_features, *, bank="mixed", chunk_size=256):
    """No depth, intrinsics or test metrics are inputs to retrieval.

    The public API also supports cross-dataset and fixed-DIODE banks. Paper
    Table 1 uses mixed. Exact score ties choose lexicographically smallest UID.
    """
    if bank not in ("mixed", "cross-dataset", "fixed-diode"):
        raise ValueError(f"Unknown reference bank: {bank}")
    q, r = np.asarray(query_features, np.float32), np.asarray(reference_features, np.float32)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1] or len(q) != len(query_rows) or len(r) != len(reference_rows):
        raise ValueError("Descriptor matrices must match query and reference rows")
    for x in (q, r):
        if not np.isfinite(x).all() or (np.linalg.norm(x, axis=1) == 0).any():
            raise ValueError("Descriptors must be finite and nonzero")
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    r = r / np.linalg.norm(r, axis=1, keepdims=True)
    ids = np.array([row["uid"] for row in reference_rows])
    datasets = np.array([row.get("dataset", "") for row in reference_rows])
    groups = np.array([row.get("scene_group", "") for row in reference_rows])
    matches = {}
    for start in range(0, len(q), chunk_size):
        similarity = q[start:start + chunk_size] @ r.T
        for offset, scores in enumerate(similarity):
            row = query_rows[start + offset]
            mask = ids != row["uid"]
            if row.get("scene_group"):
                mask &= groups != row["scene_group"]
            if bank == "cross-dataset":
                mask &= datasets != row.get("dataset", "")
            elif bank == "fixed-diode":
                mask &= datasets == "DIODE"
            if not mask.any():
                raise ValueError(f"No eligible reference for {row['uid']} in {bank}")
            best = scores[mask].max()
            matches[row["uid"]] = str(min(ids[mask & (scores == best)]))
    return matches

