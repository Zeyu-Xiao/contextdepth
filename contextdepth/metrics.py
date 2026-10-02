"""Paper metrics: fixed GT support, no alignment, clipping or depth repair."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def score_depth(depth, gt, mask=None):
    pred, target = np.asarray(depth, np.float64), np.asarray(gt, np.float64)
    valid = np.isfinite(target) & (target > 1e-6)
    if mask is not None:
        provided = np.asarray(mask)
        if provided.dtype != np.bool_ or not np.array_equal(provided, valid):
            raise ValueError("Evaluation mask must equal the fixed finite GT > 1e-6 support")
    if pred.ndim != 2 or pred.shape != target.shape or valid.shape != target.shape or valid.dtype != np.bool_:
        raise ValueError("Prediction, GT and boolean mask must share the same 2D grid")
    n = int(valid.sum())
    if not n or np.any(valid & (~np.isfinite(target) | (target <= 0))):
        raise ValueError("Empty or invalid ground-truth support")
    if np.any(valid & (~np.isfinite(pred) | (pred <= 1e-6))):
        raise ValueError("Invalid prediction on evaluation pixels; pixels cannot be silently removed")
    p, t = pred[valid], target[valid]
    ratio = np.maximum(p / t, t / p)
    # Preserve the original evaluator's accumulation order.
    absrel = float(np.sum((np.abs(p - t) / t) / n, dtype=np.float64))
    if not math.isfinite(absrel):
        raise ValueError("Non-finite AbsRel")
    return {"delta1": float(np.mean(ratio < 1.25)), "absrel": absrel, "evaluated_pixels": n}


def aggregate(records, exclude_datasets=()):
    """One image, one vote; incomplete cohorts remain explicitly visible."""
    groups = defaultdict(list)
    seen = set()
    for r in records:
        key = (r["config"], r["uid"])
        if key in seen:
            raise ValueError(f"Duplicate metric record: {key}")
        seen.add(key)
        if r["dataset"] in exclude_datasets:
            continue
        for method, metrics in r["metrics"].items():
            for dataset in (r["dataset"], "All"):
                groups[r["config"], r["setting"], dataset, method].append(metrics)
    return [dict(config=c, setting=s, dataset=d, method=m, n=len(values),
                 **{k: math.fsum(v[k] for v in values) / len(values) for k in ("delta1", "absrel")})
            for (c, s, d, m), values in sorted(groups.items())]


def compare_results(rows, expected, *, decimals=True, atol=1e-6):
    """Report all missing/mismatched cells instead of rewriting an experiment."""
    actual = {(r["config"], r["dataset"], r["method"]): r for r in rows}
    if len(actual) != len(rows):
        raise ValueError("Duplicate metric rows cannot establish reproduction")
    differences = []
    for r in expected:
        key = (r["config"], r["dataset"], r["method"])
        found = actual.get(key)
        if found is None:
            differences.append({"key": list(key), "reason": "missing row"})
            continue
        if found["n"] != r["n"]:
            differences.append({"key": list(key), "reason": "image count", "expected": r["n"], "actual": found["n"]})
        for metric in ("delta1", "absrel"):
            a, b = found[metric], r[metric]
            if decimals:
                a, b = (f"{a * 100:.2f}", f"{b * 100:.2f}") if metric == "delta1" else (f"{a:.4f}", f"{b:.4f}")
                equal = a == b
            else:
                equal = math.isfinite(a) and abs(a - b) <= atol
            if not equal:
                differences.append({"key": list(key), "metric": metric, "expected": b, "actual": a})
    return {"matches": not differences, "expected_rows": len(expected), "differences": differences}
