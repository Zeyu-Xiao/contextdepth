"""Compare a replay against saved results without modifying either directory.

Run from any working directory::

    python scripts/compare_replay.py --manifest inputs/manifest.json \
        --actual outputs/new --expected outputs/golden

Arrays and scientific observations are compared exactly; cache identities,
timings and implementation provenance are deliberately not compared. Metrics
and reference biases are also recomputed from the supplied ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextdepth.io import load_manifest, read_json, resolve
from evaluation.metrics import aggregate, score_depth
from contextdepth.selection import reference_bias


OBSERVATION_FIELDS = {
    "predictions": ("uid", "dataset", "config", "geometry", "pairwise", "candidate_cameras"),
    "references": ("uid", "dataset", "config", "geometry", "bias"),
    "selected": ("uid", "reference_uid", "reference_bias", "selected_arm", "choice",
                 "anchor", "eligible", "fallback", "responses", "camera_distances"),
}


class Comparison:
    def __init__(self):
        self.arrays = []
        self.array_inventory_exact = True
        self.differences = []
        self.roundoff = []
        self.checks = Counter()

    def different(self, location, reason, **details):
        self.differences.append(dict(location=location, reason=reason, **details))

    def value(self, location, actual, expected):
        self.checks["values"] += 1
        if actual != expected:
            self.different(location, "value differs", actual=actual, expected=expected)
            self._record_roundoff(location, actual, expected)

    def _record_roundoff(self, location, actual, expected):
        """Diagnose tiny scalar differences without changing exact pass/fail."""
        if isinstance(actual, dict) and isinstance(expected, dict):
            for key in actual.keys() & expected.keys():
                self._record_roundoff(f"{location}/{key}", actual[key], expected[key])
        elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            for index, (a, e) in enumerate(zip(actual, expected)):
                self._record_roundoff(f"{location}/{index}", a, e)
        elif isinstance(actual, float) and isinstance(expected, float) and actual != expected:
            if np.isfinite(actual) and np.isfinite(expected):
                spacing = max(abs(float(np.spacing(actual))), abs(float(np.spacing(expected))))
                ulps = abs(actual - expected) / spacing
                if ulps <= 16:
                    self.roundoff.append(dict(location=location, actual=actual, expected=expected,
                                              abs_diff=abs(actual - expected), ulps=ulps))

    def array(self, location, actual, expected):
        a, e = np.asarray(actual), np.asarray(expected)
        same_shape = a.shape == e.shape
        equal = same_shape and a.dtype == e.dtype and np.array_equal(a, e, equal_nan=True)
        maximum = None
        nonfinite_mismatches = None
        if same_shape and a.dtype.kind in "biuf" and e.dtype.kind in "biuf":
            af, ef = a.astype(np.float64), e.astype(np.float64)
            finite = np.isfinite(af) & np.isfinite(ef)
            if finite.any():
                with np.errstate(over="ignore", invalid="ignore"):
                    error = np.abs(af[finite] - ef[finite])
                maximum = float(error.max()) if np.isfinite(error).all() else "infinity"
            same_nonfinite = (af == ef) | (np.isnan(af) & np.isnan(ef))
            nonfinite_mismatches = int(np.count_nonzero(~finite & ~same_nonfinite))
        result = dict(location=location, array_equal=bool(equal),
                      actual_shape=list(a.shape), expected_shape=list(e.shape),
                      actual_dtype=str(a.dtype), expected_dtype=str(e.dtype),
                      max_abs_diff=maximum, nonfinite_mismatches=nonfinite_mismatches)
        self.arrays.append(result)
        self.checks["arrays"] += 1
        if not equal:
            self.different(location, "array differs", **{k: v for k, v in result.items() if k != "location"})

    def npz(self, location, actual, expected):
        try:
            with np.load(actual, allow_pickle=False) as a, np.load(expected, allow_pickle=False) as e:
                self.value(f"{location}: keys", sorted(a.files), sorted(e.files))
                for key in sorted(set(a.files) & set(e.files)):
                    self.array(f"{location}:{key}", a[key], e[key])
        except (OSError, ValueError, KeyError) as error:
            self.different(location, "cannot compare arrays", error=str(error))

    def record(self, location, actual, expected, fields):
        for field in fields:
            if field not in expected:
                continue  # New interface/provenance fields are not replay differences.
            if field not in actual:
                self.different(f"{location}:{field}", "missing field")
            else:
                self.value(f"{location}:{field}", actual[field], expected[field])


def _read(path, comparison, location):
    try:
        return read_json(path)
    except (OSError, ValueError) as error:
        comparison.different(location, "cannot read JSON", error=str(error))
        return None


def _metric_rows(rows, keys, comparison, location):
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in indexed:
            comparison.different(location, "duplicate metric row", key=list(key))
        indexed[key] = row
    return indexed


def _compare_rows(comparison, location, actual, expected, keys):
    a = _metric_rows(actual, keys, comparison, f"{location}/actual")
    e = _metric_rows(expected, keys, comparison, f"{location}/expected")
    comparison.value(f"{location}: row keys", sorted(a), sorted(e))
    for key in sorted(set(a) & set(e)):
        comparison.value(f"{location}:{'/'.join(map(str, key))}", a[key], e[key])


def _saved_metrics(actual, expected, configs, uids, comparison):
    cohorts = []
    for root, label in ((actual, "actual"), (expected, "expected")):
        records = _read(root / "per_image_metrics.json", comparison, f"{label}/per_image_metrics.json")
        cohorts.append([] if records is None else [r for r in records if r["config"] in configs])
    a, e = ([r for r in rows if r["uid"] in uids] for rows in cohorts)
    _compare_rows(comparison, "saved per-image metrics", a, e, ("config", "uid"))
    summaries = [_read(root / "metrics.json", comparison, f"{label}/metrics.json")
                 for root, label in ((actual, "actual"), (expected, "expected"))]
    if all(summary is not None for summary in summaries):
        comparison.value("metrics: excluded datasets", summaries[0].get("excluded_datasets", []),
                         summaries[1].get("excluded_datasets", []))
        # Compare summaries only when they describe the same entire cohort.
        cohort_keys = [{(r["config"], r["uid"]) for r in rows} for rows in cohorts]
        if cohort_keys[0] == cohort_keys[1]:
            _compare_rows(comparison, "saved aggregate metrics",
                          [r for r in summaries[0]["rows"] if r["config"] in configs],
                          [r for r in summaries[1]["rows"] if r["config"] in configs],
                          ("config", "setting", "dataset", "method"))
        else:
            comparison.checks["different_cohort_summaries_skipped"] += 1


def _recompute(root, label, manifest, data_root, configs, comparison):
    samples = {row["uid"]: row for row in manifest["samples"]}
    references = {row["uid"]: row for row in manifest.get("references", manifest["samples"])}
    records = []
    saved = _read(root / "per_image_metrics.json", comparison, f"{label}/per_image_metrics.json")
    saved = [] if saved is None else saved
    saved_index = _metric_rows(saved, ("config", "uid"), comparison, f"{label}/per_image_metrics.json")
    summary = _read(root / "metrics.json", comparison, f"{label}/metrics.json")
    excluded = set(summary.get("excluded_datasets", [])) if summary else set()
    for config in configs:
        for uid, row in samples.items():
            if row.get("dataset") in excluded:
                continue
            location = f"{label}/{config}/{uid}"
            try:
                decision = read_json(root / "selected" / config / f"{uid}.json")
                with np.load(resolve(data_root, row["depth"]), allow_pickle=False) as gt_file:
                    gt = gt_file["depth"]
                    mask = gt_file["mask"] if "mask" in gt_file else None
                with np.load(root / "predictions" / config / f"{uid}.npz", allow_pickle=False) as predictions:
                    chosen = decision["selected_arm"]
                    scores = {method: score_depth(predictions[arm], gt, mask)
                              for method, arm in (("direct", "direct"), ("seed0", "candidate0"),
                                                  ("selector", chosen))}
                    with np.load(root / "selected" / config / f"{uid}.npz", allow_pickle=False) as selected:
                        comparison.array(f"{location}: selected depth agrees with selected arm",
                                         selected["depth"], predictions[chosen])
                saved_row = saved_index.get((config, uid))
                setting = saved_row["setting"] if saved_row else (
                    "known" if decision["anchor"] is None else "unknown")
                record = dict(uid=uid, dataset=row.get("dataset", "custom"), config=config,
                              setting=setting, metrics=scores)
                records.append(record)
                if saved_row is None:
                    comparison.different(location, "missing saved per-image metrics")
                else:
                    comparison.value(f"{location}: recomputed metrics", record, saved_row)
                comparison.checks["images_rescored"] += 1
            except (OSError, ValueError, KeyError) as error:
                comparison.different(location, "cannot recompute metrics", error=str(error))

    # Every saved reference is calibrated independently of model names or caches.
    for path in sorted((root / "references").glob("*/*.json")):
        location = f"{label}/{path.relative_to(root)}"
        try:
            row = references[path.stem]
            observed = read_json(path)
            with np.load(resolve(data_root, row["depth"]), allow_pickle=False) as gt_file:
                gt = gt_file["depth"]
                mask = gt_file["mask"] if "mask" in gt_file else None
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as predictions:
                bias = reference_bias(predictions["direct"], gt, mask)
            comparison.value(f"{location}: recomputed bias", bias, observed["bias"])
            comparison.checks["references_recalibrated"] += 1
        except (OSError, ValueError, KeyError) as error:
            comparison.different(location, "cannot recompute reference bias", error=str(error))
    if summary is not None:
        # A full-cohort summary cannot be checked against a subset manifest.
        saved_cohort = {(r["config"], r["uid"]) for r in saved if r["config"] in configs}
        current_cohort = {(r["config"], r["uid"]) for r in records}
        if saved_cohort == current_cohort:
            expected_rows = [r for r in summary["rows"] if r["config"] in configs]
            _compare_rows(comparison, f"{label}: recomputed aggregate", aggregate(records),
                          expected_rows, ("config", "setting", "dataset", "method"))
        else:
            comparison.checks["full_cohort_summaries_skipped"] += 1
    return records


def compare_replay(manifest_path, actual, expected):
    manifest, data_root = load_manifest(manifest_path)
    actual, expected = Path(actual), Path(expected)
    comparison = Comparison()
    uids = {row["uid"] for row in manifest["samples"]}
    configs = sorted(path.name for path in (expected / "predictions").iterdir() if path.is_dir())
    if not configs or not uids:
        raise ValueError("Expected predictions and a nonempty manifest cohort are required")
    for stage in OBSERVATION_FIELDS:
        def paths(root):
            return {p.relative_to(root) for p in (root / stage).glob("*/*")
                    if p.suffix in (".json", ".npz")
                    and (stage == "references" or p.stem in uids)}
        expected_paths, actual_paths = paths(expected), paths(actual)
        if {p for p in expected_paths if p.suffix == ".npz"} != {p for p in actual_paths if p.suffix == ".npz"}:
            comparison.array_inventory_exact = False
        comparison.value(f"{stage}: files", sorted(map(str, actual_paths)), sorted(map(str, expected_paths)))
        for relative in sorted(actual_paths & expected_paths):
            a, e = actual / relative, expected / relative
            if relative.suffix == ".npz":
                comparison.npz(str(relative), a, e)
            else:
                ar = _read(a, comparison, f"actual/{relative}")
                er = _read(e, comparison, f"expected/{relative}")
                if ar is not None and er is not None:
                    comparison.record(str(relative), ar, er, OBSERVATION_FIELDS[stage])
    for config in configs:
        for uid in sorted(uids):
            for stage in ("predictions", "selected"):
                for suffix in (".json", ".npz"):
                    relative = Path(stage) / config / f"{uid}{suffix}"
                    if not (expected / relative).is_file():
                        comparison.different(str(relative), "expected cohort is incomplete")
                        if suffix == ".npz":
                            comparison.array_inventory_exact = False
    a_ids = _read(actual / "reference_ids.json", comparison, "actual/reference_ids.json")
    e_ids = _read(expected / "reference_ids.json", comparison, "expected/reference_ids.json")
    if a_ids is not None and e_ids is not None:
        for config in configs:
            for uid in sorted(uids):
                comparison.value(f"reference_ids:{config}/{uid}", a_ids.get(config, {}).get(uid),
                                 e_ids.get(config, {}).get(uid))
                if uid not in e_ids.get(config, {}):
                    comparison.different(f"reference_ids:{config}/{uid}", "missing expected reference ID")
    _saved_metrics(actual, expected, configs, uids, comparison)
    a_records = _recompute(actual, "actual", manifest, data_root, configs, comparison)
    e_records = _recompute(expected, "expected", manifest, data_root, configs, comparison)
    _compare_rows(comparison, "recomputed per-image metrics", a_records, e_records, ("config", "uid"))
    _compare_rows(comparison, "recomputed aggregate metrics", aggregate(a_records), aggregate(e_records),
                  ("config", "setting", "dataset", "method"))
    saved_metric_differences = [r for r in comparison.differences if
                               r["location"].startswith(("saved per-image metrics", "saved aggregate metrics"))
                               or ": recomputed metrics" in r["location"]
                               or r["location"].startswith(("actual: recomputed aggregate", "expected: recomputed aggregate"))
                               or "missing saved per-image metrics" == r["reason"]]
    recomputed_metric_differences = [r for r in comparison.differences if
                                    r["location"].startswith(("recomputed per-image metrics", "recomputed aggregate metrics"))
                                    or r["reason"] == "cannot recompute metrics"]
    return dict(matches=not comparison.differences, all_exact=not comparison.differences,
                depth_arrays_exact=comparison.array_inventory_exact and bool(comparison.arrays)
                                   and all(r["array_equal"] for r in comparison.arrays)
                                   and not any(r["reason"] == "cannot compare arrays" for r in comparison.differences),
                all_saved_metrics_exact=not saved_metric_differences,
                recomputed_metrics_exact=not recomputed_metric_differences,
                configs=configs, samples=len(uids),
                checks=dict(comparison.checks), arrays=comparison.arrays,
                saved_metric_differences=saved_metric_differences,
                roundoff=comparison.roundoff,
                differences=comparison.differences)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    args = parser.parse_args(argv)
    import json
    try:
        report = compare_replay(args.manifest, args.actual, args.expected)
    except (OSError, ValueError, KeyError) as error:
        report = {"matches": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
