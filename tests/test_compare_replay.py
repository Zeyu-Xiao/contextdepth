"""Regression checks for the read-only replay auditor itself."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from contextdepth.io import read_json, write_json, write_npz
from evaluation.metrics import aggregate, score_depth
from contextdepth.selection import reference_bias
from scripts.compare_replay import compare_replay


class ReplayComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.expected = self.root / "expected"
        self.actual = self.root / "actual"
        self.manifest = self.root / "manifest.json"
        self.depth = np.ones((2, 2), np.float32)
        write_npz(self.root / "gt.npz", depth=self.depth, mask=np.ones((2, 2), bool))
        write_json(self.manifest, dict(data_root=".", samples=[dict(uid="q", dataset="test", depth="gt.npz")],
                                       references=[dict(uid="r", dataset="test", depth="gt.npz")]))
        predictions = dict(direct=self.depth * 2, candidate0=self.depth * 1.5,
                           candidate1=self.depth, candidate2=self.depth * 3)
        write_npz(self.expected / "predictions" / "custom" / "q.npz", **predictions)
        write_json(self.expected / "predictions" / "custom" / "q.json",
                   dict(uid="q", config="custom", dataset="test", cache_key="old", seconds=99.,
                        candidate_cameras=[[0.], [.1], [.2]]))
        bias = reference_bias(self.depth * 2, self.depth)
        write_npz(self.expected / "references" / "calibration" / "r.npz", direct=self.depth * 2)
        write_json(self.expected / "references" / "calibration" / "r.json", dict(uid="r", bias=bias))
        write_npz(self.expected / "selected" / "custom" / "q.npz", depth=self.depth)
        write_json(self.expected / "selected" / "custom" / "q.json",
                   dict(uid="q", selected_arm="candidate1", anchor=0, reference_uid="r",
                        choice=1, reference_bias=bias))
        write_json(self.expected / "reference_ids.json", {"custom": {"q": "r"}})
        scores = {method: score_depth(predictions[arm], self.depth)
                  for method, arm in (("direct", "direct"), ("seed0", "candidate0"), ("selector", "candidate1"))}
        records = [dict(uid="q", dataset="test", config="custom", setting="unknown", metrics=scores)]
        write_json(self.expected / "per_image_metrics.json", records)
        write_json(self.expected / "metrics.json", dict(excluded_datasets=[], rows=aggregate(records)))
        shutil.copytree(self.expected, self.actual)

    def compare(self):
        return compare_replay(self.manifest, self.actual, self.expected)

    def test_identical_scientific_outputs_ignore_cache_and_time_and_preserve_expected(self):
        snapshot = {p.relative_to(self.expected): p.read_bytes() for p in self.expected.rglob("*") if p.is_file()}
        path = self.actual / "predictions" / "custom" / "q.json"
        observation = read_json(path)
        observation.update(cache_key="new", seconds=.01, implementation="a different wrapper")
        write_json(path, observation)
        report = self.compare()
        self.assertTrue(report["matches"], report["differences"])
        self.assertEqual(report["checks"]["images_rescored"], 2)
        self.assertEqual(snapshot, {p.relative_to(self.expected): p.read_bytes()
                                   for p in self.expected.rglob("*") if p.is_file()})

    def test_one_float32_ulp_in_a_prediction_is_reported_as_an_array_difference(self):
        path = self.actual / "predictions" / "custom" / "q.npz"
        with np.load(path, allow_pickle=False) as original:
            arrays = {key: original[key].copy() for key in original.files}
        arrays["candidate2"][0, 0] = np.nextafter(arrays["candidate2"][0, 0], np.float32(4))
        write_npz(path, **arrays)
        report = self.compare()
        self.assertFalse(report["matches"])
        changed = next(r for r in report["arrays"] if r["location"] == "predictions/custom/q.npz:candidate2")
        self.assertFalse(changed["array_equal"])
        self.assertGreater(changed["max_abs_diff"], 0)

    def test_changed_camera_is_detected_even_when_depth_and_metrics_are_equal(self):
        path = self.actual / "predictions" / "custom" / "q.json"
        record = read_json(path)
        record["candidate_cameras"][1][0] += .01
        write_json(path, record)
        report = self.compare()
        self.assertFalse(report["matches"])
        self.assertTrue(any("candidate_cameras" in r["location"] for r in report["differences"]))

    def test_saved_metric_corruption_is_detected_by_rescoring(self):
        path = self.actual / "per_image_metrics.json"
        records = read_json(path)
        records[0]["metrics"]["selector"]["absrel"] = .125
        write_json(path, records)
        report = self.compare()
        self.assertFalse(report["matches"])

    def test_saved_metric_roundoff_does_not_get_called_exact_or_a_depth_difference(self):
        path = self.actual / "per_image_metrics.json"
        records = read_json(path)
        records[0]["metrics"]["seed0"]["absrel"] = float(np.nextafter(.5, 1.))
        write_json(path, records)
        report = self.compare()
        self.assertFalse(report["all_exact"])
        self.assertFalse(report["all_saved_metrics_exact"])
        self.assertTrue(report["depth_arrays_exact"])
        self.assertTrue(report["recomputed_metrics_exact"])
        self.assertTrue(report["roundoff"])


if __name__ == "__main__":
    unittest.main()
