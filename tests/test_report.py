"""The Table 1 report, the quick subset, and results accumulated over several runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from contextdepth.config import configuration_names, load_config
from contextdepth.io import read_json, read_jsonl, write_json
from evaluation.cli import main, parser
from evaluation.data import subset_specs
from evaluation.pipeline import save_reference_ids
from evaluation.report import quick_check, report, table

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "evaluation/protocol"


def rows_for(reference, images, shift=0.0):
    rows = []
    for spec in reference["rows"]:
        for setting in ("unknown", "known"):
            for method, paper in zip(("direct", "seed0", "selector"), spec["paper"][setting]):
                rows.append(dict(config=spec[setting], setting=setting, dataset="All", method=method, n=images,
                                 delta1=(paper + shift) / 100.0, absrel=0.2))
    return rows


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.reference = read_json(PROTOCOL / "table1_crop.json")
        self.quick = read_json(PROTOCOL / "quick.json")

    def test_reference_table_covers_every_shipped_experiment_once(self):
        named = [spec[setting] for spec in self.reference["rows"] for setting in ("unknown", "known")]
        self.assertEqual(sorted(named), sorted(configuration_names(load_config(ROOT / "evaluation/models.yaml"))))
        self.assertEqual(len(self.reference["rows"]), 7)
        for spec in self.reference["rows"]:
            for setting in ("unknown", "known"):
                self.assertEqual(len(spec["paper"][setting]), 3)

    def test_full_cohort_report_shows_differences_from_the_paper(self):
        write_json(self.output / "metrics.json", {"excluded_datasets": [], "rows": rows_for(self.reference, 3406, .25)})
        text, result = report(self.output)
        self.assertTrue(result["full_cohort"])
        self.assertAlmostEqual(result["max_query_only_difference"], .25)
        self.assertIn("| MoGe-3 | 62.92 (+0.25) | 76.98 (+0.25) | 79.24 (+0.25) |", text)
        self.assertIn("Depth Pro 2", text)
        self.assertTrue((self.output / "report.md").is_file())
        self.assertEqual(read_json(self.output / "report.json")["images"], [3406])

    def test_partial_model_set_lists_only_the_models_that_ran(self):
        rows = [r for r in rows_for(self.reference, 3406) if r["config"] in ("moge2", "moge2_gt_fov")]
        write_json(self.output / "metrics.json", {"excluded_datasets": [], "rows": rows})
        entries = table(rows, self.reference)
        self.assertEqual([e["model"] for e in entries], ["MoGe-2"])
        text, _ = report(self.output)
        self.assertNotIn("MoGe-3", text)

    def quick_records(self, shift=0.0, drop=None):
        records = []
        for config in ("moge2", "moge2_gt_fov"):
            for uid, value in self.quick["query_only_delta1"][config].items():
                if uid != drop:
                    records.append(dict(uid=uid, config=config, setting="unknown", dataset="iBims-1",
                                        metrics={m: {"delta1": value + shift, "absrel": .1}
                                                 for m in ("direct", "seed0", "selector")}))
        return records

    def write_quick(self, records):
        rows = [dict(config=config, setting="unknown", dataset="All", method=method, n=6, delta1=.9, absrel=.1)
                for config in ("moge2", "moge2_gt_fov") for method in ("direct", "seed0", "selector")]
        write_json(self.output / "metrics.json", {"excluded_datasets": [], "rows": rows})
        write_json(self.output / "per_image_metrics.json", records)

    def test_quick_subset_checks_query_only_against_stored_values(self):
        self.write_quick(self.quick_records())
        text, result = report(self.output)
        self.assertFalse(result["full_cohort"])
        self.assertEqual(result["quick_check"]["failed"], 0)
        self.assertEqual(result["quick_check"]["experiments"], 2)
        self.assertIn("not comparable", text)
        with patch("builtins.print"):
            self.assertEqual(main(["report", "--output", str(self.output)]), 0)

    def test_quick_subset_fails_on_a_shifted_or_incomplete_experiment(self):
        self.write_quick(self.quick_records(shift=-.05))
        with patch("builtins.print"):
            self.assertEqual(main(["report", "--output", str(self.output)]), 1)
            self.assertEqual(main(["report", "--output", str(self.output), "--tolerance", "0.1"]), 0)
        rows = quick_check(self.quick_records(drop=self.quick["queries"][0]), self.quick["query_only_delta1"], .02)
        self.assertTrue(all(not row["complete"] and not row["ok"] for row in rows))

    def test_quick_subset_needs_only_ibims_and_binds_every_experiment(self):
        specs = read_jsonl(PROTOCOL / "roi_samples.jsonl.gz")
        chosen, queries = subset_specs(specs, PROTOCOL / "quick.json")
        self.assertEqual(queries, self.quick["queries"])
        self.assertEqual(len(queries), 6)
        self.assertEqual({spec["dataset"] for spec in chosen}, {"iBims-1"})
        self.assertEqual([spec["uid"] for spec in chosen[:6]], queries)
        settings = set(configuration_names(load_config(ROOT / "evaluation/models.yaml")))
        self.assertEqual(set(self.quick["query_only_delta1"]), settings)
        available = {spec["uid"] for spec in chosen}
        for uid in queries:
            spec = next(s for s in chosen if s["uid"] == uid)
            self.assertTrue(set(spec["paper_reference_ids"].values()) <= available)
            for setting in settings:
                self.assertIn(uid, self.quick["query_only_delta1"][setting])
        self.assertEqual(parser().parse_args(["prepare", "--eval-root", "e", "--output", "o", "--subset", "quick"]).subset,
                         "quick")

    def test_reference_bindings_accumulate_over_runs_into_one_folder(self):
        save_reference_ids(self.output, {"moge2": {"q": "a"}})
        save_reference_ids(self.output, {"depth_pro2": {"q": "b"}, "moge2": {"q": "c"}})
        self.assertEqual(json.loads((self.output / "reference_ids.json").read_text()),
                         {"moge2": {"q": "c"}, "depth_pro2": {"q": "b"}})


class PublicOutpainterInEvaluationTests(unittest.TestCase):
    """An adapter written for normal use (OutpaintingBaseline) also generates the evaluation candidates."""

    def setUp(self):
        import numpy as np
        from contextdepth.geometry import canvas_geometry
        self.np = np
        self.geometry = canvas_geometry(8, 6)
        self.rgb = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(6, 8, 3)

    def test_adapter_receives_canvas_geometry_and_seed_and_the_query_is_restored(self):
        from evaluation.pipeline import _generate_with_public_adapter
        np, g, calls = self.np, self.geometry, []

        class Engine:
            def generate_images(self, rgb, *, bbox, canvas_size, seeds):
                calls.append((rgb.copy(), bbox, canvas_size, seeds))
                return [np.full((canvas_size[1], canvas_size[0], 3), 7, np.uint8)]

        images = _generate_with_public_adapter(Engine(), [self.rgb, self.rgb], [g, g], [123, 456])
        self.assertEqual(len(images), 2)
        for image in images:
            self.assertEqual(image.shape, (g.canvas_height, g.canvas_width, 3))
            np.testing.assert_array_equal(image[g.roi_slices], self.rgb)
            self.assertEqual(int(image[0, 0, 0]), 7)
        self.assertEqual([call[1:] for call in calls],
                         [((g.roi_x, g.roi_y, g.roi_x + 8, g.roi_y + 6), (g.canvas_width, g.canvas_height), (seed,))
                          for seed in (123, 456)])
        np.testing.assert_array_equal(calls[0][0], self.rgb)

    def test_wrong_size_or_count_is_rejected(self):
        from evaluation.pipeline import _generate_with_public_adapter
        np, g = self.np, self.geometry

        class WrongSize:
            def generate_images(self, rgb, *, bbox, canvas_size, seeds):
                return [np.zeros((4, 4, 3), np.uint8)]

        class WrongCount:
            def generate_images(self, rgb, *, bbox, canvas_size, seeds):
                return []

        with self.assertRaisesRegex(ValueError, "requested canvas size"):
            _generate_with_public_adapter(WrongSize(), [self.rgb], [g], [1])
        with self.assertRaisesRegex(ValueError, "exactly one image per seed"):
            _generate_with_public_adapter(WrongCount(), [self.rgb], [g], [1])


if __name__ == "__main__":
    unittest.main()
