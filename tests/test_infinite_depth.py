"""A model may predict +inf depth (sky). It stays on the support and is never repaired."""
import math
import unittest

import numpy as np

from contextdepth.selection import pairwise_log_ratios, reference_bias
from contextdepth.types import Geometry, Prediction
from evaluation.geometry import project_prediction
from evaluation.metrics import aggregate, compare_results, score_depth


class InfiniteDepthTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.full((4, 4), 2.0, np.float32)

    def test_infinite_prediction_fails_delta1_and_leaves_absrel_undefined(self):
        depth = self.gt.copy()
        depth[0, :2] = np.inf
        score = score_depth(depth, self.gt)
        self.assertEqual(score["delta1"], 14 / 16)
        self.assertIsNone(score["absrel"])
        self.assertEqual(score["infinite_pixels"], 2)
        self.assertEqual(score["evaluated_pixels"], 16)
        finite = score_depth(self.gt, self.gt)
        self.assertEqual(finite, {"delta1": 1.0, "absrel": 0.0, "evaluated_pixels": 16})

    def test_nan_negative_infinity_and_nonpositive_predictions_still_raise(self):
        for bad in (np.nan, -np.inf, 0.0, -1.0):
            depth = self.gt.copy()
            depth[1, 1] = bad
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, "Invalid prediction"):
                score_depth(depth, self.gt)

    def test_infinite_prediction_off_the_gt_support_changes_nothing(self):
        gt = self.gt.copy()
        gt[0, 0] = np.nan
        depth = self.gt.copy()
        depth[0, 0] = np.inf
        self.assertEqual(score_depth(depth, gt)["absrel"], 0.0)

    def test_aggregate_reports_undefined_absrel_and_counts_the_images(self):
        records = [
            dict(uid="a", config="m", setting="unknown", dataset="D",
                 metrics={"direct": {"delta1": 1.0, "absrel": .1}, "seed0": {"delta1": .5, "absrel": None}}),
            dict(uid="b", config="m", setting="unknown", dataset="D",
                 metrics={"direct": {"delta1": .5, "absrel": .3}, "seed0": {"delta1": 1.0, "absrel": .2}}),
        ]
        rows = {(r["dataset"], r["method"]): r for r in aggregate(records)}
        self.assertAlmostEqual(rows["All", "direct"]["absrel"], .2)
        self.assertNotIn("undefined_absrel_images", rows["All", "direct"])
        self.assertEqual(rows["All", "seed0"]["delta1"], .75)
        self.assertIsNone(rows["All", "seed0"]["absrel"])
        self.assertEqual(rows["All", "seed0"]["undefined_absrel_images"], 1)
        listed = list(rows.values())
        self.assertTrue(compare_results(listed, listed)["matches"])
        changed = [dict(r, absrel=.5) if r["absrel"] is None else r for r in listed]
        self.assertFalse(compare_results(listed, changed)["matches"])

    def test_reference_bias_keeps_infinite_pixels_in_the_median(self):
        depth = self.gt.copy() * 2
        depth[0, :3] = np.inf
        self.assertAlmostEqual(reference_bias(depth, self.gt), math.log(2.0))
        depth[:3] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite on most"):
            reference_bias(depth, self.gt)
        depth = self.gt.copy()
        depth[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid reference prediction"):
            reference_bias(depth, self.gt)

    def test_pairwise_responses_use_finite_pixels_only(self):
        left, right = self.gt.copy() * 3, self.gt.copy()
        left[0] = np.inf
        self.assertAlmostEqual(pairwise_log_ratios([left, right])[0, 1], math.log(3.0))

    def test_resizing_to_the_evaluation_grid_keeps_infinity_and_creates_no_nan(self):
        # 8x8 input whose central 4x4 query maps onto a 2x2 evaluation grid.
        geometry = Geometry(fixed_w=2, fixed_h=2, roi_width=4, roi_height=4, roi_x=2, roi_y=2,
                            canvas_width=8, canvas_height=8)
        depth = np.full((8, 8), 5.0, np.float32)
        depth[2:4, 2:4] = np.inf
        projected = project_prediction(Prediction(depth), geometry).depth
        self.assertEqual(projected.shape, (2, 2))
        self.assertFalse(np.isnan(projected).any())
        self.assertTrue(np.isposinf(projected[0, 0]))
        np.testing.assert_array_equal(projected[1], [5.0, 5.0])
        finite = project_prediction(Prediction(np.full((8, 8), 5.0, np.float32)), geometry).depth
        np.testing.assert_array_equal(finite, np.full((2, 2), 5.0, np.float32))


if __name__ == "__main__":
    unittest.main()
