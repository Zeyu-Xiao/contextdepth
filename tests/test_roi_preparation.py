"""ROI preparation from public RGB/depth/camera files, without model inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image, PngImagePlugin

from contextdepth.io import load_rgb, read_json, write_json
from contextdepth.types import Geometry
from evaluation import data


class PaperROIGeometryTests(unittest.TestCase):
    def test_known_crops_rounding_boundaries_and_minimum_query_height(self):
        # Fixed expected values: notably 15/6 -> 2 and 21/6 -> 4, while
        # KITTI's 375/6 -> 62 and ETH3D's 1365/6 -> 228 use opposite ties.
        fixtures = (
            ((15, 21), (2, 4, 11, 13), Geometry(11, 13, 640, 756, 160, 190, 960, 1136)),
            ((21, 15), (4, 2, 13, 11), Geometry(13, 11, 640, 542, 160, 137, 960, 816)),
            ((2048, 1365), (341, 228, 1366, 909), Geometry(1366, 909, 640, 426, 160, 107, 960, 640)),
            ((1242, 375), (207, 62, 828, 251), Geometry(828, 251, 640, 194, 160, 55, 960, 304)),
            ((1200, 90), (200, 15, 800, 60), Geometry(800, 60, 640, 128, 160, 32, 960, 192)),
            # Query height ties also use half-even: 502.5 -> 502, 507.5 -> 508.
            ((384, 301), (64, 50, 256, 201), Geometry(256, 201, 640, 502, 160, 133, 960, 768)),
            ((384, 305), (64, 51, 256, 203), Geometry(256, 203, 640, 508, 160, 130, 960, 768)),
        )
        for dimensions, expected_crop, expected_geometry in fixtures:
            with self.subTest(dimensions=dimensions):
                crop, geometry = data.paper_roi_geometry(*dimensions)
                self.assertIsInstance(crop, tuple)
                self.assertEqual(crop, expected_crop)
                self.assertIsInstance(geometry, Geometry)
                self.assertEqual(geometry, expected_geometry)


class ROIPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.official = self.root / "official"
        self.frame = self.official / "synthetic/frame"
        self.frame.mkdir(parents=True)
        # Coordinate-varying RGB makes an off-by-one crop observable.
        self.rgb = np.arange(21 * 15 * 3, dtype=np.uint16).reshape(21, 15, 3).astype(np.uint8)
        Image.fromarray(self.rgb).save(self.frame / "image.png")
        self.codes = np.ones((21, 15), np.uint16)
        self.codes[0, 0] = 65535  # Invalid source pixel outside the retained ROI.
        self.codes[4, 2:5] = [0, 65535, 65534]
        self.codes[16, 12] = 65534
        self.save_depth(self.codes)
        # The first focal value retains bits which a float32 intermediate loses.
        write_json(self.frame / "meta.json", dict(intrinsics=[
            [1.0000000009313226, .03125, .375], [.0625, 1.25, .625], [0., 0., 1.],
        ]))
        self.camera = [[15.000000013969839, .46875, 3.625],
                       [1.3125, 26.25, 9.125], [0., 0., 1.]]
        self.geometry = Geometry(11, 13, 640, 756, 160, 190, 960, 1136)
        self.fixed = self.rgb[4:17, 2:13].copy()
        self.query = np.asarray(Image.fromarray(self.fixed).resize((640, 756), Image.Resampling.LANCZOS))
        self.spec = dict(
            uid="query", dataset="synthetic", scene_group="synthetic/scene",
            image="images/query.png", retrieval_image="retrieval/query.png", depth="gt/query.npz",
            candidates=[f"outpaintings/query/candidate{i}.png" for i in range(3)], seeds=[7, 8, 9],
            paper_reference_ids={"toy": "reference"}, query_rgb_sha256=self.rgb_hash(self.query),
            source=dict(image="synthetic/frame/image.png", depth="synthetic/frame/depth.png",
                        retrieval_rgb_sha256=self.rgb_hash(self.fixed), query_decoder="opencv"),
        )

    @staticmethod
    def rgb_hash(rgb):
        return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()

    def save_depth(self, codes):
        metadata = PngImagePlugin.PngInfo()
        for name, value in (("near", "2.0"), ("far", "8.0"), ("unit", "0.5")):
            metadata.add_text(name, value)
        Image.fromarray(codes).save(self.frame / "depth.png", pnginfo=metadata)

    def prepare(self, spec=None, name="prepared"):
        spec_path = self.root / f"{name}.jsonl"
        spec_path.write_text(json.dumps(self.spec if spec is None else spec) + "\n", encoding="utf-8")
        output = self.root / name
        path = data.prepare_benchmark(spec_path, self.official, output, workers=1)
        self.assertEqual(Path(path), output / "manifest.json")
        return output, read_json(path)

    def legacy_spec(self):
        spec = deepcopy(self.spec)
        spec["source"]["crop_xywh"] = [2, 4, 11, 13]
        spec["geometry"] = self.geometry.as_dict()
        spec["camera"] = deepcopy(self.camera)
        return spec

    def test_computed_roi_preserves_rgb_hashes_fixed_grid_camera_and_metric_gt(self):
        for decoder in ("opencv", "pillow"):
            with self.subTest(decoder=decoder):
                spec = deepcopy(self.spec)
                spec["source"]["query_decoder"] = decoder
                output, manifest = self.prepare(spec, decoder)
                self.assertEqual(manifest["protocol"], "paper-roi")
                self.assertEqual(manifest["data_root"], ".")
                self.assertEqual(len(manifest["samples"]), 1)
                row = manifest["samples"][0]
                self.assertEqual(row["geometry"], self.geometry.as_dict())
                np.testing.assert_array_equal(row["camera"], self.camera)
                self.assertEqual(row["query_decoder"], decoder)
                self.assertNotIn("source", row)
                for field in ("uid", "dataset", "scene_group", "candidates", "seeds", "paper_reference_ids"):
                    self.assertEqual(row[field], spec[field])
                actual_query, actual_fixed = load_rgb(output / row["image"]), load_rgb(output / row["retrieval_image"])
                np.testing.assert_array_equal(actual_query, self.query)
                np.testing.assert_array_equal(actual_fixed, self.fixed)
                self.assertEqual(self.rgb_hash(actual_query), row["query_rgb_sha256"])
                self.assertEqual(self.rgb_hash(actual_fixed), spec["source"]["retrieval_rgb_sha256"])
                with np.load(output / row["depth"], allow_pickle=False) as actual:
                    expected = np.ones((13, 11), np.float32)
                    expected[0, :3] = [np.nan, np.inf, 4.]
                    expected[-1, -1] = 4.
                    mask = np.ones((13, 11), bool)
                    mask[0, :2] = False
                    self.assertEqual(actual["depth"].dtype, np.float32)
                    np.testing.assert_array_equal(actual["depth"], expected)
                    np.testing.assert_array_equal(actual["mask"], mask)

    def test_equivalent_explicit_legacy_spec_produces_identical_results_without_meta(self):
        automatic, automatic_manifest = self.prepare(name="automatic")
        # Legacy specs supplied K directly and did not require a meta.json file.
        (self.frame / "meta.json").unlink()
        legacy, legacy_manifest = self.prepare(self.legacy_spec(), "legacy")
        self.assertEqual(legacy_manifest, automatic_manifest)
        for field in ("image", "retrieval_image"):
            filename = self.spec[field]
            np.testing.assert_array_equal(load_rgb(automatic / filename), load_rgb(legacy / filename))
        with np.load(automatic / self.spec["depth"], allow_pickle=False) as left, \
                np.load(legacy / self.spec["depth"], allow_pickle=False) as right:
            self.assertEqual(left.files, right.files)
            for field in left.files:
                np.testing.assert_array_equal(left[field], right[field])

    def test_custom_legacy_crop_is_not_replaced_by_the_default_formula(self):
        spec = self.legacy_spec()
        spec["source"]["crop_xywh"] = [1, 3, 8, 9]
        geometry = Geometry(8, 9, 16, 18, 4, 7, 24, 32)
        spec["geometry"] = geometry.as_dict()
        spec["camera"] = [[20., 0., 4.], [0., 21., 4.5], [0., 0., 1.]]
        fixed = self.rgb[3:12, 1:9].copy()
        query = np.asarray(Image.fromarray(fixed).resize((16, 18), Image.Resampling.LANCZOS))
        spec["source"]["retrieval_rgb_sha256"] = self.rgb_hash(fixed)
        spec["query_rgb_sha256"] = self.rgb_hash(query)
        output, manifest = self.prepare(spec)
        row = manifest["samples"][0]
        self.assertEqual(row["geometry"], geometry.as_dict())
        self.assertEqual(row["camera"], spec["camera"])
        np.testing.assert_array_equal(load_rgb(output / row["retrieval_image"]), fixed)
        np.testing.assert_array_equal(load_rgb(output / row["image"]), query)
        with np.load(output / row["depth"], allow_pickle=False) as result:
            self.assertEqual(result["depth"].shape, (9, 8))

    def test_partial_explicit_geometry_is_rejected(self):
        for present in (("crop",), ("geometry",), ("camera",),
                        ("crop", "geometry"), ("crop", "camera"), ("geometry", "camera")):
            with self.subTest(present=present):
                spec = self.legacy_spec()
                if "crop" not in present:
                    del spec["source"]["crop_xywh"]
                for field in ("geometry", "camera"):
                    if field not in present:
                        del spec[field]
                with self.assertRaises(ValueError):
                    self.prepare(spec)

    def test_gt_original_dimensions_must_match_rgb_even_when_crop_would_fit(self):
        for width in (14, 16):
            with self.subTest(depth_width=width):
                self.save_depth(np.ones((21, width), np.uint16))
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_computed_geometry_does_not_bypass_supplied_pixel_hash_checks(self):
        for field in ("query_rgb_sha256", "retrieval_rgb_sha256"):
            with self.subTest(field=field):
                spec = deepcopy(self.spec)
                target = spec if field == "query_rgb_sha256" else spec["source"]
                target[field] = "0" * 64
                with self.assertRaises(ValueError):
                    self.prepare(spec)


if __name__ == "__main__":
    unittest.main()
