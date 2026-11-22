"""The bundled full paper protocol must be portable and reference-complete."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import unittest
from unittest.mock import patch

from contextdepth.config import configuration_names, load_config
from contextdepth.io import read_jsonl
from evaluation.cli import main, parser


class FullProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.protocol = cls.root / "evaluation/protocol"
        cls.spec = cls.protocol / "roi_samples.jsonl.gz"
        cls.rows = read_jsonl(cls.spec)
        cls.availability = json.loads((cls.protocol / "reference_availability.json").read_text())
        cls.sources = json.loads((cls.protocol / "full.sources.json").read_text())

    def test_full_cohort_and_all_reference_bindings_are_complete(self):
        counts = {"DIODE": 771, "ETH3D": 454, "HAMMER": 775,
                  "KITTI": 652, "NYUv2": 654, "iBims-1": 100}
        self.assertEqual(len(self.rows), 3406)
        self.assertEqual(Counter(row["dataset"] for row in self.rows), counts)
        by_uid = {row["uid"]: row for row in self.rows}
        self.assertEqual(len(by_uid), 3406)
        settings = set(configuration_names(load_config(self.root / "evaluation/models.yaml")))
        self.assertEqual(len(settings), 14)
        self.assertEqual(self.sources["configuration_count"], 14)
        eligible = self.availability["eligible_uids"]
        selected_references = set()
        for row in self.rows:
            self.assertEqual(set(row["paper_reference_ids"]), settings)
            self.assertEqual(len(row["candidates"]), 3)
            self.assertEqual(len(row["seeds"]), 3)
            for setting, uid in row["paper_reference_ids"].items():
                self.assertIn(uid, by_uid)
                self.assertNotEqual(row["uid"], uid)
                self.assertNotEqual(row["scene_group"], by_uid[uid]["scene_group"])
                if setting in eligible:
                    self.assertIn(uid, eligible[setting])
                selected_references.add(uid)
        self.assertEqual(len(selected_references), 1177)
        self.assertEqual(set(eligible), {"da3nested"})
        self.assertEqual(len(eligible["da3nested"]), 1765)
        self.assertEqual(len(set(eligible["da3nested"])), 1765)
        self.assertTrue(set(eligible["da3nested"]) <= by_uid.keys())
        self.assertEqual(self.availability["reference_count"], 3406)

    def test_derived_crop_geometry_and_camera_are_not_stored_in_default_spec(self):
        for row in self.rows:
            self.assertNotIn("crop_xywh", row["source"], row["uid"])
            self.assertNotIn("geometry", row, row["uid"])
            self.assertNotIn("camera", row, row["uid"])
        self.assertEqual(int.from_bytes(self.spec.read_bytes()[4:8], "little"), 0)
        self.assertEqual(self.sources["source_spec_sha256"],
                         "d37d6cba5e43ec1b92a73db4d8d2434082fd02b4d78697440cf3711a3cbee346")

    def test_metadata_hashes_public_sources_and_portable_paths(self):
        for field in ("roi_spec", "reference_availability"):
            path = self.protocol / self.sources[field]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             self.sources[field + "_sha256"])
        self.assertEqual(self.sources["query_count"], 3406)
        self.assertEqual(self.sources["dataset_counts"],
                         Counter(row["dataset"] for row in self.rows))
        self.assertEqual({r["dataset"] for r in self.sources["archives"]},
                         set(self.sources["dataset_counts"]))
        self.assertEqual(self.sources["dataset_revision"],
                         "574610d77c0cc14d911a1ff46ec529ba934c8960")
        for row in self.rows:
            paths = [row[field] for field in ("image", "retrieval_image", "depth")]
            paths += row["candidates"] + [row["source"][field] for field in ("image", "depth")]
            for value in paths:
                path = PurePosixPath(value)
                self.assertFalse(path.is_absolute(), value)
                self.assertNotIn("..", path.parts)
        payloads = [gzip.decompress(self.spec.read_bytes()).decode(),
                    json.dumps(self.availability), json.dumps(self.sources)]
        for payload in payloads:
            for private_root in ("/opt/", "/home/", "/Users/", "/root/", "/mnt/"):
                self.assertNotIn(private_root, payload)

    def test_prepare_defaults_to_full_protocol_and_allows_explicit_override(self):
        args = parser().parse_args(["prepare", "--eval-root", "official", "--output", "prepared"])
        self.assertEqual(args.spec, self.spec)
        self.assertTrue(args.spec.is_absolute())
        override = parser().parse_args(["prepare", "--spec", "custom.jsonl",
                                        "--eval-root", "official", "--output", "prepared"])
        self.assertEqual(override.spec, Path("custom.jsonl"))
        with patch("evaluation.data.prepare_benchmark", return_value=Path("prepared/manifest.json")) as prepare, \
                patch("builtins.print"):
            self.assertEqual(main(["prepare", "--eval-root", "official", "--output", "prepared"]), 0)
        prepare.assert_called_once_with(self.spec, Path("official"), Path("prepared"), workers=8)


if __name__ == "__main__":
    unittest.main()
