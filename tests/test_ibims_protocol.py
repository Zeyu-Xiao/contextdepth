"""Public iBims metadata must form a complete query/reference cohort."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from contextdepth.io import read_jsonl
from evaluation.protocol.ibims import finalize_manifest


class IBimsProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "evaluation/protocol/ibims100_with_references.jsonl"
        cls.rows = [{k: v for k, v in row.items() if k != "source"} for row in read_jsonl(path)]

    def test_bundled_cohort_has_complete_bindings_and_splits_without_scoring_references(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"data_root": ".", "samples": self.rows}))
            finalize_manifest(path)
            result = json.loads(path.read_text())
            self.assertEqual(len(result["samples"]), 100)
            self.assertEqual(len(result["references"]), 67)
            self.assertEqual({row["dataset"] for row in result["samples"]}, {"iBims-1"})
            refs = {row["uid"]: row for row in result["references"]}
            for row in result["samples"]:
                self.assertEqual(len(row["paper_reference_ids"]), 12)
                for uid in row["paper_reference_ids"].values():
                    self.assertIn(uid, refs)
                    self.assertNotEqual(row["uid"], uid)
                    self.assertNotEqual(row["scene_group"], refs[uid]["scene_group"])
            finalize_manifest(path)
            self.assertEqual(json.loads(path.read_text()), result)

    def test_missing_reference_or_changed_binding_fails_without_rewriting_manifest(self):
        missing = self.rows[1:]
        changed = deepcopy(self.rows)
        query = next(row for row in changed if row["dataset"] == "iBims-1")
        query["paper_reference_ids"]["moge2"] = query["uid"]
        for rows in (missing, changed, [*self.rows, self.rows[0]]):
            with self.subTest(rows=len(rows)), TemporaryDirectory() as directory:
                path = Path(directory) / "manifest.json"
                payload = json.dumps({"data_root": ".", "samples": rows})
                path.write_text(payload)
                with self.assertRaises(ValueError):
                    finalize_manifest(path)
                self.assertEqual(path.read_text(), payload)


if __name__ == "__main__":
    unittest.main()
