"""DIODE retrieval contract tests with tiny local images and a fake encoder."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, PngImagePlugin

from contextdepth import reference
from contextdepth.io import load_rgb, write_json


class FakeEncoder:
    def __init__(self):
        self.batch_sizes = []
        self.close_calls = 0

    def encode(self, images):
        self.batch_sizes.append(len(images))
        for image in images:
            assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        return np.array([[*image.mean(axis=(0, 1)), 1.] for image in images], np.float32)

    def close(self):
        self.close_calls += 1


class ReferenceBankTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "DIODE"
        self.root.mkdir()
        self.checkpoint = Path(self.temporary.name) / "siglip"
        self.checkpoint.mkdir()
        self.rows = [
            "val/indoors/scene_00001/scan_00001/frame_red",
            "val/indoors/scene_00001/scan_00001/frame_pink",
            "val/outdoor/scene_00002/scan_00002/frame_blue",
            "val/outdoor/scene_00003/scan_00003/frame_green",
        ]
        colors = ((200, 0, 0), (200, 10, 0), (0, 0, 200), (0, 200, 0))
        for row, color in zip(self.rows, colors):
            directory = self.root / row
            directory.mkdir(parents=True)
            Image.fromarray(np.full((3, 4, 3), color, np.uint8)).save(directory / "image.jpg")
            self.depth(directory / "depth.png")
            write_json(directory / "meta.json", {"intrinsics": [[1.25, 0, .5], [0, 1.5, .5], [0, 0, 1]]})
        (self.root / ".index.txt").write_text("\n".join(self.rows) + "\n")
        self.encoders = []

    def depth(self, path, *, codes=None, near=2., far=8., unit=None):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("near", str(near))
        metadata.add_text("far", str(far))
        if unit is not None:
            metadata.add_text("unit", str(unit))
        codes = np.ones((3, 4), np.uint16) if codes is None else np.asarray(codes, np.uint16)
        Image.fromarray(codes).save(path, pnginfo=metadata)

    def factory(self, *, checkpoint, device):
        self.assertEqual(Path(checkpoint), self.checkpoint)
        self.assertEqual(device, "cpu")
        encoder = FakeEncoder()
        self.encoders.append(encoder)
        return encoder

    def bank(self):
        bank = reference.DIODEReferenceBank(self.root, checkpoint=self.checkpoint,
                                             device="cpu", batch_size=2, _encoder_factory=self.factory)
        self.addCleanup(bank.close)
        return bank

    def test_missing_download_is_reported_before_loading_an_encoder(self):
        for root in (self.root / "missing", self.checkpoint):
            with self.subTest(root=root):
                with self.assertRaisesRegex(FileNotFoundError, "Download and extract DIODE"):
                    reference.DIODEReferenceBank(root, checkpoint=self.checkpoint,
                                                device="cpu", _encoder_factory=self.factory)
        self.assertEqual(self.encoders, [])

    def test_non_diode_index_and_duplicate_entries_are_rejected(self):
        for content, message in (("custom/image\n", "MoGe-format DIODE"),
                                 (self.rows[0] + "\n" + self.rows[0], "Duplicate DIODE")):
            with self.subTest(content=content):
                (self.root / ".index.txt").write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    self.bank()
        self.assertEqual(self.encoders, [])

    def test_encoder_and_features_are_reused_and_gt_is_read_only_after_selection(self):
        selected_path = self.root / self.rows[0]
        # Unselected labels are deliberately unreadable. Retrieval must still
        # build all RGB features, select a reference, and return its valid label.
        for row in self.rows[1:]:
            (self.root / row / "depth.png").write_text("not a depth file")
            (self.root / row / "meta.json").write_text("not camera JSON")
        bank = self.bank()
        self.assertEqual(self.encoders, [])
        query = np.full((3, 4, 3), (250, 0, 0), np.uint8)
        with patch.object(reference, "_read_depth", wraps=reference._read_depth) as read_depth:
            first = bank.retrieve(query)
            self.assertEqual(len(self.encoders), 1)
            self.assertEqual(self.encoders[0].batch_sizes, [2, 2, 1])
            self.assertEqual(read_depth.call_args.args[0], selected_path / "depth.png")
            self.assertEqual(read_depth.call_count, 1)
            np.testing.assert_array_equal(first.depth, np.full((3, 4), 2., np.float32))
            np.testing.assert_array_equal(first.camera, [[5., 0, 2.], [0, 4.5, 1.5], [0, 0, 1]])
            # Altering the selected GT changes returned calibration data, not
            # the RGB retrieval choice or the cached reference descriptors.
            self.depth(selected_path / "depth.png", unit=3.)
            second = bank.retrieve(query)
            self.assertEqual(second.uid, first.uid)
            np.testing.assert_array_equal(second.depth, np.full((3, 4), 6., np.float32))
            self.assertEqual(len(self.encoders), 1)
            self.assertEqual(self.encoders[0].batch_sizes, [2, 2, 1, 1])
            self.assertEqual(read_depth.call_count, 2)
        bank.close()
        bank.close()
        self.assertEqual(self.encoders[0].close_calls, 1)
        self.assertIsNone(bank._features)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            bank.retrieve(query)

    def test_source_path_excludes_all_images_from_its_scene(self):
        bank = self.bank()
        # No exact hash match; both red/pink images would win over green/blue.
        query = np.full((3, 4, 3), (180, 9, 0), np.uint8)
        picked = bank.retrieve(query, source_path=self.root / self.rows[0] / "image.jpg")
        self.assertIn("scene_00003", picked.uid)
        self.assertNotIn("scene_00001", picked.uid)

    def test_identical_external_rgb_excludes_its_matching_bank_scene(self):
        bank = self.bank()
        query = load_rgb(self.root / self.rows[0] / "image.jpg")
        picked = bank.retrieve(query, source_path=Path(self.temporary.name) / "external.png")
        # Excluding only the identical frame would incorrectly choose pink.
        self.assertNotIn("scene_00001", picked.uid)

    def test_no_eligible_scene_fails_without_reading_ground_truth(self):
        (self.root / ".index.txt").write_text("\n".join(self.rows[:2]) + "\n")
        bank = self.bank()
        query = load_rgb(self.root / self.rows[0] / "image.jpg")
        with patch.object(reference, "_read_depth", side_effect=AssertionError("GT entered retrieval")):
            with self.assertRaisesRegex(ValueError, "No DIODE reference remains"):
                bank.retrieve(query)

    def test_exact_score_ties_use_stable_uid_order(self):
        (self.root / ".index.txt").write_text("\n".join(reversed(self.rows)) + "\n")
        bank = self.bank()
        with patch.object(FakeEncoder, "encode", side_effect=lambda images: np.ones((len(images), 2), np.float32)):
            picked = bank.retrieve(np.full((3, 4, 3), 37, np.uint8))
        self.assertEqual(picked.uid, min(entry.uid for entry in bank._entries))

    def test_log_depth_decoding_preserves_units_nan_and_infinity(self):
        path = self.root / self.rows[0] / "depth.png"
        self.depth(path, codes=[[0, 1], [65534, 65535]], near=2., far=8., unit=.5)
        depth = reference._read_depth(path)
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_array_equal(depth, [[np.nan, 1.], [4., np.inf]])

    def test_nonfinite_rgb_descriptors_fail_before_loading_gt(self):
        bank = self.bank()
        with patch.object(FakeEncoder, "encode", side_effect=lambda images: np.full((len(images), 2), np.nan)), \
             patch.object(reference, "_read_depth", side_effect=AssertionError("GT entered retrieval")):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                bank.retrieve(np.full((3, 4, 3), 37, np.uint8))


if __name__ == "__main__":
    unittest.main()
