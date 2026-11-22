"""CPU-only public outpainting and retained paper-call contract checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from contextdepth.generation import OutpaintingBaseline, create_outpainter, expansion_geometry
from contextdepth.geometry import canvas_geometry
from contextdepth.outpainting import OneRewardBaseline, OneRewardOutpainter, seed_from_uid


class Paint(OutpaintingBaseline):
    def generate_images(self, rgb, *, bbox, canvas_size, seeds):
        # A backend may alter its private input or paint over the entire ROI.
        # The public contract must still preserve the caller's original pixels.
        rgb[:] = 0
        w, h = canvas_size
        return [np.full((h, w, 3), seed % 256, np.uint8) for seed in seeds]


class Generator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class NativePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        generators = kwargs["generator"]
        generators = generators if isinstance(generators, list) else [generators]
        return SimpleNamespace(images=[Image.fromarray(np.full(
            (kwargs["height"], kwargs["width"], 3), generator.seed % 256, np.uint8)) for generator in generators])


class OutpaintingTests(unittest.TestCase):
    def test_default_expansion_three_seeds_and_original_pixels(self):
        rgb = np.arange(25 * 35 * 3, dtype=np.uint8).reshape(25, 35, 3)
        original = rgb.copy()
        engine = Paint()
        engine.metadata = {"nested": {"value": 1}}
        outputs = engine.generate(rgb)
        self.assertEqual([item.seed for item in outputs], [0, 1, 2])
        for item in outputs:
            self.assertEqual(item.rgb.shape, (31, 43, 3))
            self.assertEqual(item.bbox, (4, 3, 39, 28))
            np.testing.assert_array_equal(item.rgb[3:28, 4:39], original)
            self.assertEqual(int(item.rgb[0, 0, 0]), item.seed)
        np.testing.assert_array_equal(rgb, original)
        outputs[0].metadata["nested"]["value"] = 9
        self.assertEqual(outputs[1].metadata["nested"]["value"], 1)
        self.assertEqual(engine.metadata["nested"]["value"], 1)

    def test_asymmetric_expansion_and_camera_mapping(self):
        image = np.full((10, 20, 3), 99, np.uint8)
        result = Paint().generate(image, expansion={"top": .2, "bottom": 0, "left": .15, "right": .4}, seeds=(17,))[0]
        self.assertEqual(result.rgb.shape, (12, 31, 3))
        self.assertEqual(result.bbox, (3, 2, 23, 12))
        camera = np.array([[500, 2, 10], [0, 600, 5], [0, 0, 1]], np.float32)
        np.testing.assert_array_equal(result.map_intrinsics(camera), [[500, 2, 13], [0, 600, 7], [0, 0, 1]])
        np.testing.assert_array_equal(camera, [[500, 2, 10], [0, 600, 5], [0, 0, 1]])
        self.assertEqual(expansion_geometry(20, 10, (.2, 0, .15, .4)), ((31, 12), (3, 2, 23, 12)))

    def test_invalid_inputs_and_backend_outputs_fail(self):
        engine = Paint()
        image = np.zeros((10, 10, 3), np.uint8)
        for expansion in (-.1, float("nan"), True, {"top": .1}, (.1, .2)):
            with self.subTest(expansion=expansion), self.assertRaises(ValueError):
                engine.generate(image, expansion=expansion)
        for seeds in ((1, 1, 2), (), (-1,), (True,), (2**63,)):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                engine.generate(image, seeds=seeds)
        with self.assertRaises(ValueError):
            engine.generate(image.astype(np.float32))
        for outputs in ([], [image] * 3, [np.zeros((12, 12, 3), np.float32)] * 3):
            with patch.object(engine, "generate_images", return_value=outputs), self.assertRaises(ValueError):
                engine.generate(image)
            self.assertFalse(engine._busy)
        self.assertEqual(len(engine.generate(image)), 3)
        engine.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.generate(image)

    def test_file_class_and_object_injection_without_registry(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "my_generator.py"
            path.write_text("from contextdepth.generation import OutpaintingBaseline\n"
                            "import numpy as np\n"
                            "class Custom(OutpaintingBaseline):\n"
                            "    def generate_images(self, rgb, *, bbox, canvas_size, seeds):\n"
                            "        w, h = canvas_size\n"
                            "        return [np.full((h, w, 3), s, np.uint8) for s in seeds]\n")
            engine = create_outpainter({"implementation": f"{path}:Custom", "options": {"test": 1}}, device="cpu")
            result = engine.generate(np.full((10, 10, 3), 9, np.uint8))
            self.assertEqual(result[2].rgb.shape, (12, 12, 3))
            self.assertEqual(engine.paths["options"], {"test": 1})
            self.assertEqual(engine.device, "cpu")
        self.assertIsInstance(create_outpainter(device="cpu", implementation=Paint), Paint)
        ready = Paint()
        self.assertIs(create_outpainter(implementation=ready), ready)
        with self.assertRaises(TypeError):
            create_outpainter(implementation=object())

    def test_onereward_builds_requested_canvas_and_mask_not_paper_canvas(self):
        native = SimpleNamespace(pipe=NativePipe(), _torch=SimpleNamespace(Generator=Generator),
                                 pipeline_sha256="fake", close=Mock())
        with patch("contextdepth.outpainting.OneRewardOutpainter", return_value=native) as constructor:
            engine = OneRewardBaseline({"options": {"offload": "model"}}, device="cuda:2")
        constructor.assert_called_once_with({"options": {"offload": "model"}}, "cuda:2", offload="model")
        rgb = np.full((25, 35, 3), 79, np.uint8)
        outputs = engine.generate(rgb)
        self.assertEqual(len(native.pipe.calls), 3)
        for call, item in zip(native.pipe.calls, outputs):
            # 43x31 requested; 48x32 for FLUX, versus the paper's 64x48.
            self.assertEqual((call["width"], call["height"]), (48, 32))
            self.assertEqual(call["generator"].seed, item.seed)
            self.assertEqual(call["generator"].device, "cpu")
            self.assertIsInstance(call["prompt"], str)
            self.assertEqual(item.rgb.shape, (31, 43, 3))
            mask = np.asarray(call["mask_image"])
            expected = np.full((32, 48), 255, np.uint8)
            expected[3:28, 4:39] = 0
            np.testing.assert_array_equal(mask, expected)
            np.testing.assert_array_equal(np.asarray(call["image"])[3:28, 4:39], rgb)
            np.testing.assert_array_equal(item.rgb[3:28, 4:39], rgb)
        engine.close()
        engine.close()
        native.close.assert_called_once()

    def test_retained_paper_api_keeps_aligned_canvas_uid_seeds_and_scalar_call(self):
        engine = object.__new__(OneRewardOutpainter)
        engine._torch, engine.pipe = SimpleNamespace(Generator=Generator), NativePipe()
        rgb = np.full((25, 35, 3), 71, np.uint8)
        geometry = canvas_geometry(35, 25)
        output = engine.generate([rgb], ["test_query"], [2], [geometry])[0]
        call = engine.pipe.calls[0]
        self.assertEqual((call["width"], call["height"]), (64, 48))
        self.assertEqual(call["generator"].seed, seed_from_uid("test_query", 2))
        self.assertIsInstance(call["prompt"], str)
        self.assertIsInstance(call["image"], Image.Image)
        self.assertEqual((call["guidance_scale"], call["true_cfg"], call["num_inference_steps"]), (1.0, 4.0, 50))
        np.testing.assert_array_equal(output[geometry.roi_slices], rgb)
        engine.generate([rgb, rgb], ["test_query", "test_query"], [0, 1], [geometry, geometry])
        call = engine.pipe.calls[-1]
        self.assertEqual(len(call["prompt"]), 2)
        self.assertEqual(len(call["negative_prompt"]), 2)
        self.assertEqual([g.seed for g in call["generator"]], [seed_from_uid("test_query", i) for i in (0, 1)])

    def test_cpu_offload_load_never_moves_full_pipeline_to_cuda_first(self):
        pipe = SimpleNamespace(vae=SimpleNamespace(enable_tiling=Mock(), enable_slicing=Mock()),
                               to=Mock(side_effect=AssertionError("Premature whole-pipeline CUDA allocation")),
                               enable_model_cpu_offload=Mock(), set_progress_bar_config=Mock())
        upstream = SimpleNamespace(FluxFillCFGPipeline=SimpleNamespace(from_pretrained=Mock(return_value=pipe)))
        fake_torch = SimpleNamespace(bfloat16="bf16", device=lambda _: SimpleNamespace(type="cuda", index=2))
        fake_diffusers = SimpleNamespace(FluxTransformer2DModel=SimpleNamespace(from_pretrained=Mock(return_value=object())))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source/src").mkdir(parents=True)
            (root / "source/src/pipeline_flux_fill_with_cfg.py").write_text("# fake native module")
            (root / "weights").mkdir()
            (root / "base").mkdir()
            resources = {"source": root / "source", "weights": root / "weights", "base_model": root / "base"}
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None))
            with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}), \
                    patch("contextdepth.outpainting.importlib.util.spec_from_file_location", return_value=spec), \
                    patch("contextdepth.outpainting.importlib.util.module_from_spec", return_value=upstream):
                OneRewardOutpainter(resources, "cuda:2", offload="model")
        pipe.to.assert_not_called()
        pipe.enable_model_cpu_offload.assert_called_once_with(gpu_id=2, device="cuda")


if __name__ == "__main__":
    unittest.main()
