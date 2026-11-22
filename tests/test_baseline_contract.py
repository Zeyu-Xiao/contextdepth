"""CPU integration checks for third-party adapters and the experiment contract."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from contextdepth.config import configuration_names, experiment, load_config, model_paths
from contextdepth.geometry import canvas_geometry
from contextdepth.io import read_json, write_json, write_npz
from contextdepth.models import create_model
from contextdepth.types import Geometry, InferenceInput


PLUGIN = '''
import numpy as np
from contextdepth.baseline import DepthBaseline
from contextdepth.types import CameraEstimate, Prediction

class Baseline(DepthBaseline):
    supports_batch = True
    supports_setting_reuse = True
    supports_query_camera = True
    matmul_tf32 = (False, True)

    def __init__(self, name, paths, device):
        super().__init__(name, paths, device)
        self.native_calls = []
        self.native_contexts = []
        self.transferred_calls = []

    def preprocess(self, requests):
        # Known input must reach the model, and query labels must never reach it.
        for request in requests:
            assert not hasattr(request, "depth")
            assert not hasattr(request, "mask")
            assert (request.camera is not None) == (request.setting == "known")
        return requests

    def infer_model(self, requests):
        self.native_calls.append([(request.uid, request.arm) for request in requests])
        self.native_contexts.append(dict(name=self.name, setting=self.setting, known=self.known,
                                        request_settings=[request.setting for request in requests],
                                        matmul_tf32=self.torch.backends.cuda.matmul.allow_tf32))
        self.transferred_calls.extend((request.arm, request.query_camera) for request in requests
                                      if request.query_camera is not None)
        # A candidate that reuses the query camera predicts ten times its native depth.
        return [2.0 if request.uid == "reference" else {
            "direct": 2.0, "candidate0": 1.5, "candidate1": 1.0, "candidate2": 3.0,
        }[request.arm] * (10.0 if request.query_camera is not None else 1.0) for request in requests]

    def postprocess(self, outputs, requests):
        predictions = []
        for request, raw in zip(requests, outputs):
            geometry = request.geometry
            shape = request.rgb.shape[:2]
            if self.options.get("crop_output"):
                shape = (geometry.fixed_h, geometry.fixed_w)
            depth = np.full(shape, raw, dtype=self.options.get("dtype", "float32"))
            if self.options.get("coordinate_depth"):
                depth += np.arange(depth.size, dtype=depth.dtype).reshape(depth.shape)
            camera = None
            if request.setting == "unknown" and self.options.get("camera_mode") != "missing":
                dimensions = self.options.get("camera_dimensions", 1)
                if self.options.get("camera_mode") == "inconsistent" and request.arm == "candidate1":
                    dimensions += 1
                value = {"direct": 0.0, "candidate0": .1, "candidate1": .2, "candidate2": .3}[request.arm]
                if request.query_camera is not None:
                    value = 9.0
                axes = np.asarray([geometry.roi_width, geometry.roi_height][:dimensions])
                camera = CameraEstimate(focal_px=np.exp(value) * axes)
            state = None
            if (request.setting == "unknown" and request.arm == "direct"
                    and self.options.get("camera_state") != "missing"):
                state = {"estimated_on": request.uid}
            predictions.append(Prediction(depth=depth, camera=camera, camera_state=state))
        return predictions

class Standalone:
    # An otherwise valid external adapter has no configure method. The worker
    # must keep its immutable camera setting rather than reusing it unsafely.
    def __init__(self, name, paths, device):
        self.inner = Baseline(name, paths, device)
        self.max_batch_size = self.inner.max_batch_size
        self.native_calls = self.inner.native_calls
        self.native_contexts = self.inner.native_contexts

    def predict(self, requests):
        return self.inner.predict(requests)
'''


class BaselineFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "baselines").mkdir()
        self.plugin = self.root / "baselines" / "new_camera_model.py"
        self.plugin.write_text(textwrap.dedent(PLUGIN), encoding="utf-8")
        self.config_path = self.root / "custom.yaml"
        self.config_data = {
            "models": {"brand_new_backend": {
                "implementation": "baselines/new_camera_model.py:Baseline",
                "rng_policy": "caller", "options": {"camera_dimensions": 1},
            }},
            "experiments": {
                "novel_unknown": dict(baseline="brand_new_backend", camera="unknown",
                                      context_fraction=.25, reference="novel_calibration"),
                "novel_known": dict(baseline="brand_new_backend", camera="known",
                                    context_fraction=.25, reference="novel_calibration"),
                "novel_calibration": dict(baseline="brand_new_backend", camera="known",
                                          context_fraction=.25, reference="novel_calibration",
                                          reference_only=True),
            },
        }
        self.geometry = canvas_geometry(4, 4)
        self.camera = [[4., 0., 2.], [0., 4., 2.], [0., 0., 1.]]
        self.pool_initializations = []
        self.worker_engines = []

    def config(self, **options):
        self.config_data["models"]["brand_new_backend"]["options"].update(options)
        self.config_path.write_text(yaml.safe_dump(self.config_data), encoding="utf-8")
        return load_config(self.config_path)

    def request(self, known=False, camera=None):
        return InferenceInput("query", "direct", np.zeros((4, 4, 3), np.uint8),
                              self.geometry.direct(), camera, "known" if known else "unknown")

    def manifest(self):
        rgb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        Image.fromarray(rgb).save(self.root / "query.png")
        Image.fromarray(rgb).save(self.root / "reference.png")
        candidates = []
        for index in range(3):
            expanded = np.full((self.geometry.canvas_height, self.geometry.canvas_width, 3), index, np.uint8)
            expanded[self.geometry.roi_slices] = rgb
            path = self.root / f"candidate{index}.png"
            Image.fromarray(expanded).save(path)
            candidates.append(path.name)
        write_npz(self.root / "gt.npz", depth=np.ones((4, 4), np.float32), mask=np.ones((4, 4), bool))
        query = dict(uid="query", dataset="synthetic", image="query.png", depth="gt.npz",
                     camera=self.camera, scene_group="query_scene", geometry=self.geometry.as_dict(),
                     candidates=candidates)
        reference = dict(uid="reference", dataset="synthetic", image="reference.png", depth="gt.npz",
                         camera=self.camera, scene_group="reference_scene", geometry=self.geometry.as_dict())
        path = self.root / "manifest.json"
        write_json(path, dict(data_root=".", samples=[query], references=[reference]))
        return path


class BaselineContractTests(BaselineFixture):
    def test_file_plugin_loads_under_an_unregistered_result_name(self):
        config = self.config()
        backend = create_model("arbitrary_result_label", model_paths(config, "novel_unknown"), device="cpu")
        result, = backend.predict([self.request()])
        np.testing.assert_array_equal(result.depth, np.full((4, 4), 2., np.float32))
        self.assertEqual(result.camera.focal_px.shape, (1,))
        self.assertEqual(backend.name, "arbitrary_result_label")

    def test_external_rgb_inference_returns_the_complete_input_grid(self):
        backend = create_model("external", model_paths(self.config(), "novel_unknown"), device="cpu")
        rgb = np.zeros((5, 7, 3), np.uint8)
        result = backend.infer(rgb)
        self.assertEqual(result.depth.shape, rgb.shape[:2])
        self.assertEqual(result.depth.dtype, np.float32)
        np.testing.assert_array_equal(result.camera.focal_px, [7.])

    def test_legacy_geometry_is_cropped_only_by_evaluation_projection(self):
        import cv2
        from evaluation.geometry import project_prediction
        backend = create_model("external", model_paths(self.config(coordinate_depth=True), "novel_unknown"), device="cpu")
        geometry = Geometry(4, 3, 6, 4, 2, 1, 10, 8)
        request = InferenceInput("query", "candidate0", np.zeros((8, 10, 3), np.uint8), geometry)
        result = backend.infer(request)
        expected_full = np.arange(80, dtype=np.float32).reshape(8, 10) + 1.5
        np.testing.assert_array_equal(result.depth, expected_full)
        projected = project_prediction(result, geometry)
        expected_fixed = cv2.resize(expected_full[geometry.roi_slices], (4, 3), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(projected.depth, expected_fixed)
        np.testing.assert_allclose(projected.camera_log_focal, [.1], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(result.depth, expected_full)

    def test_adapter_cannot_return_a_legacy_evaluation_crop(self):
        backend = create_model("external", model_paths(self.config(crop_output=True), "novel_unknown"), device="cpu")
        geometry = Geometry(4, 3, 6, 4, 2, 1, 10, 8)
        request = InferenceInput("query", "candidate0", np.zeros((8, 10, 3), np.uint8), geometry)
        with self.assertRaisesRegex(ValueError, "full-input-grid"):
            backend.infer(request)

    def test_compatible_images_use_one_native_batch_call(self):
        self.config_data["depth_batch_size"] = 4
        backend = create_model("batched", model_paths(self.config(), "novel_unknown"), device="cpu")
        requests = [InferenceInput(f"query{i}", arm, np.zeros((4, 4, 3), np.uint8))
                    for i, arm in enumerate(("candidate0", "candidate1", "candidate2"))]
        results = backend.predict(requests)
        self.assertEqual(backend.native_calls, [[(r.uid, r.arm) for r in requests]])
        self.assertEqual([float(p.depth[0, 0]) for p in results], [1.5, 1., 3.])
        self.assertEqual([p.metadata["execution_batch_size"] for p in results], [3, 3, 3])

    def test_heterogeneous_batch_keys_group_and_restore_caller_order(self):
        self.config_data["depth_batch_size"] = 4
        backend = create_model("batched", model_paths(self.config(), "novel_unknown"), device="cpu")
        requests = [InferenceInput(f"query{i}", arm, np.zeros((*shape, 3), np.uint8))
                    for i, (arm, shape) in enumerate((("candidate0", (4, 4)),
                                                     ("candidate1", (3, 5)),
                                                     ("candidate2", (4, 4))))]
        results = backend.predict(requests)
        self.assertEqual(backend.native_calls, [[("query0", "candidate0"), ("query2", "candidate2")],
                                               [("query1", "candidate1")]])
        self.assertEqual([p.depth.shape for p in results], [(4, 4), (3, 5), (4, 4)])
        self.assertEqual([float(p.depth[0, 0]) for p in results], [1.5, 1., 3.])

    def test_native_batches_do_not_exceed_the_configured_limit(self):
        self.config_data["depth_batch_size"] = 2
        backend = create_model("batched", model_paths(self.config(), "novel_unknown"), device="cpu")
        requests = [InferenceInput(f"query{i}", "direct", np.zeros((4, 4, 3), np.uint8)) for i in range(5)]
        backend.predict(requests)
        self.assertEqual([len(batch) for batch in backend.native_calls], [2, 2, 1])

    def test_explicit_experiments_do_not_inherit_a_builtin_registry(self):
        config = self.config()
        self.assertEqual(set(config["models"]), {"brand_new_backend"})
        self.assertEqual(set(configuration_names(config)), {"novel_known", "novel_unknown"})
        self.assertEqual(experiment(config, "novel_unknown").reference, "novel_calibration")
        self.assertEqual(model_paths(config, "novel_known")["setting"], "known")
        self.assertEqual(model_paths(config, "novel_unknown")["implementation"], f"{self.plugin}:Baseline")

    def test_known_camera_requires_k_before_calling_the_model(self):
        backend = create_model("novel_known", model_paths(self.config(), "novel_known"), device="cpu")
        with self.assertRaisesRegex(ValueError, "camera"):
            backend.predict([self.request(known=True)])
        result, = backend.predict([self.request(known=True, camera=np.asarray(self.camera))])
        self.assertEqual(result.depth.dtype, np.float32)

    def test_unknown_camera_does_not_accept_query_calibration(self):
        backend = create_model("novel_unknown", model_paths(self.config(), "novel_unknown"), device="cpu")
        with self.assertRaisesRegex(ValueError, "camera"):
            backend.predict([self.request(camera=np.asarray(self.camera))])

    def test_adapter_must_return_the_declared_depth_dtype(self):
        backend = create_model("novel_unknown", model_paths(self.config(dtype="float64"), "novel_unknown"), device="cpu")
        with self.assertRaisesRegex(ValueError, "float32"):
            backend.predict([self.request()])

    def test_plugin_edits_invalidate_depth_cache_even_with_unchanged_mtime(self):
        from evaluation.cache import model_revision
        paths = model_paths(self.config(), "novel_unknown")
        before = model_revision(paths)
        stat = self.plugin.stat()
        self.plugin.write_text(self.plugin.read_text().replace('"direct": 2.0', '"direct": 4.0'))
        os.utime(self.plugin, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(model_revision(paths), before)

    def test_cli_and_metric_edits_do_not_invalidate_depth_or_generation(self):
        from evaluation import cache
        paths = model_paths(self.config(), "novel_unknown")
        real_hash = cache.file_sha256
        def edited_hash(path):
            return "changed" if Path(path).name in ("cli.py", "metrics.py", "selection.py") else real_hash(path)
        for stage in ("depth", "generation"):
            with self.subTest(stage=stage):
                before = cache.model_revision(paths, stage=stage)
                with patch.object(cache, "file_sha256", side_effect=edited_hash):
                    after = cache.model_revision(paths, stage=stage)
                self.assertEqual(after, before)

    def test_ordinary_pipeline_is_not_a_dependency_of_evaluation_caches(self):
        from evaluation import cache
        paths = model_paths(self.config(), "novel_unknown")
        ordinary_pipeline = Path(__file__).resolve().parents[1] / "contextdepth" / "pipeline.py"
        original_read = Path.read_text
        original_hash = cache.file_sha256
        def guarded_read(path, *args, **kwargs):
            if path == ordinary_pipeline:
                raise AssertionError("Evaluation must not fingerprint the ordinary inference pipeline")
            return original_read(path, *args, **kwargs)
        def changed_ordinary_hash(path):
            return "ordinary-api-change" if Path(path) == ordinary_pipeline else original_hash(path)
        for stage in ("depth", "generation"):
            with self.subTest(stage=stage):
                before = cache.model_revision(paths, stage=stage)
                with patch.object(Path, "read_text", new=guarded_read), \
                     patch.object(cache, "file_sha256", side_effect=changed_ordinary_hash):
                    self.assertEqual(cache.model_revision(paths, stage=stage), before)

    def test_evaluation_preparation_changes_invalidate_only_the_relevant_stage(self):
        from evaluation import cache, pipeline
        paths = model_paths(self.config(), "novel_unknown")
        pipeline_path = Path(pipeline.__file__)
        source = pipeline_path.read_text()
        old = 'camera = np.asarray(row["camera"], np.float64) if known else None'
        self.assertIn(old, source)
        original_read = Path.read_text
        def changed_read(path, *args, **kwargs):
            if path == pipeline_path:
                return source.replace(old, old.replace("np.float64", "np.float32"))
            return original_read(path, *args, **kwargs)
        before_depth = cache.model_revision(paths)
        before_generation = cache.model_revision(paths, stage="generation")
        with patch.object(Path, "read_text", new=changed_read):
            self.assertNotEqual(cache.model_revision(paths), before_depth)
            self.assertEqual(cache.model_revision(paths, stage="generation"), before_generation)

    def test_old_evaluation_imports_reexport_the_canonical_functions(self):
        from contextdepth import data, evaluation_geometry, metrics
        from evaluation import data as canonical_data, geometry, metrics as canonical_metrics
        self.assertIs(data.prepare_benchmark, canonical_data.prepare_benchmark)
        self.assertIs(evaluation_geometry.project_prediction, geometry.project_prediction)
        self.assertIs(metrics.score_depth, canonical_metrics.score_depth)
        self.assertIs(metrics.aggregate, canonical_metrics.aggregate)

    def test_observation_changes_invalidate_depth_but_not_generation(self):
        from evaluation import cache
        from contextdepth import selection
        paths = model_paths(self.config(), "novel_unknown")
        selection_path = Path(selection.__file__)
        source = selection_path.read_text()
        original_read = Path.read_text
        before_depth = cache.model_revision(paths)
        before_generation = cache.model_revision(paths, stage="generation")
        edits = (
            ("DEPTH_EPS = 1e-6", "DEPTH_EPS = 1e-5"),
            ("np.log(left[support]) - np.log(right[support])", "np.log(left[support]) + np.log(right[support])"),
            ("np.median(np.log(prediction[support]) - np.log(target[support]))",
             "np.mean(np.log(prediction[support]) - np.log(target[support]))"),
        )
        for old, new in edits:
            with self.subTest(edit=old):
                self.assertIn(old, source)
                def edited_read(path, *args, **kwargs):
                    return source.replace(old, new) if path == selection_path else original_read(path, *args, **kwargs)
                with patch.object(Path, "read_text", new=edited_read):
                    self.assertNotEqual(cache.model_revision(paths), before_depth)
                    self.assertEqual(cache.model_revision(paths, stage="generation"), before_generation)

    def test_selector_decision_changes_do_not_invalidate_existing_observations(self):
        from evaluation import cache
        from contextdepth import selection
        paths = model_paths(self.config(), "novel_unknown")
        selection_path = Path(selection.__file__)
        source = selection_path.read_text()
        old = "choice = len(values) if fallback else int(signed.argmax())"
        self.assertIn(old, source)
        original_read = Path.read_text
        def edited_read(path, *args, **kwargs):
            return source.replace(old, "choice = 0") if path == selection_path else original_read(path, *args, **kwargs)
        before = cache.model_revision(paths)
        with patch.object(Path, "read_text", new=edited_read):
            self.assertEqual(cache.model_revision(paths), before)

    def test_explicit_cache_dependencies_resolve_relative_to_the_yaml(self):
        from evaluation.cache import model_revision
        helper = self.root / "extra" / "camera_conversion.py"
        helper.parent.mkdir()
        helper.write_text("SCALE = 1\n")
        self.config_data["models"]["brand_new_backend"]["cache_dependencies"] = ["extra/camera_conversion.py"]
        paths = model_paths(self.config(), "novel_unknown")
        self.assertEqual(paths["cache_dependencies"], [str(helper)])
        before = model_revision(paths)
        helper.write_text("SCALE = 2\n")
        self.assertNotEqual(model_revision(paths), before)

    def test_creating_or_updating_runtime_cache_does_not_change_the_model_recipe(self):
        from evaluation.cache import model_revision
        self.config_data["models"]["brand_new_backend"]["cache_dir"] = "compiler-cache"
        paths = model_paths(self.config(), "novel_unknown")
        cache_dir = Path(paths["cache_dir"])
        self.assertFalse(cache_dir.exists())
        before = model_revision(paths)
        cache_dir.mkdir()
        runtime_file = cache_dir / "runtime.json"
        runtime_file.write_text('{"compiled": 1}\n')
        self.assertEqual(model_revision(paths), before)
        runtime_file.write_text('{"compiled": 2, "autotune": "complete"}\n')
        self.assertEqual(model_revision(paths), before)

    def test_depth_batch_size_is_part_of_the_prediction_cache_identity(self):
        from evaluation.cache import model_revision
        config = self.config()
        singleton = model_paths(config, "novel_unknown")
        config["depth_batch_size"] = 4
        batched = model_paths(config, "novel_unknown")
        self.assertEqual((singleton["batch_size"], batched["batch_size"]), (1, 4))
        self.assertNotEqual(model_revision(singleton), model_revision(batched))

    def test_resident_configuration_switches_setting_label_and_numerical_flags(self):
        config = self.config()
        known_paths, unknown_paths = (model_paths(config, name) for name in ("novel_known", "novel_unknown"))
        backend = create_model("novel_known", known_paths, device="cpu")
        original_calls = backend.native_calls
        known = backend.infer(self.request(known=True, camera=np.asarray(self.camera)))
        backend.configure("novel_unknown", unknown_paths)
        unknown = backend.infer(self.request())
        backend.configure("known_again", known_paths)
        repeated_known = backend.infer(self.request(known=True, camera=np.asarray(self.camera)))
        self.assertIs(backend.native_calls, original_calls)
        self.assertEqual([context["setting"] for context in backend.native_contexts], ["known", "unknown", "known"])
        self.assertEqual([context["matmul_tf32"] for context in backend.native_contexts], [True, False, True])
        self.assertEqual([result.metadata["model"] for result in (known, unknown, repeated_known)],
                         ["novel_known", "novel_unknown", "known_again"])
        self.assertEqual([result.metadata["numerical_flags"]["cuda_matmul_tf32"]
                          for result in (known, unknown, repeated_known)], [True, False, True])
        self.assertIsNone(known.camera)
        self.assertIsNotNone(unknown.camera)

    def test_known_and_unknown_recipes_remain_distinct_when_weights_are_shared(self):
        from evaluation.cache import model_revision
        config = self.config()
        known, unknown = (model_paths(config, name) for name in ("novel_known", "novel_unknown"))
        self.assertEqual({k: v for k, v in known.items() if k != "setting"},
                         {k: v for k, v in unknown.items() if k != "setting"})
        self.assertNotEqual(model_revision(known), model_revision(unknown))

    def test_resident_configure_rejects_a_changed_loading_configuration(self):
        paths = model_paths(self.config(), "novel_unknown")
        backend = create_model("resident", paths, device="cpu")
        for field, value in (("weights", "different-checkpoint.pth"),
                             ("options", {"camera_dimensions": 2}), ("batch_size", 3)):
            with self.subTest(field=field):
                changed = deepcopy(paths)
                changed[field] = value
                with self.assertRaisesRegex(ValueError, "loading configuration changed"):
                    backend.configure("changed", changed)
                self.assertEqual(backend.name, "resident")
                self.assertEqual(backend.paths, paths)


class RetrievalPluginTests(BaselineFixture):
    def test_new_experiment_on_paper_manifest_computes_retrieval(self):
        from evaluation import cli
        from contextdepth import retrieval
        manifest = read_json(self.manifest())
        manifest["samples"][0]["paper_reference_ids"] = {"historical_model": "reference"}
        config = self.config()
        config["retrieval"] = {"checkpoint": "unused-in-this-CPU-test"}
        args = SimpleNamespace(retrieval=None, bank="mixed", recompute_retrieval=False,
                               output=self.root / "results", models=["novel_unknown"],
                               gpus=[0], encoder_batch_size=2, io_workers=1)
        expected = {"query": "reference"}
        with patch.object(retrieval, "encode_images", return_value=np.ones((2, 3), np.float32)) as encoder, \
             patch.object(retrieval, "retrieve_top1", return_value=expected) as search:
            matches = cli._retrieval(args, manifest, self.root, config)
        self.assertEqual(matches, expected)
        encoder.assert_called_once()
        search.assert_called_once()
        self.assertEqual(read_json(args.output / "retrieval.json"), expected)


class PipelinePluginTests(BaselineFixture):
    @contextmanager
    def cpu_workers(self):
        """Replace only GPU worker startup; run real preparation and persistence."""
        from evaluation import pipeline

        @contextmanager
        def cpu_pool(gpus, workers_per_gpu, initializer_args, max_tasks=None):
            role, paths, name, root, output, fraction, overwrite = initializer_args
            self.pool_initializations.append(name)
            engine = create_model(name, paths, device="cpu")
            self.worker_engines.append(engine)
            previous = dict(pipeline._WORKER)
            pipeline._WORKER.clear()
            pipeline._WORKER.update(engine=engine,
                                    root=Path(root), output=Path(output), name=name, role=role,
                                    fraction=fraction, overwrite=overwrite, gpu=None,
                                    setting=paths["setting"], options=paths.get("options", {}), paths=paths,
                                    model_instance=f"cpu-test-model-{len(self.worker_engines)}", model_load_seconds=0.)
            class Pool:
                def map(self, function, jobs, chunksize=1):
                    return map(function, jobs)
            try:
                yield Pool()
            finally:
                pipeline._WORKER.clear()
                pipeline._WORKER.update(previous)

        with patch.object(pipeline, "_pool", cpu_pool):
            yield pipeline

    def run_prediction(self, config, names):
        path = self.manifest()
        output = self.root / "results"
        with self.cpu_workers() as pipeline:
            pipeline.predict(path, config, names, output=output, retrieval={"query": "reference"})
        return path, output

    def paper_manifest(self):
        path = self.manifest()
        manifest = read_json(path)
        manifest["protocol"] = "paper-roi"
        write_json(path, manifest)
        return path, manifest

    @staticmethod
    def portable_rng_state(seed):
        """Create valid CPU RNG inputs without importing or initializing CUDA."""
        import torch
        numpy_state = np.random.RandomState(seed).get_state()
        metadata = dict(numpy_name=numpy_state[0], numpy_position=numpy_state[2],
                        numpy_has_gauss=numpy_state[3], numpy_cached_gaussian=numpy_state[4],
                        python=random.Random(seed).getstate())
        return dict(rng_torch_cpu=torch.Generator(device="cpu").manual_seed(seed).get_state().numpy(),
                    rng_torch_cuda=np.empty(0, np.uint8), rng_numpy_keys=numpy_state[1],
                    rng_json=np.asarray(json.dumps(metadata)))

    def test_query_camera_transfer_repredicts_candidates_and_keeps_native_camera_observations(self):
        self.config_data["experiments"]["novel_unknown"]["camera_source"] = "query"
        config = self.config()
        self.assertEqual(experiment(config, "novel_unknown").camera_source, "query")
        _, output = self.run_prediction(config, ["novel_unknown"])
        record = read_json(output / "predictions" / "novel_unknown" / "query.json")
        self.assertEqual(record["camera_source"], "query")
        # Selection observes the camera each candidate estimates natively, never the reused one.
        np.testing.assert_allclose(record["candidate_cameras"], [[.1], [.2], [.3]], rtol=0, atol=1e-12)
        with np.load(output / "predictions" / "novel_unknown" / "query.npz", allow_pickle=False) as arrays:
            self.assertTrue(np.all(arrays["direct"] == 2.0))
            for arm, native in (("candidate0", 1.5), ("candidate1", 1.0), ("candidate2", 3.0)):
                self.assertTrue(np.all(arrays[arm] == np.float32(10 * native)), arm)
        # Pairwise responses come from the saved (transferred) candidate depths.
        self.assertAlmostEqual(record["pairwise"][1][0], float(np.log(15.0 / 2.0)), places=6)
        engine = self.worker_engines[-1]
        self.assertEqual(engine.transferred_calls,
                         [(f"candidate{i}", {"estimated_on": "query"}) for i in range(3)])
        # The same manifest without transfer has a different cache identity.
        self.config_data["experiments"]["novel_unknown"].pop("camera_source")
        from evaluation.cache import input_revision
        manifest = read_json(self.manifest())
        native = experiment(self.config(), "novel_unknown")
        self.assertNotEqual(input_revision(manifest["samples"][0], self.root, native, "query"),
                            input_revision(manifest["samples"][0], self.root, experiment(config, "novel_unknown"), "query"))

    def test_query_camera_transfer_needs_adapter_support_and_a_camera_state(self):
        self.config_data["experiments"]["novel_unknown"]["camera_source"] = "query"
        with self.assertRaisesRegex(ValueError, "no reusable query camera"):
            self.run_prediction(self.config(camera_state="missing"), ["novel_unknown"])
        self.config_data["models"]["brand_new_backend"]["implementation"] = "baselines/new_camera_model.py:Standalone"
        with self.assertRaisesRegex(TypeError, "supports_query_camera"):
            self.run_prediction(self.config(camera_state="present"), ["novel_unknown"])

    def test_query_camera_transfer_is_an_unknown_camera_option(self):
        self.config_data["experiments"]["novel_known"]["camera_source"] = "query"
        with self.assertRaisesRegex(ValueError, "camera_source: query requires camera: unknown"):
            self.config()
        self.config_data["experiments"]["novel_known"]["camera_source"] = "elsewhere"
        with self.assertRaisesRegex(ValueError, "camera_source must be native or query"):
            self.config()

    def test_default_settings_predict_fresh_candidates_without_rng_assets(self):
        # Keep the shipped experiments, replacing only heavyweight backends.
        config = load_config()
        names = configuration_names(config)
        # Seven models, each without and with the ground-truth camera.
        self.assertEqual(len(names), 14)
        self.assertEqual(sorted(name for name in names if experiment(config, name).camera_source == "query"),
                         ["unidepth_v2_vitl14", "unik3d_unknown"])
        self.assertTrue(all(experiment(config, name).context_fraction == .25 for name in names))
        self.assertTrue(all(not experiment(config, name).requires_rng for name in names))
        config["models"] = {name: dict(implementation=f"{self.plugin}:Baseline", rng_policy="caller")
                            for name in config["models"]}
        path, manifest = self.paper_manifest()
        self.assertNotIn("rng_states", manifest["samples"][0])
        output = self.root / "results"
        with self.cpu_workers() as pipeline:
            pipeline.predict(path, config, names, output=output, retrieval={"query": "reference"})
        for name in names:
            with self.subTest(setting=name):
                record = read_json(output / "predictions" / name / "query.json")
                self.assertEqual(len(record["inference"]), 4)
                self.assertTrue(all(item["legacy_rng_policy"] == "caller" for item in record["inference"]))

    def test_explicit_strict_rng_requirement_rejects_missing_states_before_startup(self):
        from evaluation import pipeline
        self.config_data["experiments"]["novel_unknown"]["requires_rng"] = True
        config = self.config()
        path, _ = self.paper_manifest()
        with patch.object(pipeline, "_pool") as pool:
            with self.assertRaisesRegex(ValueError, "requires exported before-input RNG states"):
                pipeline.predict(path, config, ["novel_unknown"], output=self.root / "results",
                                 retrieval={"query": "reference"})
        pool.assert_not_called()

    def test_imported_query_and_reference_rng_are_restored_and_invalidate_their_caches(self):
        import torch
        from contextdepth import baseline
        previous_torch, previous_numpy, previous_python = torch.get_rng_state(), np.random.get_state(), random.getstate()
        self.addCleanup(torch.set_rng_state, previous_torch)
        self.addCleanup(np.random.set_state, previous_numpy)
        self.addCleanup(random.setstate, previous_python)
        config = self.config()
        path, manifest = self.paper_manifest()
        output = self.root / "results"
        query_record = output / "predictions" / "novel_unknown" / "query.json"
        reference_record = output / "references" / "novel_calibration" / "reference.json"

        def predict():
            with self.cpu_workers() as pipeline:
                pipeline.predict(path, config, ["novel_unknown"], output=output, retrieval={"query": "reference"})

        predict()
        old_keys = [read_json(record)["cache_key"] for record in (query_record, reference_record)]
        arms = ("direct", "candidate0", "candidate1", "candidate2")
        states = {arm: self.portable_rng_state(index + 10) for index, arm in enumerate(arms)}
        reference_state = self.portable_rng_state(100)
        query_rng, reference_rng = self.root / "query-rng.npz", self.root / "reference-rng.npz"
        write_npz(query_rng, **{f"{arm}__{key}": value for arm, state in states.items() for key, value in state.items()})
        write_npz(reference_rng, **{f"direct__{key}": value for key, value in reference_state.items()})
        manifest["samples"][0]["rng_states"] = {"novel_unknown": query_rng.name}
        manifest["references"][0]["reference_rng_states"] = {"novel_calibration": reference_rng.name}
        write_json(path, manifest)
        with patch.object(baseline, "restore_rng_state", wraps=baseline.restore_rng_state) as restore:
            predict()
        self.assertEqual(restore.call_count, 5)
        for call, expected in zip(restore.call_args_list, [reference_state, *states.values()]):
            self.assertEqual(set(call.args[0]), set(expected))
            for key in expected:
                np.testing.assert_array_equal(call.args[0][key], expected[key])
        keys = [read_json(record)["cache_key"] for record in (query_record, reference_record)]
        self.assertTrue(all(old != new for old, new in zip(old_keys, keys)))
        for record in (query_record, reference_record):
            self.assertTrue(all(item["legacy_rng_policy"] == "supplied_before_input_state"
                                for item in read_json(record)["inference"]))

        # Requiring an already supplied state changes validation, not inference.
        config["experiments"]["novel_unknown"]["requires_rng"] = True
        predict()
        self.assertTrue(read_json(output / "execution.json")["cache_hit_only"])
        self.assertEqual([read_json(record)["cache_key"] for record in (query_record, reference_record)], keys)

        states["candidate1"] = self.portable_rng_state(200)
        stamp = query_rng.stat()
        write_npz(query_rng, **{f"{arm}__{key}": value for arm, state in states.items() for key, value in state.items()})
        os.utime(query_rng, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
        with patch.object(baseline, "restore_rng_state", wraps=baseline.restore_rng_state) as restore:
            predict()
        self.assertEqual(restore.call_count, 4)
        self.assertNotEqual(read_json(query_record)["cache_key"], keys[0])
        self.assertEqual(read_json(reference_record)["cache_key"], keys[1])
        for key in states["candidate1"]:
            np.testing.assert_array_equal(restore.call_args_list[2].args[0][key], states["candidate1"][key])

    def test_optional_rng_does_not_silently_ignore_incomplete_imported_states(self):
        config = self.config()
        path, manifest = self.paper_manifest()
        states = self.root / "incomplete-rng.npz"
        manifest["samples"][0]["rng_states"] = {"novel_unknown": states.name}
        write_json(path, manifest)
        cases = (
            ({"direct__rng_torch_cpu": np.zeros(1, np.uint8)}, "Missing before-input RNG state"),
            ({f"{arm}__rng_torch_cpu": np.zeros(1, np.uint8)
              for arm in ("direct", "candidate0", "candidate1", "candidate2")}, "RNG state is missing"),
        )
        for arrays, error in cases:
            with self.subTest(error=error):
                write_npz(states, **arrays)
                with self.cpu_workers() as pipeline, self.assertRaisesRegex(ValueError, error):
                    pipeline.predict(path, config, ["novel_unknown"], output=self.root / "results",
                                     retrieval={"query": "reference"})

    def batch_manifest(self):
        path = self.manifest()
        manifest = read_json(path)
        original = manifest["samples"][0]
        for index in (1, 2):
            row = deepcopy(original)
            row.update(uid=f"query{index}", scene_group=f"query_scene{index}")
            row["image"] = f"query{index}.png"
            (self.root / row["image"]).write_bytes((self.root / original["image"]).read_bytes())
            row["candidates"] = []
            for candidate, source in enumerate(original["candidates"]):
                destination = f"query{index}_candidate{candidate}.png"
                (self.root / destination).write_bytes((self.root / source).read_bytes())
                row["candidates"].append(destination)
            manifest["samples"].append(row)
        write_json(path, manifest)
        return path, {row["uid"]: "reference" for row in manifest["samples"]}

    def test_partial_batch_resume_recomputes_its_original_window(self):
        self.config_data["depth_batch_size"] = 2
        config = self.config()
        path, retrieval = self.batch_manifest()
        output = self.root / "results"
        with self.cpu_workers() as pipeline:
            pipeline.predict(path, config, ["novel_unknown"], output=output, retrieval=retrieval)
        first_engine = self.worker_engines[-1]
        original_window_calls = [batch for batch in first_engine.native_calls
                                 if all(uid in ("query", "query1") for uid, _ in batch)]
        folder = output / "predictions" / "novel_unknown"
        old_keys = {uid: read_json(folder / f"{uid}.json")["cache_key"] for uid in retrieval}
        (folder / "query1.npz").unlink()
        with self.cpu_workers() as pipeline:
            pipeline.predict(path, config, ["novel_unknown"], output=output, retrieval=retrieval,
                             gpus=(0, 1), workers_per_gpu=2)
        self.assertEqual(self.worker_engines[-1].native_calls, original_window_calls)
        self.assertEqual({uid for batch in self.worker_engines[-1].native_calls for uid, _ in batch},
                         {"query", "query1"})
        self.assertEqual(self.pool_initializations, ["novel_calibration", "novel_unknown"])
        for uid in retrieval:
            self.assertEqual(read_json(folder / f"{uid}.json")["cache_key"], old_keys[uid])
        self.assertEqual(read_json(folder / "query.json")["batch_inputs"], ["query", "query1"])

    def test_changed_batch_neighbor_invalidates_prediction_selection_and_evaluation(self):
        self.config_data["depth_batch_size"] = 2
        config = self.config()
        path, retrieval = self.batch_manifest()
        output = self.root / "results"
        with self.cpu_workers() as pipeline:
            pipeline.predict(path, config, ["novel_unknown"], output=output, retrieval=retrieval)
        pipeline.select(path, config, ["novel_unknown"], output=output, workers=1)
        folder = output / "predictions" / "novel_unknown"
        old_keys = {uid: read_json(folder / f"{uid}.json")["cache_key"] for uid in retrieval}
        changed = self.root / "query1_candidate0.png"
        stamp = changed.stat()
        with Image.open(changed) as image:
            pixels = np.array(image, copy=True)
        self.assertGreater(self.geometry.roi_x, 0)
        pixels[0, 0, 0] ^= 1
        Image.fromarray(pixels).save(changed)
        os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
        # query itself is unchanged: its cached observation must still be
        # rejected because the other image in its numerical batch changed.
        for stage in (pipeline.select, pipeline.evaluate):
            with self.subTest(stage=stage.__name__):
                with self.assertRaisesRegex(ValueError, "novel_unknown/query; rerun predict"):
                    stage(path, config, ["novel_unknown"], output=output, workers=1)
        with self.cpu_workers() as pipeline:
            pipeline.predict(path, config, ["novel_unknown"], output=output, retrieval=retrieval)
        observed = {uid for batch in self.worker_engines[-1].native_calls for uid, _ in batch}
        self.assertEqual(observed, {"query", "query1"})
        for uid in ("query", "query1"):
            self.assertNotEqual(read_json(folder / f"{uid}.json")["cache_key"], old_keys[uid])
        self.assertEqual(read_json(folder / "query2.json")["cache_key"], old_keys["query2"])
        pipeline.select(path, config, ["novel_unknown"], output=output, workers=1)
        pipeline.evaluate(path, config, ["novel_unknown"], output=output, workers=1)

    def test_unregistered_backend_runs_both_selectors_and_scores_the_selected_depth(self):
        from evaluation import pipeline
        config = self.config()
        names = ["novel_unknown", "novel_known"]
        path, output = self.run_prediction(config, names)
        pipeline.select(path, config, names, output=output, workers=1)
        rows = pipeline.evaluate(path, config, names, output=output, workers=1)
        for name in names:
            decision = read_json(output / "selected" / name / "query.json")
            self.assertEqual(decision["selected_arm"], "candidate1")
            with np.load(output / "selected" / name / "query.npz", allow_pickle=False) as result:
                np.testing.assert_array_equal(result["depth"], np.ones((4, 4), np.float32))
            metrics = {r["method"]: r for r in rows if r["config"] == name and r["dataset"] == "All"}
            self.assertEqual(metrics["selector"]["delta1"], 1.0)
            self.assertEqual(metrics["selector"]["absrel"], 0.0)
            self.assertEqual(metrics["direct"]["absrel"], 1.0)
            self.assertEqual(metrics["seed0"]["absrel"], .5)

    def test_unknown_selector_rejects_a_backend_without_camera_observations(self):
        with self.assertRaisesRegex(ValueError, "[Cc]amera"):
            self.run_prediction(self.config(camera_mode="missing"), ["novel_unknown"])

    def test_unknown_selector_rejects_inconsistent_camera_dimensions(self):
        with self.assertRaisesRegex(ValueError, "[Cc]amera|dimension|shape"):
            self.run_prediction(self.config(camera_mode="inconsistent"), ["novel_unknown"])

    def test_reference_and_query_for_one_experiment_share_one_model_initialization(self):
        self.config_data["experiments"]["novel_known"]["reference"] = "novel_known"
        self.config_data["depth_batch_size"] = 3
        config = self.config()
        path, output = self.run_prediction(config, ["novel_known"])
        self.assertEqual(self.pool_initializations, ["novel_known"])
        self.assertEqual(len(self.worker_engines), 1)
        requests = [r for batch in self.worker_engines[0].native_calls for r in batch]
        self.assertEqual(requests, [("reference", "reference"), ("query", "direct"),
                                    ("query", "candidate0"), ("query", "candidate1"), ("query", "candidate2")])
        self.assertEqual([len(batch) for batch in self.worker_engines[0].native_calls], [1, 1, 3])
        execution = read_json(output / "execution.json")
        self.assertEqual(execution["model_initializations"], 1)
        self.assertEqual({job["role"] for job in execution["jobs"]}, {"reference", "query"})
        with self.cpu_workers() as pipeline:
            pipeline.predict(path, config, ["novel_known"], output=output, retrieval={"query": "reference"})
        self.assertEqual(self.pool_initializations, ["novel_known"])
        self.assertTrue(read_json(output / "execution.json")["cache_hit_only"])

    def test_reference_known_and_unknown_share_one_resident_model_with_correct_context(self):
        config = self.config()
        names = ["novel_unknown", "novel_known"]
        path, output = self.run_prediction(config, names)
        self.assertEqual(len(self.pool_initializations), 1)
        self.assertEqual(len(self.worker_engines), 1)
        contexts = self.worker_engines[0].native_contexts
        self.assertEqual({context["name"] for context in contexts},
                         {"novel_calibration", "novel_unknown", "novel_known"})
        for context in contexts:
            known = context["name"] != "novel_unknown"
            self.assertEqual(context["setting"], "known" if known else "unknown")
            self.assertEqual(context["known"], known)
            self.assertEqual(context["matmul_tf32"], known)
            self.assertEqual(set(context["request_settings"]), {context["setting"]})
        for stage, name, known in (("references", "novel_calibration", True),
                                   ("predictions", "novel_known", True),
                                   ("predictions", "novel_unknown", False)):
            uid = "reference" if stage == "references" else "query"
            record = read_json(output / stage / name / f"{uid}.json")
            self.assertEqual(record["config"], name)
            for metadata in record["inference"]:
                self.assertEqual(metadata["model"], name)
                self.assertEqual(metadata["camera_setting"], "known" if known else "unknown")
                self.assertEqual(metadata["numerical_flags"]["cuda_matmul_tf32"], known)
        known_record = read_json(output / "predictions" / "novel_known" / "query.json")
        unknown_record = read_json(output / "predictions" / "novel_unknown" / "query.json")
        self.assertNotEqual(known_record["cache_key"], unknown_record["cache_key"])
        self.assertEqual(read_json(output / "execution.json")["model_initializations"], 1)
        with self.cpu_workers() as pipeline:
            pipeline.predict(path, config, names, output=output, retrieval={"query": "reference"})
        self.assertEqual(len(self.pool_initializations), 1)
        self.assertTrue(read_json(output / "execution.json")["cache_hit_only"])

    def test_different_weight_resources_cannot_share_a_resident_model(self):
        weight_a, weight_b = self.root / "weights-a.pth", self.root / "weights-b.pth"
        weight_a.write_bytes(b"first fake checkpoint")
        weight_b.write_bytes(b"second fake checkpoint")
        self.config_data["models"]["brand_new_backend"]["weights"] = str(weight_a)
        alternative = deepcopy(self.config_data["models"]["brand_new_backend"])
        alternative["weights"] = str(weight_b)
        self.config_data["models"]["alternative_backend"] = alternative
        self.config_data["experiments"]["novel_known"]["baseline"] = "alternative_backend"
        self.run_prediction(self.config(), ["novel_unknown", "novel_known"])
        self.assertEqual(len(self.worker_engines), 2)
        self.assertEqual({engine.paths["weights"] for engine in self.worker_engines}, {str(weight_a), str(weight_b)})

    def test_different_model_options_cannot_share_a_resident_model(self):
        self.config_data["experiments"]["novel_known"]["options"] = {"coordinate_depth": True}
        self.run_prediction(self.config(), ["novel_unknown", "novel_known"])
        self.assertEqual(len(self.worker_engines), 2)
        self.assertEqual({bool(engine.options.get("coordinate_depth")) for engine in self.worker_engines}, {False, True})

    def test_plugin_without_configure_runs_each_context_without_reuse(self):
        self.config_data["models"]["brand_new_backend"]["implementation"] = "baselines/new_camera_model.py:Standalone"
        path, output = self.run_prediction(self.config(), ["novel_unknown", "novel_known"])
        self.assertEqual(len(self.worker_engines), 3)
        for engine in self.worker_engines:
            self.assertFalse(callable(getattr(engine, "configure", None)))
            contexts = engine.native_contexts
            self.assertEqual(len({(row["name"], row["setting"]) for row in contexts}), 1)
        for name in ("novel_unknown", "novel_known"):
            record = read_json(output / "predictions" / name / "query.json")
            self.assertTrue(all(row["model"] == name for row in record["inference"]))
            self.assertTrue(all(row["camera_setting"] == ("known" if name == "novel_known" else "unknown")
                                for row in record["inference"]))
        with self.cpu_workers() as pipeline:
            pipeline.predict(path, self.config(), ["novel_unknown", "novel_known"],
                             output=output, retrieval={"query": "reference"})
        self.assertEqual(len(self.worker_engines), 3)

    def test_adding_another_camera_setting_keeps_the_existing_known_cache(self):
        config = self.config()
        path, output = self.run_prediction(config, ["novel_known"])
        folder = output / "predictions" / "novel_known"
        files = [folder / "query.json", folder / "query.npz"]
        before = [(file.read_bytes(), file.stat().st_mtime_ns) for file in files]
        previous_engines = len(self.worker_engines)
        with self.cpu_workers() as pipeline:
            pipeline.predict(path, config, ["novel_unknown", "novel_known"],
                             output=output, retrieval={"query": "reference"})
        self.assertEqual([(file.read_bytes(), file.stat().st_mtime_ns) for file in files], before)
        new_contexts = [context for engine in self.worker_engines[previous_engines:] for context in engine.native_contexts]
        self.assertTrue(new_contexts)
        self.assertEqual({context["name"] for context in new_contexts}, {"novel_unknown"})
        fresh_output = self.root / "mixed-fresh"
        with self.cpu_workers() as pipeline:
            pipeline.predict(path, config, ["novel_unknown", "novel_known"],
                             output=fresh_output, retrieval={"query": "reference"})
        fresh = fresh_output / "predictions" / "novel_known"
        self.assertEqual(read_json(fresh / "query.json")["cache_key"], read_json(files[0])["cache_key"])
        with np.load(fresh / "query.npz") as fresh_arrays, np.load(files[1]) as original_arrays:
            self.assertEqual(fresh_arrays.files, original_arrays.files)
            for arm in original_arrays.files:
                np.testing.assert_array_equal(fresh_arrays[arm], original_arrays[arm])

    def test_selection_and_evaluation_reject_changed_query_or_reference_recipes(self):
        from evaluation import pipeline
        config = self.config()
        names = ["novel_unknown"]
        path, output = self.run_prediction(config, names)
        pipeline.select(path, config, names, output=output, workers=1)
        for label, options in (("novel_unknown", {"camera_dimensions": 2}),
                               ("novel_calibration", {"matmul_tf32": True})):
            changed = deepcopy(config)
            changed["experiments"][label]["options"] = options
            for stage in (pipeline.select, pipeline.evaluate):
                with self.subTest(experiment=label, stage=stage.__name__):
                    with self.assertRaisesRegex(ValueError, "rerun predict"):
                        stage(path, changed, names, output=output, workers=1)
        # Rejected recipes have not invalidated or rewritten the original run.
        rows = pipeline.evaluate(path, config, names, output=output, workers=1)
        self.assertEqual(next(r["absrel"] for r in rows
                              if r["dataset"] == "All" and r["method"] == "selector"), 0.0)


if __name__ == "__main__":
    unittest.main()
