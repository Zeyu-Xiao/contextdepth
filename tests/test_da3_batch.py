"""DA3 recovery isolation without upstream packages, checkpoints, or a GPU."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.da3 import DA3Metric, DA3Nested, _defer_recovery, _slice_batch
from contextdepth.baseline import DepthBaseline
from contextdepth.types import InferenceInput


class RecoveryBranch:
    """Reductions deliberately mix images unless recovery receives B=1."""

    def __init__(self, metric=False):
        self.metric = metric
        self.recovery_shapes = []

    def forward(self, images):
        b, n, _, h, w = images.shape
        depth = images[:, :, 0].clone() + 2
        if self.metric:
            depth = depth.square()
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(b, n, 1, 1)
        intrinsics[..., 0, 0] = w + images[:, :, 0, 0, 0]
        intrinsics[..., 1, 1] = h + images[:, :, 0, 0, 0]
        return self._process_mono_sky_estimation(dict(depth=depth, intrinsics=intrinsics))

    def _process_mono_sky_estimation(self, output):
        self.recovery_shapes.append(tuple(output["depth"].shape))
        # Like sky filling, a global reduction is only independent after split.
        output["depth"] = output["depth"] + output["depth"].amax() / 20
        return output


class NestedRecovery:
    def __init__(self):
        self.da3 = RecoveryBranch()
        self.da3_metric = RecoveryBranch(metric=True)
        self.recovery_shapes = []

    def forward(self, images):
        output, metric = self.da3.forward(images), self.da3_metric.forward(images)
        output = self._apply_metric_scaling(output, metric)
        output = self._apply_depth_alignment(output, metric)
        return self._handle_sky_regions(output, metric)

    def _apply_metric_scaling(self, output, metric):
        self.recovery_shapes.append(tuple(output["depth"].shape))
        metric["depth"] = metric["depth"] * output["intrinsics"][..., 0, 0].mean() / 30
        return output

    def _apply_depth_alignment(self, output, metric):
        self.recovery_shapes.append(tuple(output["depth"].shape))
        scale = metric["depth"].mean() / output["depth"].mean()
        output["depth"] = output["depth"] * scale + torch.rand(()) / 100
        return output

    def _handle_sky_regions(self, output, metric):
        self.recovery_shapes.append(tuple(output["depth"].shape))
        output["depth"][..., 0, :] = metric["depth"].amax()
        return output


class TinyAPI:
    def __init__(self, nested):
        self.model = NestedRecovery() if nested else RecoveryBranch(metric=True)
        self.forward_shapes = []

    def _run_model_forward(self, images, *args):
        self.forward_shapes.append(tuple(images.shape))
        return self.model.forward(images)

    def _convert_to_prediction(self, output):
        assert output["depth"].shape[:2] == (1, 1)
        return SimpleNamespace(depth=output["depth"][0].numpy())


def adapter(cls, known):
    result = cls.__new__(cls)
    result.source_files = ()
    DepthBaseline.__init__(result, "arbitrary-label", dict(
        setting="known" if known else "unknown", batch_size=2), "cpu")
    result.model = TinyAPI(result.nested)
    return result


def prepared_requests(known):
    k = np.array([[40., 0, 21], [0, 35., 14], [0, 0, 1]], np.float32)
    items, images = [], []
    for index, level in enumerate((.2, 2.5)):
        request = InferenceInput(str(index), "direct", np.zeros((28, 42, 3), np.uint8),
                                 camera=k.copy() if known else None,
                                 setting="known" if known else "unknown")
        items.append(dict(request=request, geometry=request.geometry, known=known,
                          resolution=42, camera=request.camera))
        images.append(torch.linspace(0, 1, 28 * 42).reshape(1, 28, 42).repeat(3, 1, 1) + level)
    return dict(items=items, images=torch.stack(images)[:, None], extrinsics=None, intrinsics=None)


def run_stages(backend, prepared):
    with torch.inference_mode():
        return backend.postprocess(backend.infer_model(prepared), prepared)


class DA3BatchRecoveryTests(unittest.TestCase):
    def setUp(self):
        for attribute in ("is_available", "is_bf16_supported"):
            self.enterContext(patch.object(torch.cuda, attribute, return_value=False))

    def test_recovery_is_per_image_and_invariant_to_batch_order(self):
        for cls, known in ((DA3Nested, False), (DA3Nested, True), (DA3Metric, True)):
            with self.subTest(adapter=cls.__name__, known=known):
                prepared = prepared_requests(known)
                singleton, batched = adapter(cls, known), adapter(cls, known)
                expected = []
                for index in range(2):
                    one = dict(prepared, items=prepared["items"][index:index + 1],
                               images=prepared["images"][index:index + 1])
                    expected.extend(run_stages(singleton, one))
                actual = run_stages(batched, prepared)
                self.assertEqual(batched.model.forward_shapes, [(2, 1, 3, 28, 42)])
                self.assertEqual(singleton.model.forward_shapes, [(1, 1, 3, 28, 42)] * 2)
                reverse = dict(prepared, items=list(reversed(prepared["items"])),
                               images=prepared["images"].flip(0))
                reordered = run_stages(batched, reverse)
                for target, observed, reverse_observed in zip(expected, actual, reversed(reordered)):
                    np.testing.assert_array_equal(observed.depth, target.depth)
                    np.testing.assert_array_equal(reverse_observed.depth, target.depth)
                    self.assertEqual(observed.metadata["inference_seed"], target.metadata["inference_seed"])
                    if target.camera is not None:
                        np.testing.assert_array_equal(observed.camera.intrinsics, target.camera.intrinsics)
                network = batched.model.model
                components = (network, network.da3, network.da3_metric) if batched.nested else (network,)
                for component in components:
                    self.assertTrue(component.recovery_shapes)
                    self.assertTrue(all(shape[:2] == (1, 1) for shape in component.recovery_shapes))

    def test_recovery_callbacks_are_restored_after_a_forward_exception(self):
        network = NestedRecovery()
        owners = ((network.da3, "_process_mono_sky_estimation"),
                  (network.da3_metric, "_process_mono_sky_estimation"),
                  (network, "_apply_metric_scaling"),
                  (network, "_apply_depth_alignment"), (network, "_handle_sky_regions"))
        originals = [getattr(owner, name) for owner, name in owners]
        with self.assertRaisesRegex(RuntimeError, "synthetic neural failure"):
            with _defer_recovery(network, True):
                raise RuntimeError("synthetic neural failure")
        for (owner, name), original in zip(owners, originals):
            self.assertNotIn(name, owner.__dict__)
            self.assertEqual(getattr(owner, name), original)

    def test_recovery_mutations_cannot_change_raw_outputs_or_other_images(self):
        raw = dict(intrinsics=torch.eye(3).repeat(2, 1, 1),
                   nested=[torch.arange(4).reshape(2, 2)])
        first, second = (_slice_batch(raw, index, 2, torch) for index in range(2))
        first["intrinsics"][0, 0, 0] = 100
        first["nested"][0][0, 0] = 100
        self.assertEqual(float(raw["intrinsics"][0, 0, 0]), 1.)
        self.assertEqual(float(second["intrinsics"][0, 0, 0]), 1.)
        self.assertEqual(int(raw["nested"][0][0, 0]), 0)
        self.assertEqual(int(second["nested"][0][0, 0]), 2)


if __name__ == "__main__":
    unittest.main()
