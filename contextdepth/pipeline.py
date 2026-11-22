"""Normal RGB inference: generation and depth adapters, with optional DIODE selection.

This module does not load evaluation manifests, crop dataset ROIs, or score GT.
The paper runner lives in ``evaluation`` and uses the same model adapters and
selection mathematics with its recorded preprocessing protocol.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import camera_features, crop_prediction, pixel_intrinsics
from .config import load_runtime_config
from .generation import Outpainting, create_outpainter
from .io import load_rgb, write_json, write_npz
from .models import load_implementation
from .selection import (Selection, pairwise_log_ratios, reference_bias,
                        reference_camera, select_known, select_unknown)
from .types import InferenceInput, Prediction


def _image(value):
    if isinstance(value, (str, Path)):
        return load_rgb(value)
    if isinstance(value, Image.Image):
        value = np.asarray(value.convert("RGB"))
    rgb = np.asarray(value)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected a nonempty uint8 RGB array, PIL image, or image path")
    return np.ascontiguousarray(rgb)


def _save_prediction(path, prediction):
    arrays = {"depth": prediction.depth}
    if prediction.camera is not None:
        for key in ("intrinsics", "focal_px", "rays"):
            value = getattr(prediction.camera, key)
            if value is not None:
                arrays[key] = value
    write_npz(path, **arrays)


@dataclass
class ContextPrediction:
    outpainting: Outpainting
    prediction: Prediction


@dataclass
class ContextDepthResult:
    """Full-image outputs; selected is present only when DIODE is enabled."""
    direct: Prediction
    candidates: list[ContextPrediction]
    selected: Prediction | None = None
    selection: Selection | None = None
    reference_uid: str | None = None

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _save_prediction(directory / "direct.npz", self.direct)
        records = []
        for i, candidate in enumerate(self.candidates):
            Image.fromarray(candidate.outpainting.rgb).save(directory / f"candidate{i}.png")
            _save_prediction(directory / f"candidate{i}.npz", candidate.prediction)
            records.append({"seed": candidate.outpainting.seed,
                            "query_bbox": list(candidate.outpainting.bbox)})
        if self.selected is not None:
            _save_prediction(directory / "selected.npz", self.selected)
        elif (directory / "selected.npz").is_file():
            (directory / "selected.npz").unlink()
        write_json(directory / "result.json", {
            "candidates": records, "reference_uid": self.reference_uid,
            "selected_arm": None if self.selection is None else self.selection.selected_arm,
            "depth_unit": "metres", "depth_semantics": "camera Z"})
        return directory


class ContextDepthPipeline:
    """Load OneReward and MoGe-2 lazily, or supply custom black-box adapters.

    ``generate`` and ``estimate_depth`` can be used separately. Calling the
    pipeline returns the original image prediction and all generated-image
    predictions. Reference selection is disabled by default; opt in with
    ``reference_bank='diode', diode_root=...`` after downloading DIODE.
    """

    def __init__(self, *, config=None, depth_model=None, outpainter=None,
                 device="cuda:0", depth_batch_size=None, expansion=None, seeds=None,
                 reference_bank=None, diode_root=None, depth_options=None,
                 generation_options=None, outpainter_paths=None):
        self.config = load_runtime_config(config)
        self.device = str(device)
        self.depth_model = self.config.get("default_depth_model", "moge2") if depth_model is None else depth_model
        self._depth_owned = not (hasattr(self.depth_model, "predict") and not isinstance(self.depth_model, type))
        self.expansion = self.config.get("expansion", 0.1) if expansion is None else expansion
        self.seeds = tuple(self.config.get("seeds", (0, 1, 2)) if seeds is None else seeds)
        configured_batch = (self.config.get("depth_batch_size", 1) if self._depth_owned else
                            getattr(self.depth_model, "max_batch_size", 1))
        self.depth_batch_size = configured_batch if depth_batch_size is None else depth_batch_size
        if (isinstance(self.depth_batch_size, bool) or not isinstance(self.depth_batch_size, int)
                or self.depth_batch_size < 1):
            raise ValueError("depth_batch_size must be a positive integer")
        if not self._depth_owned and self.depth_batch_size != configured_batch:
            raise ValueError("A supplied depth instance keeps its configured batch size")
        self.depth_options = deepcopy(depth_options or {})
        self.generation_options = deepcopy(generation_options or {})
        self.outpainter_paths = deepcopy(outpainter_paths or {})
        self._outpainter_spec, self._outpainter = outpainter, None
        self._outpainter_owned = outpainter is None or isinstance(outpainter, (str, type))
        self._depth, self._depth_setting = None, None
        self._reference_calibration = {}
        if reference_bank not in (None, False, "diode"):
            raise ValueError("Only reference_bank='diode' is supported; omit it to disable selection")
        if reference_bank == "diode":
            if diode_root is None:
                raise ValueError("Download DIODE and provide diode_root to enable reference selection")
            from .reference import DIODEReferenceBank
            self._bank = DIODEReferenceBank(
                diode_root, checkpoint=self.config.get("retrieval", {}).get("checkpoint"), device=self.device)
        else:
            if diode_root is not None:
                raise ValueError("Set reference_bank='diode' to use diode_root")
            self._bank = None

    def generate(self, rgb, *, expansion=None, seeds=None) -> list[Outpainting]:
        image = _image(rgb)
        if self._outpainter is None:
            paths = deepcopy(self.config.get("outpainting", {})) if self._outpainter_spec is None else {}
            paths.update(self.outpainter_paths)
            paths["options"] = {**paths.get("options", {}), **self.generation_options}
            self._outpainter = create_outpainter(paths, device=self.device, implementation=self._outpainter_spec)
        return self._outpainter.generate(image,
            expansion=self.expansion if expansion is None else expansion,
            seeds=self.seeds if seeds is None else seeds)

    def _depth_config(self, known):
        model = self.depth_model
        if isinstance(model, str) and model in self.config.get("models", {}):
            paths = deepcopy(self.config["models"][model])
            factory = load_implementation(paths["implementation"])
            name = model
        elif isinstance(model, (str, type)):
            factory = load_implementation(model) if isinstance(model, str) else model
            paths = {}
            name = getattr(factory, "__name__", "custom_depth")
        else:
            factory = None
            paths = deepcopy(getattr(model, "paths", {}))
            name = getattr(model, "name", type(model).__name__)
        # Resource paths for a custom adapter can be supplied without a YAML
        # experiment. Options are passed through; the pipeline never branches
        # on model class or upstream model names.
        paths.update(deepcopy(self.depth_options))
        paths.update(setting="known" if known else "unknown", batch_size=self.depth_batch_size)
        return factory, paths, name

    def _get_depth(self, known):
        factory, paths, name = self._depth_config(known)
        setting = paths["setting"]
        if self._depth is None:
            self._depth = factory(name, paths, self.device) if factory is not None else self.depth_model
            self._depth_setting = getattr(self._depth, "setting", setting)
        if self._depth_setting != setting:
            configure = getattr(self._depth, "configure", None)
            if getattr(self._depth, "supports_setting_reuse", False) and callable(configure):
                configure(name, paths)
            elif self._depth_owned:
                self.release_depth()
                self._depth = factory(name, paths, self.device)
            else:
                raise ValueError("The supplied depth instance cannot switch camera settings")
            self._depth_setting = setting
        if not callable(getattr(self._depth, "predict", None)):
            raise TypeError("A depth adapter must implement predict(requests)")
        return self._depth

    def estimate_depths(self, images, *, cameras=None) -> list[Prediction]:
        images = [_image(image) for image in images]
        cameras = [None] * len(images) if cameras is None else list(cameras)
        if len(images) != len(cameras):
            raise ValueError("Supply one optional camera per RGB image")
        cameras = [None if camera is None else pixel_intrinsics(camera) for camera in cameras]
        result = [None] * len(images)
        for known in (False, True):
            indices = [i for i, camera in enumerate(cameras) if (camera is not None) == known]
            if not indices:
                continue
            engine = self._get_depth(known)
            requests = [InferenceInput(str(i), "direct", images[i], camera=cameras[i],
                                       setting="known" if known else "unknown") for i in indices]
            predictions = list(engine.predict(requests))
            if len(predictions) != len(indices):
                raise ValueError("Depth adapter returned an incomplete batch")
            for i, prediction in zip(indices, predictions):
                if (not isinstance(prediction, Prediction) or prediction.depth.dtype != np.float32
                        or prediction.depth.shape != images[i].shape[:2]):
                    raise ValueError("Depth adapter must return complete input-grid float32 depth")
                result[i] = prediction
        return result

    def estimate_depth(self, rgb, *, camera=None) -> Prediction:
        return self.estimate_depths([rgb], cameras=[camera])[0]

    def infer(self, rgb, *, camera=None, expansion=None, seeds=None,
              outpaintings=None) -> ContextDepthResult:
        image = _image(rgb)
        camera = None if camera is None else pixel_intrinsics(camera)
        contexts = (self.generate(image, expansion=expansion, seeds=seeds)
                    if outpaintings is None else list(outpaintings))
        if not contexts:
            raise ValueError("At least one outpainting is required")
        for context in contexts:
            if not isinstance(context, Outpainting):
                raise TypeError("Saved outpaintings must be Outpainting objects with a query bbox and seed")
            x0, y0, x1, y1 = context.bbox
            if not np.array_equal(context.rgb[y0:y1, x0:x1], image):
                raise ValueError("Outpainting must preserve the complete original RGB pixels")
        images = [image, *(context.rgb for context in contexts)]
        cameras = [camera, *(None if camera is None else context.map_intrinsics(camera) for context in contexts)]
        predictions = self.estimate_depths(images, cameras=cameras)
        result = ContextDepthResult(predictions[0], [ContextPrediction(context, prediction)
                                   for context, prediction in zip(contexts, predictions[1:])])
        if self._bank is not None:
            reference = self._bank.retrieve(image, source_path=rgb if isinstance(rgb, (str, Path)) else None)
            if reference.uid not in self._reference_calibration:
                prediction = self.estimate_depth(reference.rgb, camera=reference.camera)
                self._reference_calibration[reference.uid] = reference_bias(prediction.depth, reference.depth)
            bias = self._reference_calibration[reference.uid]
            candidates = [crop_prediction(item.prediction, item.outpainting.bbox) for item in result.candidates]
            pairwise = pairwise_log_ratios([result.direct.depth, *(p.depth for p in candidates)])
            if camera is not None:
                selection = select_known(pairwise[1:, 0], bias)
            else:
                height, width = image.shape[:2]
                observations = [camera_features(p.camera, width, height) for p in candidates]
                ref_h, ref_w = reference.rgb.shape[:2]
                ref_camera = reference_camera(reference.camera, ref_w, ref_h, len(observations[0]))
                selection = select_unknown(observations, pairwise[1:, 1:], ref_camera, bias)
            result.selection = selection
            result.reference_uid = reference.uid
            result.selected = result.direct if selection.is_direct else candidates[selection.choice]
        return result

    __call__ = infer

    def release_generation(self):
        if (self._outpainter is not None and self._outpainter_owned
                and callable(getattr(self._outpainter, "close", None))):
            self._outpainter.close()
        self._outpainter = None

    def release_depth(self):
        if self._depth is not None and self._depth_owned:
            close = getattr(self._depth, "close", None)
            if callable(close):
                close()
        self._depth, self._depth_setting = None, None

    def close(self):
        self.release_generation()
        self.release_depth()
        if self._bank is not None:
            self._bank.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
