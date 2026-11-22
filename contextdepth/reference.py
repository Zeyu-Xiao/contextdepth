"""Optional RGB-only retrieval from the downloaded MoGe-format DIODE bank."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .io import load_rgb


@dataclass(frozen=True)
class Reference:
    uid: str
    rgb: np.ndarray
    depth: np.ndarray
    camera: np.ndarray


@dataclass(frozen=True)
class _Entry:
    uid: str
    directory: Path
    scene: str


def _pixel_hash(rgb):
    digest = hashlib.sha256(str(rgb.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()


def _read_depth(path):
    """Decode the released 16-bit logarithmic PNG, including legacy units."""
    with Image.open(path) as image:
        try:
            near, far = float(image.info["near"]), float(image.info["far"])
            unit = float(image.info.get("unit", 1.))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Expected a MoGe-format DIODE depth PNG with near/far metadata: {path}") from error
        codes = np.asarray(image)
    if codes.ndim != 2 or not np.issubdtype(codes.dtype, np.integer) or codes.dtype.itemsize < 2:
        raise ValueError(f"Expected a two-dimensional 16-bit depth PNG: {path}")
    if not (np.isfinite([near, far, unit]).all() and 0 < near <= far and unit > 0):
        raise ValueError(f"Invalid depth range or unit in {path}")
    if np.any(codes < 0) or np.any(codes > 65535):
        raise ValueError(f"Depth codes outside the 16-bit range: {path}")
    t = (codes.astype(np.float32) - 1) / 65533
    depth = near ** (1 - t) * far ** t
    if unit != 1.:
        depth = depth * unit
    depth[codes == 0], depth[codes == 65535] = np.nan, np.inf
    return np.ascontiguousarray(depth, dtype=np.float32)


class _SiglipEncoder:
    """One local vision tower kept alive across bank and query batches."""

    def __init__(self, checkpoint, device):
        import torch
        from transformers import SiglipImageProcessor, SiglipVisionModel

        self.torch, self.device = torch, torch.device(device)
        self.processor = SiglipImageProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.model = SiglipVisionModel.from_pretrained(
            checkpoint, torch_dtype=torch.float32, local_files_only=True)
        self.model.to(self.device).eval().requires_grad_(False)

    def encode(self, rgbs):
        torch = self.torch
        inputs = self.processor(images=[Image.fromarray(rgb) for rgb in rgbs], return_tensors="pt")["pixel_values"]
        if self.device.type == "cuda":
            inputs = inputs.pin_memory()
        inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.backends.cudnn.benchmark)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                features = self.model(pixel_values=inputs).pooler_output
            return features.detach().float().cpu().numpy()
        finally:
            (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark) = flags

    def close(self):
        self.model = None
        self.processor = None


class DIODEReferenceBank:
    """Retrieve one different-scene DIODE reference using only RGB features.

    ``root`` is the extracted DIODE directory containing ``.index.txt`` from
    the MoGe evaluation release. ``checkpoint`` is a local SigLIP2 vision
    tower. Neither dataset nor weights are downloaded automatically.

    The encoder, image hashes, and reference features are cached in memory
    until close(). Depth and intrinsics are loaded only after selection.
    """

    def __init__(self, root, *, checkpoint, device="cuda:0", batch_size=16, _encoder_factory=None):
        if root is None:
            raise ValueError("DIODE reference selection requires diode_root; download and extract MoGe-format DIODE first")
        self.root = Path(root).expanduser().resolve()
        index = self.root / ".index.txt"
        if not self.root.is_dir() or not index.is_file():
            raise FileNotFoundError(
                f"DIODE data is not installed at {self.root}. Download and extract DIODE from "
                "https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation "
                "and point diode_root to the directory containing .index.txt.")
        if checkpoint is None or not Path(checkpoint).expanduser().is_dir():
            raise FileNotFoundError("DIODE retrieval requires a downloaded local SigLIP2 vision-tower checkpoint directory")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Reference encoder batch_size must be a positive integer")
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.device, self.batch_size = device, batch_size
        self._encoder_factory = _encoder_factory or _SiglipEncoder
        self._encoder, self._features = None, None
        self._pixel_scenes = {}
        self._closed = False
        entries, seen = [], set()
        for line in index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            relative = PurePosixPath(line.strip())
            parts = relative.parts
            if (relative.is_absolute() or ".." in parts or len(parts) != 5
                    or parts[1] not in ("indoors", "outdoor")
                    or not parts[2].startswith("scene_") or not parts[3].startswith("scan_")):
                raise ValueError(f"Expected MoGe-format DIODE split/environment/scene/scan/frame entry, got {line!r}")
            if str(relative) in seen:
                raise ValueError(f"Duplicate DIODE index entry: {relative}")
            seen.add(str(relative))
            directory = self.root.joinpath(*parts)
            if not (directory / "image.jpg").is_file():
                raise FileNotFoundError(f"DIODE download is incomplete; missing {directory / 'image.jpg'}")
            entries.append(_Entry("diode_" + "__".join(parts), directory, "/".join(parts[:3])))
        if not entries:
            raise ValueError(f"DIODE index is empty: {index}")
        self._entries = sorted(entries, key=lambda entry: entry.uid)

    def _encode(self, rgbs):
        features = np.asarray(self._encoder.encode(rgbs), dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != len(rgbs) or not features.shape[1]:
            raise ValueError("RGB encoder must return one feature vector per image")
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        if not np.isfinite(features).all() or not np.isfinite(norm).all() or np.any(norm <= 0):
            raise ValueError("RGB encoder returned a nonfinite or zero feature vector")
        return features / norm

    def _build_features(self):
        if self._features is not None:
            return
        if self._encoder is None:
            self._encoder = self._encoder_factory(checkpoint=self.checkpoint, device=self.device)
        batches, pixel_scenes = [], {}
        for start in range(0, len(self._entries), self.batch_size):
            entries = self._entries[start:start + self.batch_size]
            rgbs = [load_rgb(entry.directory / "image.jpg") for entry in entries]
            for entry, rgb in zip(entries, rgbs):
                pixel_scenes.setdefault(_pixel_hash(rgb), set()).add(entry.scene)
            batches.append(self._encode(rgbs))
        self._features = np.concatenate(batches, axis=0)
        self._pixel_scenes = pixel_scenes

    def _load_reference(self, entry):
        rgb = load_rgb(entry.directory / "image.jpg")
        depth = _read_depth(entry.directory / "depth.png")
        if depth.shape != rgb.shape[:2] or not np.any(np.isfinite(depth) & (depth > 1e-6)):
            raise ValueError(f"DIODE reference depth must match its RGB and have valid metric pixels: {entry.directory}")
        with (entry.directory / "meta.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        camera = np.asarray(metadata["intrinsics"], dtype=np.float64)
        if (camera.shape != (3, 3) or not np.isfinite(camera).all()
                or not np.array_equal(camera[2], [0, 0, 1]) or camera[0, 0] <= 0 or camera[1, 1] <= 0):
            raise ValueError(f"Invalid normalized DIODE camera: {entry.directory / 'meta.json'}")
        camera = camera.copy()
        camera[0] *= rgb.shape[1]
        camera[1] *= rgb.shape[0]
        return Reference(entry.uid, rgb, depth, camera)

    def retrieve(self, rgb, source_path=None):
        if self._closed:
            raise RuntimeError("DIODE reference bank is closed")
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError("Reference retrieval expects a nonempty uint8 RGB image")
        self._build_features()
        excluded = set(self._pixel_scenes.get(_pixel_hash(rgb), ()))
        if source_path is not None:
            try:
                parts = Path(source_path).expanduser().resolve().relative_to(self.root).parts
            except ValueError:
                parts = ()
            if len(parts) >= 3:
                excluded.add("/".join(parts[:3]))
        eligible = np.array([entry.scene not in excluded for entry in self._entries])
        if not eligible.any():
            raise ValueError("No DIODE reference remains after excluding the query and its scene")
        query = self._encode([rgb])
        if query.shape[1] != self._features.shape[1]:
            raise ValueError("Query and reference RGB feature dimensions differ")
        scores = self._features @ query[0]
        # Entries are sorted by stable UID; argmax keeps that order on ties.
        index = int(np.flatnonzero(eligible)[np.argmax(scores[eligible])])
        return self._load_reference(self._entries[index])

    def close(self):
        if self._encoder is not None:
            close = getattr(self._encoder, "close", None)
            if callable(close):
                close()
        self._encoder, self._features = None, None
        self._pixel_scenes.clear()
        self._closed = True
