"""OneReward generation through the separately installed upstream pipeline.

No model weights or upstream pipeline source are bundled here. One instance
owns one GPU and must be called sequentially; use separate worker processes
for multiple GPUs. A call with several same-size items is a real model batch.
"""
from __future__ import annotations

import hashlib
import importlib.util
from numbers import Integral
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .geometry import _rgb, _validate_geometry, canvas_geometry, restore_query_pixels
from .generation import OutpaintingBaseline
from .types import Geometry


PROMPT = "high-definition, perfect composition"
NEGATIVE_PROMPT = "nsfw"
UPSTREAM_COMMIT = "0d8d826d9e383c32ad97b39f01bd6872e3ea7aa4"
UPSTREAM_PIPELINE_SHA256 = "b3a9b352a648f782f8cf35e593c8f7fa49766a264cc095e0fe3c42f93a9bddb4"


def seed_from_uid(uid: str, candidate_index: int) -> int:
    """Historical candidate0/1/2 seeds; candidate0 does not mean seed value 0."""
    if not isinstance(uid, str) or not uid:
        raise ValueError("uid must be a nonempty string")
    if (isinstance(candidate_index, (bool, np.bool_)) or not isinstance(candidate_index, Integral)
            or not 0 <= candidate_index < 3):
        raise ValueError("candidate_index must be 0, 1, or 2")
    digest = hashlib.sha256(f"{uid}|generated-demonstration-bank-v1|{int(candidate_index)}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


class OneRewardOutpainter:
    """Load a local OneReward transformer and FLUX.1-Fill-dev base on one device.

    paths has ``source`` (OneReward checkout), ``base_model`` (FLUX base), and
    ``weights`` (the OneReward transformer directory). Set the outer worker's
    batch size to 1 for the historical call shape. Larger same-canvas batches
    use one CPU generator per item; true CFG evaluates a transformer batch
    twice that size, so they need more activation memory.
    """

    def __init__(self, paths: Mapping[str, str | Path], device: str = "cuda:0", *, offload=None):
        import torch
        from diffusers import FluxTransformer2DModel

        if offload is None:
            # The evaluation constructs this class from its config section alone, so the
            # memory option can also be given there: outpainting: {options: {offload: model}}.
            offload = dict(paths.get("options") or {}).get("offload")
        if offload not in (None, "model"):
            raise ValueError("offload must be omitted or 'model'")
        if offload == "model" and torch.device(device).type != "cuda":
            raise ValueError("Model CPU offload requires a CUDA device")
        self.paths = {name: Path(paths[name]).expanduser().resolve() for name in ("source", "base_model", "weights")}
        pipeline_path = self.paths["source"] / "src/pipeline_flux_fill_with_cfg.py"
        if not pipeline_path.is_file():
            raise FileNotFoundError(f"OneReward pipeline is missing: {pipeline_path}")
        for name in ("base_model", "weights"):
            if not self.paths[name].is_dir():
                raise FileNotFoundError(f"Local {name} directory is missing: {self.paths[name]}")
        # Load that checkout's standalone module without a global `src` import
        # collision with a depth-model repository. The upstream file is intact.
        spec = importlib.util.spec_from_file_location("contextdepth_onereward_pipeline", pipeline_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.pipeline_sha256 = hashlib.sha256(pipeline_path.read_bytes()).hexdigest()
        self.device, self._torch = device, torch
        transformer = FluxTransformer2DModel.from_pretrained(
            str(self.paths["weights"]), torch_dtype=torch.bfloat16, local_files_only=True)
        self.pipe = module.FluxFillCFGPipeline.from_pretrained(
            str(self.paths["base_model"]), transformer=transformer,
            torch_dtype=torch.bfloat16, local_files_only=True)
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        if offload == "model":
            target = torch.device(device)
            self.pipe.enable_model_cpu_offload(gpu_id=target.index, device=target.type)
        else:
            self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, query_rgbs: list[np.ndarray], uids: list[str], candidate_indices: list[int],
                 geometries: list[Geometry], seeds: list[int] | None = None) -> list[np.ndarray]:
        """Generate and center-lock items in caller order; never resize a query.

        Geometry must be the historical 25% aligned canvas. Crop a returned
        image with ``crop_context`` when a depth model uses 10% context.
        Different canvas sizes belong in different calls. The same geometry
        may be used three times for a query's three independent candidates.
        """
        count = len(query_rgbs)
        if not (len(uids) == len(candidate_indices) == len(geometries) == count):
            raise ValueError("Query, uid, candidate-index and geometry lists must have equal lengths")
        if seeds is not None and len(seeds) != count:
            raise ValueError("There must be exactly one seed per generated item")
        derived = [seed_from_uid(uid, index) for uid, index in zip(uids, candidate_indices)]
        seeds = derived if seeds is None else seeds
        if not count:
            return []
        if any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral)
               or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("Seeds must be integers between 0 and 2**63 - 1")
        canvases, masks = [], []
        canvas_size = (geometries[0].canvas_width, geometries[0].canvas_height)
        for rgb, geometry in zip(query_rgbs, geometries):
            _validate_geometry(geometry)
            expected = canvas_geometry(geometry.roi_width, geometry.roi_height, geometry.fixed_w, geometry.fixed_h)
            if geometry != expected:
                raise ValueError("Generate the aligned 25% canvas first; smaller contexts must be cropped afterwards")
            size = (geometry.canvas_width, geometry.canvas_height)
            if size != canvas_size:
                raise ValueError("Batch items must have the same canvas size; group shapes before calling generate")
            rgb = _rgb(rgb, geometry.roi_width, geometry.roi_height, "Query")
            canvas = Image.new("RGB", size, "white")
            canvas.paste(Image.fromarray(rgb), (geometry.roi_x, geometry.roi_y))
            mask = Image.new("L", size, 255)
            mask.paste(0, (geometry.roi_x, geometry.roi_y,
                           geometry.roi_x + geometry.roi_width, geometry.roi_y + geometry.roi_height))
            canvases.append(canvas)
            masks.append(mask)
        generators = [self._torch.Generator("cpu").manual_seed(int(seed)) for seed in seeds]
        # Keep the exact scalar API for historical batch=1. The upstream
        # negative branch must receive B prompts too when true_cfg > 1.
        result = self.pipe(
            prompt=PROMPT if count == 1 else [PROMPT] * count,
            negative_prompt=NEGATIVE_PROMPT if count == 1 else [NEGATIVE_PROMPT] * count,
            image=canvases[0] if count == 1 else canvases,
            mask_image=masks[0] if count == 1 else masks,
            height=canvas_size[1], width=canvas_size[0], guidance_scale=1.0, true_cfg=4.0,
            num_inference_steps=50, generator=generators[0] if count == 1 else generators,
        ).images
        if len(result) != count:
            raise RuntimeError(f"OneReward returned {len(result)} images for {count} items")
        return [restore_query_pixels(np.asarray(image.convert("RGB")), query, geometry)
                for image, query, geometry in zip(result, query_rgbs, geometries)]

    def close(self):
        """Release this instance's weights before loading another GPU model."""
        import gc
        self.pipe = None
        gc.collect()
        if str(self.device).startswith("cuda") and self._torch.cuda.is_available():
            with self._torch.cuda.device(self.device):
                self._torch.cuda.empty_cache()


class OneRewardBaseline(OutpaintingBaseline):
    """Public OneReward adapter for exact requested expansions, default 10%.

    Unlike the retained paper API above, this constructs the requested canvas
    directly. FLUX requires multiples of 16: only its temporary canvas is padded
    on the right/bottom, and those extra pixels are removed from the result.
    The supplied query is never resized. Default candidate inference is serial
    to avoid silently tripling full FLUX activation memory.
    """

    def __init__(self, paths, device="cuda:0"):
        super().__init__(paths, device)
        offload = self.paths.get("options", {}).get("offload")
        self._engine = OneRewardOutpainter(paths, device, offload=offload)
        self.metadata = {"implementation": "OneReward", "pipeline_sha256": self._engine.pipeline_sha256,
                         "alignment": "minimal right/bottom padding to 16 pixels", "query_resized": False,
                         "offload": offload}

    def generate_images(self, rgb, *, bbox, canvas_size, seeds):
        width, height = canvas_size
        native_width, native_height = ((width + 15) // 16) * 16, ((height + 15) // 16) * 16
        canvas = Image.new("RGB", (native_width, native_height), "white")
        canvas.paste(Image.fromarray(rgb), bbox[:2])
        mask = Image.new("L", (native_width, native_height), 255)
        mask.paste(0, bbox)
        images = []
        for seed in seeds:
            generator = self._engine._torch.Generator("cpu").manual_seed(seed)
            result = self._engine.pipe(
                prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT, image=canvas, mask_image=mask,
                height=native_height, width=native_width, guidance_scale=1.0, true_cfg=4.0,
                num_inference_steps=50, generator=generator,
            ).images
            if len(result) != 1:
                raise RuntimeError(f"OneReward returned {len(result)} images for one seed")
            image = np.asarray(result[0].convert("RGB"))
            if image.shape != (native_height, native_width, 3):
                raise ValueError("OneReward output does not match its requested aligned canvas")
            images.append(np.array(image[:height, :width], copy=True, order="C"))
        return images

    def close(self):
        super().close()
        if self._engine is not None:
            self._engine.close()
            self._engine = None
