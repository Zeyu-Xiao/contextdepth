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

    def __init__(self, paths: Mapping[str, str | Path], device: str = "cuda:0"):
        import torch
        from diffusers import FluxTransformer2DModel

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
