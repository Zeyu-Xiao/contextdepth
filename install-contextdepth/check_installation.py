#!/usr/bin/env python3
"""Import the real OneReward/MoGe2 classes without weights, downloads or GPU use."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path(__file__).resolve().parents[1] / "third_party")
    parser.add_argument("--with-bank", action="store_true", help="Also import the optional SigLIP vision encoder")
    args = parser.parse_args()
    # This check intentionally proves imports, not model inference or CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sources = args.sources.expanduser().resolve()
    paths = [sources / "MoGe2", sources / "utils3d", sources / "OneReward" / "src"]
    expected = [paths[0] / "moge/model/v2.py", paths[1] / "utils3d/__init__.py",
                paths[2] / "pipeline_flux_fill_with_cfg.py"]
    for path in expected:
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned source: {path}; run setup_upstreams.py --install --group core")
    sys.path[:0] = [str(path) for path in paths]
    import cv2
    import numpy as np
    import scipy
    import torch
    import torchvision
    from diffusers import FluxTransformer2DModel
    from moge.model.v2 import MoGeModel
    from pipeline_flux_fill_with_cfg import FluxFillCFGPipeline

    # Exercise the compiled torchvision extension and NumPy/OpenCV boundary on CPU.
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]])
    assert torchvision.ops.nms(boxes, torch.tensor([.9, .8]), .5).tolist() == [0, 1]
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    assert cv2.resize(rgb, (2, 2)).shape == (2, 2, 3)
    assert torch.from_numpy(rgb).shape == (4, 4, 3)
    classes = [FluxTransformer2DModel.__name__, FluxFillCFGPipeline.__name__, MoGeModel.__name__]
    if args.with_bank:
        from transformers import SiglipImageProcessor, SiglipVisionModel
        classes.extend([SiglipImageProcessor.__name__, SiglipVisionModel.__name__])
    names = ("torch", "torchvision", "numpy", "scipy", "Pillow", "opencv-python", "diffusers",
             "transformers", "tokenizers", "huggingface-hub", "accelerate", "protobuf")
    print(json.dumps({
        "status": "imports_and_cpu_operators_passed", "python": sys.version.split()[0],
        "versions": {name: importlib.metadata.version(name) for name in names},
        "classes": classes, "sources": str(sources), "torch_cuda_build": torch.version.cuda,
        "gpu_used": False, "weights_loaded": False, "model_inference_tested": False,
    }, indent=2))


if __name__ == "__main__":
    main()
