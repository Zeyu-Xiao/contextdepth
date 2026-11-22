"""Normal-use CLI; historical dataset commands are forwarded to evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parser():
    p = argparse.ArgumentParser(description="ContextDepth: RGB outpainting and metric depth inference",
        epilog="Paper ROI preparation and evaluation: python -m evaluation --help")
    stages = p.add_subparsers(dest="command", required=True)
    for name, help_text in (("generate", "Generate RGB context without depth estimation"),
                            ("depth", "Estimate full-image metric depth without generation"),
                            ("infer", "Generate context and estimate original/candidate depths")):
        stage = stages.add_parser(name, help=help_text)
        stage.add_argument("--image", type=Path, required=True)
        stage.add_argument("--output", type=Path, required=True)
        stage.add_argument("--config", type=Path, default=None)
        stage.add_argument("--device", default="cuda:0")
        if name != "generate":
            stage.add_argument("--depth-model", default=None,
                               help="Backend name or module:Class/file.py:Class")
            stage.add_argument("--camera", type=Path, help="JSON containing pixel-space 3x3 K")
            stage.add_argument("--depth-batch-size", type=int, default=None)
        if name != "depth":
            stage.add_argument("--outpainter", help="Custom module:Class/file.py:Class")
            stage.add_argument("--expansion", type=float, default=None,
                               help="Expansion fraction per side, default 0.1")
            stage.add_argument("--seeds", type=int, nargs="+", default=None)
            stage.add_argument("--offload", choices=["model"], help="Offload OneReward to CPU between components")
        if name == "infer":
            stage.add_argument("--reference-bank", choices=["diode"], default=None)
            stage.add_argument("--diode-root", type=Path)
    return p


def _legacy_infer(argv):
    """Recognize historical infer options without taking ordinary DIODE calls."""
    if not argv or argv[0] != "infer":
        return False
    for index, argument in enumerate(argv[1:], start=1):
        if argument == "--models" or argument.startswith("--models="):
            return True
        if argument.startswith("--reference-bank="):
            value = argument.partition("=")[2]
        elif (argument == "--reference-bank" and index + 1 < len(argv)
              and not argv[index + 1].startswith("--")):
            value = argv[index + 1]
        else:
            continue
        if value != "diode":
            return True
    return False


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Keep old scripts runnable, while ordinary commands have no manifests or
    # scoring switches. The evaluation parser and runtime are independent.
    legacy = {"prepare", "import-assets", "retrieve", "predict", "select", "evaluate", "run"}
    if (argv and (argv[0] in legacy or argv[0] == "eval" or _legacy_infer(argv)
                  or (argv[0] in {"generate", "infer"} and
                      ("--manifest" in argv or "--prepare-only" in argv)))):
        from evaluation.cli import main as evaluate
        return evaluate(argv[1:] if argv[0] == "eval" else argv)
    args = parser().parse_args(argv)
    from .io import read_json, write_json
    from .pipeline import ContextDepthPipeline, _save_prediction
    from PIL import Image
    options = {}
    if getattr(args, "offload", None):
        options["offload"] = args.offload
    with ContextDepthPipeline(
            config=args.config, device=args.device,
            depth_model=getattr(args, "depth_model", None),
            outpainter=getattr(args, "outpainter", None),
            depth_batch_size=getattr(args, "depth_batch_size", None),
            expansion=getattr(args, "expansion", None), seeds=getattr(args, "seeds", None),
            generation_options=options,
            reference_bank=getattr(args, "reference_bank", None),
            diode_root=getattr(args, "diode_root", None)) as pipeline:
        camera_path = getattr(args, "camera", None)
        camera = read_json(camera_path) if camera_path else None
        args.output.mkdir(parents=True, exist_ok=True)
        if args.command == "generate":
            candidates = pipeline.generate(args.image)
            for i, candidate in enumerate(candidates):
                Image.fromarray(candidate.rgb).save(args.output / f"candidate{i}.png")
            write_json(args.output / "outpaintings.json", [
                {"image": f"candidate{i}.png", "bbox": list(item.bbox), "seed": item.seed}
                for i, item in enumerate(candidates)])
        elif args.command == "depth":
            _save_prediction(args.output / "depth.npz", pipeline.estimate_depth(args.image, camera=camera))
        else:
            pipeline(args.image, camera=camera).save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
