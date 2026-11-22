"""Command-line entry for preparing and evaluating the full paper cohort."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from contextdepth.config import configuration_names, experiment, load_config
from contextdepth.io import load_manifest, read_json, resolve, write_json, write_npz


def _names(value, config):
    available = configuration_names(config)
    if value is None:
        return [config.get("default_experiment", available[0])]
    if value == ["all"]:
        return available
    if "all" in value or any(name not in available for name in value):
        raise ValueError(f"Use --models all alone, or choose experiment names: {available}")
    return value


def _retrieval(args, manifest, root, config):
    if args.retrieval:
        return read_json(args.retrieval)
    names = _names(args.models, config)
    if (args.bank == "mixed" and all(all(name in row.get("paper_reference_ids", {}) for name in names)
                                     for row in manifest["samples"])
            and not args.recompute_retrieval):
        from .pipeline import reference_ids, save_reference_ids
        save_reference_ids(args.output, reference_ids(manifest, names))
        return None
    from contextdepth.retrieval import encode_images, retrieve_top1
    query_rows = manifest["samples"]
    reference_rows = manifest.get("references", query_rows)
    distinct = {r["uid"]: r for r in [*query_rows, *reference_rows]}
    rows = list(distinct.values())
    config_encoder = config["retrieval"]
    descriptors = encode_images(rows, root, config_encoder["checkpoint"], device=f"cuda:{args.gpus[0]}",
                                batch_size=args.encoder_batch_size, workers=args.io_workers)
    index = {r["uid"]: i for i, r in enumerate(rows)}
    query_features = descriptors[[index[r["uid"]] for r in query_rows]]
    availability = manifest.get("reference_availability", {}) if args.bank != "fixed-diode" else {}
    if availability:
        matches, computed = {}, {}
        for name in _names(args.models, config):
            eligible = frozenset(availability[name]) if name in availability else frozenset(r["uid"] for r in reference_rows)
            if eligible not in computed:
                pool = [r for r in reference_rows if r["uid"] in eligible]
                computed[eligible] = retrieve_top1(query_rows, pool, query_features,
                    descriptors[[index[r["uid"]] for r in pool]], bank=args.bank)
            matches[name] = computed[eligible]
    else:
        matches = retrieve_top1(query_rows, reference_rows, query_features,
            descriptors[[index[r["uid"]] for r in reference_rows]], bank=args.bank)
    write_npz(args.output / "descriptors.npz", uids=list(index), descriptors=descriptors)
    write_json(args.output / "retrieval.json", matches)
    return matches


def _demo_manifest(args):
    from contextdepth.geometry import canvas_geometry
    from contextdepth.io import load_rgb

    bank, bank_root = load_manifest(args.reference_bank)
    references = bank["references"] if "references" in bank else bank["samples"]
    reference_rows = []
    for r in references:
        row = dict(r)
        for field in ("image", "retrieval_image", "depth"):
            if field in row:
                row[field] = str(resolve(bank_root, row[field]).resolve())
        for field in ("rng_states", "reference_rng_states"):
            if field in row:
                row[field] = {name: str(resolve(bank_root, value).resolve())
                              for name, value in row[field].items()}
        reference_rows.append(row)
    image = args.image.resolve()
    rgb = load_rgb(image)
    geometry = canvas_geometry(rgb.shape[1], rgb.shape[0])
    uid = args.uid or image.stem
    row = dict(uid=uid, image=str(image), scene_group=args.scene_group or uid,
               dataset="custom", geometry=geometry.as_dict(),
               candidates=[str((args.output / "outpaintings" / uid / f"candidate{i}.png").resolve()) for i in range(3)])
    if args.camera:
        row["camera"] = read_json(args.camera)
    manifest = dict(data_root=".", samples=[row], references=reference_rows)
    path = args.output / "input.json"
    write_json(path, manifest)
    return path


def parser():
    p = argparse.ArgumentParser(description="ContextDepth evaluation: prepare data, run the paper protocol, and score depth")
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Compute paper ROIs and cameras from the official evaluation data")
    prepare.add_argument("--spec", type=Path,
                         default=Path(__file__).resolve().parent / "protocol/roi_samples.jsonl.gz",
                         help="Sample metadata (default: full paper cohort; ROI coordinates are computed from source dimensions)")
    prepare.add_argument("--eval-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--io-workers", type=int, default=8)
    prepare.add_argument("--subset", choices=["quick"],
                         help="Prepare only a listed handful of queries and their references "
                              "(quick: iBims-1 only, to check an installation end to end)")
    summary = sub.add_parser("report", help="Print results in the layout of the paper's Table 1 (crop block)")
    summary.add_argument("--output", type=Path, required=True, help="Folder written by evaluation run")
    summary.add_argument("--reference", type=Path, help="Expected table (default: protocol/table1_crop.json)")
    summary.add_argument("--tolerance", type=float, default=0.02,
                         help="Quick check: allowed difference of an experiment's mean Query-only delta1 (0.02 = 2 points)")
    summary.add_argument("--save", type=Path, help="Also write the Markdown report here (default: <output>/report.md)")
    for command in ("generate", "retrieve", "predict", "select", "evaluate", "run", "infer"):
        stage = sub.add_parser(command)
        stage.set_defaults(gpus=[0], workers_per_gpu=1, io_workers=8,
                           generation_batch_size=1, encoder_batch_size=16,
                           retrieval=None, recompute_retrieval=False, bank="mixed",
                           overwrite=False, skip_generation=False, exclude_datasets=[])
        stage.add_argument("--manifest", type=Path, required=command != "infer")
        stage.add_argument("--data-root", type=Path)
        stage.add_argument("--config", type=Path, default=Path("evaluation/models.yaml"))
        stage.add_argument("--models", nargs="+", help="Experiment names from the configuration, or all")
        stage.add_argument("--depth-batch-size", type=int,
                           help="Maximum independent images per depth forward (default: config, or 1 for exact replay)")
        stage.add_argument("--output", type=Path, default=Path("outputs/run"))
        if command in ("generate", "retrieve", "predict", "run", "infer"):
            stage.add_argument("--gpus", type=int, nargs="+", default=[0])
        if command in ("predict", "run", "infer"):
            stage.add_argument("--workers-per-gpu", type=int, default=1)
        stage.add_argument("--io-workers", type=int, default=8)
        if command in ("generate", "run", "infer"):
            stage.add_argument("--generation-batch-size", type=int, default=1)
            stage.add_argument("--skip-generation", action="store_true", help="Use the existing three candidates")
        if command in ("retrieve", "predict", "run", "infer"):
            stage.add_argument("--encoder-batch-size", type=int, default=16)
            stage.add_argument("--retrieval", type=Path, help="Existing RGB-retrieval UID mapping")
            stage.add_argument("--recompute-retrieval", action="store_true")
            stage.add_argument("--bank", choices=["mixed", "cross-dataset", "fixed-diode"], default="mixed")
        if command in ("generate", "predict", "run", "infer"):
            stage.add_argument("--overwrite", action="store_true", help="Recompute this stage's existing outputs")
        if command in ("evaluate", "run", "infer"):
            stage.add_argument("--exclude-datasets", nargs="*", default=[])
        if command == "infer":
            stage.add_argument("--image", type=Path, required=True)
            stage.add_argument("--reference-bank", type=Path, required=True)
            stage.add_argument("--camera", type=Path, help="Pixel 3x3 K on the input image grid")
            stage.add_argument("--uid")
            stage.add_argument("--scene-group")
            stage.add_argument("--prepare-only", action="store_true", help="Write input.json for separate generation/depth environments")
    assets = sub.add_parser("import-assets", help="Import externally supplied candidate images and RNG states")
    assets.add_argument("--manifest", type=Path, required=True)
    assets.add_argument("--index", type=Path, required=True)
    assets.add_argument("--source", type=Path, required=True)
    assets.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    assets.add_argument("--io-workers", type=int, default=8)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_benchmark
        if args.subset:
            print(prepare_benchmark(args.spec, args.eval_root, args.output, workers=args.io_workers,
                                    subset=args.subset))
        else:
            print(prepare_benchmark(args.spec, args.eval_root, args.output, workers=args.io_workers))
        return 0
    if args.command == "report":
        from .report import report
        text, result = report(args.output, reference_path=args.reference, tolerance=args.tolerance, save=args.save)
        print(text)
        return 1 if result.get("quick_check", {}).get("failed") else 0
    if args.command == "import-assets":
        from contextdepth.assets import import_assets
        print(import_assets(args.manifest, args.index, args.source,
                            mode=args.mode, workers=args.io_workers))
        return 0
    from . import pipeline
    config = load_config(args.config)
    if args.depth_batch_size is not None:
        if args.depth_batch_size < 1:
            raise ValueError("--depth-batch-size must be positive")
        config["depth_batch_size"] = args.depth_batch_size
    names = _names(args.models, config)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate model configurations")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "infer":
        if len(names) != 1:
            raise ValueError("The single-image demo runs one model configuration")
        if experiment(config, names[0]).camera == "known" and args.camera is None:
            raise ValueError("The known-camera configuration requires --camera")
        args.manifest = _demo_manifest(args)
        if args.prepare_only:
            print(args.manifest)
            return 0
    manifest, root = load_manifest(args.manifest, args.data_root)
    common = dict(data_root=args.data_root, output=args.output)
    command = "run" if args.command == "infer" else args.command
    if command in ("generate", "run") and not args.skip_generation:
        pipeline.generate(args.manifest, config, data_root=args.data_root, gpus=args.gpus,
                          batch_size=args.generation_batch_size, workers_per_gpu=1, overwrite=args.overwrite)
    if command in ("retrieve", "predict", "run"):
        retrieval = _retrieval(args, manifest, root, config)
        if command != "retrieve":
            pipeline.predict(args.manifest, config, names, **common, gpus=args.gpus,
                             workers_per_gpu=args.workers_per_gpu, retrieval=retrieval, overwrite=args.overwrite)
    if command in ("select", "run"):
        pipeline.select(args.manifest, config, names, **common, workers=args.io_workers)
    has_gt = ["depth" in r for r in manifest["samples"]]
    if command == "evaluate" and not all(has_gt) or command == "run" and any(has_gt) and not all(has_gt):
        raise ValueError("Evaluation requires GT for every sample; partial cohorts are not scored")
    if command in ("evaluate", "run") and all(has_gt):
        rows = pipeline.evaluate(args.manifest, config, names, **common, workers=args.io_workers,
                                 exclude_datasets=args.exclude_datasets)
        for r in rows:
            if r["dataset"] == "All":
                absrel = (f"{r['absrel']:.4f}" if r["absrel"] is not None else
                          f"undefined ({r['undefined_absrel_images']} images predict infinite depth on GT pixels)")
                print(f"{r['config']:24} {r['method']:9} n={r['n']} δ1={100*r['delta1']:.2f}% AbsRel={absrel}")
    if command in ("select", "run"):
        print(f"Selected metric depths: {args.output / 'selected'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
