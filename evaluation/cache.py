"""Cache identities for the file-based evaluation and paper replay protocol."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

from contextdepth.cache import digest, file_revision, file_sha256
from contextdepth.io import resolve


def _definitions_revision(path, names):
    """Hash only stage dependencies in a module shared by several stages."""
    tree = ast.parse(Path(path).read_text())
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            identifiers = [target.id for target in targets if isinstance(target, ast.Name)]
        else:
            continue
        for name in set(identifiers) & set(names):
            definitions[name] = ast.dump(node, include_attributes=False)
    if definitions.keys() != set(names):
        raise ValueError(f"Missing cache dependencies in {path}: {set(names) - definitions.keys()}")
    return digest(definitions)


def model_revision(paths, *, stage="depth"):
    revisions = {}
    for key, value in paths.items():
        # Compiler caches are generated state, not model input resources. Their
        # creation during the first inference must not invalidate that result.
        if key == "cache_dir" or not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_file():
            revisions[key] = file_revision(path)
        elif path.is_dir():
            entries = path.rglob("*") if key in ("base_model", "weights", "checkpoint") else path.iterdir()
            # FLUX also loads tokenizer data outside the weight/config suffixes.
            # Bind those two assets without invalidating caches for README text.
            tokenizer_files = {"tokenizer/merges.txt", "tokenizer_2/spiece.model"} if key == "base_model" else set()
            files = [p for p in entries if p.is_file() and (p.suffix in
                     (".json", ".yaml", ".pt", ".pth", ".bin", ".safetensors")
                     or p.relative_to(path).as_posix() in tokenizer_files)]
            revisions[key] = [file_revision(p) for p in sorted(files)]
            if "source" in key:
                revisions[key + "_python"] = [file_revision(p) for p in sorted(path.rglob("*.py"))]
    package = Path(__file__).resolve().parents[1] / "contextdepth"
    evaluation = Path(__file__).parent
    # Selection, metrics and CLI changes cannot alter generated images or model
    # predictions. Only fingerprint dependencies of this computational stage.
    shared = ["types.py", "geometry.py", "io.py"]
    if stage == "depth":
        shared += ["baseline.py", "models.py", "model_compat.py"]
    recipe = {name: file_sha256(package / name) for name in shared}
    if stage == "depth":
        recipe["evaluation_geometry"] = file_sha256(evaluation / "geometry.py")
    functions = ["sample_geometry", "candidate_path", "_device_init", "_pool"]
    if stage == "depth":
        functions += ["_prepare", "_predict_chunk", "_predict_job", "_transfer_query_camera",
                      "_finish_prediction", "_depth_tasks", "_depth_jobs"]
        recipe["load_configuration"] = _definitions_revision(package / "config.py", ["model_load_config"])
        recipe["observations"] = _definitions_revision(package / "selection.py",
            ["DEPTH_EPS", "_array", "reference_bias", "pairwise_log_ratios"])
    elif stage == "generation":
        functions += ["_generate_with_public_adapter", "_generate_batch", "generate"]
    else:
        raise ValueError(f"Unknown cache stage: {stage}")
    recipe["preparation"] = _definitions_revision(evaluation / "pipeline.py", functions)
    entrypoint = paths.get("implementation")
    if entrypoint:
        module = entrypoint.split(":", 1)[0]
        origin = module if module.endswith(".py") else importlib.util.find_spec(module).origin
        recipe["implementation"] = file_sha256(origin)
        for helper in Path(origin).parent.glob("_*.py"):
            recipe[str(helper)] = file_sha256(helper)
    for dependency in paths.get("cache_dependencies", []):
        recipe[str(dependency)] = file_sha256(dependency)
    return digest({"config": paths, "files": revisions, "recipe": recipe,
                   "seed": paths.get("seed", os.environ.get("DEPTH_EVAL_SEED", "20260621"))})


def input_revision(row, root, spec, role):
    name = spec.name
    known = role == "reference" or spec.camera == "known"
    fields = ("image", "depth") if role == "reference" else ("image",)
    value = {"uid": row["uid"], "geometry": row.get("geometry"), "role": role,
             "files": {f: file_revision(resolve(root, row[f])) for f in fields}}
    if known:
        value["camera"] = row["camera"]
    if role == "query":
        value["fraction"] = spec.context_fraction
        if spec.camera_source != "native":
            value["camera_source"] = spec.camera_source
        candidates = row.get("candidates", [f"outpaintings/{row['uid']}/candidate{i}.png" for i in range(3)])
        value["candidates"] = [file_revision(resolve(root, p)) for p in candidates]
    state_field = "reference_rng_states" if role == "reference" else "rng_states"
    if name in row.get(state_field, {}):
        value["rng"] = file_revision(resolve(root, row[state_field][name]))
    return digest(value)
