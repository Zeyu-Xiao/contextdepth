"""Small cache identities: recipe, configuration and input file revisions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .io import resolve


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_revision(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_revision(paths):
    revisions = {}
    for key, value in paths.items():
        if not isinstance(value, str):
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
    package = Path(__file__).parent
    recipe = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.glob("*.py"))}
    return digest({"config": paths, "files": revisions, "recipe": recipe,
                   "seed": paths.get("seed", os.environ.get("DEPTH_EVAL_SEED", "20260621"))})


def input_revision(row, root, name, role):
    from .config import CONFIGS
    known = role == "reference" or CONFIGS[name][1] == "known"
    fields = ("image", "depth") if role == "reference" else ("image",)
    value = {"uid": row["uid"], "geometry": row.get("geometry"), "role": role,
             "files": {f: file_revision(resolve(root, row[f])) for f in fields}}
    if known:
        value["camera"] = row["camera"]
    if role == "query":
        value["fraction"] = CONFIGS[name][2]
        candidates = row.get("candidates", [f"outpaintings/{row['uid']}/candidate{i}.png" for i in range(3)])
        value["candidates"] = [file_revision(resolve(root, p)) for p in candidates]
    state_field = "reference_rng_states" if role == "reference" else "rng_states"
    if name in row.get(state_field, {}):
        value["rng"] = file_revision(resolve(root, row[state_field][name]))
    return digest(value)
