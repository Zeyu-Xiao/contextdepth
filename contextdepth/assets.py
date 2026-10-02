"""Copy/link and verify a portable bank of archived candidates and RNG states."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile

from .cache import file_sha256
from .io import load_manifest, read_jsonl, resolve, write_json


def _transfer(source, destination, expected, mode):
    source, destination = Path(source).resolve(), Path(destination)
    if file_sha256(source) != expected:
        raise ValueError(f"Asset checksum mismatch: {source}")
    if destination.exists():
        if file_sha256(destination) != expected:
            raise ValueError(f"Refusing to overwrite a different asset: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        temporary = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    else:
        raise ValueError("Asset mode must be copy, symlink or hardlink")


def build_assets(manifest_path, output, *, mode="copy", workers=8):
    manifest, root = load_manifest(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    query_ids = {r["uid"] for r in manifest["samples"]}
    rows = {r["uid"]: dict(r) for r in manifest["samples"]}
    for ref in manifest.get("references", []):
        row = rows.setdefault(ref["uid"], dict(ref))
        row["reference_rng_states"] = {**row.get("reference_rng_states", {}), **ref.get("reference_rng_states", {})}

    def one(row):
        record = {"uid": row["uid"], "candidates": [], "rng_states": {}, "reference_rng_states": {}}
        candidates = row.get("candidates", [f"outpaintings/{row['uid']}/candidate{i}.png" for i in range(3)])
        if row["uid"] not in query_ids:
            candidates = []
        sources = [("candidates", i, path, f"outpaintings/{row['uid']}/candidate{i}.png")
                   for i, path in enumerate(candidates)]
        sources += [("rng_states", name, path, f"rng/{name}/{row['uid']}.npz")
                    for name, path in row.get("rng_states", {}).items()]
        sources += [("reference_rng_states", name, path, f"rng/reference_{name}/{row['uid']}.npz")
                    for name, path in row.get("reference_rng_states", {}).items()]
        for field, key, source_path, relative in sources:
            source = resolve(root, source_path)
            checksum = file_sha256(source)
            _transfer(source, output / relative, checksum, mode)
            entry = {"path": relative, "sha256": checksum, "bytes": source.stat().st_size}
            if field == "candidates":
                record[field].append(entry)
            else:
                record[field][key] = entry
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(one, rows.values()))
    content = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records).encode()
    (output / "assets.jsonl.gz").write_bytes(gzip.compress(content, mtime=0))
    return output / "assets.jsonl.gz"


def import_assets(manifest_path, index_path, source, *, mode="symlink", workers=8):
    manifest, root = load_manifest(manifest_path)
    entries = read_jsonl(index_path)
    bank = {r["uid"]: r for r in entries}
    if len(bank) != len(entries):
        raise ValueError("Asset index contains duplicate UIDs")

    def one(row, reference=False):
        record = bank[row["uid"]]
        if not reference and len(record["candidates"]) != 3:
            raise ValueError(f"Exactly three archived candidates required: {row['uid']}")
        row = dict(row)
        if not reference:
            row["candidates"], row["candidate_sha256"], row["rng_states"] = [], [], {}
        row["reference_rng_states"] = {}
        fields = ("reference_rng_states",) if reference else ("candidates", "rng_states", "reference_rng_states")
        for field in fields:
            items = enumerate(record[field]) if field == "candidates" else record.get(field, {}).items()
            for key, item in items:
                relative = Path(item["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Asset paths must stay inside the bundle")
                _transfer(Path(source) / relative, root / relative, item["sha256"], mode)
                if field == "candidates":
                    row[field].append(str(relative))
                    row["candidate_sha256"].append(item["sha256"])
                else:
                    row[field][key] = str(relative)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        manifest["samples"] = list(pool.map(one, manifest["samples"]))
        if "references" in manifest:
            manifest["references"] = list(pool.map(lambda row: one(row, reference=True), manifest["references"]))

    # Preserve the manifest's on-disk format.  ``load_manifest`` accepts both
    # JSON objects and JSONL/JSONL.GZ rows; rewriting a JSONL path as a JSON
    # object would make the next run fail to parse it.
    manifest_path = Path(manifest_path)
    if ".jsonl" in manifest_path.name:
        rows = manifest["samples"]
        if "references" in manifest:
            rows = [*rows, *manifest["references"]]
        payload = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=manifest_path.parent, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            if manifest_path.name.endswith(".gz"):
                with gzip.open(temporary, "wt", encoding="utf-8", newline="") as stream:
                    stream.write(payload)
            else:
                temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    else:
        write_json(manifest_path, manifest)
    return Path(manifest_path)
