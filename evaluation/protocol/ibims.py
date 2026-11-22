"""Split prepared public ROI records into the 100 iBims queries and 67 references."""
from __future__ import annotations

import argparse
from pathlib import Path

from contextdepth.io import read_json, read_jsonl, write_json


def finalize_manifest(path):
    path = Path(path)
    specs = read_jsonl(Path(__file__).with_name("ibims100_with_references.jsonl"))
    expected = {row["uid"]: row for row in specs}
    query_ids = [row["uid"] for row in specs if row["dataset"] == "iBims-1"]
    reference_ids = sorted({uid for query in query_ids for uid in expected[query]["paper_reference_ids"].values()})
    if (len(expected), len(query_ids), len(reference_ids)) != (113, 100, 67):
        raise ValueError("The bundled iBims protocol must contain 113 images, 100 queries and 67 references")
    manifest = read_json(path)
    rows = {row["uid"]: row for row in manifest["samples"]}
    references = {row["uid"]: row for row in manifest.get("references", [])}
    if len(rows) != len(manifest["samples"]) or len(references) != len(manifest.get("references", [])):
        raise ValueError("Prepared manifest contains duplicate image identifiers")
    available = {**references, **rows}
    if set(available) != set(expected):
        raise ValueError("Prepare all records from ibims100_with_references.jsonl before splitting this cohort")
    for uid, row in available.items():
        for field in ("dataset", "scene_group", "camera", "geometry", "query_rgb_sha256"):
            if row.get(field) != expected[uid][field]:
                raise ValueError(f"Prepared metadata differs from the bundled protocol: {uid}/{field}")
    for uid in query_ids:
        if available[uid].get("paper_reference_ids") != expected[uid]["paper_reference_ids"]:
            raise ValueError(f"Prepared reference bindings differ from the bundled protocol: {uid}")
    manifest["samples"] = [available[uid] for uid in query_ids]
    manifest["references"] = [references.get(uid, available[uid]) for uid in reference_ids]
    manifest["cohort"] = "iBims-1-100"
    manifest["protocol"] = "paper-roi"
    write_json(path, manifest)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest produced by evaluation prepare")
    args = parser.parse_args(argv)
    print(finalize_manifest(args.manifest))
    print("100 iBims queries; 67 historical references; 113 unique images")


if __name__ == "__main__":
    main()
