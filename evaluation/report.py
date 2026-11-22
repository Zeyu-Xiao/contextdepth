"""Lay out evaluation results like the crop block of the paper's Table 1.

Reads ``metrics.json`` and ``per_image_metrics.json`` written by
``python -m evaluation run`` and prints delta1 for Query-only, Outpaint and
Outpaint + Select, without and with the ground-truth camera, next to the
paper's values. For the quick subset it instead checks every Query-only score
against the stored per-image value, which is the part of the pipeline that
does not depend on freshly generated images.
"""
from __future__ import annotations

from pathlib import Path

from contextdepth.io import read_json, write_json

PROTOCOL = Path(__file__).resolve().parent / "protocol"
METHODS = ("direct", "seed0", "selector")
SETTINGS = ("unknown", "known")


def _cell(value, paper=None):
    if value is None:
        return "-"
    text = f"{value:.2f}"
    return text if paper is None else f"{text} ({value - paper:+.2f})"


def table(rows, reference, dataset="All"):
    """One entry per model: measured delta1 (%) and its difference from the paper."""
    found = {(r["config"], r["method"]): r for r in rows if r["dataset"] == dataset}
    result = []
    for spec in reference["rows"]:
        entry = {"model": spec["model"], "measured": {}, "paper": spec["paper"], "images": {}}
        for setting in SETTINGS:
            values = [found.get((spec[setting], method)) for method in METHODS]
            entry["measured"][setting] = [None if v is None else 100.0 * v["delta1"] for v in values]
            counts = {v["n"] for v in values if v is not None}
            entry["images"][setting] = counts.pop() if len(counts) == 1 else None
        if any(v is not None for setting in SETTINGS for v in entry["measured"][setting]):
            result.append(entry)
    return result


def render(entries, reference, *, compare=True):
    heads = [reference["methods"][m] for m in METHODS]
    lines = ["| Model | " + " | ".join(f"{h}, no GT camera" for h in heads) + " | "
             + " | ".join(f"{h}, GT camera" for h in heads) + " |",
             "|---|" + "---:|" * 6]
    for entry in entries:
        cells = [_cell(value, paper if compare else None)
                 for setting in SETTINGS
                 for value, paper in zip(entry["measured"][setting], entry["paper"][setting])]
        lines.append(f"| {entry['model']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def quick_check(records, expected, tolerance):
    """Compare Query-only delta1 with the stored values, experiment by experiment.

    An experiment passes when its mean over the subset is within ``tolerance`` of the stored
    mean. A single image can move more than that when many of its pixels sit near the 1.25
    threshold, so the largest per-image difference is reported but does not decide the result.
    """
    measured = {}
    for record in records:
        stored = expected.get(record["config"], {}).get(record["uid"])
        if stored is not None:
            measured.setdefault(record["config"], {})[record["uid"]] = (record["metrics"]["direct"]["delta1"], stored)
    rows = []
    for config, pairs in sorted(measured.items()):
        complete = set(pairs) == set(expected[config])
        mine = sum(a for a, _ in pairs.values()) / len(pairs)
        theirs = sum(b for _, b in pairs.values()) / len(pairs)
        rows.append(dict(config=config, images=len(pairs), complete=complete, measured=mine, expected=theirs,
                         largest_image_difference=max(abs(a - b) for a, b in pairs.values()),
                         ok=complete and abs(mine - theirs) <= tolerance))
    return rows


def report(output, *, reference_path=None, quick_path=None, tolerance=0.02, save=None):
    output = Path(output)
    reference = read_json(reference_path or PROTOCOL / "table1_crop.json")
    metrics = read_json(output / "metrics.json")
    entries = table(metrics["rows"], reference)
    if not entries:
        raise ValueError(f"{output / 'metrics.json'} has no experiment of the Table 1 crop protocol")
    counts = {n for e in entries for n in e["images"].values() if n is not None}
    full = counts == {reference["images"]} and not metrics.get("excluded_datasets")
    lines = [f"# {reference['title']}: {reference['metric']}", ""]
    result = dict(full_cohort=full, images=sorted(counts), entries=entries)
    if full:
        lines += [f"{reference['images']} images. Each cell: measured (difference from the paper in points).", "",
                  render(entries, reference), "", reference["note"]]
        query = [abs(value - paper) for e in entries for s in SETTINGS
                 for value, paper in list(zip(e["measured"][s], e["paper"][s]))[:1] if value is not None]
        result["max_query_only_difference"] = max(query) if query else None
        lines += ["", f"Largest Query-only difference from the paper: {max(query):.2f} points."] if query else []
    else:
        lines += [f"{sorted(counts)} images per experiment: a subset, so the paper's {reference['images']}-image "
                  "numbers are not comparable. Measured delta1 (%):", "", render(entries, reference, compare=False)]
        quick_file = Path(quick_path or PROTOCOL / "quick.json")
        if quick_file.is_file():
            quick = read_json(quick_file)
            rows = quick_check(read_json(output / "per_image_metrics.json"), quick["query_only_delta1"], tolerance)
            if rows:
                failed = [r for r in rows if not r["ok"]]
                result["quick_check"] = dict(experiments=len(rows), failed=len(failed), tolerance=tolerance, rows=rows)
                lines += ["", f"Query-only check against the stored values (mean delta1 within {100 * tolerance:g} "
                              f"points): {len(rows) - len(failed)} of {len(rows)} experiments pass.", "",
                          "| Experiment | Measured | Stored | Largest per-image difference | |", "|---|---:|---:|---:|---|"]
                lines += [f"| {r['config']} | {100 * r['measured']:.2f} | {100 * r['expected']:.2f} | "
                          f"{100 * r['largest_image_difference']:.2f} | "
                          f"{'ok' if r['ok'] else 'DIFFERS' if r['complete'] else 'INCOMPLETE'} |" for r in rows]
                lines += ["", "Outpaint and Outpaint + Select use images generated on this machine; on a handful of "
                              "images they only show that generation, depth, selection and scoring run end to end."]
    text = "\n".join(lines)
    write_json(output / "report.json", result)
    Path(save or output / "report.md").write_text(text + "\n", encoding="utf-8")
    return text, result
