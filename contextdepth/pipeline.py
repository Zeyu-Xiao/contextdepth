"""Resumable stages with persistent GPU workers and bounded CPU prefetch.

The only dependencies between stages are ordinary files. Generation and depth
can run in different environments or on different machines using the same
manifest. No model, image encoder or reference is loaded per query.
"""
from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from .cache import digest, file_revision, file_sha256, input_revision, model_revision
from .config import CONFIGS, model_paths
from .geometry import canvas_geometry, crop_context
from .io import load_manifest, load_rgb, read_json, resolve, write_json, write_npz
from .metrics import aggregate, score_depth
from .selection import pairwise_log_ratios, reference_bias, reference_camera, select_known, select_unknown
from .types import Geometry, InferenceInput

_WORKER = {}


def sample_geometry(row, rgb=None):
    if "geometry" in row:
        return Geometry.from_dict(row["geometry"])
    if rgb is None:
        raise ValueError("Geometry or decoded RGB required")
    return canvas_geometry(rgb.shape[1], rgb.shape[0])


def candidate_path(row, index):
    paths = row.get("candidates", [f"outpaintings/{row['uid']}/candidate{i}.png" for i in range(3)])
    if len(paths) != 3:
        raise ValueError(f"Exactly three candidates required for {row['uid']}")
    return paths[index]


def _device_init(devices, role, paths, model_name=None, root=None, output=None, fraction=None, overwrite=False):
    gpu = devices.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    if role == "generation":
        from .outpainting import OneRewardOutpainter
        engine = OneRewardOutpainter(paths, device="cuda:0")
    else:
        from .models import create_model
        engine = create_model(model_name, paths, device="cuda:0")
    _WORKER.update(engine=engine, root=Path(root), output=Path(output), name=model_name,
                   fraction=fraction, overwrite=overwrite, role=role, gpu=gpu)


def _pool(gpus, workers_per_gpu, initializer_args, max_tasks=None):
    if not gpus or workers_per_gpu < 1:
        raise ValueError("Choose at least one GPU and one worker per GPU")
    # Set limits before spawn imports NumPy/BLAS in the children.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    context = mp.get_context("spawn")
    devices = context.Queue()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    available = visible.split(",") if visible else None
    slots = []
    for _ in range(workers_per_gpu):
        for gpu in gpus:
            slots.append(available[gpu] if available is not None else gpu)
    if max_tasks is not None:
        slots = slots[:max_tasks]
    for slot in slots:
        devices.put(slot)
    return ProcessPoolExecutor(max_workers=len(slots),
                               mp_context=context, initializer=_device_init,
                               initargs=(devices, *initializer_args))


def _generate_batch(batch):
    root, engine = _WORKER["root"], _WORKER["engine"]
    with ThreadPoolExecutor(max_workers=min(4, len(batch))) as pool:
        rgbs = list(pool.map(lambda item: load_rgb(resolve(root, item[0]["image"])), batch))
    rows, indices = zip(*batch)
    geometries = [sample_geometry(row, rgb) for row, rgb in zip(rows, rgbs)]
    from .outpainting import seed_from_uid
    seeds = [row["seeds"][i] if "seeds" in row else seed_from_uid(row["uid"], i) for row, i in batch]
    images = engine.generate(rgbs, [r["uid"] for r in rows], list(indices), geometries, seeds)
    if len(images) != len(batch):
        raise ValueError("Outpainter returned an incomplete batch")

    def save(item):
        row, i, rgb = item
        path = resolve(root, candidate_path(row, i))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
        Image.fromarray(rgb).save(temporary, format="PNG", compress_level=3)
        os.replace(temporary, path)
        write_json(path.with_suffix(".json"), {"uid": row["uid"], "candidate": i,
                   "cache_key": row.get("_generation_cache_key"), "sha256": file_sha256(path)})

    with ThreadPoolExecutor(max_workers=min(4, len(batch))) as pool:
        list(pool.map(save, [(r, i, rgb) for (r, i), rgb in zip(batch, images)]))
    return len(batch)


def generate(manifest_path, config, *, data_root=None, gpus=(0,), batch_size=1, workers_per_gpu=1, overwrite=False):
    manifest, root = load_manifest(manifest_path, data_root)
    if batch_size < 1:
        raise ValueError("Generation batch size must be positive")
    groups = defaultdict(list)
    protected = {resolve(root, r[field]).resolve() for r in [*manifest["samples"], *manifest.get("references", [])]
                 for field in ("image", "retrieval_image", "depth") if field in r}
    protected.add(Path(manifest_path).resolve())
    protected.update(resolve(root, value).resolve()
                     for r in [*manifest["samples"], *manifest.get("references", [])]
                     for field in ("rng_states", "reference_rng_states")
                     for value in r.get(field, {}).values())
    destinations = set()
    recipe = model_revision(config["outpainting"])
    from .outpainting import seed_from_uid
    for row in manifest["samples"]:
        rgb = None if "geometry" in row else load_rgb(resolve(root, row["image"]))
        geometry = sample_geometry(row, rgb)
        for index in range(3):
            path = resolve(root, candidate_path(row, index))
            marker = path.with_suffix(".json")
            outputs = {path.resolve(), marker.resolve()}
            if len(outputs) != 2 or outputs & protected or outputs & destinations:
                raise ValueError(f"Candidate output or completion marker would overwrite an input or another output: {path}")
            destinations.update(outputs)
            key = digest({"input": file_revision(resolve(root, row["image"])),
                          "geometry": geometry.as_dict(), "recipe": recipe, "batch_size": batch_size,
                          "seed": row["seeds"][index] if "seeds" in row else seed_from_uid(row["uid"], index)})
            reusable = False
            if not overwrite and path.is_file():
                if marker.is_file():
                    metadata = read_json(marker)
                    reusable = metadata.get("cache_key") == key and metadata.get("sha256") == file_sha256(path)
                elif "candidate_sha256" in row:
                    reusable = row["candidate_sha256"][index] == file_sha256(path)
            if not reusable:
                groups[geometry.canvas_width, geometry.canvas_height].append(({**row, "_generation_cache_key": key}, index))
    jobs = [items[i:i + batch_size] for _, items in sorted(groups.items()) for i in range(0, len(items), batch_size)]
    if not jobs:
        return
    with _pool(gpus, workers_per_gpu, ("generation", config["outpainting"], None, root, root, None, overwrite), len(jobs)) as pool:
        with tqdm(total=sum(map(len, jobs)), desc="Outpainting") as progress:
            for completed in pool.map(_generate_batch, jobs, chunksize=1):
                progress.update(completed)


def _prepare(row):
    root, role = _WORKER["root"], _WORKER["role"]
    rgb = load_rgb(resolve(root, row["image"]))
    geometry = sample_geometry(row, rgb)
    if rgb.shape != (geometry.roi_height, geometry.roi_width, 3):
        raise ValueError(f"Query dimensions differ from the manifest: {row['uid']}")
    known = role == "reference" or CONFIGS[_WORKER["name"]][1] == "known"
    camera = np.asarray(row["camera"], np.float64) if known else None
    direct_arm = "reference" if role == "reference" and _WORKER["name"] == "moge3_known" else "direct"
    requests = [InferenceInput(row["uid"], direct_arm, rgb, geometry.direct(), camera, "known" if known else "unknown")]
    if role != "reference":
        for i in range(3):
            expanded = load_rgb(resolve(root, candidate_path(row, i)))
            if expanded.shape != (geometry.canvas_height, geometry.canvas_width, 3):
                raise ValueError(f"Candidate {i} dimensions differ for {row['uid']}")
            if not np.array_equal(expanded[geometry.roi_slices], rgb):
                raise ValueError(f"Candidate {i} changed query pixels for {row['uid']}")
            current = geometry
            # The historical 25% canvas includes its alignment border.
            if _WORKER["fraction"] != .25:
                expanded, current = crop_context(expanded, geometry, _WORKER["fraction"])
            requests.append(InferenceInput(row["uid"], f"candidate{i}", expanded, current, camera, "known" if known else "unknown"))
    state_field = "reference_rng_states" if role == "reference" else "rng_states"
    state_path = row.get(state_field, {}).get(_WORKER["name"])
    if state_path:
        with np.load(resolve(root, state_path), allow_pickle=False) as states:
            for request in requests:
                prefix = ("direct" if request.arm == "reference" else request.arm) + "__"
                request.rng_state = {k[len(prefix):]: np.array(states[k], copy=True)
                                     for k in states.files if k.startswith(prefix)}
                if not request.rng_state:
                    raise ValueError(f"Missing before-input RNG state: {row['uid']}/{request.arm}")
    return row, requests


def _result_paths(row):
    folder = _WORKER["output"] / ("references" if _WORKER["role"] == "reference" else "predictions") / _WORKER["name"]
    return folder / f"{row['uid']}.npz", folder / f"{row['uid']}.json"


def _finish_prediction(row, requests, predictions, elapsed):
    if len(predictions) != len(requests):
        raise ValueError(f"Depth backend returned an incomplete batch for {row['uid']}")
    for req, pred in zip(requests, predictions):
        if pred.depth.shape != (req.geometry.fixed_h, req.geometry.fixed_w):
            raise ValueError(f"Depth backend returned the wrong output grid for {row['uid']}/{req.arm}")
    arrays_path, record_path = _result_paths(row)
    arrays = {"direct" if req.arm == "reference" else req.arm: pred.depth for req, pred in zip(requests, predictions)}
    record = dict(uid=row["uid"], dataset=row.get("dataset", "custom"), config=_WORKER["name"],
                  seconds=elapsed, geometry=requests[-1].geometry.as_dict(),
                  input_revision=row.get("_input_revision"), cache_key=row.get("_cache_key"))
    if _WORKER["role"] == "reference":
        with np.load(resolve(_WORKER["root"], row["depth"]), allow_pickle=False) as gt_file:
            gt = gt_file["depth"]
            mask = gt_file["mask"] if "mask" in gt_file else None
        record["bias"] = reference_bias(predictions[0].depth, gt, mask)
    else:
        pairwise = pairwise_log_ratios([pred.depth for pred in predictions])
        record["pairwise"] = pairwise.tolist()
        if CONFIGS[_WORKER["name"]][1] == "unknown":
            record["candidate_cameras"] = [np.asarray(pred.camera_log_focal, np.float64).tolist() for pred in predictions[1:]]
    record["inference"] = [pred.metadata for pred in predictions]
    write_npz(arrays_path, **arrays)
    # The JSON is the completion marker; an interrupted array write is rerun.
    write_json(record_path, record)
    return row["uid"]


def _predict_chunk(rows):
    needed = rows  # Parent checks cache identities before loading GPU models.
    engine = _WORKER["engine"]
    # Preload two queries and overlap depth execution with CPU observations and
    # uncompressed array writes. Memory stays bounded independently of cohort size.
    with ThreadPoolExecutor(max_workers=2) as readers, ThreadPoolExecutor(max_workers=2) as writers:
        prefetch, finishing = {}, deque()
        for i in range(min(2, len(needed))):
            prefetch[i] = readers.submit(_prepare, needed[i])
        for i in range(len(needed)):
            row, requests = prefetch.pop(i).result()
            if i + 2 < len(needed):
                prefetch[i + 2] = readers.submit(_prepare, needed[i + 2])
            start = time.monotonic()
            maximum = max(1, getattr(engine, "max_batch_size", 1))
            predictions = list(engine.predict(requests[:1]))
            # Direct and context have different shapes; do not stack those into
            # one model call. Adapters expose a larger batch only when supported.
            candidates = requests[1:]
            for begin in range(0, len(candidates), maximum):
                predictions.extend(engine.predict(candidates[begin:begin + maximum]))
            finishing.append(writers.submit(_finish_prediction, row, requests, predictions, time.monotonic() - start))
            if len(finishing) >= 2:
                finishing.popleft().result()
        for future in finishing:
            future.result()
    return len(rows)


def _depth_jobs(name, rows, role, config, root, output, gpus, workers_per_gpu, overwrite):
    if not rows:
        return
    folder = Path(output) / ("references" if role == "reference" else "predictions") / name
    recipe = model_revision(model_paths(config, name))
    needed = []
    for row in rows:
        revision = input_revision(row, root, name, role)
        key = digest({"input": revision, "model": name, "recipe": recipe})
        record = folder / f"{row['uid']}.json"
        if not overwrite and record.is_file() and (folder / f"{row['uid']}.npz").is_file():
            if read_json(record).get("cache_key") == key:
                continue
        needed.append({**row, "_input_revision": revision, "_cache_key": key})
    rows = needed
    if not rows:
        return
    fraction = CONFIGS[name][2] if role == "query" else None
    # Small dynamic shards share the queue across GPUs; a slow dataset does not
    # strand cards assigned to faster datasets.
    shard_size = max(1, min(16, (len(rows) + len(gpus) * workers_per_gpu - 1) // (len(gpus) * workers_per_gpu)))
    jobs = [rows[i:i + shard_size] for i in range(0, len(rows), shard_size)]
    with _pool(gpus, workers_per_gpu, (role, model_paths(config, name), name, root, output, fraction, overwrite), len(jobs)) as pool:
        with tqdm(total=len(rows), desc=f"{name}: {role}") as progress:
            for completed in pool.map(_predict_chunk, jobs, chunksize=1):
                progress.update(completed)


def reference_ids(manifest, names, retrieval=None):
    """Use saved paper retrieval IDs or a newly computed RGB-only retrieval."""
    ids = {}
    references = {r["uid"]: r for r in manifest.get("references", manifest["samples"])}
    for name in names:
        ids[name] = {}
        for row in manifest["samples"]:
            if retrieval is not None:
                mapping = retrieval[name] if isinstance(retrieval.get(name), dict) else retrieval
                ids[name][row["uid"]] = mapping[row["uid"]]
            else:
                ids[name][row["uid"]] = row["paper_reference_ids"][name]
            ref = references[ids[name][row["uid"]]]
            if ref["uid"] == row["uid"] or (row.get("scene_group") and ref.get("scene_group") == row["scene_group"]):
                raise ValueError(f"Reference must exclude the query and its scene: {row['uid']} → {ref['uid']}")
    return ids


def predict(manifest_path, config, names, *, output, data_root=None, gpus=(0,), workers_per_gpu=1, retrieval=None, overwrite=False):
    manifest, root = load_manifest(manifest_path, data_root)
    if manifest.get("protocol") == "paper-roi":
        for name in names:
            if name.startswith("da3") and any(name not in r.get("rng_states", {}) for r in manifest["samples"]):
                raise ValueError("Paper DA3 reproduction requires the exported before-input RNG states; import the benchmark assets first")
    ids = reference_ids(manifest, names, retrieval)
    references = {r["uid"]: r for r in manifest.get("references", manifest["samples"])}
    needed = defaultdict(set)
    for name in names:
        needed[CONFIGS[name][3]].update(ids[name].values())
    for name, uids in needed.items():
        _depth_jobs(name, [references[uid] for uid in sorted(uids)], "reference", config, root, output, gpus, workers_per_gpu, overwrite)
    for name in names:
        _depth_jobs(name, manifest["samples"], "query", config, root, output, gpus, workers_per_gpu, overwrite)
    write_json(Path(output) / "reference_ids.json", ids)


def select(manifest_path, names, *, output, data_root=None, workers=8):
    manifest, root = load_manifest(manifest_path, data_root)
    output = Path(output)
    ids = read_json(output / "reference_ids.json")
    for name in names:
        reference_ids(manifest, [name], ids[name])
    references = {r["uid"]: r for r in manifest.get("references", manifest["samples"])}
    calibration = {}
    for name in names:
        ref_name = CONFIGS[name][3]
        for uid in set(ids[name].values()):
            record = read_json(output / "references" / ref_name / f"{uid}.json")
            if record.get("input_revision") != input_revision(references[uid], root, ref_name, "reference"):
                raise ValueError(f"Reference inputs changed since calibration: {uid}; rerun predict")
            calibration[name, uid] = record

    def one(job):
        name, row = job
        uid = row["uid"]
        observation = read_json(output / "predictions" / name / f"{uid}.json")
        if observation.get("input_revision") != input_revision(row, root, name, "query"):
            raise ValueError(f"Inputs changed since depth inference: {uid}; rerun predict")
        reference_uid = ids[name][uid]
        reference_record = calibration[name, reference_uid]
        bias = reference_record["bias"]
        matrix = np.asarray(observation["pairwise"], np.float64)
        if CONFIGS[name][1] == "known":
            result = select_known(matrix[1:, 0], bias)
        else:
            ref = references[reference_uid]
            if "geometry" in ref:
                geometry = Geometry.from_dict(ref["geometry"])
            else:
                geometry = sample_geometry(ref, load_rgb(resolve(root, ref["image"])))
            camera = reference_camera(ref["camera"], geometry.fixed_w, geometry.fixed_h, 1 if name == "depth_pro" else 2)
            result = select_unknown(observation["candidate_cameras"], matrix[1:, 1:], camera, bias)
        arm = "direct" if result.choice == 3 else f"candidate{result.choice}"
        with np.load(output / "predictions" / name / f"{uid}.npz", allow_pickle=False) as arrays:
            write_npz(output / "selected" / name / f"{uid}.npz", depth=arrays[arm])
        write_json(output / "selected" / name / f"{uid}.json",
                   {"uid": uid, "reference_uid": reference_uid, "reference_bias": bias,
                    "prediction_cache_key": observation["cache_key"],
                    "reference_cache_key": reference_record["cache_key"],
                    "selected_arm": arm, **asdict(result)})

    jobs = [(name, row) for name in names for row in manifest["samples"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(one, jobs), total=len(jobs), desc="Selecting"))


def evaluate(manifest_path, names, *, output, data_root=None, workers=8, exclude_datasets=()):
    manifest, root = load_manifest(manifest_path, data_root)
    output = Path(output)
    ids = read_json(output / "reference_ids.json")
    references = {r["uid"]: r for r in manifest.get("references", manifest["samples"])}

    def one(job):
        name, row = job
        uid = row["uid"]
        selected = read_json(output / "selected" / name / f"{uid}.json")
        observation = read_json(output / "predictions" / name / f"{uid}.json")
        if selected.get("prediction_cache_key") != observation["cache_key"]:
            raise ValueError(f"Predictions changed since selection: {uid}; rerun select")
        ref_uid = ids[name][uid]
        ref_name = CONFIGS[name][3]
        ref_record = read_json(output / "references" / ref_name / f"{ref_uid}.json")
        if selected["reference_uid"] != ref_uid or selected.get("reference_cache_key") != ref_record["cache_key"]:
            raise ValueError(f"Reference changed since selection: {uid}; rerun select")
        if ref_record.get("input_revision") != input_revision(references[ref_uid], root, ref_name, "reference"):
            raise ValueError(f"Reference inputs changed since calibration: {ref_uid}; rerun predict")
        if observation.get("input_revision") != input_revision(row, root, name, "query"):
            raise ValueError(f"Inputs changed since depth inference: {uid}; rerun predict")
        choice = selected["selected_arm"]
        with np.load(resolve(root, row["depth"]), allow_pickle=False) as file:
            gt = file["depth"]
            mask = file["mask"] if "mask" in file else None
        with np.load(output / "predictions" / name / f"{uid}.npz", allow_pickle=False) as arrays:
            scores = {method: score_depth(arrays[arm], gt, mask)
                      for method, arm in (("direct", "direct"), ("seed0", "candidate0"), ("selector", choice))}
        return dict(uid=uid, dataset=row.get("dataset", "custom"), config=name, setting=CONFIGS[name][1], metrics=scores)

    jobs = [(name, row) for name in names for row in manifest["samples"] if row.get("dataset") not in exclude_datasets]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(tqdm(pool.map(one, jobs), total=len(jobs), desc="Evaluating"))
    rows = aggregate(records)
    write_json(output / "per_image_metrics.json", records)
    write_json(output / "metrics.json", {"image_weighting": "equal per image", "excluded_datasets": list(exclude_datasets), "rows": rows})
    return rows
