"""The twelve paper configurations and their shared model/reference backends."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


CONFIGS = {
    "moge3_unknown": ("moge3", "unknown", .25, "moge3_known"),
    "moge3_known": ("moge3", "known", .25, "moge3_known"),
    "da3nested": ("da3nested", "unknown", .25, "da3nested_reference"),
    "da3metric_gt_focal": ("da3metric", "known", .25, "da3metric_gt_focal"),
    "moge2": ("moge2", "unknown", .25, "moge2_gt_fov"),
    "moge2_gt_fov": ("moge2", "known", .25, "moge2_gt_fov"),
    "unik3d_unknown": ("unik3d", "unknown", .10, "unik3d_known"),
    "unik3d_known": ("unik3d", "known", .25, "unik3d_known"),
    "depth_pro": ("depth_pro", "unknown", .10, "depth_pro_gt_focal"),
    "depth_pro_gt_focal": ("depth_pro", "known", .25, "depth_pro_gt_focal"),
    "unidepth_v2_vitl14": ("unidepth", "unknown", .10, "unidepth_gt_k"),
    "unidepth_gt_k": ("unidepth", "known", .25, "unidepth_gt_k"),
}


def load_config(path):
    path = Path(path).resolve()
    with path.open() as f:
        config = yaml.safe_load(f)
    base = path.parent
    # Configured paths are relative to the directory containing this YAML file.
    for section in [*config.get("models", {}).values(), config.get("outpainting", {}), config.get("retrieval", {})]:
        for key, value in section.items():
            if key in ("source", "weights", "base_model", "flex_gemm_source", "utils3d_source", "checkpoint", "ptxas", "cache_dir", "config") and value:
                expanded = Path(os.path.expandvars(str(value))).expanduser()
                section[key] = str(expanded if expanded.is_absolute() else base / expanded)
    return config


def model_paths(config, name):
    family = "da3nested" if name == "da3nested_reference" else CONFIGS[name][0]
    return config["models"][family]
