#!/usr/bin/env python3
"""Explicit checkpoint downloads and CPU-only, verified weight preparation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

from weights import FLUX_FILL_BASE, ONEREWARD_BF16, SIGLIP2_VISION


ROOT = Path(__file__).resolve().parents[1]
# An unavailable historical Hub revision is left unset; byte hashes still gate
# acceptance, so a changed default-branch checkpoint cannot silently replace it.
DEPTH_ASSETS = (
    ("Ruicheng/moge-2-vitl-normal", None, "model.pt", "moge-2-vitl-normal", "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01"),
    ("Ruicheng/moge-3-vitg", "6ef26c5a4b4148dab5ccaacdb08b72dc66380475", "model.pt", "moge-3-vitg", "ce7c15417e9105c2ace7b4272e2cc69e36940921211eb7fa05d4d0bb03f0a00c"),
    ("depth-anything/DA3NESTED-GIANT-LARGE-1.1", None, "model.safetensors", "DA3NESTED-GIANT-LARGE-1.1", "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c"),
    ("depth-anything/DA3METRIC-LARGE", None, "model.safetensors", "DA3METRIC-LARGE", "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776"),
    ("lpiccinelli/unik3d-vitl", "008aa54308b8bc142f74c9c7b098833aaede06bd", "model.safetensors", "unik3d-vitl", "890be79699dd68abb3c9a3aab24b528bf09d97eca4dd073e3e83c0a2028566b2"),
    ("lpiccinelli/unidepth-v2-vitl14", "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9", "pytorch_model.bin", "unidepth-v2-vitl14", "101d4a941854e7f7ec6839a5a39bfe35bd5cc59ed65d8c5994a55607d8481541"),
    # Depth Pro 2 (H+) keeps its focal-length estimator in a separate checkpoint.
    ("apple/DepthPro2-H", "68426c789f75a7c760273e857222625b4fe69576", "model.safetensors", "depth_pro2/DepthPro2-H", "cfc532336019abd2128b04be2c2967175522e46de69222ef72b87c6675ee3b17"),
    ("apple/DepthPro2-focal", "622659d44a2002cf9ae4333f2569afa938290d9a", "model.safetensors", "depth_pro2/DepthPro2-focal", "8f567d1c97ff916137288668f6bd306ce2094da2e2b852b0e4539dbb139c8c86"),
)
DEPTH_PRO_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
DEPTH_PRO_SHA256 = "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"
DEPTH_MODEL_DIRECTORIES = {
    "moge2": "moge-2-vitl-normal",
    "moge3": "moge-3-vitg",
    "da3": "DA3NESTED-GIANT-LARGE-1.1",
    "da3_metric": "DA3METRIC-LARGE",
    "unik3d": "unik3d-vitl",
    "unidepth": "unidepth-v2-vitl14",
    "depth_pro": "depth_pro.pt",
    "depth_pro2": ("depth_pro2/DepthPro2-H", "depth_pro2/DepthPro2-focal"),
}


def _directories(names=None):
    """Weight folders (or the Depth Pro file) of the chosen models; all models by default."""
    values = DEPTH_MODEL_DIRECTORIES.values() if not names else (DEPTH_MODEL_DIRECTORIES[name] for name in names)
    return {directory for value in values for directory in (value if isinstance(value, tuple) else (value,))}
DEPTH_CONFIG_SHA256 = {
    "DA3NESTED-GIANT-LARGE-1.1": "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468",
    "DA3METRIC-LARGE": "a336f3e76fe375aaae17a9aed9130c9f2aa061535d317ec57dcb2f1f02e1dd53",
    "unik3d-vitl": "2cb006c03081f593b31d90238891cd5e0def149f9edfe7fa033be22588645b21",
    "unidepth-v2-vitl14": "09eb0ea8de53a6c9a1d428ac98c79847fe2602ea417701261f3f628099a30816",
}


def specs_sha(specs):
    payload = json.dumps(specs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected):
    observed = file_sha(path)
    if observed != expected:
        raise ValueError(f"Checkpoint mismatch: {path}; expected {expected}, observed {observed}")
    return observed


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


def staging_directory(output):
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".preparing-", dir=output.parent))
    print(f"Preparing in {temporary}", flush=True)
    return output, temporary


def download_depth(root, *, check_only=False, models=None):
    selected = _directories(models)
    if not check_only:
        from huggingface_hub import snapshot_download
        root.mkdir(parents=True, exist_ok=True)
    verified = []
    for repo, revision, filename, directory, expected in DEPTH_ASSETS:
        if directory not in selected:
            continue
        destination = root / directory
        target = destination / filename
        if target.exists():
            require_hash(target, expected)
        if not check_only and (not target.exists() or (filename != "model.pt" and not (destination / "config.json").is_file())):
            snapshot_download(repo_id=repo, revision=revision, local_dir=destination,
                              allow_patterns=[filename, "config.json", "README.md", "LICENSE*", "NOTICE*"])
        observed = require_hash(target, expected)
        if filename != "model.pt" and not (destination / "config.json").is_file():
            raise FileNotFoundError(f"Missing model config: {destination / 'config.json'}")
        item = {"repo": repo, "revision": revision, "file": str(target.relative_to(root)), "sha256": observed}
        if directory in DEPTH_CONFIG_SHA256:
            item["config_sha256"] = require_hash(destination / "config.json", DEPTH_CONFIG_SHA256[directory])
        verified.append(item)
    if "depth_pro.pt" in selected:
        target = root / "depth_pro.pt"
        if not target.exists() and not check_only:
            with tempfile.NamedTemporaryFile(prefix="depth_pro.", suffix=".download", dir=root, delete=False) as stream:
                temporary = Path(stream.name)
            urllib.request.urlretrieve(DEPTH_PRO_URL, temporary)
            require_hash(temporary, DEPTH_PRO_SHA256)
            temporary.rename(target)
        observed = require_hash(target, DEPTH_PRO_SHA256)
        verified.append({"url": DEPTH_PRO_URL, "file": "depth_pro.pt", "sha256": observed})
    report = {"status": "verified", "assets": verified}
    if not check_only:
        suffix = "_" + "_".join(sorted(set(models))) if models else ""
        write_json(root / f"depth_weights_verified{suffix}.json", report)
    print(json.dumps(report, indent=2))


def convert_onereward(source, output):
    """Keep tensor names/values; cast to BF16 and verify all 1160 paper hashes."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    expected = ONEREWARD_BF16
    config = json.loads((source / "config.json").read_text())
    if config.get("_class_name") != "FluxTransformer2DModel" or config.get("in_channels") != 384:
        raise ValueError("Expected the OneReward FLUX Fill transformer config (in_channels=384)")
    sources = sorted(source.glob("*.safetensors"))
    if not sources:
        raise FileNotFoundError(f"No Diffusers safetensors in {source}")
    output, temporary = staging_directory(output)
    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(path), framework="pt", device="cpu")) for path in sources]
        key_to_handle = {}
        for handle in handles:
            for key in handle.keys():
                if key in key_to_handle:
                    raise ValueError(f"Duplicate tensor {key}; source must contain only the original transformer shards")
                key_to_handle[key] = handle
        if set(key_to_handle) != set(expected["tensors"]):
            raise ValueError("Source tensor names differ from the paper's OneReward transformer")
        groups, current, size = [], [], 0
        for key in sorted(key_to_handle):
            info = expected["tensors"][key]
            if info["dtype"] != "BF16":
                raise ValueError(f"Unexpected expected dtype for {key}")
            elements = 1
            for dimension in info["shape"]:
                elements *= dimension
            tensor_bytes = elements * 2
            if current and size + tensor_bytes > 5_000_000_000:
                groups.append(current)
                current, size = [], 0
            current.append(key)
            size += tensor_bytes
        if current:
            groups.append(current)
        weight_map, total_size = {}, 0
        for index, group in enumerate(groups, 1):
            filename = f"diffusion_pytorch_model-{index:05d}-of-{len(groups):05d}.safetensors"
            tensors = {}
            for key in group:
                tensor = key_to_handle[key].get_tensor(key).to(torch.bfloat16).contiguous()
                info = expected["tensors"][key]
                actual = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
                if list(tensor.shape) != info["shape"] or actual != info["sha256"]:
                    raise ValueError(f"OneReward BF16 tensor differs from the experiment: {key}")
                tensors[key] = tensor
                weight_map[key] = filename
                total_size += tensor.numel() * tensor.element_size()
            save_file(tensors, str(temporary / filename), metadata={"format": "pt"})
            del tensors
            print(f"Verified and saved shard {index}/{len(groups)}", flush=True)
    shutil.copyfile(source / "config.json", temporary / "config.json")
    write_json(temporary / "diffusion_pytorch_model.safetensors.index.json",
               {"metadata": {"total_size": total_size}, "weight_map": weight_map})
    receipt = {"status": "verified", "tensor_count": len(weight_map), "all_expected_tensor_hashes_match": True,
               "expected_tensor_specs_sha256": specs_sha(expected), "torch": torch.__version__,
               "dtype": "bfloat16", "conversion": "CPU tensor-by-tensor cast, no quantization or training",
               "output_files_sha256": {path.name: file_sha(path) for path in sorted(temporary.iterdir())}}
    write_json(temporary / "preparation.json", receipt)
    temporary.rename(output)
    print(json.dumps({"output": str(output), **receipt}, indent=2))


def export_siglip2(source, output):
    import torch
    from transformers import SiglipVisionModel

    output, temporary = staging_directory(output)
    model = SiglipVisionModel.from_pretrained(str(source), torch_dtype=torch.float32,
                                             local_files_only=True).eval().requires_grad_(False)
    config = model.config
    if (config.hidden_size, config.num_hidden_layers, config.image_size, config.patch_size) != (1152, 27, 384, 16):
        raise ValueError("Expected google/siglip2-so400m-patch16-384 vision tower")
    expected = SIGLIP2_VISION["tensors"]
    state = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("SigLIP2 vision tensor names differ from the experiment")
    for key, tensor in state.items():
        actual = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        info = expected[key]
        if info["dtype"] != "F32" or list(tensor.shape) != info["shape"] or actual != info["sha256"]:
            raise ValueError(f"SigLIP2 vision tensor differs from the experiment: {key}")
    model.save_pretrained(str(temporary), safe_serialization=True)
    shutil.copyfile(source / "preprocessor_config.json", temporary / "preprocessor_config.json")
    # Processor bytes and saved tensor bytes are frozen independently of JSON
    # formatting in the generated vision-only config.
    require_hash(temporary / "preprocessor_config.json", "fb2817d3523ca3b666c859f15320c7138416bc38ffc515e2963f78c868c51c90")
    receipt = {"status": "verified", "model": "google/siglip2-so400m-patch16-384", "dtype": "float32",
               "vision_only": True, "torch": torch.__version__,
               "tensor_count": len(state), "all_expected_tensor_hashes_match": True,
               "expected_tensor_specs_sha256": specs_sha(expected),
               "output_files_sha256": {path.name: file_sha(path) for path in sorted(temporary.iterdir())}}
    write_json(temporary / "preparation.json", receipt)
    temporary.rename(output)
    print(json.dumps({"output": str(output), **receipt}, indent=2))


def check_flux_base(source):
    expected = FLUX_FILL_BASE
    verified = {name: require_hash(source / name, value) for name, value in expected.items()}
    print(json.dumps({"status": "verified", "loaded_asset_count": len(verified),
                      "source_modified": False, "files_sha256": verified}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    depth = commands.add_parser("depth", help="Download and hash-check the checkpoints of the built-in depth models")
    depth.add_argument("--weights-root", type=Path, default=ROOT / "weights")
    depth.add_argument("--check-only", action="store_true", help="Read local weights without downloading or modifying files")
    depth.add_argument("--model", action="append", choices=tuple(DEPTH_MODEL_DIRECTORIES),
                       help="Download/check only this model (repeatable); default: every built-in depth model")
    siglip = commands.add_parser("siglip2", help="Export a downloaded SigLIP2 vision tower on CPU")
    siglip.add_argument("--source", type=Path, required=True)
    siglip.add_argument("--output", type=Path, default=ROOT / "weights/siglip2-so400m-patch16-384-vision")
    reward = commands.add_parser("onereward", help="Convert downloaded original OneReward transformer shards to verified BF16")
    reward.add_argument("--source", type=Path, required=True)
    reward.add_argument("--output", type=Path, default=ROOT / "weights/OneReward/diffusers-bf16")
    flux = commands.add_parser("flux-base", help="Read-only hash check of every FLUX base asset used by OneReward")
    flux.add_argument("--source", type=Path, default=ROOT / "weights/FLUX.1-Fill-dev")
    args = parser.parse_args()
    if args.command == "depth":
        download_depth(args.weights_root.expanduser().resolve(), check_only=args.check_only, models=args.model)
    elif args.command == "siglip2":
        export_siglip2(args.source.expanduser().resolve(), args.output)
    elif args.command == "flux-base":
        check_flux_base(args.source.expanduser().resolve())
    else:
        convert_onereward(args.source.expanduser().resolve(), args.output)


if __name__ == "__main__":
    main()
