#!/usr/bin/env python3
"""Fetch pinned upstream source trees; never install packages or weights."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PATCHES = Path(__file__).resolve().parent / "patches"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    revision: str
    group: str = "depth"
    patch: str | None = None


SOURCES = (
    Source("MoGe2", "https://github.com/microsoft/MoGe.git", "07444410f1e33f402353b99d6ccd26bd31e469e8"),
    Source("utils3d", "https://github.com/EasternJournalist/utils3d.git", "3fab839f0be9931dac7c8488eb0e1600c236e183"),
    Source("MoGe3", "https://github.com/microsoft/MoGe.git", "74fbce054ebed49800de42d0ad0e83495065719a"),
    Source("utils3d-moge", "https://github.com/EasternJournalist/utils3d-moge.git", "62f09d58509485564e24d5d9f6aac9ee9ebc0c37"),
    Source("FlexGEMM", "https://github.com/JeffreyXiang/FlexGEMM.git", "b2fadb29d41846c7981ade6801ffc689fae119cf", patch="flexgemm_triton33.patch"),
    Source("Depth-Anything-3", "https://github.com/ByteDance-Seed/Depth-Anything-3.git", "41736238f5bced4debf3f2a12375d2466874866d"),
    Source("UniDepth", "https://github.com/lpiccinelli-eth/UniDepth.git", "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"),
    Source("UniK3D", "https://github.com/lpiccinelli-eth/UniK3D.git", "29f7862f3c3ee79c53ef0153dfc901ce511df575", patch="unik3d_optional_logging.patch"),
    Source("ml-depth-pro", "https://github.com/apple/ml-depth-pro.git", "9efe5c1def37a26c5367a71df664b18e1306c708"),
    Source("ml-depth-pro-2", "https://github.com/apple-aiml-research/ml-depth-pro-2.git", "aa389cbb0b74341664ed663a0966c859905cd6c0"),
    Source("OneReward", "https://github.com/bytedance/OneReward.git", "0d8d826d9e383c32ad97b39f01bd6872e3ea7aa4", group="generation"),
)


def git(path, *args, check=True):
    return subprocess.run(["git", "-C", str(path), *args], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def check_patched_contents(target, patch, names):
    """Check exact patched bytes, including edits outside the patch's hunks."""
    with tempfile.TemporaryDirectory(prefix="contextdepth-patch-check-") as temporary:
        root = Path(temporary)
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Unsafe path in bundled patch: {name}")
            original = subprocess.check_output(["git", "-C", str(target), "show", f"HEAD:{name}"])
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
        subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
        for name in names:
            if (target / name).read_bytes() != (root / name).read_bytes():
                raise ValueError(f"Unexpected changes beyond the bundled patch: {target / name}")


def inspect(source, directory, *, install=False):
    target = directory / source.name
    if not target.exists():
        if not install:
            raise FileNotFoundError(f"Missing source tree: {target}")
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", source.url, str(target)], check=True)
        git(target, "checkout", "--detach", source.revision)
    if not (target / ".git").exists():
        raise ValueError(f"Refusing to overwrite a non-git directory: {target}")
    actual = git(target, "rev-parse", "HEAD").stdout.strip()
    if actual != source.revision:
        raise ValueError(f"{target} is at {actual}, expected {source.revision}; use a separate clean checkout")
    result = {"name": source.name, "revision": actual, "patch": source.patch}
    changed = git(target, "diff", "--name-only", "HEAD").stdout.splitlines()
    if not source.patch:
        if changed:
            raise ValueError(f"Tracked source changes in {target}: {changed}")
        result["patch_state"] = "not_required"
        return result
    patch = PATCHES / source.patch
    text = patch.read_text()
    expected_changes = {line[6:] for line in text.splitlines() if line.startswith("+++ b/")}
    if set(changed) - expected_changes:
        raise ValueError(f"Unrelated source changes in {target}: {changed}")
    if git(target, "apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
        result["patch_state"] = "already_applied"
    elif not changed and git(target, "apply", "--check", str(patch), check=False).returncode == 0:
        if not install:
            raise ValueError(f"Required patch is not applied in {target}: {source.patch}")
        git(target, "apply", str(patch))
        git(target, "apply", "--reverse", "--check", str(patch))
        result["patch_state"] = "applied"
    else:
        raise ValueError(f"Patch cannot be applied cleanly: {target}/{source.patch}")
    check_patched_contents(target, patch, expected_changes)
    result["patch_sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--install", action="store_true", help="Clone missing pinned trees and apply the two compatibility patches")
    mode.add_argument("--check", action="store_true", help="Check existing trees and patches without changing them")
    parser.add_argument("--group", choices=("all", "core", "depth", "generation"), default="all",
                        help="core fetches only OneReward, MoGe2 and its utils3d dependency")
    parser.add_argument("--directory", type=Path, default=ROOT / "third_party")
    args = parser.parse_args()
    core = {"MoGe2", "utils3d", "OneReward"}
    selected = [source for source in SOURCES if args.group == "all" or source.group == args.group
                or (args.group == "core" and source.name in core)]
    if not args.install and not args.check:
        print(json.dumps({"mode": "plan_only", "directory": str(args.directory),
                          "sources": [asdict(source) for source in selected]}, indent=2))
        return
    rows = [inspect(source, args.directory.resolve(), install=args.install) for source in selected]
    print(json.dumps({"status": "passed", "sources": rows, "environment_modified": False,
                      "weights_downloaded": False}, indent=2))


if __name__ == "__main__":
    main()
