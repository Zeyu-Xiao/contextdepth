#!/usr/bin/env python3
"""Extract the verified historical ptxas from a local wheel, without executing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import zipfile


WHEEL_SHA256 = "2d6dc36fb7cb5ac9c0b8825bc13d193c35487a315664007287d0126531238011"
PTXAS_SHA256 = "c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca"
WHEEL_MEMBER = "nvidia/cuda_nvcc/bin/ptxas"
REPO_ROOT = Path(__file__).resolve().parent.parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_existing(path: Path) -> bool:
    """Accept an existing regular file only if its contents match exactly."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Refusing non-regular output (including symlinks): {path}")
    actual = file_sha256(path)
    if actual != PTXAS_SHA256:
        raise ValueError(
            f"Output already exists with a different SHA-256; it was not overwritten: "
            f"{path} (actual {actual}, expected {PTXAS_SHA256})"
        )
    return True


def install_payload(payload: bytes, output: Path) -> str:
    if verify_existing(output):
        if os.name == "posix":
            output.chmod(0o755)
        return "already_present"

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            if os.name == "posix":
                os.fchmod(stream.fileno(), 0o755)
            os.fsync(stream.fileno())
        if file_sha256(temporary) != PTXAS_SHA256:
            raise ValueError("Temporary ptxas file failed SHA-256 verification")
        try:
            # Link a fully written file into place atomically. Unlike replace(),
            # this cannot overwrite an output created by another process.
            os.link(temporary, output)
            status = "installed"
        except FileExistsError:
            if not verify_existing(output):
                raise ValueError(f"Output disappeared during installation: {output}")
            if os.name == "posix":
                output.chmod(0o755)
            status = "already_present"
        if not verify_existing(output):
            raise ValueError(f"Output disappeared after installation: {output}")
        return status
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel", required=True, type=Path,
        help=("Locally downloaded nvidia_cuda_nvcc_cu12-12.8.93-py3-none-"
              "manylinux2010_x86_64.manylinux_2_12_x86_64.whl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "tools" / "cuda-12.8" / "bin" / "ptxas",
        help="Destination file; default: <repository>/tools/cuda-12.8/bin/ptxas",
    )
    arguments = parser.parse_args()
    try:
        wheel = arguments.wheel.expanduser().resolve(strict=True)
        # Do not resolve the final component: an existing symlink must be rejected.
        output = Path(os.path.abspath(arguments.output.expanduser()))
        actual_wheel_sha256 = file_sha256(wheel)
        if actual_wheel_sha256 != WHEEL_SHA256:
            raise ValueError(
                f"Wheel SHA-256 mismatch: actual {actual_wheel_sha256}, "
                f"expected {WHEEL_SHA256}"
            )
        with zipfile.ZipFile(wheel) as archive:
            if archive.namelist().count(WHEEL_MEMBER) != 1:
                raise ValueError(f"Wheel must contain exactly one {WHEEL_MEMBER}")
            payload = archive.read(WHEEL_MEMBER)
        actual_payload_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_payload_sha256 != PTXAS_SHA256:
            raise ValueError(
                f"ptxas SHA-256 mismatch: actual {actual_payload_sha256}, "
                f"expected {PTXAS_SHA256}"
            )
        status = install_payload(payload, output)
        print(json.dumps({
            "status": status,
            "wheel": str(wheel),
            "wheel_sha256": actual_wheel_sha256,
            "path": str(output),
            "sha256": file_sha256(output),
            "ptxas_executed": False,
        }, indent=2))
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, RuntimeError) as error:
        print(f"prepare_compiler: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
