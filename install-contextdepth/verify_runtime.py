#!/usr/bin/env python3
"""Read-only comparison with the observed depth runtime; no model inference.

Use the depth environment's Python and explicitly select the compiler:
    python install-contextdepth/verify_runtime.py --ptxas /path/to/ptxas

Only --output writes a file. Missing dependencies, metadata, or identity
mismatches produce a JSON report and a nonzero exit status. A match is not
evidence of numerical or prediction equivalence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


# Suppress bytecode writes during dependency imports in this inspection tool.
sys.dont_write_bytecode = True
REFERENCE = Path(__file__).with_name("validated-depth-runtime.json")


def error_description(error: Exception) -> str:
    """Keep diagnostic categories without leaking local paths from exceptions."""
    description = type(error).__name__
    if isinstance(error, ModuleNotFoundError) and error.name:
        description += f" (missing module: {error.name})"
    return description


class Report:
    def __init__(self):
        self.matches = []
        self.mismatches = []

    def check(self, name, expected, actual, error=None):
        entry = {"check": name, "expected": expected, "actual": actual}
        if error:
            entry["error"] = error
        passed = error is None and type(actual) is type(expected) and actual == expected
        (self.matches if passed else self.mismatches).append(entry)

    def inspect(self, name, expected, function):
        try:
            # Keep stdout as one JSON document even if a dependency is noisy.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                actual = function()
        except Exception as error:
            self.check(name, expected, None, error_description(error))
        else:
            self.check(name, expected, actual)


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def module_version(name):
    return str(importlib.import_module(name).__version__)


def cv_jpeg_build():
    cv2 = importlib.import_module("cv2")
    for line in cv2.getBuildInformation().splitlines():
        # JPEG 2000 is a different codec and must not satisfy this check.
        match = re.match(r"^\s*JPEG:\s*(.*?)\s*$", line)
        if match:
            return " ".join(match.group(1).split())
    return None


def conda_identity(name, fields):
    directory = Path(sys.prefix) / "conda-meta"
    if not directory.is_dir():
        raise RuntimeError("No conda-meta directory")
    records = []
    for filename in directory.glob(name + "-*.json"):
        with filename.open(encoding="utf-8") as stream:
            record = json.load(stream)
        if record.get("name") == name:
            records.append(record)
    if len(records) != 1:
        raise RuntimeError("Expected exactly one matching Conda record")
    return {field: records[0].get(field) for field in fields}


def check_conda(report, expected):
    for name, identity in expected.items():
        try:
            actual = conda_identity(name, identity)
        except Exception as error:
            # A missing/ambiguous record is a mismatch, never a skipped check.
            reason = error_description(error)
            if isinstance(error, RuntimeError):
                reason += ": " + str(error)
            report.check("conda." + name, identity, None, reason)
        else:
            report.check("conda." + name, identity, actual)


def check_ptxas(report, expected, binary):
    try:
        digest = hashlib.sha256()
        with binary.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
    except Exception as error:
        actual_sha256 = None
        report.check("ptxas.sha256", expected["sha256"], None, error_description(error))
    else:
        report.check("ptxas.sha256", expected["sha256"], actual_sha256)
    try:
        completed = subprocess.run(
            [str(binary.resolve()), "--version"],
            check=False, capture_output=True, text=True, timeout=15,
        )
        text = completed.stdout + "\n" + completed.stderr
        version = re.search(r"release\s+([\d.]+),\s*V([\d.]+)", text)
        build = re.search(r"^\s*Build\s+(\S+)\s*$", text, re.MULTILINE)
        actual = {
            "cuda_release": version.group(1) if version else None,
            "version": version.group(2) if version else None,
            "build": build.group(1) if build else None,
        }
    except Exception as error:
        for field in ("cuda_release", "version", "build"):
            report.check("ptxas." + field, expected[field], None, error_description(error))
    else:
        report.check("ptxas.version_command_exit_code", 0, completed.returncode)
        for field, value in actual.items():
            report.check("ptxas." + field, expected[field], value)


def verify(reference, binary):
    report = Report()
    report.check("python.version", reference["python"], platform.python_version())
    for name, expected in reference["packages"].items():
        report.inspect("distribution." + name, expected, lambda name=name: package_version(name))

    # Import the actual numerical/image runtime in addition to reading metadata.
    # Transformers, generation packages, models, and weights are not imported.
    for module, distribution in (
        ("torch", "torch"), ("torchvision", "torchvision"),
        ("numpy", "numpy"), ("scipy", "scipy"),
        ("PIL", "Pillow"), ("triton", "triton"),
    ):
        report.inspect(
            "module." + module + ".version", reference["packages"][distribution],
            lambda module=module: module_version(module),
        )
    torch_expected = reference["torch"]
    report.inspect("torch.cuda", torch_expected["cuda"], lambda: importlib.import_module("torch").version.cuda)
    report.inspect("torch.cudnn", torch_expected["cudnn"], lambda: importlib.import_module("torch").backends.cudnn.version())
    report.inspect("torch.cuda_available", True, lambda: importlib.import_module("torch").cuda.is_available())
    report.inspect("torch.gpu_0", torch_expected["gpu"], lambda: importlib.import_module("torch").cuda.get_device_name(0))

    images = reference["image_libraries"]
    report.inspect("opencv.jpeg_build", images["opencv_jpeg_build"], cv_jpeg_build)
    report.inspect(
        "pillow.jpeg_version", images["pillow_jpeg_version"],
        lambda: importlib.import_module("PIL.features").version_codec("jpg"),
    )
    report.inspect(
        "pillow.libjpeg_turbo_version", images["pillow_libjpeg_turbo_version"],
        lambda: importlib.import_module("PIL.features").version_feature("libjpeg_turbo"),
    )
    check_conda(report, reference["conda_packages"])
    check_ptxas(report, reference["ptxas"], binary)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptxas", required=True, type=Path, help="Explicit compiler binary to hash and inspect; PATH is not searched")
    parser.add_argument("--output", type=Path, help="Optional JSON report file; no environment files are modified")
    args = parser.parse_args()
    try:
        with REFERENCE.open(encoding="utf-8") as stream:
            reference = json.load(stream)
        report = verify(reference, args.ptxas)
    except Exception as error:
        report = Report()
        report.check("verification_completed", True, False, error_description(error))
    result = {
        "schema_version": 1,
        "reference": REFERENCE.name,
        "runtime_identity_matches": not report.mismatches,
        "prediction_equivalence_verified": False,
        "scope": "Runtime identity only; no inputs, preprocessing, weights, model settings, or predictions were validated.",
        "matched_count": len(report.matches),
        "mismatched_count": len(report.mismatches),
        "matches": report.matches,
        "mismatches": report.mismatches,
    }
    if args.output:
        try:
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        except Exception as error:
            result["report_write_error"] = error_description(error)
            print(json.dumps(result, indent=2))
            return 2
    print(json.dumps(result, indent=2))
    return 0 if result["runtime_identity_matches"] else 1


if __name__ == "__main__":
    sys.exit(main())
