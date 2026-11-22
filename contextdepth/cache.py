"""Generic file fingerprints and compatibility access to evaluation caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


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


def model_revision(paths, *, stage="depth"):
    """Compatibility entry for the evaluation recipe; ordinary APIs do not use it."""
    from evaluation.cache import model_revision as evaluation_model_revision
    return evaluation_model_revision(paths, stage=stage)


def input_revision(row, root, spec, role):
    """Compatibility entry for the evaluation manifest input fingerprint."""
    from evaluation.cache import input_revision as evaluation_input_revision
    return evaluation_input_revision(row, root, spec, role)
