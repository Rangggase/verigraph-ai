"""Integrity guard for joblib model artifacts.

joblib/pickle files can execute arbitrary code when loaded. Models produced by
``scripts/train_ml_baseline.py`` therefore get a ``model.joblib.sha256`` sidecar,
and the app refuses to load any artifact whose digest is missing or different.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SIDECAR_SUFFIX = ".sha256"


class ModelIntegrityError(RuntimeError):
    """Raised when a model artifact is unsigned or has been modified."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(model_path: Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(model_path.name + SIDECAR_SUFFIX)


def write_sidecar(model_path: Path) -> str:
    digest = file_sha256(model_path)
    sidecar_path(model_path).write_text(digest + "\n", encoding="ascii")
    return digest


def verify_model(model_path: Path) -> None:
    """Raise ``ModelIntegrityError`` unless the file matches its sidecar digest."""
    model_path = Path(model_path)
    sidecar = sidecar_path(model_path)
    if not model_path.is_file():
        raise ModelIntegrityError("Model file not found.")
    if not sidecar.is_file():
        raise ModelIntegrityError("Model has no integrity signature; refusing to load it.")
    expected = sidecar.read_text(encoding="ascii", errors="replace").strip().lower()
    if expected != file_sha256(model_path):
        raise ModelIntegrityError("Model file does not match its signature; refusing to load it.")


def load_trusted_model(model_path: Path):
    """Verify then load a project-local model. Never call on uploaded files."""
    verify_model(model_path)
    import joblib

    return joblib.load(model_path)
