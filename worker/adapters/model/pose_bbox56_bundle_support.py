"""Verified bundle-member helpers shared by pose+bbox56 runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from worker.adapters.model.errors import ModelLoadError


def member_digest(manifest: object, relative_path: str) -> str:
    """Return one verified member's digest from a valid bundle manifest."""
    if not isinstance(manifest, dict):
        raise ModelLoadError("invalid bundle-manifest.json")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ModelLoadError("invalid bundle-manifest.json")
    for item in files:
        if isinstance(item, dict) and item.get("relative_path") == relative_path:
            digest = item.get("sha256")
            if isinstance(digest, str):
                return digest
    raise ModelLoadError(f"bundle manifest does not list {relative_path}")


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelLoadError(f"cannot read {path.name}") from exc


def verify_bundle(root: Path, manifest: object) -> None:
    """Verify every listed member before a runner deserializes model data."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ModelLoadError("invalid bundle-manifest.json")
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise ModelLoadError("invalid bundle-manifest file entry")
        relative, digest, size = item.get("relative_path"), item.get("sha256"), item.get("size")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ModelLoadError("invalid bundle-manifest path")
        if not isinstance(digest, str) or len(digest) != 64 or not isinstance(size, int):
            raise ModelLoadError("invalid bundle-manifest digest")
        path = root / relative
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ModelLoadError(f"missing bundle member {relative}") from exc
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise ModelLoadError(f"bundle member identity mismatch: {relative}")
