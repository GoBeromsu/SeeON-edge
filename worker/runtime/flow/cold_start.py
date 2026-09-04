"""Fail-closed prebuilt-engine admission for the DeepStream Flow profile."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path


class EngineIdentityError(RuntimeError):
    """A prebuilt Flow engine is absent or does not match its identity proof."""


class FlowWarmupTimeout(RuntimeError):
    """The Flow bootstrap source did not yield an accepted metadata frame."""


@dataclass(frozen=True, slots=True)
class FlowColdStart:
    """Verify immutable artifacts, then warm the already-built media plane."""

    engine_path: Path
    identity_path: Path
    files: Mapping[str, Path]
    warmup: Callable[[], None]

    def run(self) -> None:
        verify_engine_identity(self.engine_path, self.identity_path, self.files)
        self.warmup()


def verify_engine_identity(
    engine_path: Path,
    identity_path: Path,
    files: Mapping[str, Path],
) -> None:
    """Verify all identity-file digests without ever building at runtime."""
    if not engine_path.is_file():
        raise EngineIdentityError(
            f"Flow engine is absent: {engine_path}; run edge-engine-build before activation"
        )
    if not identity_path.is_file():
        raise EngineIdentityError(f"Flow engine identity is absent: {identity_path}")
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EngineIdentityError(f"Flow engine identity is unreadable: {identity_path}") from error
    if not isinstance(identity, dict):
        raise EngineIdentityError("Flow engine identity must be a JSON object")
    expected = {"engine_sha256": engine_path, **dict(files)}
    for key, path in expected.items():
        digest = identity.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise EngineIdentityError(f"Flow engine identity lacks valid {key}")
        if not path.is_file():
            raise EngineIdentityError(f"Flow artifact is absent for {key}: {path}")
        actual = _sha256(path)
        if actual != digest:
            raise EngineIdentityError(f"Flow artifact digest mismatch for {key}: {path}")
    image_digest = identity.get("image_digest")
    if not isinstance(image_digest, str) or not image_digest:
        raise EngineIdentityError("Flow engine identity lacks image_digest")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["EngineIdentityError", "FlowColdStart", "FlowWarmupTimeout", "verify_engine_identity"]
