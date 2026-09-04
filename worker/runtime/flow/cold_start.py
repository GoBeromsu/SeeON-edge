"""Fail-closed prebuilt-engine admission for the DeepStream Flow profile."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final


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
) -> dict[str, str]:
    """Verify all identity-file digests without ever building at runtime.

    Returns the verified identity so the composition root can name the
    engine's digest as the pose component's artifact identity.
    """
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
    return {key: str(value) for key, value in identity.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["EngineIdentityError", "FlowColdStart", "FlowWarmupTimeout", "verify_engine_identity"]


#: Every env key the flow media plane needs before a camera may be admitted.
FLOW_BOOT_ENV: Final = (
    "ML_WORKER_FLOW_ENGINE_PATH",
    "ML_WORKER_FLOW_ENGINE_IDENTITY_PATH",
    "ML_WORKER_FLOW_INFER_CONFIG",
    "ML_WORKER_FLOW_TRACKER_CONFIG",
    "ML_WORKER_FLOW_TRACKER_LIBRARY",
    "ML_WORKER_FLOW_ONNX_PATH",
    "ML_WORKER_FLOW_PARSER_LIBRARY",
    "ML_WORKER_FLOW_RECORD_DIR",
    "ML_WORKER_FLOW_RECORD_CACHE_SECONDS",
    "ML_WORKER_FLOW_FRAME_WIDTH",
    "ML_WORKER_FLOW_FRAME_HEIGHT",
)

#: Identity keys whose recorded digest must match the file on disk.
FLOW_IDENTITY_FILES: Final = {
    "infer_config_sha256": "ML_WORKER_FLOW_INFER_CONFIG",
    "tracker_config_sha256": "ML_WORKER_FLOW_TRACKER_CONFIG",
    "tracker_library_sha256": "ML_WORKER_FLOW_TRACKER_LIBRARY",
    "onnx_sha256": "ML_WORKER_FLOW_ONNX_PATH",
    "parser_lib_sha256": "ML_WORKER_FLOW_PARSER_LIBRARY",
}


def verify_flow_boot_inputs(env: Mapping[str, str]) -> dict[str, str]:
    """Fail closed on the flow profile's wiring and its engine identity.

    The flow plane has no native manifest, so this is the boot gate's only
    proof that what will run is what ``edge-engine-build`` produced. Every
    recorded digest is checked against the file on disk; nothing is ever built
    here (ADR-0002).
    """
    missing = [key for key in FLOW_BOOT_ENV if not env.get(key)]
    if missing:
        raise EngineIdentityError(f"flow profile wiring is missing: {', '.join(missing)}")
    return verify_engine_identity(
        Path(env["ML_WORKER_FLOW_ENGINE_PATH"]),
        Path(env["ML_WORKER_FLOW_ENGINE_IDENTITY_PATH"]),
        {key: Path(env[name]) for key, name in FLOW_IDENTITY_FILES.items()},
    )
