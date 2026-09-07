"""Durable, identity-bound clip re-analysis artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile

from shared.events.clip_analysis_wire import (
    ClipAnalysisResult,
    ClipAnalysisWireError,
    decode_clip_analysis,
    encode_clip_analysis,
)


class ClipAnalysisArtifactError(ValueError):
    """An analysis artifact is malformed or has the wrong immutable identity."""


@dataclass(frozen=True, slots=True)
class ClipAnalysisArtifactIdentity:
    clip_id: str
    clip_sha256: str
    pose_model_sha256: str
    bed_model_sha256: str
    analysis_profile_sha256: str
    decoder_identity: str


def artifact_path(clip_path: Path) -> Path:
    return clip_path.parent / "clip.analysis.json"


def _sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def _matches(result: ClipAnalysisResult, identity: ClipAnalysisArtifactIdentity) -> bool:
    return (
        result.clip_id == identity.clip_id
        and result.clip_sha256 == identity.clip_sha256
        and result.pose_model_sha256 == identity.pose_model_sha256
        and result.bed_model_sha256 == identity.bed_model_sha256
        and result.analysis_profile_sha256 == identity.analysis_profile_sha256
        and result.decoder_identity == identity.decoder_identity
    )


def _decode_checked(payload: bytes, identity: ClipAnalysisArtifactIdentity) -> ClipAnalysisResult:
    try:
        result = decode_clip_analysis(payload)
    except ClipAnalysisWireError as exc:
        raise ClipAnalysisArtifactError("invalid_clip_analysis") from exc
    if not _matches(result, identity):
        raise ClipAnalysisArtifactError("identity_mismatch")
    return result


def validate_scratch(path: Path, identity: ClipAnalysisArtifactIdentity) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ClipAnalysisArtifactError("scratch_unreadable") from exc
    _decode_checked(payload, identity)
    return payload


def load_clip_analysis(
    clip_path: Path, identity: ClipAnalysisArtifactIdentity
) -> ClipAnalysisResult | None:
    path = artifact_path(clip_path)
    sidecar = _sidecar_path(path)
    try:
        payload = path.read_bytes()
        expected_digest = sidecar.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if expected_digest != sha256(payload).hexdigest():
        return None
    try:
        return _decode_checked(payload, identity)
    except ClipAnalysisArtifactError:
        return None


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
        file.write(payload)
        file.flush()
    try:
        replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_clip_analysis(
    clip_path: Path, scratch_path: Path, identity: ClipAnalysisArtifactIdentity
) -> Path:
    """Validate then atomically publish the one artifact valid for *identity*."""
    payload = validate_scratch(scratch_path, identity)
    existing = load_clip_analysis(clip_path, identity)
    target = artifact_path(clip_path)
    if existing is not None:
        return target
    # Re-encode to require canonical wire bytes even when the child wrote valid JSON.
    result = _decode_checked(payload, identity)
    canonical = encode_clip_analysis(result)
    _atomic_write(target, canonical)
    _atomic_write(_sidecar_path(target), f"{sha256(canonical).hexdigest()}\n".encode("ascii"))
    return target


__all__ = [
    "ClipAnalysisArtifactError",
    "ClipAnalysisArtifactIdentity",
    "artifact_path",
    "load_clip_analysis",
    "publish_clip_analysis",
    "validate_scratch",
]
