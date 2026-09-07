"""Durable, immutable identity-bound clip re-analysis artifacts."""

from __future__ import annotations

import logging
import os
import time
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

LOGGER = logging.getLogger(__name__)


class ClipAnalysisArtifactError(ValueError):
    """An analysis artifact is malformed or violates immutable publication."""


@dataclass(frozen=True, slots=True)
class ClipAnalysisArtifactIdentity:
    clip_id: str
    clip_sha256: str
    pose_model_sha256: str
    bed_model_sha256: str
    analysis_profile_sha256: str
    decoder_identity: str

    @property
    def digest(self) -> str:
        material = "|".join(
            (
                self.clip_sha256,
                self.pose_model_sha256,
                self.bed_model_sha256,
                self.decoder_identity,
                self.analysis_profile_sha256,
            )
        )
        return sha256(material.encode("utf-8")).hexdigest()[:16]


def artifact_path(clip_path: Path, identity: ClipAnalysisArtifactIdentity) -> Path:
    return clip_path.parent / f"clip.analysis.{identity.digest}.json"


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


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    try:
        # Readable like manifest.json: the backend reads the store as another user.
        temporary.chmod(0o644)
        replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_clip_analysis(
    clip_path: Path, scratch_path: Path, identity: ClipAnalysisArtifactIdentity
) -> Path:
    payload = validate_scratch(scratch_path, identity)
    result = _decode_checked(payload, identity)
    canonical = encode_clip_analysis(result)
    target = artifact_path(clip_path, identity)
    sidecar = _sidecar_path(target)
    if target.exists() or sidecar.exists():
        existing_payload = _verified_payload(target, sidecar)
        if existing_payload is not None:
            try:
                existing = _decode_checked(existing_payload, identity)
            except ClipAnalysisArtifactError:
                existing = None
            if existing is not None and existing_payload == canonical:
                return target
            raise ClipAnalysisArtifactError("identity_collision")
        if target.exists() and target.read_bytes() == canonical and not sidecar.exists():
            _atomic_write(sidecar, f"{sha256(canonical).hexdigest()}\n".encode("ascii"))
            return target
        _quarantine_unverified(target)
        _quarantine_unverified(sidecar)
    _atomic_write(sidecar, f"{sha256(canonical).hexdigest()}\n".encode("ascii"))
    _atomic_write(target, canonical)
    return target


def _quarantine_unverified(path: Path) -> None:
    if not path.exists():
        return
    quarantined = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
    try:
        replace(path, quarantined)
    except OSError as exc:
        raise ClipAnalysisArtifactError("corrupt_quarantine_failed") from exc
    LOGGER.warning("quarantined unverified clip analysis artifact %s as %s", path, quarantined)


def _verified_payload(target: Path, sidecar: Path) -> bytes | None:
    try:
        payload = target.read_bytes()
        expected_digest = sidecar.read_text(encoding="ascii").strip()
    except OSError:
        return None
    return payload if expected_digest == sha256(payload).hexdigest() else None


__all__ = [
    "ClipAnalysisArtifactError",
    "ClipAnalysisArtifactIdentity",
    "artifact_path",
    "publish_clip_analysis",
    "validate_scratch",
]
