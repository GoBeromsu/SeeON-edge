from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from worker.runtime.provenance.models import (
    AppliedRuntimeManifest,
    AppliedRuntimeManifestError,
)


@dataclass(frozen=True, slots=True)
class AppliedRuntimeRecord:
    manifest_sha256: str
    boot_instance_id: str
    applied_at: str


@dataclass(frozen=True, slots=True)
class ProvenanceRetentionPolicy:
    """Bounded local retention for immutable applied-manifest records."""

    max_boots: int = 512
    max_boots_per_camera: int = 128

    def __post_init__(self) -> None:
        if self.max_boots < 1 or self.max_boots_per_camera < 1:
            raise ValueError("provenance retention bounds must be positive")


DEFAULT_PROVENANCE_RETENTION_POLICY = ProvenanceRetentionPolicy()


class AppliedRuntimeManifestStore:
    """Persist local provenance without sending it through the alert queue."""

    def __init__(
        self,
        database_path: Path,
        retention: ProvenanceRetentionPolicy = DEFAULT_PROVENANCE_RETENTION_POLICY,
    ) -> None:
        self.directory = database_path.parent / "runtime-provenance"
        self._latest_path = database_path.parent / "applied-runtime-manifest.json"
        self._retention = retention

    def persist(
        self,
        manifest: AppliedRuntimeManifest,
        *,
        boot_instance_id: str,
        applied_at: str,
    ) -> AppliedRuntimeRecord:
        if not boot_instance_id or not applied_at:
            raise AppliedRuntimeManifestError("boot instance and applied time must be resolved")
        record = {
            "applied_at": applied_at,
            "boot_instance_id": boot_instance_id,
            "canonical_json": manifest.canonical_json,
            "manifest_sha256": manifest.sha256,
        }
        payload = _canonical_bytes(record)
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        _fsync_directory(self.directory.parent)
        record_path = (
            self.directory / f"{hashlib.sha256(boot_instance_id.encode()).hexdigest()}.json"
        )
        if record_path.exists():
            try:
                existing = record_path.read_bytes()
            except OSError as exc:
                raise AppliedRuntimeManifestError("runtime manifest record is unavailable") from exc
            if existing != payload:
                raise AppliedRuntimeManifestError(
                    "runtime manifest boot identity conflicts with prior record"
                )
        else:
            _atomic_write(record_path, payload)
        self._write_latest_readback(payload)
        self._prune_records()
        return AppliedRuntimeRecord(manifest.sha256, boot_instance_id, applied_at)

    def _write_latest_readback(self, payload: bytes) -> None:
        _atomic_write(self._latest_path, payload)

    def _prune_records(self) -> None:
        records = sorted(
            self.directory.glob("*.json"),
            key=_record_order,
            reverse=True,
        )
        for path in records[self._retention.max_boots :]:
            path.unlink()
        _fsync_directory(self.directory)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _record_order(path: Path) -> tuple[str, str]:
    try:
        content = json.loads(path.read_bytes())
        applied_at = content.get("applied_at")
    except (OSError, ValueError, TypeError):
        applied_at = ""
    return (applied_at if isinstance(applied_at, str) else "", path.name)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "AppliedRuntimeManifestStore",
    "AppliedRuntimeRecord",
    "ProvenanceRetentionPolicy",
]
