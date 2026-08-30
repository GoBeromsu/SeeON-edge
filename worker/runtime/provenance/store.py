from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from shared.events.delivery_queue import DeliveryQueue, EventEntry
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
    """Backend owns canonical manifest retention; retained for constructor shape."""

    max_boots: int = 512
    max_boots_per_camera: int = 128

    def __post_init__(self) -> None:
        if self.max_boots < 1 or self.max_boots_per_camera < 1:
            raise ValueError("provenance retention bounds must be positive")


DEFAULT_PROVENANCE_RETENTION_POLICY = ProvenanceRetentionPolicy()


class AppliedRuntimeManifestStore:
    """Publish applied manifests for backend-owned canonical retention."""

    def __init__(
        self,
        database_path: Path,
        retention: ProvenanceRetentionPolicy = DEFAULT_PROVENANCE_RETENTION_POLICY,
    ) -> None:
        self.directory = database_path.parent / "delivery-queue"

    def persist(
        self,
        manifest: AppliedRuntimeManifest,
        *,
        boot_instance_id: str,
        applied_at: str,
    ) -> AppliedRuntimeRecord:
        if not boot_instance_id or not applied_at:
            raise AppliedRuntimeManifestError("boot instance and applied time must be resolved")
        entry = _manifest_entry(manifest, boot_instance_id, applied_at)
        result = DeliveryQueue(self.directory).try_admit(entry)
        if not result.accepted:
            raise AppliedRuntimeManifestError(
                f"runtime manifest queue admission failed: {result.fault}"
            )
        self._write_latest_readback(
            manifest, boot_instance_id=boot_instance_id, applied_at=applied_at
        )
        return AppliedRuntimeRecord(manifest.sha256, boot_instance_id, applied_at)

    def _write_latest_readback(
        self, manifest: AppliedRuntimeManifest, *, boot_instance_id: str, applied_at: str
    ) -> None:
        self.directory.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        path = self.directory.parent / "applied-runtime-manifest.json"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = json.dumps(
            {
                "applied_at": applied_at,
                "boot_instance_id": boot_instance_id,
                "canonical_json": manifest.canonical_json,
                "manifest_sha256": manifest.sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise


def _manifest_entry(
    manifest: AppliedRuntimeManifest, boot_instance_id: str, applied_at: str
) -> EventEntry:
    payload = json.dumps(
        {
            "applied_at": applied_at,
            "boot_instance_id": boot_instance_id,
            "canonical_json": manifest.canonical_json,
            "manifest_schema_version": manifest.schema_version,
            "manifest_sha256": manifest.sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    identity = hashlib.sha256(boot_instance_id.encode()).hexdigest()
    return EventEntry(
        edge_event_id=f"manifest-{identity}",
        event_type="runtime.manifest",
        detected_at=_safe_text(applied_at),
        camera_id="runtime",
        facility_id="local",
        decision_trace=b"",
        values=payload,
    )


def _safe_text(value: str) -> str:
    if value.isascii() and value.isprintable():
        return value
    return hashlib.sha256(value.encode()).hexdigest()


__all__ = [
    "AppliedRuntimeManifestStore",
    "AppliedRuntimeRecord",
    "ProvenanceRetentionPolicy",
]
