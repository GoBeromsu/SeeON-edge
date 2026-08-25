from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    DerivativeArtifact,
    DerivativeKind,
)
from worker.pipeline.output.derivative_producer import DerivativeProducer
from worker.pipeline.output.evidence.durability import fsync_directory, fsync_file


@dataclass(frozen=True, slots=True)
class StoredDerivativeArtifact:
    incident_id: str
    derivative_kind: DerivativeKind
    derivative_id: str
    media_relpath: str
    sha256: str
    size_bytes: int
    mime_type: str
    width: int
    height: int
    start_time_ms: int
    end_time_ms: int


class DerivativeArtifactConflictError(RuntimeError):
    pass


class DerivativeArtifactCapacityError(RuntimeError):
    pass


class DerivativeArtifactStore:
    """Content-addressed derivative files; backend records their receipt."""

    def __init__(self, store_root: Path, *, max_disk_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
        if max_disk_bytes <= 0:
            raise ValueError("derivative disk bound must be positive")
        self.store_root = store_root
        self.max_disk_bytes = max_disk_bytes

    def publish(
        self, job: AnnotatedDerivativeJob, artifact: DerivativeArtifact
    ) -> StoredDerivativeArtifact:
        if artifact.mime_type != job.derivative_kind.mime_type:
            raise DerivativeArtifactConflictError("derivative kind and MIME type differ")
        producer = (
            DerivativeProducer.NATIVE_GPU
            if artifact.render_backend == DerivativeProducer.NATIVE_GPU.value
            else DerivativeProducer.CPU_REFERENCE
        )
        producer_facts = {
            DerivativeProducer.CPU_REFERENCE: ("cpu", "host"),
            DerivativeProducer.NATIVE_GPU: ("gpu", "encoded-source"),
        }
        if (artifact.render_device, artifact.input_memory_kind) != producer_facts[producer]:
            raise DerivativeArtifactConflictError("derivative producer facts are invalid")
        if _facts(job.primary_media_path) != (job.primary_sha256, job.source_size_bytes):
            raise DerivativeArtifactConflictError("primary source media changed or is unavailable")
        self.store_root.mkdir(parents=True, exist_ok=True)
        relative = (
            PurePosixPath("derivatives")
            / "objects"
            / f"{job.identity}{job.derivative_kind.extension}"
        )
        destination = self.store_root / Path(relative)
        self._require_contained(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fsync_directory(destination.parent.parent)
        if destination.exists():
            if _facts(destination) != (artifact.sha256, artifact.size_bytes):
                raise DerivativeArtifactConflictError("content-addressed derivative differs")
        else:
            if (
                _disk_usage(self.store_root / "derivatives") + artifact.size_bytes
                > self.max_disk_bytes
            ):
                raise DerivativeArtifactCapacityError("derivative store capacity exceeded")
            staging = destination.with_name(f".{destination.name}.pending")
            if staging.is_symlink():
                raise DerivativeArtifactConflictError("derivative staging path is not regular")
            if artifact.path.resolve() != staging.resolve():
                shutil.copyfile(artifact.path, staging)
            fsync_file(staging)
            if _facts(staging) != (artifact.sha256, artifact.size_bytes):
                staging.unlink(missing_ok=True)
                raise DerivativeArtifactConflictError("staged derivative facts changed")
            os.replace(staging, destination)
            fsync_file(destination)
            fsync_directory(destination.parent)
        return StoredDerivativeArtifact(
            job.incident_id,
            job.derivative_kind,
            job.identity,
            relative.as_posix(),
            artifact.sha256,
            artifact.size_bytes,
            artifact.mime_type,
            artifact.width,
            artifact.height,
            artifact.start_time_ms,
            artifact.end_time_ms,
        )

    def receipt_for(self, job: AnnotatedDerivativeJob) -> StoredDerivativeArtifact | None:
        """Return a prior command's immutable artifact without rendering again."""
        relative = (
            PurePosixPath("derivatives")
            / "objects"
            / f"{job.identity}{job.derivative_kind.extension}"
        )
        path = self.store_root / Path(relative)
        if not path.is_file() or path.is_symlink():
            return None
        digest, size = _facts(path)
        return StoredDerivativeArtifact(
            job.incident_id,
            job.derivative_kind,
            job.identity,
            relative.as_posix(),
            digest,
            size,
            job.derivative_kind.mime_type,
            0,
            0,
            0,
            0,
        )

    def _require_contained(self, path: Path) -> None:
        try:
            path.resolve(strict=False).relative_to(self.store_root.resolve(strict=True))
        except (FileNotFoundError, ValueError) as error:
            raise DerivativeArtifactConflictError("derivative path escapes store") from error


def _facts(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _disk_usage(root: Path) -> int:
    return (
        0
        if not root.exists()
        else sum(
            path.stat().st_size
            for path in root.glob("*/*")
            if path.is_file() and not path.is_symlink()
        )
    )


__all__ = [
    "DerivativeArtifactCapacityError",
    "DerivativeArtifactConflictError",
    "DerivativeArtifactStore",
    "StoredDerivativeArtifact",
]
