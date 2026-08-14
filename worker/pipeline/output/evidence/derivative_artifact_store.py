from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shared.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    DerivativeArtifact,
    DerivativeCancelled,
    DerivativeKind,
)
from worker.pipeline.output.evidence.durability import fsync_directory, fsync_file
from worker.pipeline.output.evidence.evidence_record_stage import ensure_media_object


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
    """Crash-recoverable content-addressed STILL/VIDEO publication."""

    def __init__(
        self,
        database_path: Path,
        store_root: Path,
        *,
        max_disk_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        if max_disk_bytes <= 0:
            raise ValueError("derivative disk bound must be positive")
        self.database_path = database_path
        self.store_root = store_root
        self.max_disk_bytes = max_disk_bytes

    def publish(
        self,
        job: AnnotatedDerivativeJob,
        artifact: DerivativeArtifact,
        *,
        updated_at: str,
    ) -> StoredDerivativeArtifact:
        if artifact.mime_type != job.derivative_kind.mime_type:
            raise DerivativeArtifactConflictError("derivative kind and MIME type differ")
        if artifact.render_device != "cpu" or artifact.input_memory_kind != "host":
            raise DerivativeArtifactConflictError("derivative render memory facts are invalid")
        if _facts(job.primary_media_path) != (job.primary_sha256, job.source_size_bytes):
            raise DerivativeArtifactConflictError("primary source media changed or is unavailable")
        self.store_root.mkdir(parents=True, exist_ok=True)
        relative = (
            PurePosixPath("derivatives")
            / "objects"
            / f"{artifact.sha256}{job.derivative_kind.extension}"
        )
        destination = self.store_root / Path(relative)
        self._require_contained(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fsync_directory(destination.parent.parent)
        created_new = False
        if destination.exists():
            if _facts(destination) != (artifact.sha256, artifact.size_bytes):
                raise DerivativeArtifactConflictError("content-addressed derivative differs")
        else:
            usage = _disk_usage(self.store_root / "derivatives")
            if usage + artifact.size_bytes > self.max_disk_bytes:
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
            created_new = True
        try:
            _ = self._record(job, artifact, relative.as_posix(), updated_at)
        except DerivativeCancelled:
            if created_new:
                self._quarantine_unreferenced(destination)
            raise
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

    def _record(
        self,
        job: AnnotatedDerivativeJob,
        artifact: DerivativeArtifact,
        relative: str,
        updated_at: str,
    ) -> str:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            with write_transaction(connection):
                slot = connection.execute(
                    "SELECT request_id,state,media_id,revision,cancel_requested "
                    "FROM derivative_jobs WHERE incident_id=? AND derivative_kind=?",
                    (job.incident_id, job.derivative_kind.value),
                ).fetchone()
                if slot is None or str(slot[0]) != job.identity:
                    raise DerivativeArtifactConflictError("derivative request slot differs")
                state = str(slot[1])
                media_id_existing = None if slot[2] is None else str(slot[2])
                revision = int(slot[3])
                cancel_requested = bool(slot[4])
                if state == "CANCELLED" or (state in {"PENDING", "RUNNING"} and cancel_requested):
                    raise DerivativeCancelled("derivative job was cancelled", job)
                if state not in {"PENDING", "RUNNING", "AVAILABLE"}:
                    raise DerivativeArtifactConflictError("derivative request is terminal")
                media_id = ensure_media_object(
                    connection,
                    sha256=artifact.sha256,
                    size_bytes=artifact.size_bytes,
                    mime_type=artifact.mime_type,
                    relpath=relative,
                    created_at=updated_at,
                )
                if state == "AVAILABLE" and media_id_existing != media_id:
                    raise DerivativeArtifactConflictError("derivative request is terminal")
                values = (
                    job.incident_id,
                    job.derivative_kind.value,
                    job.identity,
                    media_id,
                    job.primary_clip_id,
                    job.primary_sha256,
                    job.decision_trace_id,
                    job.runtime_manifest_sha256,
                    artifact.scene_id,
                    1,
                    artifact.render_backend,
                    artifact.render_device,
                    artifact.input_memory_kind,
                    artifact.render_version,
                    artifact.width,
                    artifact.height,
                    artifact.start_time_ms,
                    artifact.end_time_ms,
                    updated_at,
                )
                if state in {"PENDING", "RUNNING"}:
                    changed = connection.execute(
                        "UPDATE derivative_jobs SET state='AVAILABLE',media_id=?,"
                        "cancel_requested=0,revision=revision+1,updated_at=? "
                        "WHERE incident_id=? AND derivative_kind=? AND revision=? "
                        "AND state IN ('PENDING','RUNNING') AND cancel_requested=0",
                        (
                            media_id,
                            updated_at,
                            job.incident_id,
                            job.derivative_kind.value,
                            revision,
                        ),
                    ).rowcount
                    if changed != 1:
                        current = connection.execute(
                            "SELECT state,cancel_requested FROM derivative_jobs "
                            "WHERE incident_id=? AND derivative_kind=?",
                            (job.incident_id, job.derivative_kind.value),
                        ).fetchone()
                        if current is not None and (
                            str(current[0]) == "CANCELLED" or bool(current[1])
                        ):
                            raise DerivativeCancelled("derivative job was cancelled", job)
                        raise DerivativeArtifactConflictError("derivative request revision changed")
                connection.execute(
                    "INSERT OR IGNORE INTO derivative_artifacts VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    values,
                )
                actual = connection.execute(
                    "SELECT incident_id,derivative_kind,derivative_id,media_id,primary_clip_id,"
                    "primary_media_sha256,decision_trace_id,runtime_manifest_sha256,scene_id,"
                    "scene_schema_version,render_backend,render_device,input_memory_kind,"
                    "render_version,width,height,start_time_ms,end_time_ms,created_at "
                    "FROM derivative_artifacts WHERE incident_id=? AND derivative_kind=?",
                    (job.incident_id, job.derivative_kind.value),
                ).fetchone()
                if actual != values:
                    raise DerivativeArtifactConflictError("immutable derivative artifact differs")
                return media_id
        finally:
            connection.close()

    def reconcile(self, *, updated_at: str) -> tuple[int, int, int]:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        recovered = corrupt = 0
        try:
            rows = connection.execute(
                "SELECT job.incident_id,job.derivative_kind,job.state,job.revision,"
                "job.cancel_requested,artifact.media_id,media.contained_relpath,"
                "media.content_sha256,media.size_bytes "
                "FROM derivative_artifacts AS artifact "
                "JOIN derivative_jobs AS job USING(incident_id,derivative_kind) "
                "JOIN evidence_media_objects AS media USING(media_id)"
            ).fetchall()
            for (
                incident_id,
                kind,
                state,
                revision,
                cancel_requested,
                media_id,
                relpath,
                digest,
                size,
            ) in rows:
                valid = self._contained_match(str(relpath), str(digest), int(size))
                if str(state) in {"PENDING", "RUNNING"} and bool(cancel_requested):
                    # Cancellation already accepted; never promote a cancelled slot.
                    # Artifact rows are immutable — leave any residue for retention.
                    target, reason, next_media = "CANCELLED", "CANCELLED", None
                    recovered += 1
                elif str(state) in {"PENDING", "RUNNING"} and valid:
                    target, reason, next_media = "AVAILABLE", None, str(media_id)
                    recovered += 1
                elif str(state) in {"PENDING", "RUNNING", "AVAILABLE"} and not valid:
                    target, reason, next_media = "CORRUPT", "MISSING_OR_MUTATED", str(media_id)
                    corrupt += 1
                else:
                    continue
                with write_transaction(connection):
                    connection.execute(
                        "UPDATE derivative_jobs SET state=?,media_id=?,reason=?,"
                        "revision=revision+1,updated_at=? WHERE incident_id=? "
                        "AND derivative_kind=? AND revision=?",
                        (
                            target,
                            next_media,
                            reason,
                            updated_at,
                            str(incident_id),
                            str(kind),
                            int(revision),
                        ),
                    )
        finally:
            connection.close()
        orphaned = self._quarantine_orphans()
        return recovered, corrupt, orphaned

    def _contained_match(self, relative: str, digest: str, size: int) -> bool:
        path = PurePosixPath(relative)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            return False
        try:
            candidate = self.store_root / Path(path)
            self._require_contained(candidate)
            return _facts(candidate) == (digest, size)
        except (OSError, DerivativeArtifactConflictError):
            return False

    def _quarantine_orphans(self) -> int:
        root = self.store_root / "derivatives"
        if not root.exists():
            return 0
        known = self._known_derivative_relpaths()
        quarantine = self.store_root / ".derivative-quarantine"
        moved = 0
        for path in sorted(root.glob("*/*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(self.store_root).as_posix()
            if relative in known:
                continue
            if self._move_to_quarantine(path):
                moved += 1
        if moved:
            fsync_directory(quarantine)
        return moved

    def _quarantine_unreferenced(self, path: Path) -> None:
        """Drop a just-staged CAS object only when no durable media row claims it."""
        if not path.exists() or path.is_symlink():
            return
        try:
            relative = path.relative_to(self.store_root).as_posix()
        except ValueError:
            return
        if relative in self._known_derivative_relpaths():
            return
        if self._move_to_quarantine(path):
            fsync_directory(self.store_root / ".derivative-quarantine")

    def _known_derivative_relpaths(self) -> set[str]:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            return {
                str(row[0])
                for row in connection.execute(
                    "SELECT contained_relpath FROM evidence_media_objects "
                    "WHERE contained_relpath LIKE 'derivatives/%'"
                ).fetchall()
            }
        finally:
            connection.close()

    def _move_to_quarantine(self, path: Path) -> bool:
        quarantine = self.store_root / ".derivative-quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        relative = path.relative_to(self.store_root).as_posix()
        identity = hashlib.sha256(relative.encode()).hexdigest()
        suffix = "".join(path.suffixes)[-16:]
        destination = quarantine / f"{identity}{suffix}"
        if destination.exists() and _facts(destination) != _facts(path):
            raise DerivativeArtifactConflictError("derivative quarantine collision")
        if destination.exists():
            path.unlink()
        else:
            os.replace(path, destination)
            fsync_file(destination)
        return True

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
