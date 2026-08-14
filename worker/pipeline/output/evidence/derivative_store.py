from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shared.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from worker.pipeline.output.annotated_derivative import AnnotatedDerivativeJob, DerivativeArtifact
from worker.pipeline.output.evidence.durability import fsync_directory, fsync_file
from worker.pipeline.output.evidence.evidence_record_stage import ensure_media_object


@dataclass(frozen=True, slots=True)
class StoredDerivative:
    derivative_id: str
    media_relpath: str
    sha256: str
    size_bytes: int
    mime_type: str


class DerivativeConflictError(RuntimeError):
    pass


class DerivativeCapacityError(RuntimeError):
    pass


class DerivativeStore:
    """Two-phase immutable derivative publication linked to central evidence."""

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
        self, job: AnnotatedDerivativeJob, artifact: DerivativeArtifact, *, updated_at: str
    ) -> StoredDerivative:
        if artifact.render_device != "cpu" or artifact.input_memory_kind != "host":
            raise ValueError("CPU derivative publication requires truthful host render facts")
        self._verify_primary(job)
        relative = PurePosixPath("derivatives") / job.incident_id / f"{artifact.sha256}.mp4"
        destination = self.store_root / Path(relative)
        try:
            destination.resolve(strict=False).relative_to(self.store_root.resolve(strict=True))
        except (FileNotFoundError, ValueError) as error:
            raise DerivativeConflictError("derivative destination escapes store") from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        fsync_directory(destination.parent.parent)
        if destination.exists():
            if _facts(destination) != (artifact.sha256, artifact.size_bytes):
                raise DerivativeConflictError("existing derivative differs from content identity")
        else:
            usage = _derivative_disk_usage(self.store_root / "derivatives")
            if usage + artifact.size_bytes > self.max_disk_bytes:
                raise DerivativeCapacityError("derivative store capacity exceeded")
            staging = destination.with_suffix(".mp4.pending")
            if staging.is_symlink():
                raise DerivativeConflictError("derivative staging path is not regular")
            if artifact.path.resolve() != staging.resolve():
                shutil.copyfile(artifact.path, staging)
            fsync_file(staging)
            if _facts(staging) != (artifact.sha256, artifact.size_bytes):
                staging.unlink(missing_ok=True)
                raise DerivativeConflictError("staged derivative facts changed")
            os.replace(staging, destination)
            fsync_file(destination)
            fsync_directory(destination.parent)
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            with write_transaction(connection):
                self._record_available(connection, job, artifact, relative.as_posix(), updated_at)
        finally:
            connection.close()
        return StoredDerivative(
            job.identity,
            relative.as_posix(),
            artifact.sha256,
            artifact.size_bytes,
            artifact.mime_type,
        )

    def _record_available(
        self,
        connection: sqlite3.Connection,
        job: AnnotatedDerivativeJob,
        artifact: DerivativeArtifact,
        relpath: str,
        updated_at: str,
    ) -> None:
        slot = connection.execute(
            "SELECT state, media_id, reason, revision FROM derivative_evidence_slots "
            "WHERE incident_id = ? AND derivative_kind = 'ANNOTATED_CLIP'",
            (job.incident_id,),
        ).fetchone()
        if slot is None:
            raise DerivativeConflictError("central derivative slot is absent")
        media_id = ensure_media_object(
            connection,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            mime_type=artifact.mime_type,
            relpath=relpath,
            created_at=updated_at,
        )
        created = connection.execute(
            "SELECT created_at FROM derivative_render_records WHERE incident_id=? "
            "AND derivative_kind='ANNOTATED_CLIP'",
            (job.incident_id,),
        ).fetchone()
        created_at = updated_at if created is None else str(created[0])
        values = (
            job.incident_id,
            "ANNOTATED_CLIP",
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
            created_at,
        )
        connection.execute(
            "INSERT OR IGNORE INTO derivative_render_records VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            values,
        )
        actual = connection.execute(
            "SELECT incident_id, derivative_kind, derivative_id, media_id, "
            "primary_clip_id, primary_media_sha256, decision_trace_id, "
            "runtime_manifest_sha256, scene_id, scene_schema_version, render_backend, "
            "render_device, input_memory_kind, render_version, width, height, "
            "start_time_ms, end_time_ms, created_at FROM derivative_render_records "
            "WHERE incident_id = ? AND derivative_kind = 'ANNOTATED_CLIP'",
            (job.incident_id,),
        ).fetchone()
        if actual != values:
            raise DerivativeConflictError("immutable derivative identity differs")
        if slot[0] == "PENDING":
            connection.execute(
                "UPDATE derivative_evidence_slots SET state='AVAILABLE', media_id=?, "
                "revision=revision+1, updated_at=? WHERE incident_id=? "
                "AND derivative_kind='ANNOTATED_CLIP' AND revision=?",
                (media_id, updated_at, job.incident_id, int(slot[3])),
            )
            self._complete_incident(connection, job.incident_id, updated_at)
        elif (slot[0], slot[1], slot[2]) != ("AVAILABLE", media_id, None):
            raise DerivativeConflictError("central derivative slot is already resolved")

    def mark_unavailable(self, incident_id: str, reason: str, *, updated_at: str) -> None:
        if not reason or len(reason) > 128:
            raise ValueError("derivative unavailable reason is invalid")
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            with write_transaction(connection):
                row = connection.execute(
                    "SELECT state, revision FROM derivative_evidence_slots WHERE incident_id=? "
                    "AND derivative_kind='ANNOTATED_CLIP'",
                    (incident_id,),
                ).fetchone()
                if row is None:
                    raise DerivativeConflictError("central derivative slot is absent")
                if row[0] == "PENDING":
                    connection.execute(
                        "UPDATE derivative_evidence_slots SET state='UNAVAILABLE', reason=?, "
                        "revision=revision+1, updated_at=? WHERE incident_id=? AND revision=?",
                        (reason, updated_at, incident_id, int(row[1])),
                    )
                    self._complete_incident(connection, incident_id, updated_at)
                elif row[0] != "UNAVAILABLE":
                    raise DerivativeConflictError("central derivative slot is already resolved")
        finally:
            connection.close()

    def reconcile(self, *, updated_at: str) -> tuple[int, int]:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        available = corrupt = 0
        try:
            rows = connection.execute(
                "SELECT record.incident_id, media.contained_relpath, media.content_sha256, "
                "media.size_bytes, slot.state, slot.revision, record.media_id "
                "FROM derivative_render_records "
                "AS record JOIN evidence_media_objects AS media USING(media_id) "
                "JOIN derivative_evidence_slots AS slot USING(incident_id, derivative_kind)"
            ).fetchall()
            for incident_id, relpath, digest, size, state, revision, media_id in rows:
                valid = self._contained_match(str(relpath), str(digest), int(size))
                if state == "PENDING" and valid:
                    with write_transaction(connection):
                        connection.execute(
                            "UPDATE derivative_evidence_slots SET state='AVAILABLE', media_id=?, "
                            "revision=revision+1, updated_at=? WHERE incident_id=? AND revision=?",
                            (str(media_id), updated_at, str(incident_id), int(revision)),
                        )
                        self._complete_incident(connection, str(incident_id), updated_at)
                    available += 1
                elif state == "PENDING" and not valid:
                    with write_transaction(connection):
                        connection.execute(
                            "UPDATE derivative_evidence_slots SET state='CORRUPT', media_id=?, "
                            "reason='MISSING_OR_MUTATED', revision=revision+1, updated_at=? "
                            "WHERE incident_id=? AND revision=?",
                            (str(media_id), updated_at, str(incident_id), int(revision)),
                        )
                        self._complete_incident(connection, str(incident_id), updated_at)
                    corrupt += 1
                elif state == "AVAILABLE" and not valid:
                    with write_transaction(connection):
                        connection.execute(
                            "UPDATE derivative_evidence_slots SET state='CORRUPT', "
                            "reason='MISSING_OR_MUTATED', revision=revision+1, updated_at=? "
                            "WHERE incident_id=? AND revision=?",
                            (updated_at, str(incident_id), int(revision)),
                        )
                    corrupt += 1
        finally:
            connection.close()
        self._quarantine_orphans()
        return available, corrupt

    def _verify_primary(self, job: AnnotatedDerivativeJob) -> None:
        if _facts(job.primary_media_path) != (job.primary_sha256, job.source_size_bytes):
            raise DerivativeConflictError("primary source media changed or is unavailable")

    def _contained_match(self, relpath: str, digest: str, size: int) -> bool:
        path = PurePosixPath(relpath)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            return False
        try:
            return _facts(self.store_root / Path(path)) == (digest, size)
        except OSError:
            return False

    def _quarantine_orphans(self) -> None:
        root = self.store_root / "derivatives"
        if not root.exists():
            return
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            known = {
                str(row[0])
                for row in connection.execute(
                    "SELECT contained_relpath FROM evidence_media_objects WHERE contained_relpath "
                    "LIKE 'derivatives/%'"
                ).fetchall()
            }
        finally:
            connection.close()
        quarantine = self.store_root / ".derivative-quarantine"
        candidates = tuple(root.glob("*/*.mp4")) + tuple(root.glob("*/*.mp4.pending"))
        for path in sorted(candidates):
            relative = path.relative_to(self.store_root).as_posix()
            if relative in known:
                continue
            quarantine.mkdir(parents=True, exist_ok=True)
            identity = hashlib.sha256(relative.encode("utf-8")).hexdigest()
            suffix = ".pending" if path.name.endswith(".pending") else ".mp4"
            quarantined = quarantine / f"{identity}{suffix}"
            if quarantined.exists():
                if _facts(quarantined) == _facts(path):
                    path.unlink()
                    continue
                raise DerivativeConflictError("derivative quarantine identity collision")
            os.replace(path, quarantined)
            fsync_file(quarantined)
            fsync_directory(quarantine)

    @staticmethod
    def _complete_incident(
        connection: sqlite3.Connection, incident_id: str, updated_at: str
    ) -> None:
        connection.execute(
            "UPDATE evidence_incidents SET lifecycle_state='COMPLETE', revision=revision+1, "
            "updated_at=? WHERE incident_id=? AND lifecycle_state='DERIVATIVE_PENDING'",
            (updated_at, incident_id),
        )


def _derivative_disk_usage(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.glob("*/*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def _facts(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


__all__ = [
    "DerivativeCapacityError",
    "DerivativeConflictError",
    "DerivativeStore",
    "StoredDerivative",
]
