from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from shared.edge_db.connection import (
    RuntimeActor,
    open_runtime_database,
    write_transaction,
)
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
    max_boots: int = 512
    max_boots_per_camera: int = 128

    def __post_init__(self) -> None:
        if self.max_boots < 1 or self.max_boots_per_camera < 1:
            raise ValueError("provenance retention bounds must be positive")


DEFAULT_PROVENANCE_RETENTION_POLICY = ProvenanceRetentionPolicy()


class AppliedRuntimeManifestStore:
    """DDL-free worker owner for immutable manifest contents and boot references."""

    def __init__(
        self,
        database_path: Path,
        retention: ProvenanceRetentionPolicy = DEFAULT_PROVENANCE_RETENTION_POLICY,
    ) -> None:
        self.database_path = database_path
        self.retention = retention

    def persist(
        self,
        manifest: AppliedRuntimeManifest,
        *,
        boot_instance_id: str,
        applied_at: str,
    ) -> AppliedRuntimeRecord:
        if not boot_instance_id or not applied_at:
            raise AppliedRuntimeManifestError("boot instance and applied time must be resolved")
        camera_ids = _manifest_camera_ids(manifest)
        connection = open_runtime_database(
            self.database_path,
            actor=RuntimeActor.WORKER,
        )
        try:
            with write_transaction(connection):
                existing_content = connection.execute(
                    "SELECT canonical_json FROM runtime_manifest_contents "
                    "WHERE manifest_sha256 = ?",
                    (manifest.sha256,),
                ).fetchone()
                if existing_content is not None and existing_content != (manifest.canonical_json,):
                    raise AppliedRuntimeManifestError(
                        "runtime manifest hash resolves to contradictory canonical content"
                    )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO runtime_manifest_contents (
                        manifest_sha256, manifest_schema_version, canonical_json, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        manifest.sha256,
                        manifest.schema_version,
                        manifest.canonical_json,
                        applied_at,
                    ),
                )
                existing_boot = connection.execute(
                    "SELECT manifest_sha256 FROM runtime_manifest_boots WHERE boot_instance_id = ?",
                    (boot_instance_id,),
                ).fetchone()
                if existing_boot is not None and existing_boot != (manifest.sha256,):
                    raise AppliedRuntimeManifestError(
                        "boot instance identity is already bound to another runtime manifest"
                    )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO runtime_manifest_boots (
                        boot_instance_id, manifest_sha256, applied_at
                    ) VALUES (?, ?, ?)
                    """,
                    (boot_instance_id, manifest.sha256, applied_at),
                )
                for camera_id in camera_ids:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO runtime_manifest_cameras (
                            boot_instance_id, camera_id, manifest_sha256, applied_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (boot_instance_id, camera_id, manifest.sha256, applied_at),
                    )
                self._prune_history(connection)
        finally:
            connection.close()
        return AppliedRuntimeRecord(manifest.sha256, boot_instance_id, applied_at)

    def _prune_history(self, connection: sqlite3.Connection) -> None:
        execute = connection.execute
        pruned_cameras = 0
        blocked_cameras = 0
        camera_ids = tuple(
            str(row[0])
            for row in execute(
                "SELECT DISTINCT camera_id FROM runtime_manifest_cameras ORDER BY camera_id"
            ).fetchall()
        )
        for camera_id in camera_ids:
            rows = execute(
                """
                SELECT boot_instance_id FROM runtime_manifest_cameras
                WHERE camera_id = ?
                ORDER BY applied_at DESC, boot_instance_id DESC
                """,
                (camera_id,),
            ).fetchall()
            for row in rows[self.retention.max_boots_per_camera :]:
                boot_id = str(row[0])
                referenced = execute(
                    "SELECT 1 FROM runtime_analysis_traces "
                    "WHERE worker_boot_id = ? AND camera_id = ? LIMIT 1",
                    (boot_id, camera_id),
                ).fetchone()
                if referenced is not None:
                    blocked_cameras += 1
                    continue
                pruned_cameras += execute(
                    "DELETE FROM runtime_manifest_cameras "
                    "WHERE boot_instance_id = ? AND camera_id = ?",
                    (boot_id, camera_id),
                ).rowcount

        pruned_boots = 0
        blocked_boots = 0
        boots = execute(
            "SELECT boot_instance_id FROM runtime_manifest_boots "
            "ORDER BY applied_at DESC, boot_instance_id DESC"
        ).fetchall()
        for row in boots[self.retention.max_boots :]:
            boot_id = str(row[0])
            referenced = execute(
                "SELECT 1 FROM runtime_analysis_traces WHERE worker_boot_id = ? LIMIT 1",
                (boot_id,),
            ).fetchone()
            if referenced is not None:
                blocked_boots += 1
                continue
            pruned_cameras += execute(
                "DELETE FROM runtime_manifest_cameras WHERE boot_instance_id = ?",
                (boot_id,),
            ).rowcount
            pruned_boots += execute(
                "DELETE FROM runtime_manifest_boots WHERE boot_instance_id = ?",
                (boot_id,),
            ).rowcount

        # Camera pruning can leave boot rows with no projected camera. They are
        # bounded history, not immutable content, and have no evidence reference.
        orphan_boots = execute(
            """
            SELECT boot.boot_instance_id FROM runtime_manifest_boots AS boot
            WHERE NOT EXISTS (
                SELECT 1 FROM runtime_manifest_cameras AS camera
                WHERE camera.boot_instance_id = boot.boot_instance_id
            ) AND NOT EXISTS (
                SELECT 1 FROM runtime_analysis_traces AS analysis
                WHERE analysis.worker_boot_id = boot.boot_instance_id
            )
            """
        ).fetchall()
        for row in orphan_boots:
            pruned_boots += execute(
                "DELETE FROM runtime_manifest_boots WHERE boot_instance_id = ?",
                (str(row[0]),),
            ).rowcount
        execute(
            """
            UPDATE runtime_provenance_retention SET
                pruned_boots = pruned_boots + ?,
                pruned_camera_bindings = pruned_camera_bindings + ?,
                retention_blocked_boots = retention_blocked_boots + ?,
                retention_blocked_camera_bindings =
                    retention_blocked_camera_bindings + ?
            WHERE id = 1
            """,
            (pruned_boots, pruned_cameras, blocked_boots, blocked_cameras),
        )

    def get(self, manifest_sha256: str) -> AppliedRuntimeManifest | None:
        connection = open_runtime_database(
            self.database_path,
            actor=RuntimeActor.WORKER,
        )
        try:
            row = connection.execute(
                "SELECT canonical_json FROM runtime_manifest_contents WHERE manifest_sha256 = ?",
                (manifest_sha256,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return AppliedRuntimeManifest.parse(str(row[0]), manifest_sha256)

    def latest_for_camera(self, camera_id: str) -> AppliedRuntimeManifest | None:
        connection = open_runtime_database(
            self.database_path,
            actor=RuntimeActor.WORKER,
        )
        try:
            row = connection.execute(
                """
                SELECT content.manifest_sha256, content.canonical_json
                FROM runtime_manifest_cameras AS camera
                JOIN runtime_manifest_contents AS content USING (manifest_sha256)
                WHERE camera.camera_id = ?
                ORDER BY camera.applied_at DESC, camera.boot_instance_id DESC
                LIMIT 1
                """,
                (camera_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return AppliedRuntimeManifest.parse(str(row[1]), str(row[0]))


def _manifest_camera_ids(manifest: AppliedRuntimeManifest) -> tuple[str, ...]:
    import json

    content = json.loads(manifest.canonical_json)
    cameras = content.get("cameras") if isinstance(content, dict) else None
    if not isinstance(cameras, list):
        raise AppliedRuntimeManifestError("runtime manifest camera identities are invalid")
    camera_ids: list[str] = []
    for camera in cameras:
        camera_id = camera.get("camera_id") if isinstance(camera, dict) else None
        if not isinstance(camera_id, str) or not camera_id:
            raise AppliedRuntimeManifestError("runtime manifest camera identity is unresolved")
        camera_ids.append(camera_id)
    if len(camera_ids) != len(set(camera_ids)):
        raise AppliedRuntimeManifestError("runtime manifest camera identities are duplicated")
    return tuple(camera_ids)


__all__ = [
    "AppliedRuntimeManifestStore",
    "AppliedRuntimeRecord",
    "ProvenanceRetentionPolicy",
]
