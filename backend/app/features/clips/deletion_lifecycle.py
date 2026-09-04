"""Backend-owned durable lifecycle for worker-owned clip-byte deletion."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException

from backend.app.edge_db import RuntimeActor, open_runtime_database
from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.edge_db.connection import write_transaction
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    empty_detail,
)
from backend.app.features.audit.http import AuditUnavailableError
from backend.app.features.audit.store import AuditEvent, AuditStore, utc_now
from backend.app.features.clips.deletion_control import preflight_clip_deletion

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    completed: tuple[str, ...]
    retained_pending: tuple[str, ...]


class ClipDeletionLifecycle:
    def __init__(self, database_path: Path, app: FastAPI) -> None:
        self.database_path = database_path
        self.app = app

    def state(self, clip_id: str) -> str | None:
        with closing(
            open_runtime_database(self.database_path, actor=RuntimeActor.API)
        ) as connection:
            row = connection.execute(
                "SELECT retention_state FROM clips WHERE clip_id = ?", (clip_id,)
            ).fetchone()
        return None if row is None else str(row[0])

    def begin(self, clip_id: str, actor_id: str) -> str | None:
        """Commit PENDING and its request audit together, exactly once."""
        return self._transition(
            clip_id,
            actor_id=actor_id,
            action=AuditAction.CLIP_DELETE_REQUEST,
            expected="RETAINED",
            update="""
                UPDATE clips
                SET retention_state = 'PENDING',
                    retention_reason = 'OPERATOR_REQUESTED',
                    retention_requested_at = ?, retention_updated_at = ?,
                    revision = revision + 1, updated_at = ?
                WHERE clip_id = ? AND retention_state = 'RETAINED'
            """,
            completion=False,
        )

    def complete(
        self,
        clip_id: str,
        *,
        actor_id: str,
        system: bool = False,
    ) -> bool:
        """Commit PURGED projection and completion audit together, exactly once."""
        result = self._transition(
            clip_id,
            actor_id=actor_id,
            action=AuditAction.CLIP_DELETE_COMPLETE,
            expected="PENDING",
            update="""
                UPDATE clips
                SET local_state = 'UNAVAILABLE', local_reason = 'PURGED',
                    manifest_relpath = NULL, media_relpath = NULL,
                    thumbnail_relpath = NULL, manifest_sha256 = NULL,
                    media_sha256 = NULL, thumbnail_sha256 = NULL,
                    manifest_size_bytes = NULL, media_size_bytes = NULL,
                    thumbnail_size_bytes = NULL,
                    retention_state = 'PURGED', retention_reason = 'OPERATOR_DELETED',
                    retention_updated_at = ?, revision = revision + 1, updated_at = ?
                WHERE clip_id = ? AND retention_state = 'PENDING'
            """,
            completion=True,
            system=system,
        )
        return result == "PURGED"

    def pending_clip_ids(self) -> tuple[str, ...]:
        with closing(
            open_runtime_database(self.database_path, actor=RuntimeActor.API)
        ) as connection:
            rows = connection.execute(
                "SELECT clip_id FROM clips WHERE retention_state = 'PENDING' ORDER BY clip_id"
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def _transition(
        self,
        clip_id: str,
        *,
        actor_id: str,
        action: AuditAction,
        expected: str,
        update: str,
        completion: bool,
        system: bool = False,
    ) -> str | None:
        timestamp = utc_now()
        try:
            with (
                closing(
                    open_runtime_database(self.database_path, actor=RuntimeActor.API)
                ) as connection,
                write_transaction(connection),
            ):
                row = connection.execute(
                    "SELECT retention_state FROM clips WHERE clip_id = ?", (clip_id,)
                ).fetchone()
                if row is None:
                    return None
                current = str(row[0])
                if current != expected:
                    return current
                if completion:
                    changed = connection.execute(update, (timestamp, timestamp, clip_id)).rowcount
                    connection.execute(
                        """
                            UPDATE artifacts
                            SET state = 'PURGED', reason = 'CLIP_PURGED',
                                contained_relpath = NULL, codec = NULL,
                                revision = revision + 1, updated_at = ?
                            WHERE clip_id = ? AND kind = 'PRIMARY_CLIP'
                              AND state IN ('AVAILABLE','UNAVAILABLE','CORRUPT')
                              AND artifact_id IS NOT NULL
                            """,
                        (timestamp, clip_id),
                    )
                    transitioned = "PURGED"
                else:
                    changed = connection.execute(
                        update, (timestamp, timestamp, timestamp, clip_id)
                    ).rowcount
                    transitioned = "PENDING"
                if changed != 1:
                    raise sqlite3.IntegrityError("clip retention transition lost")
                self._audit_store().append(
                    AuditEvent(
                        occurred_at=timestamp,
                        actor_id=actor_id,
                        action=action,
                        target_id=clip_id,
                        detail=empty_detail(action),
                        actor_type=(AuditActorType.SYSTEM if system else AuditActorType.USER),
                        auth_mechanism=(
                            AuditAuthMechanism.INTERNAL
                            if system
                            else AuditAuthMechanism.DASHBOARD_SESSION
                        ),
                    ),
                    connection=connection,
                )
                return transitioned
        except (OSError, sqlite3.Error, EdgeDatabaseError) as error:
            raise AuditUnavailableError from error

    def _audit_store(self) -> AuditStore:
        candidate = getattr(self.app.state, "audit_store", None)
        if isinstance(candidate, AuditStore):
            return candidate
        store = AuditStore(self.database_path)
        self.app.state.audit_store = store
        return store


def reconcile_pending_clip_deletions(app: FastAPI, database_path: Path) -> ReconciliationReport:
    """Complete only worker-verified missing PENDING clips during backend startup."""
    lifecycle = ClipDeletionLifecycle(database_path, app)
    completed: list[str] = []
    retained: list[str] = []
    for clip_id in lifecycle.pending_clip_ids():
        try:
            payload = preflight_clip_deletion(app, clip_id)
        except HTTPException:  # worker/transport refusal keeps durable PENDING truth
            _LOGGER.warning("pending clip deletion remains pending: %s", clip_id)
            retained.append(clip_id)
            continue
        if payload == {"clip_id": clip_id, "status": "MISSING"}:
            if lifecycle.complete(clip_id, actor_id="startup-reconciler", system=True):
                completed.append(clip_id)
        else:
            retained.append(clip_id)
    return ReconciliationReport(tuple(completed), tuple(retained))


__all__ = [
    "ClipDeletionLifecycle",
    "ReconciliationReport",
    "reconcile_pending_clip_deletions",
]
