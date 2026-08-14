"""Exact schema validation and relation planning for clip repair."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from worker.pipeline.output.evidence.clip_consistency_schema import (
    canonical_evidence_relation_fingerprint,
    canonical_worker_schema_fingerprint,
    evidence_relation_fingerprint,
    is_supported_schema_version,
    is_worker_state_schema,
    schema_fingerprint,
)
from worker.pipeline.output.evidence.clip_consistency_types import ClipConsistencyError


@dataclass(frozen=True, slots=True)
class RelationPlan:
    relations_before: int
    relations_after: int
    mismatch_clips: int
    mismatch_tuples: int
    delete_event_ids: tuple[str, ...]
    insert_rows: tuple[tuple[str, str, int], ...]
    quarantine_clip_ids: tuple[str, ...]
    before_sha256: str
    after_sha256: str

    def authority_payload(self) -> dict[str, object]:
        return {
            "delete_event_ids": list(self.delete_event_ids),
            "insert_rows": [list(row) for row in self.insert_rows],
            "quarantine_clip_ids": list(self.quarantine_clip_ids),
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }

    @property
    def plan_sha256(self) -> str:
        encoded = json.dumps(
            self.authority_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def validate_database(
    connection: sqlite3.Connection,
    *,
    now: float,
    check_leases: bool = True,
) -> int:
    """Validate schema authority and return the accepted user_version."""
    version_row = connection.execute("PRAGMA user_version").fetchone()
    if version_row is None or not is_supported_schema_version(int(version_row[0])):
        raise ClipConsistencyError("schema_drift", "database schema version is unsupported")
    version = int(version_row[0])
    if is_worker_state_schema(version):
        if schema_fingerprint(connection) != canonical_worker_schema_fingerprint():
            raise ClipConsistencyError(
                "schema_drift", "worker-state schema fingerprint differs"
            )
    elif evidence_relation_fingerprint(connection) != canonical_evidence_relation_fingerprint():
        raise ClipConsistencyError(
            "schema_drift", "central evidence relation fingerprint differs"
        )
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ClipConsistencyError("integrity_drift", "database integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ClipConsistencyError("foreign_key_drift", "database foreign keys are invalid")
    if check_leases:
        _reject_active_leases(connection, now)
    return version


def plan_relations(
    connection: sqlite3.Connection,
    desired: dict[str, tuple[str, ...]],
    *,
    quarantine_clip_ids: tuple[str, ...] = (),
) -> RelationPlan:
    clip_ids = {
        str(row[0]) for row in connection.execute("SELECT clip_id FROM evidence_clips")
    }
    event_ids = {
        str(row[0])
        for row in connection.execute("SELECT edge_event_id FROM evidence_events")
    }
    if quarantine_clip_ids != tuple(sorted(set(quarantine_clip_ids))):
        raise ClipConsistencyError("manifest_conflict", "staging authority is not canonical")
    if not set(quarantine_clip_ids) <= set(desired):
        raise ClipConsistencyError("manifest_conflict", "staging lacks final authority")
    desired_refs = [event_id for refs in desired.values() for event_id in refs]
    if len(desired_refs) != len(set(desired_refs)):
        raise ClipConsistencyError("manifest_conflict", "event has multiple authorities")
    if not set(desired) <= clip_ids:
        raise ClipConsistencyError("database_drift", "final clip is absent from database")
    if not set(desired_refs) <= event_ids:
        raise ClipConsistencyError("database_drift", "manifest event is absent from database")

    existing_by_clip = {
        clip_id: tuple(
            (str(row[0]), int(row[1]))
            for row in connection.execute(
                "SELECT edge_event_id, ordinal FROM clip_events "
                "WHERE clip_id = ? ORDER BY ordinal",
                (clip_id,),
            )
        )
        for clip_id in desired
    }
    desired_by_clip = {
        clip_id: tuple((event_id, ordinal) for ordinal, event_id in enumerate(refs))
        for clip_id, refs in desired.items()
    }
    changed = {
        clip_id
        for clip_id in desired
        if existing_by_clip[clip_id] != desired_by_clip[clip_id]
    }
    owners = {
        str(row[0]): str(row[1])
        for row in connection.execute(
            _select_event_owners_sql(len(desired_refs)), desired_refs
        )
    } if desired_refs else {}
    delete_ids = {
        str(row[0])
        for clip_id in changed
        for row in connection.execute(
            "SELECT edge_event_id FROM clip_events WHERE clip_id = ?", (clip_id,)
        )
    }
    delete_ids.update(
        event_id
        for event_id, owner in owners.items()
        if owner not in desired
    )
    inserts = tuple(
        (clip_id, event_id, ordinal)
        for clip_id in sorted(changed)
        for event_id, ordinal in desired_by_clip[clip_id]
    )
    before_rows = relation_rows(connection)
    after_rows = _project_after(before_rows, delete_ids, inserts)
    current_scope = {
        row
        for row in before_rows
        if row[0] in desired or row[1] in set(desired_refs)
    }
    desired_scope = {
        (clip_id, event_id, ordinal)
        for clip_id, rows in desired_by_clip.items()
        for event_id, ordinal in rows
    }
    return RelationPlan(
        relations_before=len(before_rows),
        relations_after=len(after_rows),
        mismatch_clips=len(changed),
        mismatch_tuples=len(current_scope.symmetric_difference(desired_scope)),
        delete_event_ids=tuple(sorted(delete_ids)),
        insert_rows=inserts,
        quarantine_clip_ids=quarantine_clip_ids,
        before_sha256=_rows_sha256(before_rows),
        after_sha256=_rows_sha256(after_rows),
    )


def relation_rows(connection: sqlite3.Connection) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (str(row[0]), str(row[1]), int(row[2]))
        for row in connection.execute(
            "SELECT clip_id, edge_event_id, ordinal FROM clip_events "
            "ORDER BY clip_id, ordinal, edge_event_id"
        )
    )


def relation_state_sha256(connection: sqlite3.Connection) -> str:
    return _rows_sha256(relation_rows(connection))


def _project_after(
    before: tuple[tuple[str, str, int], ...],
    delete_ids: set[str],
    inserts: tuple[tuple[str, str, int], ...],
) -> tuple[tuple[str, str, int], ...]:
    rows = [row for row in before if row[1] not in delete_ids]
    rows.extend(inserts)
    return tuple(sorted(rows, key=lambda row: (row[0], row[2], row[1])))


def _rows_sha256(rows: tuple[tuple[str, str, int], ...]) -> str:
    payload = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _select_event_owners_sql(count: int) -> str:
    placeholders = ",".join("?" for _ in range(count))
    return (
        "SELECT edge_event_id, clip_id FROM clip_events "
        f"WHERE edge_event_id IN ({placeholders})"
    )


def _reject_active_leases(connection: sqlite3.Connection, now: float) -> None:
    active_event = connection.execute(
        "SELECT 1 FROM evidence_events WHERE state = 'IN_FLIGHT' "
        "AND lease_expires_at > ? LIMIT 1",
        (now,),
    ).fetchone()
    active_clip = connection.execute(
        "SELECT 1 FROM evidence_clips WHERE publish_state = 'IN_FLIGHT' "
        "AND publish_lease_expires_at > ? LIMIT 1",
        (now,),
    ).fetchone()
    if active_event is not None or active_clip is not None:
        raise ClipConsistencyError("active_lease", "active evidence lease exists")


__all__ = [
    "RelationPlan",
    "plan_relations",
    "relation_rows",
    "relation_state_sha256",
    "validate_database",
]
