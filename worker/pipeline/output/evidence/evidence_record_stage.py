"""Atomic central incident creation beside immutable outbox staging."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from worker.pipeline.output.evidence.evidence_outbox_types import StagedEvent


def central_records_available(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'evidence_incidents'"
        ).fetchone()
        is not None
    )


def stage_central_incident(
    connection: sqlite3.Connection,
    event: StagedEvent,
    *,
    runtime_manifest_sha256: str | None,
    decision_trace_id: str | None,
) -> None:
    if not central_records_available(connection):
        return
    payload = json.loads(event.payload_json)
    if not isinstance(payload, dict):
        raise TypeError("central evidence event payload must be an object")
    camera_id = _required_text(payload, "camera_id")
    event_type = _required_text(payload, "event_type")
    qualified = _qualification(
        connection,
        camera_id=camera_id,
        runtime_manifest_sha256=runtime_manifest_sha256,
        decision_trace_id=decision_trace_id,
    )
    provenance_state = "QUALIFIED" if qualified is not None else "MISSING"
    missing_reason = (
        None
        if qualified is not None
        else _missing_reason(runtime_manifest_sha256, decision_trace_id)
    )
    module_id, policy_id, effective_policy_id, resolved_manifest = (
        (None, None, None, runtime_manifest_sha256) if qualified is None else qualified
    )
    values = (
        event.edge_event_id,
        event.edge_event_id,
        camera_id,
        event_type,
        event.detected_at,
        resolved_manifest,
        decision_trace_id,
        module_id,
        policy_id,
        effective_policy_id,
        provenance_state,
        missing_reason,
        event.detected_at,
        event.detected_at,
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO evidence_incidents (
            incident_id, edge_event_id, camera_id, event_type, detected_at,
            runtime_manifest_sha256, decision_trace_id, module_qualified_id,
            policy_qualified_id, effective_policy_id, provenance_state,
            provenance_missing_reason, lifecycle_state, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'STAGING', ?, ?)
        """,
        values,
    )
    existing = connection.execute(
        """
        SELECT incident_id, edge_event_id, camera_id, event_type, detected_at,
               runtime_manifest_sha256, decision_trace_id, module_qualified_id,
               policy_qualified_id, effective_policy_id, provenance_state,
               provenance_missing_reason, created_at, updated_at
        FROM evidence_incidents WHERE edge_event_id = ?
        """,
        (event.edge_event_id,),
    ).fetchone()
    if existing != values:
        raise ValueError("central evidence replay conflicts with immutable incident facts")
    _ensure_slot(
        connection,
        str(event.edge_event_id),
        "PRIMARY_CLIP",
        "PENDING",
        None,
        None,
        event.detected_at,
    )
    snapshot = payload.get("snapshot")
    if snapshot is None:
        _ensure_slot(
            connection,
            str(event.edge_event_id),
            "SNAPSHOT",
            "UNAVAILABLE",
            None,
            "NOT_CAPTURED",
            event.detected_at,
        )
    elif isinstance(snapshot, dict):
        _stage_snapshot_pending(
            connection,
            str(event.edge_event_id),
            camera_id,
            snapshot,
            event.detected_at,
        )
    else:
        raise ValueError("central evidence snapshot metadata must be an object")


def _qualification(
    connection: sqlite3.Connection,
    *,
    camera_id: str,
    runtime_manifest_sha256: str | None,
    decision_trace_id: str | None,
) -> tuple[str, str, str, str] | None:
    if decision_trace_id is None:
        return None
    row = connection.execute(
        """
        SELECT decision.module_qualified_id, decision.policy_qualified_id,
               decision.effective_policy_id, decision.runtime_manifest_sha256,
               analysis.camera_id
        FROM evidence_decision_traces AS decision
        LEFT JOIN runtime_analysis_traces AS analysis
          ON analysis.trace_id = decision.analysis_trace_id
        WHERE decision.trace_id = ?
        """,
        (decision_trace_id,),
    ).fetchone()
    if row is None:
        raise ValueError("central evidence decision trace is missing")
    if str(row[4]) != camera_id:
        raise ValueError("central evidence camera differs from its decision trace")
    resolved_manifest = str(row[3])
    if runtime_manifest_sha256 is not None and resolved_manifest != runtime_manifest_sha256:
        raise ValueError("central evidence runtime manifest differs from its decision trace")
    return str(row[0]), str(row[1]), str(row[2]), resolved_manifest


def _stage_snapshot_pending(
    connection: sqlite3.Connection,
    incident_id: str,
    camera_id: str,
    snapshot: dict[str, Any],
    timestamp: str,
) -> None:
    facts = _snapshot_facts(snapshot, incident_id=incident_id, camera_id=camera_id)
    row = connection.execute(
        "SELECT state FROM evidence_artifact_slots "
        "WHERE incident_id = ? AND slot_name = 'SNAPSHOT'",
        (incident_id,),
    ).fetchone()
    if row is None:
        _ensure_slot(connection, incident_id, "SNAPSHOT", "PENDING", None, None, timestamp)
        return
    if str(row[0]) == "PENDING":
        return
    if str(row[0]) == "AVAILABLE":
        _require_snapshot_relation(connection, incident_id, facts)
        return
    raise ValueError("central evidence snapshot replay conflicts with terminal state")


def attach_snapshot_record(
    connection: sqlite3.Connection,
    incident_id: str,
    snapshot: dict[str, Any],
) -> None:
    """Attach immutable facts only after the staged bytes were atomically published."""
    incident = connection.execute(
        "SELECT camera_id, edge_event_id FROM evidence_incidents WHERE incident_id = ?",
        (incident_id,),
    ).fetchone()
    if incident is None:
        raise ValueError("central evidence incident is missing")
    camera_id = str(incident[0])
    event_id = str(incident[1])
    facts = _snapshot_facts(snapshot, incident_id=event_id, camera_id=camera_id)
    payload_row = connection.execute(
        "SELECT payload_json FROM evidence_events WHERE edge_event_id = ?",
        (event_id,),
    ).fetchone()
    if payload_row is None:
        raise ValueError("central evidence event is missing")
    payload = json.loads(str(payload_row[0]))
    staged_snapshot = payload.get("snapshot") if isinstance(payload, dict) else None
    if (
        not isinstance(staged_snapshot, dict)
        or _snapshot_facts(staged_snapshot, incident_id=event_id, camera_id=camera_id) != facts
    ):
        raise ValueError("central evidence snapshot differs from staged event")
    snapshot_id, relpath, sha256, size_bytes, mime_type, captured_at = facts
    media_id = ensure_media_object(
        connection,
        sha256=sha256,
        size_bytes=size_bytes,
        mime_type=mime_type,
        relpath=relpath,
        created_at=captured_at,
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO evidence_incident_snapshots (
            incident_id, snapshot_id, media_id, captured_at, camera_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (incident_id, snapshot_id, media_id, captured_at, camera_id, captured_at),
    )
    _require_snapshot_relation(connection, incident_id, facts)
    slot = connection.execute(
        "SELECT state, media_id, revision FROM evidence_artifact_slots "
        "WHERE incident_id = ? AND slot_name = 'SNAPSHOT'",
        (incident_id,),
    ).fetchone()
    if slot is None:
        raise ValueError("central evidence snapshot transition is missing")
    if str(slot[0]) == "AVAILABLE":
        if str(slot[1]) != media_id:
            raise ValueError("central evidence snapshot slot conflicts")
        return
    if str(slot[0]) != "PENDING":
        raise ValueError("central evidence snapshot transition is terminal")
    changed = connection.execute(
        """
        UPDATE evidence_artifact_slots
        SET state = 'AVAILABLE', media_id = ?, revision = revision + 1, updated_at = ?
        WHERE incident_id = ? AND slot_name = 'SNAPSHOT'
          AND state = 'PENDING' AND revision = ?
        """,
        (media_id, captured_at, incident_id, int(slot[2])),
    ).rowcount
    if changed != 1:
        raise sqlite3.IntegrityError("central evidence snapshot revision changed")


def _snapshot_facts(
    snapshot: dict[str, Any],
    *,
    incident_id: str,
    camera_id: str,
) -> tuple[str, str, str, int, str, str]:
    snapshot_id = _required_text(snapshot, "snapshot_id")
    if _required_text(snapshot, "camera_id") != camera_id:
        raise ValueError("central evidence snapshot camera differs from incident")
    if _required_text(snapshot, "edge_event_id") != incident_id:
        raise ValueError("central evidence snapshot event differs from incident")
    relpath = _contained_relpath(_required_text(snapshot, "path"))
    expected_path = _snapshot_relpath(
        camera_id,
        _required_text(snapshot, "captured_at"),
        snapshot_id,
    )
    if relpath != expected_path:
        raise ValueError("central evidence snapshot path differs from identity")
    sha256 = _sha256(snapshot.get("sha256"))
    size_bytes = snapshot.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ValueError("central evidence snapshot size is invalid")
    mime_type = _required_text(snapshot, "mime_type")
    if mime_type != "image/jpeg":
        raise ValueError("central evidence snapshot MIME type is invalid")
    captured_at = _required_text(snapshot, "captured_at")
    return snapshot_id, relpath, sha256, size_bytes, mime_type, captured_at


def _require_snapshot_relation(
    connection: sqlite3.Connection,
    incident_id: str,
    facts: tuple[str, str, str, int, str, str],
) -> None:
    snapshot_id, relpath, sha256, size_bytes, mime_type, captured_at = facts
    media_id = f"sha256:{sha256}:{size_bytes}"
    row = connection.execute(
        """
        SELECT snapshot.snapshot_id, media.contained_relpath, media.content_sha256,
               media.size_bytes, media.mime_type, snapshot.captured_at
        FROM evidence_incident_snapshots AS snapshot
        JOIN evidence_media_objects AS media USING (media_id)
        WHERE snapshot.incident_id = ? AND snapshot.media_id = ?
        """,
        (incident_id, media_id),
    ).fetchone()
    if row != (snapshot_id, relpath, sha256, size_bytes, mime_type, captured_at):
        raise ValueError("central evidence snapshot replay conflicts")


def _ensure_slot(
    connection: sqlite3.Connection,
    incident_id: str,
    slot_name: str,
    state: str,
    media_id: str | None,
    reason: str | None,
    timestamp: str,
) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO evidence_artifact_slots (
            incident_id, slot_name, state, media_id, reason, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (incident_id, slot_name, state, media_id, reason, timestamp, timestamp),
    )
    row = connection.execute(
        "SELECT state, media_id, reason FROM evidence_artifact_slots "
        "WHERE incident_id = ? AND slot_name = ?",
        (incident_id, slot_name),
    ).fetchone()
    if row != (state, media_id, reason):
        raise ValueError("central evidence artifact replay conflicts")


def ensure_media_object(
    connection: sqlite3.Connection,
    *,
    sha256: str,
    size_bytes: int,
    mime_type: str,
    relpath: str,
    created_at: str,
) -> str:
    media_id = f"sha256:{sha256}:{size_bytes}"
    values = (media_id, sha256, size_bytes, mime_type, relpath, PurePosixPath(relpath).name)
    connection.execute(
        """
        INSERT OR IGNORE INTO evidence_media_objects (
            media_id, content_sha256, size_bytes, mime_type,
            contained_relpath, basename, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (*values, created_at),
    )
    row = connection.execute(
        "SELECT media_id, content_sha256, size_bytes, mime_type, contained_relpath, basename "
        "FROM evidence_media_objects WHERE media_id = ?",
        (media_id,),
    ).fetchone()
    if row != values:
        raise ValueError("content identity resolves to contradictory media facts")
    return media_id


def _snapshot_relpath(camera_id: str, captured_at: str, snapshot_id: str) -> str:
    try:
        date = datetime.fromisoformat(captured_at.replace("Z", "+00:00")).date().isoformat()
    except ValueError as exc:
        raise ValueError("central evidence snapshot captured_at is invalid") from exc
    camera_key = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()[:16]
    snapshot_key = hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()
    return f"snapshots/{camera_key}/{date}/{snapshot_key}.jpg"


def _missing_reason(runtime: str | None, decision: str | None) -> str:
    if runtime is None and decision is None:
        return "RUNTIME_AND_DECISION_TRACE_NOT_RECORDED"
    if runtime is None:
        return "RUNTIME_MANIFEST_NOT_RECORDED"
    return "DECISION_TRACE_NOT_RECORDED"


def _contained_relpath(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("central evidence media path is not contained")
    return path.as_posix()


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"central evidence {key} is missing")
    return value


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("central evidence SHA-256 is invalid")
    return value


__all__ = [
    "attach_snapshot_record",
    "central_records_available",
    "ensure_media_object",
    "stage_central_incident",
]
