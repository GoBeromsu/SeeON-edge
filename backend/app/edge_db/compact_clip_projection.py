"""Project legacy clip identity and canonical publication facts."""

from __future__ import annotations

import sqlite3
from typing import TypeAlias

SqliteValue: TypeAlias = None | int | float | str | bytes


def _facet(value: SqliteValue) -> str:
    return str(value) if value in {"fall", "bed-exit"} else "other"


def _insert_unavailable(
    target: sqlite3.Connection,
    *,
    clip_id: str,
    camera_id: str,
    event_type: SqliteValue,
    started_at: str,
    codec: SqliteValue,
    mime_type: SqliteValue,
    publish_state: str = "WAITING",
    published_at: str | None = None,
    revision: int = 1,
) -> None:
    target.execute(
        "INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            clip_id,
            camera_id,
            _facet(event_type),
            started_at,
            started_at,
            None,
            codec,
            mime_type,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "UNAVAILABLE",
            "LEGACY_CATALOG_ONLY",
            publish_state,
            published_at,
            None,
            "RETAINED",
            None,
            None,
            None,
            revision,
            started_at,
            started_at,
        ),
    )


def _publication(state: SqliteValue, fallback_at: str) -> tuple[str, str | None]:
    if state == "PUBLISHED":
        return "PUBLISHED", fallback_at
    if state in {"PERMANENT", "COMPATIBILITY"}:
        return str(state), None
    return "WAITING", None


def project_clip_facts(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    """Add non-manifest identities, then overlay canonical publication state."""
    existing = {str(row[0]) for row in target.execute("SELECT clip_id FROM clips")}
    for row in source.execute(
        "SELECT clip_id,camera_id,event_type,state,started_at,mime_type,encoder "
        "FROM clips ORDER BY clip_id"
    ):
        clip_id = str(row[0])
        if clip_id in existing:
            continue
        publish_state, published_at = _publication(str(row[3]).upper(), str(row[4]))
        _insert_unavailable(
            target,
            clip_id=clip_id,
            camera_id=str(row[1] or "unknown"),
            event_type=row[2],
            started_at=str(row[4]),
            codec=row[6],
            mime_type=row[5],
            publish_state=publish_state,
            published_at=published_at,
        )
        existing.add(clip_id)
    relations = source.execute(
        "SELECT primary_record.clip_id,incident.camera_id,incident.event_type,"
        "incident.detected_at,primary_record.codec,clip.state_version,clip.publish_state "
        "FROM evidence_primary_clips AS primary_record "
        "JOIN evidence_incidents AS incident ON incident.incident_id=primary_record.incident_id "
        "JOIN evidence_clips AS clip ON clip.clip_id=primary_record.clip_id "
        "ORDER BY primary_record.clip_id"
    ).fetchall()
    for row in relations:
        clip_id = str(row[0])
        publish_state, published_at = _publication(row[6], str(row[3]))
        if clip_id not in existing:
            _insert_unavailable(
                target,
                clip_id=clip_id,
                camera_id=str(row[1]),
                event_type=row[2],
                started_at=str(row[3]),
                codec=row[4],
                mime_type=None,
                publish_state=publish_state,
                published_at=published_at,
                revision=int(row[5]),
            )
            existing.add(clip_id)
        elif publish_state != "WAITING":
            current = target.execute(
                "SELECT revision,updated_at,started_at FROM clips WHERE clip_id=?", (clip_id,)
            ).fetchone()
            if current is None:
                raise sqlite3.DatabaseError("projected clip disappeared")
            target.execute(
                "UPDATE clips SET publish_state=?,published_at=?,revision=?,updated_at=? "
                "WHERE clip_id=?",
                (
                    publish_state,
                    str(current[2]) if published_at is not None else None,
                    int(current[0]) + 1,
                    str(current[1]),
                    clip_id,
                ),
            )


__all__ = ["project_clip_facts"]
