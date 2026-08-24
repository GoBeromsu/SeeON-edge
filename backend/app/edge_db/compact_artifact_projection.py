"""Project retained primary-clip and snapshot artifact facts."""

from __future__ import annotations

import sqlite3


def project_artifacts(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    """Map only PRIMARY_CLIP and SNAPSHOT slots with verified source relations."""
    media = {
        str(row[0]): row
        for row in source.execute(
            "SELECT media_id,content_sha256,size_bytes,mime_type,contained_relpath "
            "FROM evidence_media_objects"
        )
    }
    primary = {
        str(row[0]): row
        for row in source.execute(
            "SELECT incident_id,clip_id,media_id,codec,created_at FROM evidence_primary_clips"
        )
    }
    snapshots = {
        str(row[0]): row
        for row in source.execute(
            "SELECT incident_id,snapshot_id,media_id,captured_at,created_at "
            "FROM evidence_incident_snapshots"
        )
    }
    detected = {
        str(row[0]): str(row[1])
        for row in source.execute("SELECT incident_id,detected_at FROM evidence_incidents")
    }
    target_clips = {str(row[0]) for row in target.execute("SELECT clip_id FROM clips").fetchall()}
    slots = source.execute(
        "SELECT incident_id,slot_name,state,media_id,reason,revision,created_at,updated_at "
        "FROM evidence_artifact_slots WHERE slot_name IN ('PRIMARY_CLIP','SNAPSHOT') "
        "ORDER BY incident_id,slot_name"
    ).fetchall()
    for slot in slots:
        incident_id, kind, state = str(slot[0]), str(slot[1]), str(slot[2])
        relation = (
            primary.get(incident_id) if kind == "PRIMARY_CLIP" else snapshots.get(incident_id)
        )
        media_id = slot[3]
        media_row = None if media_id is None else media.get(str(media_id))
        artifact_id = None
        clip_id = None
        captured_at = None if kind == "PRIMARY_CLIP" else detected[incident_id]
        codec = None
        if relation is not None:
            if kind == "PRIMARY_CLIP":
                clip_id = str(relation[1]) if str(relation[1]) in target_clips else None
                media_id = relation[2] if media_id is None else media_id
                codec = relation[3]
            else:
                artifact_id = relation[1]
                media_id = relation[2] if media_id is None else media_id
                captured_at = relation[3]
            media_row = None if media_id is None else media.get(str(media_id))
        if state in {"AVAILABLE", "CORRUPT"} and media_row is None:
            state = "UNAVAILABLE"
        if state == "AVAILABLE" and kind == "PRIMARY_CLIP" and clip_id is None:
            state = "UNAVAILABLE"
        reason = slot[4]
        if state == "UNAVAILABLE":
            reason = str(reason or "NOT_RECORDED")[:64]
            artifact_id = None
            clip_id = None
            media_row = None
            codec = None
        elif state == "PENDING":
            artifact_id = None
            clip_id = None
            media_row = None
            reason = None
            codec = None
        elif media_row is not None:
            artifact_id = str(media_row[0]) if artifact_id is None else str(artifact_id)
        target.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                incident_id,
                kind,
                artifact_id,
                clip_id,
                state,
                reason,
                None if media_row is None else media_row[4],
                None if media_row is None else media_row[1],
                None if media_row is None else media_row[2],
                None if media_row is None else media_row[3],
                codec,
                captured_at,
                slot[5],
                slot[6],
                slot[7],
            ),
        )


__all__ = ["project_artifacts"]
