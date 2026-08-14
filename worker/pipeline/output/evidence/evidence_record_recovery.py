"""Descriptor-bound verification for immutable central snapshot evidence."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from pathlib import Path, PurePosixPath

from worker.pipeline.output.evidence.evidence_record_stage import central_records_available
from worker.pipeline.output.evidence.snapshot_store import StoredSnapshot


def verify_snapshot_records(
    connection: sqlite3.Connection,
    store_dir: Path,
    *,
    updated_at: str,
) -> int:
    if not central_records_available(connection):
        return 0
    rows = connection.execute(
        """
        SELECT incident.incident_id, incident.lifecycle_state, incident.revision,
               slot.revision, media.contained_relpath, media.content_sha256,
               media.size_bytes
        FROM evidence_incident_snapshots AS snapshot
        JOIN evidence_incidents AS incident USING (incident_id)
        JOIN evidence_artifact_slots AS slot
          ON slot.incident_id = incident.incident_id AND slot.slot_name = 'SNAPSHOT'
        JOIN evidence_media_objects AS media ON media.media_id = snapshot.media_id
        WHERE slot.state = 'AVAILABLE'
        ORDER BY incident.incident_id
        """
    ).fetchall()
    corrupt = 0
    for incident_id, lifecycle, incident_revision, slot_revision, relpath, digest, size in rows:
        if _matches(store_dir, str(relpath), str(digest), int(size)):
            continue
        connection.execute(
            """
            UPDATE evidence_artifact_slots
            SET state = 'CORRUPT', reason = 'MISSING_OR_MUTATED',
                revision = revision + 1, updated_at = ?
            WHERE incident_id = ? AND slot_name = 'SNAPSHOT' AND revision = ?
            """,
            (updated_at, str(incident_id), int(slot_revision)),
        )
        if str(lifecycle) != "FAILED":
            connection.execute(
                """
                UPDATE evidence_incidents
                SET lifecycle_state = 'FAILED', failure_reason = 'CORRUPT',
                    revision = revision + 1, updated_at = ?
                WHERE incident_id = ? AND revision = ?
                """,
                (updated_at, str(incident_id), int(incident_revision)),
            )
        corrupt += 1
    return corrupt


def pending_snapshot_records(connection: sqlite3.Connection) -> tuple[StoredSnapshot, ...]:
    if not central_records_available(connection):
        return ()
    rows = connection.execute(
        """
        SELECT event.payload_json
        FROM evidence_incidents AS incident
        JOIN evidence_events AS event USING (edge_event_id)
        JOIN evidence_artifact_slots AS slot
          ON slot.incident_id = incident.incident_id AND slot.slot_name = 'SNAPSHOT'
        WHERE slot.state = 'PENDING'
        ORDER BY incident.incident_id
        """
    ).fetchall()
    records: list[StoredSnapshot] = []
    for (payload_json,) in rows:
        payload = json.loads(str(payload_json))
        snapshot = payload.get("snapshot") if isinstance(payload, dict) else None
        if not isinstance(snapshot, dict):
            raise TypeError("pending central snapshot metadata is missing")
        records.append(_stored_snapshot(snapshot))
    return tuple(records)


def available_snapshot_records(connection: sqlite3.Connection) -> tuple[StoredSnapshot, ...]:
    if not central_records_available(connection):
        return ()
    rows = connection.execute(
        """
        SELECT snapshot.snapshot_id, media.contained_relpath, media.content_sha256,
               media.size_bytes, media.mime_type, snapshot.captured_at,
               snapshot.camera_id, incident.edge_event_id
        FROM evidence_incident_snapshots AS snapshot
        JOIN evidence_incidents AS incident USING (incident_id)
        JOIN evidence_artifact_slots AS slot
          ON slot.incident_id = incident.incident_id AND slot.slot_name = 'SNAPSHOT'
        JOIN evidence_media_objects AS media USING (media_id)
        WHERE slot.state = 'AVAILABLE'
        ORDER BY snapshot.snapshot_id
        """
    ).fetchall()
    return tuple(
        StoredSnapshot(
            snapshot_id=str(row[0]),
            path=str(row[1]),
            sha256=str(row[2]),
            size_bytes=int(row[3]),
            mime_type=str(row[4]),
            captured_at=str(row[5]),
            camera_id=str(row[6]),
            edge_event_id=str(row[7]),
        )
        for row in rows
    )


def mark_pending_snapshot_unavailable(
    connection: sqlite3.Connection,
    snapshot_id: str,
    *,
    reason: str,
    updated_at: str,
) -> bool:
    if not central_records_available(connection):
        return False
    if not reason:
        raise ValueError("snapshot unavailable reason must be set")
    changed = connection.execute(
        """
        UPDATE evidence_artifact_slots
        SET state = 'UNAVAILABLE', reason = ?, revision = revision + 1, updated_at = ?
        WHERE slot_name = 'SNAPSHOT' AND state = 'PENDING'
          AND incident_id = (
              SELECT incident_id FROM evidence_incidents WHERE edge_event_id = ?
          )
        """,
        (reason, updated_at, snapshot_id),
    ).rowcount
    return changed == 1


def _stored_snapshot(payload: dict[str, object]) -> StoredSnapshot:
    expected = {
        "snapshot_id",
        "path",
        "sha256",
        "size_bytes",
        "mime_type",
        "captured_at",
        "camera_id",
        "edge_event_id",
    }
    if set(payload) != expected:
        raise ValueError("pending central snapshot metadata fields are invalid")
    snapshot_id = payload["snapshot_id"]
    path = payload["path"]
    sha256 = payload["sha256"]
    size_bytes = payload["size_bytes"]
    mime_type = payload["mime_type"]
    captured_at = payload["captured_at"]
    camera_id = payload["camera_id"]
    edge_event_id = payload["edge_event_id"]
    texts = (snapshot_id, path, sha256, mime_type, captured_at, camera_id)
    if not all(isinstance(value, str) for value in texts):
        raise TypeError("pending central snapshot metadata types are invalid")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
        raise TypeError("pending central snapshot size type is invalid")
    if edge_event_id is not None and not isinstance(edge_event_id, str):
        raise TypeError("pending central snapshot event type is invalid")
    assert isinstance(snapshot_id, str)
    assert isinstance(path, str)
    assert isinstance(sha256, str)
    assert isinstance(mime_type, str)
    assert isinstance(captured_at, str)
    assert isinstance(camera_id, str)
    return StoredSnapshot(
        snapshot_id=snapshot_id,
        path=path,
        sha256=sha256,
        size_bytes=size_bytes,
        mime_type=mime_type,
        captured_at=captured_at,
        camera_id=camera_id,
        edge_event_id=edge_event_id,
    )


def _matches(root: Path, raw_relpath: str, expected_hash: str, expected_size: int) -> bool:
    relative = PurePosixPath(raw_relpath)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        return False
    descriptors: list[int] = []
    try:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(descriptor)
        for component in relative.parts[:-1]:
            descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            descriptors.append(descriptor)
        media = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        descriptors.append(media)
        info = os.fstat(media)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
            return False
        digest = hashlib.sha256()
        while chunk := os.read(media, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest() == expected_hash
    except OSError:
        return False
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


__all__ = [
    "available_snapshot_records",
    "mark_pending_snapshot_unavailable",
    "pending_snapshot_records",
    "verify_snapshot_records",
]
