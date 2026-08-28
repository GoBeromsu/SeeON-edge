"""Project retained primary-clip and snapshot artifact facts."""

from __future__ import annotations

import sqlite3
from dataclasses import astuple, dataclass
from typing import TypeAlias

SqliteValue: TypeAlias = int | float | str | bytes | None


@dataclass(frozen=True, slots=True)
class Artifact:
    incident_id: str
    kind: str
    artifact_id: str | None
    clip_id: str | None
    state: str
    reason: str | None
    contained_relpath: str | None
    content_sha256: str | None
    size_bytes: int | None
    mime_type: str | None
    codec: str | None
    captured_at: str | None
    revision: int
    created_at: str
    updated_at: str


def _integer(value: SqliteValue) -> int:
    if not isinstance(value, int):
        raise sqlite3.DatabaseError("artifact integer authority is invalid")
    return value


def _insert(target: sqlite3.Connection, artifact: Artifact) -> None:
    target.execute(
        "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        astuple(artifact),
    )


def _normalized(artifact: Artifact) -> Artifact:
    if artifact.state in {"AVAILABLE", "CORRUPT"} and (
        artifact.artifact_id is None
        or artifact.contained_relpath is None
        or artifact.content_sha256 is None
        or artifact.size_bytes is None
        or artifact.mime_type is None
    ):
        return Artifact(
            artifact.incident_id,
            artifact.kind,
            None,
            None,
            "UNAVAILABLE",
            "NOT_RECORDED",
            None,
            None,
            None,
            None,
            None,
            artifact.captured_at,
            artifact.revision,
            artifact.created_at,
            artifact.updated_at,
        )
    if artifact.state == "UNAVAILABLE":
        return Artifact(
            artifact.incident_id,
            artifact.kind,
            None,
            None,
            artifact.state,
            (artifact.reason or "NOT_RECORDED")[:64],
            None,
            None,
            None,
            None,
            None,
            artifact.captured_at,
            artifact.revision,
            artifact.created_at,
            artifact.updated_at,
        )
    return artifact


def _primary_artifacts(
    source: sqlite3.Connection,
    target_clips: set[str],
    media: dict[str, tuple[SqliteValue, ...]],
    slots: dict[tuple[str, str], tuple[SqliteValue, ...]],
) -> tuple[Artifact, ...]:
    artifacts: list[Artifact] = []
    relations: dict[str, tuple[SqliteValue, ...]] = {
        str(row[0]): row
        for row in source.execute(
            "SELECT incident_id,clip_id,media_id,codec,created_at FROM evidence_primary_clips"
        )
    }
    for incident_id, relation in relations.items():
        slot = slots.get((incident_id, "PRIMARY_CLIP"))
        media_id = relation[2] if slot is None or slot[3] is None else slot[3]
        media_row = None if media_id is None else media.get(str(media_id))
        clip_id = str(relation[1]) if str(relation[1]) in target_clips else None
        state = (
            ("AVAILABLE" if media_row is not None else "UNAVAILABLE")
            if slot is None
            else str(slot[2])
        )
        reason = None if slot is None else slot[4]
        revision = 1 if slot is None else _integer(slot[5])
        created_at = str(relation[4]) if slot is None else str(slot[6])
        updated_at = created_at if slot is None else str(slot[7])
        artifacts.append(
            _normalized(
                Artifact(
                    incident_id,
                    "PRIMARY_CLIP",
                    None if media_row is None else str(media_row[0]),
                    clip_id,
                    state,
                    None if reason is None else str(reason),
                    None if media_row is None else str(media_row[4]),
                    None if media_row is None else str(media_row[1]),
                    None if media_row is None else _integer(media_row[2]),
                    None if media_row is None else str(media_row[3]),
                    None if relation[3] is None else str(relation[3]),
                    None,
                    revision,
                    created_at,
                    updated_at,
                )
            )
        )
    return tuple(artifacts)


def _snapshot_artifacts(
    source: sqlite3.Connection,
    media: dict[str, tuple[SqliteValue, ...]],
    slots: dict[tuple[str, str], tuple[SqliteValue, ...]],
) -> tuple[Artifact, ...]:
    event_incidents = {
        str(event_id): str(incident_id)
        for incident_id, event_id in source.execute(
            "SELECT incident_id,edge_event_id FROM evidence_incidents"
        )
    }
    generic: dict[str, tuple[SqliteValue, ...]] = {
        event_incidents[str(row[2])]: row
        for row in source.execute(
            "SELECT snapshot_id,camera_id,edge_event_id,captured_at,path,sha256,"
            "size_bytes,mime_type FROM snapshots"
        )
        if str(row[2]) in event_incidents
    }
    relations: dict[str, tuple[SqliteValue, ...]] = {
        str(row[0]): row
        for row in source.execute(
            "SELECT incident_id,snapshot_id,media_id,captured_at,created_at "
            "FROM evidence_incident_snapshots"
        )
    }
    incident_ids = sorted(
        set(relations)
        | set(generic)
        | {incident_id for incident_id, kind in slots if kind == "SNAPSHOT"}
    )
    artifacts: list[Artifact] = []
    for incident_id in incident_ids:
        relation = relations.get(incident_id)
        catalog = generic.get(incident_id)
        slot = slots.get((incident_id, "SNAPSHOT"))
        if slot is None:
            state = "AVAILABLE"
            reason = None
            revision = 1
        else:
            state = str(slot[2])
            reason = slot[4]
            revision = _integer(slot[5])
        media_id = None if relation is None else relation[2]
        if media_id is None and slot is not None:
            media_id = slot[3]
        media_row = None if media_id is None else media.get(str(media_id))
        artifact_id = (
            str(relation[1])
            if relation is not None
            else (str(catalog[0]) if catalog is not None else None)
        )
        captured_at = (
            str(relation[3])
            if relation is not None
            else (str(catalog[3]) if catalog is not None else None)
        )
        if slot is not None:
            created_at = str(slot[6])
        elif relation is not None:
            created_at = str(relation[4])
        elif catalog is not None:
            created_at = str(catalog[3])
        else:
            raise sqlite3.DatabaseError("snapshot authority has no source row")
        direct = None if relation is not None or catalog is None else catalog
        artifact = Artifact(
            incident_id,
            "SNAPSHOT",
            artifact_id,
            None,
            state,
            None if reason is None else str(reason),
            (
                str(media_row[4])
                if media_row is not None
                else (None if direct is None or direct[4] is None else str(direct[4]))
            ),
            (
                str(media_row[1])
                if media_row is not None
                else (None if direct is None or direct[5] is None else str(direct[5]))
            ),
            (
                _integer(media_row[2])
                if media_row is not None
                else (None if direct is None or direct[6] is None else _integer(direct[6]))
            ),
            (
                str(media_row[3])
                if media_row is not None
                else (None if direct is None or direct[7] is None else str(direct[7]))
            ),
            None,
            captured_at,
            revision,
            created_at,
            str(slot[7]) if slot is not None else created_at,
        )
        artifacts.append(_normalized(artifact))
    return tuple(artifacts)


def project_artifacts(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    media: dict[str, tuple[SqliteValue, ...]] = {
        str(row[0]): row
        for row in source.execute(
            "SELECT media_id,content_sha256,size_bytes,mime_type,contained_relpath "
            "FROM evidence_media_objects"
        )
    }
    slots: dict[tuple[str, str], tuple[SqliteValue, ...]] = {
        (str(row[0]), str(row[1])): row
        for row in source.execute(
            "SELECT incident_id,slot_name,state,media_id,reason,revision,created_at,updated_at "
            "FROM evidence_artifact_slots WHERE slot_name IN ('PRIMARY_CLIP','SNAPSHOT')"
        )
    }
    target_clips = {str(row[0]) for row in target.execute("SELECT clip_id FROM clips")}
    for artifact in (
        *_primary_artifacts(source, target_clips, media, slots),
        *_snapshot_artifacts(source, media, slots),
    ):
        _insert(target, artifact)


__all__ = ["project_artifacts"]
