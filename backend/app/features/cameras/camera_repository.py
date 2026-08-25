"""Typed SQL projection helpers for the compact camera registry."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from backend.app.edge_db.configuration import ensure_edge_site, utc_now
from backend.app.features.cameras.camera_values import (
    CameraRegistryData,
    CameraStatus,
    normalize_stream_identity,
    parse_legacy_floor,
)

_CAMERA_SELECT = (
    "SELECT camera_id,label,rtsp_url,space_id,backend_camera_id,mapping_state,"
    "decode_backend,floor_override,created_at,last_probed_at,last_ok_at,never_connected,"
    "edge_ref,room_location_id FROM cameras"
)


def read_registry(
    connection: sqlite3.Connection, statuses: Mapping[str, CameraStatus]
) -> CameraRegistryData:
    version_row = connection.execute("SELECT registry_version FROM edge_site WHERE id=1").fetchone()
    rows = connection.execute(_CAMERA_SELECT + " ORDER BY camera_id").fetchall()
    cameras = [camera_from_row(row) for row in rows]
    for camera in cameras:
        camera["status"] = statuses.get(str(camera["id"]), "unknown")
    return {
        "registry_version": 0 if version_row is None else int(version_row[0]),
        "cameras": cameras,
    }


def get_camera(
    connection: sqlite3.Connection,
    camera_id: str,
    statuses: Mapping[str, CameraStatus],
) -> dict[str, object] | None:
    row = connection.execute(_CAMERA_SELECT + " WHERE camera_id=?", (camera_id,)).fetchone()
    if row is None:
        return None
    record = camera_from_row(row)
    record["status"] = statuses.get(camera_id, "unknown")
    return record


def find_duplicate(
    connection: sqlite3.Connection,
    rtsp_url: str,
    *,
    exclude_camera_id: str | None = None,
) -> dict[str, object] | None:
    identity = normalize_stream_identity(rtsp_url)
    if exclude_camera_id is None:
        row = connection.execute(
            _CAMERA_SELECT + " WHERE normalized_stream_identity=?", (identity,)
        ).fetchone()
    else:
        row = connection.execute(
            _CAMERA_SELECT + " WHERE normalized_stream_identity=? AND camera_id<>?",
            (identity, exclude_camera_id),
        ).fetchone()
    return None if row is None else camera_from_row(row)


def migrate_legacy_floors(connection: sqlite3.Connection) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    rows = connection.execute(
        "SELECT camera_id,floor_override FROM cameras WHERE floor_override IS NOT NULL"
    ).fetchall()
    for camera_id, stored_floor in rows:
        parsed = parse_legacy_floor(stored_floor, camera_id=str(camera_id))
        if parsed is None or str(stored_floor) == str(parsed):
            continue
        connection.execute(
            "UPDATE cameras SET floor_override=?,revision=revision+1,updated_at=? "
            "WHERE camera_id=?",
            (str(parsed), utc_now(), str(camera_id)),
        )
        changes.append({"camera_id": str(camera_id), "old": stored_floor, "new": parsed})
    if changes:
        record_registry_mutation(connection)
    return changes


def record_registry_mutation(connection: sqlite3.Connection) -> None:
    ensure_edge_site(connection)
    now = utc_now()
    connection.execute(
        "UPDATE edge_site SET registry_version=registry_version+1,"
        "topology_dirty_registry_version=registry_version+1,"
        "topology_dirty_created_at=?,updated_at=? WHERE id=1",
        (now, now),
    )


@contextmanager
def camera_transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    connection.execute("BEGIN IMMEDIATE")
    with connection:
        yield connection


def camera_from_row(row: tuple[object, ...]) -> dict[str, object]:
    floor = None if row[7] is None else parse_legacy_floor(str(row[7]), camera_id=str(row[0]))
    return {
        "id": str(row[0]),
        "label": str(row[1]),
        "rtsp_url": str(row[2]),
        "space_id": _text(row[3]),
        "backend_camera_id": _text(row[4]),
        "mapping_pending": row[5] == "PENDING",
        "status": "unknown",
        "decode_backend": _text(row[6]),
        "floor": floor,
        "created_at": str(row[8]),
        "last_probed_at": _text(row[9]),
        "last_ok_at": _text(row[10]),
        "never_connected": bool(row[11]),
        "edge_ref": _text(row[12]),
        "room_edge_ref": _text(row[13]),
    }


def _text(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "camera_transaction",
    "find_duplicate",
    "get_camera",
    "migrate_legacy_floors",
    "read_registry",
    "record_registry_mutation",
]
