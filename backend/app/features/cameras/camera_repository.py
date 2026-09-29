"""Native SQL on borrowed transactions; writers hold the edge_site row lock."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from backend.app.edge_db.postgres import PostgresError
from backend.app.features.cameras.camera_values import (
    CameraRegistryData,
    CameraStatus,
    normalize_stream_identity,
    parse_legacy_floor,
)

_CAMERA_COLUMNS = (
    "c.camera_id,c.incarnation,c.label,c.rtsp_url,c.space_id,c.backend_camera_id,c.mapping_state,"
    "c.decode_backend,c.floor_override,c.created_at,c.last_probed_at,c.last_ok_at,"
    "c.never_connected,c.edge_ref,c.room_location_id"
)
_CAMERA_SELECT = "SELECT " + _CAMERA_COLUMNS + " FROM cameras AS c"


class CameraRegistryNotInitialized(PostgresError):
    def __init__(self) -> None:
        super().__init__("camera registry bootstrap row is missing")


class CameraRegistryWriteError(PostgresError):
    def __init__(self) -> None:
        super().__init__("camera registry write rejected by a database constraint")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def lock_registry(connection: psycopg.Connection) -> None:
    """Serialize writers after authority admission, before any registry row access."""
    row = connection.execute("SELECT id FROM edge_site WHERE id=1 FOR UPDATE").fetchone()
    if row is None:
        raise CameraRegistryNotInitialized()


def read_registry(
    connection: psycopg.Connection, statuses: Mapping[str, tuple[UUID, CameraStatus]]
) -> CameraRegistryData:
    # A single statement has one MVCC snapshot even under READ COMMITTED.
    # LEFT JOIN retains the bootstrap row when the registry is empty.
    with connection.cursor(row_factory=dict_row) as cursor:
        rows = cursor.execute(
            "SELECT s.registry_version,"
            + _CAMERA_COLUMNS
            + " FROM edge_site AS s LEFT JOIN cameras AS c ON true "
            'WHERE s.id=1 ORDER BY c.camera_id COLLATE "C"'
        ).fetchall()
    if not rows:
        raise CameraRegistryNotInitialized()
    cameras = []
    for row in rows:
        if row["camera_id"] is None:
            continue
        camera = camera_from_row(row)
        camera["status"] = _cached_status(row, statuses)
        cameras.append(camera)
    return {
        "registry_version": int(rows[0]["registry_version"]),
        "cameras": cameras,
    }


def get_camera(
    connection: psycopg.Connection,
    camera_id: str,
    statuses: Mapping[str, tuple[UUID, CameraStatus]],
) -> tuple[UUID, dict[str, object]] | None:
    """Keep the persisted incarnation separate from the public record."""
    with connection.cursor(row_factory=dict_row) as cursor:
        row = cursor.execute(
            "SELECT "
            + _CAMERA_COLUMNS
            + " FROM edge_site AS s LEFT JOIN cameras AS c ON c.camera_id=%s WHERE s.id=1",
            (camera_id,),
        ).fetchone()
    if row is None:
        raise CameraRegistryNotInitialized()
    if row["camera_id"] is None:
        return None
    record = camera_from_row(row)
    record["status"] = _cached_status(row, statuses)
    return cast(UUID, row["incarnation"]), record


def _cached_status(
    row: Mapping[str, object], statuses: Mapping[str, tuple[UUID, CameraStatus]]
) -> CameraStatus:
    cached = statuses.get(str(row["camera_id"]))
    if cached is not None and cached[0] == row["incarnation"]:
        return cached[1]
    return "unknown"


def find_duplicate(
    connection: psycopg.Connection,
    rtsp_url: str,
    *,
    exclude_camera_id: str | None = None,
) -> dict[str, object] | None:
    identity = normalize_stream_identity(rtsp_url)
    with connection.cursor(row_factory=dict_row) as cursor:
        if exclude_camera_id is None:
            row = cursor.execute(
                _CAMERA_SELECT + " WHERE c.normalized_stream_identity=%s", (identity,)
            ).fetchone()
        else:
            row = cursor.execute(
                _CAMERA_SELECT + " WHERE c.normalized_stream_identity=%s AND c.camera_id<>%s",
                (identity, exclude_camera_id),
            ).fetchone()
    return None if row is None else camera_from_row(row)


def migrate_legacy_floors(connection: psycopg.Connection) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    rows = connection.execute(
        "SELECT camera_id,floor_override FROM cameras WHERE floor_override IS NOT NULL"
    ).fetchall()
    for camera_id, stored_floor in rows:
        parsed = parse_legacy_floor(stored_floor, camera_id=str(camera_id))
        if parsed is None or str(stored_floor) == str(parsed):
            continue
        connection.execute(
            "UPDATE cameras SET floor_override=%s,revision=revision+1,updated_at=%s "
            "WHERE camera_id=%s",
            (str(parsed), utc_now(), str(camera_id)),
        )
        changes.append({"camera_id": str(camera_id), "old": stored_floor, "new": parsed})
    if changes:
        record_registry_mutation(connection)
    return changes


def record_registry_mutation(connection: psycopg.Connection) -> None:
    """Bump the revision under the caller's authority and singleton row locks."""
    now = utc_now()
    cursor = connection.execute(
        "UPDATE edge_site SET registry_version=registry_version+1,"
        "topology_dirty_registry_version=registry_version+1,"
        "topology_dirty_created_at=%s,updated_at=%s WHERE id=1",
        (now, now),
    )
    if cursor.rowcount != 1:
        raise CameraRegistryNotInitialized()


def camera_from_row(row: Mapping[str, object]) -> dict[str, object]:
    camera_id = str(row["camera_id"])
    floor = (
        None
        if row["floor_override"] is None
        else parse_legacy_floor(str(row["floor_override"]), camera_id=camera_id)
    )
    return {
        "id": camera_id,
        "label": str(row["label"]),
        "rtsp_url": str(row["rtsp_url"]),
        "space_id": _text(row["space_id"]),
        "backend_camera_id": _text(row["backend_camera_id"]),
        "mapping_pending": row["mapping_state"] == "PENDING",
        "status": "unknown",
        "decode_backend": _text(row["decode_backend"]),
        "floor": floor,
        "created_at": str(row["created_at"]),
        "last_probed_at": _text(row["last_probed_at"]),
        "last_ok_at": _text(row["last_ok_at"]),
        "never_connected": bool(row["never_connected"]),
        "edge_ref": _text(row["edge_ref"]),
        "room_edge_ref": _text(row["room_location_id"]),
    }


def _text(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "CameraRegistryNotInitialized",
    "CameraRegistryWriteError",
    "find_duplicate",
    "get_camera",
    "lock_registry",
    "migrate_legacy_floors",
    "read_registry",
    "record_registry_mutation",
    "utc_now",
]
