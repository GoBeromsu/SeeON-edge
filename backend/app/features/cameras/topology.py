from __future__ import annotations

import sqlite3
from enum import StrEnum, unique

from backend.app.edge_db.configuration import utc_now
from backend.app.features.cameras.topology_query import (
    RegistryTopologySnapshot,
    TopologyDirtyMarker,
    read_topology_snapshot,
)
from contracts.edge_provisioning_validation import require_canonical_id, require_edge_ref


@unique
class TopologyErrorCode(StrEnum):
    DUPLICATE_REF = "DUPLICATE_REF"
    MISSING_PARENT = "MISSING_PARENT"
    ROOM_OCCUPIED = "ROOM_OCCUPIED"
    INVALID_LEGACY_SPACE_ID = "INVALID_LEGACY_SPACE_ID"
    INVALID_BINDING = "INVALID_BINDING"


class TopologyConflictError(Exception):
    __slots__ = ("code", "edge_ref")

    def __init__(self, code: TopologyErrorCode, edge_ref: str) -> None:
        super().__init__(code, edge_ref)
        self.code = code
        self.edge_ref = edge_ref

    def __str__(self) -> str:
        return f"{self.code}: {self.edge_ref}"


class CameraTopologyStore:
    def create_floor(
        self, connection: sqlite3.Connection, *, edge_ref: str, name: str, order_index: int
    ) -> None:
        parsed_ref = _edge_ref(edge_ref)
        now = utc_now()
        try:
            connection.execute(
                "INSERT INTO locations(location_id,kind,name,order_index,created_at,updated_at) "
                "VALUES (?,'FLOOR',?,?,?,?)",
                (parsed_ref, name, order_index, now, now),
            )
        except sqlite3.IntegrityError as error:
            raise TopologyConflictError(TopologyErrorCode.DUPLICATE_REF, parsed_ref) from error

    def update_floor(
        self, connection: sqlite3.Connection, edge_ref: str, *, name: str, order_index: int
    ) -> bool:
        cursor = connection.execute(
            "UPDATE locations SET name=?,order_index=?,updated_at=? "
            "WHERE location_id=? AND kind='FLOOR'",
            (name, order_index, utc_now(), _edge_ref(edge_ref)),
        )
        return cursor.rowcount > 0

    def delete_floor(self, connection: sqlite3.Connection, edge_ref: str) -> bool:
        return _delete_location(connection, _edge_ref(edge_ref), "FLOOR")

    def create_room(
        self,
        connection: sqlite3.Connection,
        *,
        edge_ref: str,
        floor_edge_ref: str,
        name: str,
        legacy_canonical_space_id: str | None,
    ) -> None:
        parsed_ref = _edge_ref(edge_ref)
        floor_ref = _edge_ref(floor_edge_ref)
        if not _location_exists(connection, floor_ref, "FLOOR"):
            raise TopologyConflictError(TopologyErrorCode.MISSING_PARENT, floor_ref)
        now = utc_now()
        try:
            connection.execute(
                "INSERT INTO locations(location_id,kind,parent_location_id,parent_kind,name,"
                "order_index,capacity,legacy_space_id,created_at,updated_at) "
                "VALUES (?,'ROOM',?,'FLOOR',?,0,1,?,?,?)",
                (
                    parsed_ref,
                    floor_ref,
                    name,
                    _legacy_id(legacy_canonical_space_id, parsed_ref),
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise TopologyConflictError(TopologyErrorCode.DUPLICATE_REF, parsed_ref) from error

    def update_room(self, connection: sqlite3.Connection, edge_ref: str, *, name: str) -> bool:
        cursor = connection.execute(
            "UPDATE locations SET name=?,updated_at=? WHERE location_id=? AND kind='ROOM'",
            (name, utc_now(), _edge_ref(edge_ref)),
        )
        return cursor.rowcount > 0

    def delete_room(self, connection: sqlite3.Connection, edge_ref: str) -> bool:
        return _delete_location(connection, _edge_ref(edge_ref), "ROOM")

    def bind_camera(
        self,
        connection: sqlite3.Connection,
        *,
        camera_id: str,
        edge_ref: str | None,
        room_edge_ref: str | None,
    ) -> None:
        if edge_ref is None and room_edge_ref is None:
            return
        if edge_ref is None or room_edge_ref is None:
            raise TopologyConflictError(TopologyErrorCode.INVALID_BINDING, camera_id)
        parsed_ref = _edge_ref(edge_ref)
        room_ref = _edge_ref(room_edge_ref)
        if not _location_exists(connection, room_ref, "ROOM"):
            raise TopologyConflictError(TopologyErrorCode.MISSING_PARENT, room_ref)
        try:
            cursor = connection.execute(
                "UPDATE cameras SET edge_ref=?,room_location_id=?,room_location_kind='ROOM',"
                "updated_at=? WHERE camera_id=?",
                (parsed_ref, room_ref, utc_now(), camera_id),
            )
            if cursor.rowcount != 1:
                raise TopologyConflictError(TopologyErrorCode.INVALID_BINDING, camera_id)
        except sqlite3.IntegrityError as error:
            occupied = connection.execute(
                "SELECT 1 FROM cameras WHERE room_location_id=? AND camera_id<>?",
                (room_ref, camera_id),
            ).fetchone()
            code = (
                TopologyErrorCode.ROOM_OCCUPIED
                if occupied is not None
                else TopologyErrorCode.DUPLICATE_REF
            )
            raise TopologyConflictError(code, parsed_ref) from error

    def delete_camera(self, connection: sqlite3.Connection, camera_id: str) -> None:
        connection.execute(
            "UPDATE cameras SET edge_ref=NULL,room_location_id=NULL,room_location_kind=NULL "
            "WHERE camera_id=?",
            (camera_id,),
        )

    def snapshot(
        self, connection: sqlite3.Connection, *, registry_version: int, camera_ids: tuple[str, ...]
    ) -> RegistryTopologySnapshot:
        return read_topology_snapshot(
            connection, registry_version=registry_version, camera_ids=camera_ids
        )


def _delete_location(connection: sqlite3.Connection, edge_ref: str, kind: str) -> bool:
    try:
        cursor = connection.execute(
            "DELETE FROM locations WHERE location_id=? AND kind=?", (edge_ref, kind)
        )
    except sqlite3.IntegrityError as error:
        raise TopologyConflictError(TopologyErrorCode.ROOM_OCCUPIED, edge_ref) from error
    return cursor.rowcount > 0


def _location_exists(connection: sqlite3.Connection, edge_ref: str, kind: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM locations WHERE location_id=? AND kind=?", (edge_ref, kind)
        ).fetchone()
        is not None
    )


def _edge_ref(value: str) -> str:
    try:
        return require_edge_ref(value)
    except Exception as error:
        raise TopologyConflictError(TopologyErrorCode.INVALID_BINDING, value) from error


def _legacy_id(value: str | None, edge_ref: str) -> str | None:
    if value is None:
        return None
    try:
        return require_canonical_id(value)
    except Exception as error:
        raise TopologyConflictError(TopologyErrorCode.INVALID_LEGACY_SPACE_ID, edge_ref) from error


__all__ = [
    "CameraTopologyStore",
    "RegistryTopologySnapshot",
    "TopologyConflictError",
    "TopologyDirtyMarker",
    "TopologyErrorCode",
]
