from __future__ import annotations

import sqlite3
from enum import StrEnum, unique

from backend.app.features.cameras.topology_query import (
    RegistryTopologySnapshot,
    TopologyDirtyMarker,
    read_topology_snapshot,
)
from contracts.edge_provisioning_validation import require_edge_ref, require_uuid


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
        try:
            connection.execute(
                "INSERT INTO camera_topology_floors (edge_ref, name, order_index) VALUES (?, ?, ?)",
                (parsed_ref, name, order_index),
            )
        except sqlite3.IntegrityError as error:
            raise TopologyConflictError(TopologyErrorCode.DUPLICATE_REF, parsed_ref) from error

    def update_floor(
        self, connection: sqlite3.Connection, edge_ref: str, *, name: str, order_index: int
    ) -> bool:
        parsed_ref = _edge_ref(edge_ref)
        cursor = connection.execute(
            "UPDATE camera_topology_floors SET name = ?, order_index = ? WHERE edge_ref = ?",
            (name, order_index, parsed_ref),
        )
        return cursor.rowcount > 0

    def delete_floor(self, connection: sqlite3.Connection, edge_ref: str) -> bool:
        parsed_ref = _edge_ref(edge_ref)
        try:
            cursor = connection.execute(
                "DELETE FROM camera_topology_floors WHERE edge_ref = ?", (parsed_ref,)
            )
        except sqlite3.IntegrityError as error:
            raise TopologyConflictError(TopologyErrorCode.ROOM_OCCUPIED, parsed_ref) from error
        return cursor.rowcount > 0

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
        parsed_floor_ref = _edge_ref(floor_edge_ref)
        legacy_id = _legacy_id(legacy_canonical_space_id, parsed_ref)
        if not _exists(connection, "camera_topology_floors", parsed_floor_ref):
            raise TopologyConflictError(TopologyErrorCode.MISSING_PARENT, parsed_floor_ref)
        try:
            connection.execute(
                "INSERT INTO camera_topology_rooms "
                "(edge_ref, floor_edge_ref, name, room_type, capacity, legacy_canonical_space_id) "
                "VALUES (?, ?, ?, 'ROOM', 1, ?)",
                (parsed_ref, parsed_floor_ref, name, legacy_id),
            )
        except sqlite3.IntegrityError as error:
            raise TopologyConflictError(TopologyErrorCode.DUPLICATE_REF, parsed_ref) from error

    def update_room(
        self, connection: sqlite3.Connection, edge_ref: str, *, name: str
    ) -> bool:
        parsed_ref = _edge_ref(edge_ref)
        cursor = connection.execute(
            "UPDATE camera_topology_rooms SET name = ? WHERE edge_ref = ?", (name, parsed_ref)
        )
        return cursor.rowcount > 0

    def delete_room(self, connection: sqlite3.Connection, edge_ref: str) -> bool:
        parsed_ref = _edge_ref(edge_ref)
        try:
            cursor = connection.execute(
                "DELETE FROM camera_topology_rooms WHERE edge_ref = ?", (parsed_ref,)
            )
        except sqlite3.IntegrityError as error:
            raise TopologyConflictError(TopologyErrorCode.ROOM_OCCUPIED, parsed_ref) from error
        return cursor.rowcount > 0

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
        parsed_room_ref = _edge_ref(room_edge_ref)
        if not _exists(connection, "camera_topology_rooms", parsed_room_ref):
            raise TopologyConflictError(TopologyErrorCode.MISSING_PARENT, parsed_room_ref)
        try:
            connection.execute(
                "INSERT INTO camera_topology_cameras (camera_id, edge_ref, room_edge_ref) "
                "VALUES (?, ?, ?)",
                (camera_id, parsed_ref, parsed_room_ref),
            )
        except sqlite3.IntegrityError as error:
            code = (
                TopologyErrorCode.ROOM_OCCUPIED
                if _room_occupied(connection, parsed_room_ref)
                else TopologyErrorCode.DUPLICATE_REF
            )
            raise TopologyConflictError(code, parsed_ref) from error

    def delete_camera(self, connection: sqlite3.Connection, camera_id: str) -> None:
        connection.execute("DELETE FROM camera_topology_cameras WHERE camera_id = ?", (camera_id,))

    def snapshot(
        self, connection: sqlite3.Connection, *, registry_version: int, camera_ids: tuple[str, ...]
    ) -> RegistryTopologySnapshot:
        return read_topology_snapshot(
            connection, registry_version=registry_version, camera_ids=camera_ids
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
        return require_uuid(value)
    except Exception as error:
        raise TopologyConflictError(TopologyErrorCode.INVALID_LEGACY_SPACE_ID, edge_ref) from error


def _exists(connection: sqlite3.Connection, table: str, edge_ref: str) -> bool:
    row = connection.execute(
        f"SELECT 1 FROM {table} WHERE edge_ref = ?", (edge_ref,)
    ).fetchone()
    return row is not None


def _room_occupied(connection: sqlite3.Connection, room_edge_ref: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM camera_topology_cameras WHERE room_edge_ref = ?", (room_edge_ref,)
    ).fetchone() is not None


__all__ = [
    "CameraTopologyStore",
    "RegistryTopologySnapshot",
    "TopologyConflictError",
    "TopologyDirtyMarker",
    "TopologyErrorCode",
]
