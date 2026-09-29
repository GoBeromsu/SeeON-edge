from __future__ import annotations

from enum import StrEnum, unique

import psycopg
from psycopg.rows import dict_row

from backend.app.features.cameras.camera_repository import CameraRegistryNotInitialized, utc_now
from backend.app.features.cameras.topology_query import (
    RegistryTopologySnapshot,
    TopologyDirtyMarker,
)
from contracts.edge_provisioning_models import (
    EdgeErrorCode,
    TopologyCamera,
    TopologyFloor,
    TopologyRoom,
)
from contracts.edge_provisioning_validation import require_canonical_id, require_edge_ref

# All projections share one statement snapshot under READ COMMITTED.
# Project only topology fields: credentials never enter the cloud projection.
_SNAPSHOT_SQL = """
SELECT s.registry_version,s.topology_dirty_registry_version,s.topology_dirty_created_at,
    COALESCE((
        SELECT jsonb_agg(to_jsonb(l) ORDER BY l.location_id COLLATE "C",l.kind COLLATE "C")
        FROM (
            SELECT location_id,kind,parent_location_id,name,order_index,capacity,legacy_space_id
            FROM locations
        ) AS l
    ), '[]'::jsonb) AS locations,
    COALESCE((
        SELECT jsonb_agg(to_jsonb(c) ORDER BY c.camera_id COLLATE "C")
        FROM (
            SELECT camera_id,edge_ref,room_location_id,label FROM cameras
        ) AS c
    ), '[]'::jsonb) AS cameras
FROM edge_site AS s WHERE s.id=1
"""


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
    """Borrow an active transaction; mutations require the caller's registry lock."""

    def create_floor(
        self, connection: psycopg.Connection, *, edge_ref: str, name: str, order_index: int
    ) -> None:
        parsed_ref = _edge_ref(edge_ref)
        now = utc_now()
        try:
            connection.execute(
                "INSERT INTO locations(location_id,kind,name,order_index,created_at,updated_at) "
                "VALUES (%s,'FLOOR',%s,%s,%s,%s)",
                (parsed_ref, name, order_index, now, now),
            )
        except psycopg.IntegrityError:
            raise TopologyConflictError(TopologyErrorCode.DUPLICATE_REF, parsed_ref) from None

    def update_floor(
        self, connection: psycopg.Connection, edge_ref: str, *, name: str, order_index: int
    ) -> bool:
        cursor = connection.execute(
            "UPDATE locations SET name=%s,order_index=%s,updated_at=%s "
            "WHERE location_id=%s AND kind='FLOOR'",
            (name, order_index, utc_now(), _edge_ref(edge_ref)),
        )
        return cursor.rowcount > 0

    def delete_floor(self, connection: psycopg.Connection, edge_ref: str) -> bool:
        return _delete_location(connection, _edge_ref(edge_ref), "FLOOR")

    def create_room(
        self,
        connection: psycopg.Connection,
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
                "VALUES (%s,'ROOM',%s,'FLOOR',%s,0,1,%s,%s,%s)",
                (
                    parsed_ref,
                    floor_ref,
                    name,
                    _legacy_id(legacy_canonical_space_id, parsed_ref),
                    now,
                    now,
                ),
            )
        except psycopg.IntegrityError:
            raise TopologyConflictError(TopologyErrorCode.DUPLICATE_REF, parsed_ref) from None

    def update_room(self, connection: psycopg.Connection, edge_ref: str, *, name: str) -> bool:
        cursor = connection.execute(
            "UPDATE locations SET name=%s,updated_at=%s WHERE location_id=%s AND kind='ROOM'",
            (name, utc_now(), _edge_ref(edge_ref)),
        )
        return cursor.rowcount > 0

    def delete_room(self, connection: psycopg.Connection, edge_ref: str) -> bool:
        return _delete_location(connection, _edge_ref(edge_ref), "ROOM")

    def bind_camera(
        self,
        connection: psycopg.Connection,
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
                "UPDATE cameras SET edge_ref=%s,room_location_id=%s,room_location_kind='ROOM',"
                "updated_at=%s WHERE camera_id=%s",
                (parsed_ref, room_ref, utc_now(), camera_id),
            )
            if cursor.rowcount != 1:
                raise TopologyConflictError(TopologyErrorCode.INVALID_BINDING, camera_id)
        except psycopg.IntegrityError as error:
            # A failed statement aborts the transaction. Do not query it again.
            code = (
                TopologyErrorCode.ROOM_OCCUPIED
                if error.diag.constraint_name == "cameras_one_room_idx"
                else TopologyErrorCode.DUPLICATE_REF
            )
            raise TopologyConflictError(code, parsed_ref) from None

    def delete_camera(self, connection: psycopg.Connection, camera_id: str) -> None:
        connection.execute(
            "UPDATE cameras SET edge_ref=NULL,room_location_id=NULL,room_location_kind=NULL "
            "WHERE camera_id=%s",
            (camera_id,),
        )

    def snapshot(self, connection: psycopg.Connection) -> RegistryTopologySnapshot:
        with connection.cursor(row_factory=dict_row) as cursor:
            row = cursor.execute(_SNAPSHOT_SQL).fetchone()
        if row is None:
            raise CameraRegistryNotInitialized()
        cameras_by_room = {
            str(camera["room_location_id"]): TopologyCamera(
                edge_ref=str(camera["edge_ref"]), label=str(camera["label"])
            )
            for camera in row["cameras"]
            if camera["edge_ref"] is not None
        }
        rooms_by_floor: dict[str, list[TopologyRoom]] = {}
        for location in row["locations"]:
            if location["kind"] != "ROOM":
                continue
            room_ref = str(location["location_id"])
            camera = cameras_by_room.get(room_ref)
            rooms_by_floor.setdefault(str(location["parent_location_id"]), []).append(
                TopologyRoom(
                    room_ref,
                    str(location["name"]),
                    "ROOM",
                    int(location["capacity"]),
                    () if camera is None else (camera,),
                    location["legacy_space_id"],
                )
            )
        floors = tuple(
            TopologyFloor(
                str(location["location_id"]),
                str(location["name"]),
                int(location["order_index"]),
                tuple(rooms_by_floor.get(str(location["location_id"]), ())),
            )
            for location in row["locations"]
            if location["kind"] == "FLOOR"
        )
        dirty = (
            None
            if row["topology_dirty_registry_version"] is None
            else TopologyDirtyMarker(
                int(row["topology_dirty_registry_version"]),
                str(row["topology_dirty_created_at"]),
            )
        )
        unmapped = tuple(
            sorted(
                str(camera["camera_id"]) for camera in row["cameras"] if camera["edge_ref"] is None
            )
        )
        readiness = EdgeErrorCode.LEGACY_MAPPING_REQUIRED if unmapped else None
        return RegistryTopologySnapshot(
            int(row["registry_version"]), floors, dirty, readiness, unmapped
        )


def _delete_location(connection: psycopg.Connection, edge_ref: str, kind: str) -> bool:
    try:
        cursor = connection.execute(
            "DELETE FROM locations WHERE location_id=%s AND kind=%s", (edge_ref, kind)
        )
    except psycopg.IntegrityError:
        raise TopologyConflictError(TopologyErrorCode.ROOM_OCCUPIED, edge_ref) from None
    return cursor.rowcount > 0


def _location_exists(connection: psycopg.Connection, edge_ref: str, kind: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM locations WHERE location_id=%s AND kind=%s", (edge_ref, kind)
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
