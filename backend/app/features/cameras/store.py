"""Camera registry and location authority on the API-owned PostgreSQL pool."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from threading import Lock
from typing import cast

import psycopg

from backend.app.edge_db.authority import AuthorityToken
from backend.app.edge_db.postgres import PostgresDatabase, PostgresError
from backend.app.features.cameras import camera_values
from backend.app.features.cameras.camera_repository import (
    find_duplicate,
    get_camera,
    read_registry,
    record_registry_mutation,
    utc_now,
)
from backend.app.features.cameras.camera_values import (
    CameraRegistryData,
    CameraStatus,
    DuplicateCameraError,
    normalize_stream_identity,
)
from backend.app.features.cameras.location_operations import CameraLocationOperations
from backend.app.features.cameras.topology import CameraTopologyStore

DEFAULT_FLOOR = camera_values.DEFAULT_FLOOR
FLOOR_MAX = camera_values.FLOOR_MAX
FLOOR_MIN = camera_values.FLOOR_MIN
FLOOR_VALUES = camera_values.FLOOR_VALUES
ProbeErrorClass = camera_values.ProbeErrorClass
ProbeResult = camera_values.ProbeResult
floor_label = camera_values.floor_label
is_valid_floor = camera_values.is_valid_floor
mask_rtsp_url = camera_values.mask_rtsp_url
parse_legacy_floor = camera_values.parse_legacy_floor
public_camera = camera_values.public_camera
registry_expected_cameras = camera_values.registry_expected_cameras
status_from_probe = camera_values.status_from_probe
utc_now_iso = utc_now


def _text(value: object) -> str | None:
    return None if value is None else str(value)


class CameraRegistryStore(CameraLocationOperations):
    """Borrow transactions; own only process-local status, never a connection."""

    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        self.database = database
        self.authority = authority
        # This lock orders local status publication, not database clients.
        self._lock = Lock()
        self._topology = CameraTopologyStore()
        self._statuses: dict[str, tuple[uuid.UUID, CameraStatus]] = {}

    def snapshot(self) -> CameraRegistryData:
        with self._lock:
            return self.database.read(lambda connection: read_registry(connection, self._statuses))

    def create(
        self,
        *,
        camera_id: str | None = None,
        label: str,
        rtsp_url: str,
        space_id: str | None,
        status: CameraStatus,
        backend_camera_id: str | None = None,
        mapping_pending: bool = False,
        decode_backend: str | None = None,
        floor: int | None = None,
        last_probed_at: str | None = None,
        last_ok_at: str | None = None,
        never_connected: bool = True,
        edge_ref: str | None = None,
        room_edge_ref: str | None = None,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> dict[str, object]:
        def persist(connection: psycopg.Connection) -> tuple[uuid.UUID, dict[str, object]]:
            duplicate = find_duplicate(connection, rtsp_url)
            if duplicate is not None:
                raise DuplicateCameraError(duplicate)
            identifier = camera_id or str(uuid.uuid4())
            now = utc_now()
            mapping_state = (
                "MAPPED"
                if backend_camera_id is not None
                else ("PENDING" if mapping_pending else "UNMAPPED")
            )
            connection.execute(
                "INSERT INTO cameras(camera_id,backend_camera_id,label,rtsp_url,"
                "normalized_stream_identity,space_id,mapping_state,decode_backend,floor_override,"
                "never_connected,last_probed_at,last_ok_at,revision,created_at,updated_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,%s,%s)",
                (
                    identifier,
                    backend_camera_id,
                    label,
                    rtsp_url,
                    normalize_stream_identity(rtsp_url),
                    space_id,
                    mapping_state,
                    decode_backend,
                    None if floor is None else str(floor),
                    int(never_connected),
                    last_probed_at,
                    last_ok_at,
                    now,
                    now,
                ),
            )
            self._topology.bind_camera(
                connection,
                camera_id=identifier,
                edge_ref=edge_ref,
                room_edge_ref=room_edge_ref,
            )
            record_registry_mutation(connection)
            if after_write is not None:
                after_write(connection)
            candidate = get_camera(connection, identifier, self._statuses)
            if candidate is None:
                raise PostgresError("camera insert returned no row")
            incarnation, record = candidate
            record["status"] = status
            return incarnation, record

        with self._lock:
            incarnation, record = self._mutate(persist)
            self._statuses[str(record["id"])] = (incarnation, status)
            return record

    def update(
        self,
        camera_id: str,
        updates: dict[str, object],
        *,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> dict[str, object] | None:
        def persist(
            connection: psycopg.Connection,
        ) -> tuple[uuid.UUID, dict[str, object]] | None:
            current = get_camera(connection, camera_id, self._statuses)
            if current is None:
                return None
            _, current_record = current
            rtsp_url = updates.get("rtsp_url")
            if isinstance(rtsp_url, str):
                duplicate = find_duplicate(connection, rtsp_url, exclude_camera_id=camera_id)
                if duplicate is not None:
                    raise DuplicateCameraError(duplicate)
            values = {**current_record, **updates}
            backend_id = _text(values.get("backend_camera_id"))
            pending = values.get("mapping_pending") is True
            mapping_state = (
                "MAPPED" if backend_id is not None else ("PENDING" if pending else "UNMAPPED")
            )
            effective_rtsp = str(values["rtsp_url"])
            connection.execute(
                "UPDATE cameras SET backend_camera_id=%s,label=%s,rtsp_url=%s,"
                "normalized_stream_identity=%s,space_id=%s,mapping_state=%s,decode_backend=%s,"
                "floor_override=%s,never_connected=%s,last_probed_at=%s,last_ok_at=%s,"
                "revision=revision+1,updated_at=%s WHERE camera_id=%s",
                (
                    backend_id,
                    str(values["label"]),
                    effective_rtsp,
                    normalize_stream_identity(effective_rtsp),
                    _text(values.get("space_id")),
                    mapping_state,
                    _text(values.get("decode_backend")),
                    None if values.get("floor") is None else str(values["floor"]),
                    int(values.get("never_connected") is not False),
                    _text(values.get("last_probed_at")),
                    _text(values.get("last_ok_at")),
                    utc_now(),
                    camera_id,
                ),
            )
            if "edge_ref" in updates or "room_edge_ref" in updates:
                self._topology.delete_camera(connection, camera_id)
                self._topology.bind_camera(
                    connection,
                    camera_id=camera_id,
                    edge_ref=_text(values.get("edge_ref")),
                    room_edge_ref=_text(values.get("room_edge_ref")),
                )
            record_registry_mutation(connection)
            if after_write is not None:
                after_write(connection)
            updated = get_camera(connection, camera_id, self._statuses)
            if updated is None:
                raise PostgresError("camera update returned no row")
            incarnation, record = updated
            status = updates.get("status")
            if status in {"online", "offline", "starting", "unknown"}:
                record["status"] = status
            return incarnation, record

        with self._lock:
            updated = self._mutate(persist)
            if updated is None:
                return None
            incarnation, record = updated
            self._statuses[camera_id] = (incarnation, cast(CameraStatus, record["status"]))
            return record

    def delete(
        self,
        camera_id: str,
        *,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> bool:
        def persist(connection: psycopg.Connection) -> bool:
            cursor = connection.execute("DELETE FROM cameras WHERE camera_id=%s", (camera_id,))
            if cursor.rowcount == 0:
                return False
            record_registry_mutation(connection)
            if after_write is not None:
                after_write(connection)
            return True

        with self._lock:
            changed = self._mutate(persist)
            if changed:
                self._statuses.pop(camera_id, None)
            return changed

    def get(self, camera_id: str) -> dict[str, object] | None:
        with self._lock:
            result = self.database.read(
                lambda connection: get_camera(connection, camera_id, self._statuses)
            )
            return None if result is None else result[1]


__all__ = [
    "DEFAULT_FLOOR",
    "FLOOR_MAX",
    "FLOOR_MIN",
    "FLOOR_VALUES",
    "CameraRegistryStore",
    "CameraStatus",
    "DuplicateCameraError",
    "ProbeResult",
    "floor_label",
    "is_valid_floor",
    "mask_rtsp_url",
    "normalize_stream_identity",
    "parse_legacy_floor",
    "public_camera",
    "registry_expected_cameras",
    "status_from_probe",
    "utc_now_iso",
]
