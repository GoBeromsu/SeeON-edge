"""Schema-18 camera registry and location authority."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from threading import Lock

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.configuration import open_configuration_database, utc_now
from backend.app.features.cameras import camera_values
from backend.app.features.cameras.camera_repository import (
    camera_transaction,
    find_duplicate,
    get_camera,
    read_registry,
    record_registry_mutation,
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
    """Relational camera registry backed only by ``edge_site`` and ``cameras``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._connection = open_configuration_database(self.path)
        self._topology = CameraTopologyStore()
        self._statuses: dict[str, CameraStatus] = {}

    @classmethod
    def from_env(cls) -> CameraRegistryStore:
        return cls(EDGE_DATABASE_PATH)

    def snapshot(self) -> CameraRegistryData:
        with self._lock:
            return read_registry(self._connection, self._statuses)

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
    ) -> dict[str, object]:
        with self._lock, camera_transaction(self._connection) as connection:
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
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)",
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
            record = get_camera(connection, identifier, self._statuses)
            if record is None:
                raise sqlite3.DatabaseError("camera insert returned no row")
            self._statuses[identifier] = status
            record["status"] = status
            return record

    def update(self, camera_id: str, updates: dict[str, object]) -> dict[str, object] | None:
        with self._lock, camera_transaction(self._connection) as connection:
            current = get_camera(connection, camera_id, self._statuses)
            if current is None:
                return None
            rtsp_url = updates.get("rtsp_url")
            if isinstance(rtsp_url, str):
                duplicate = find_duplicate(connection, rtsp_url, exclude_camera_id=camera_id)
                if duplicate is not None:
                    raise DuplicateCameraError(duplicate)
            values = {**current, **updates}
            backend_id = _text(values.get("backend_camera_id"))
            pending = values.get("mapping_pending") is True
            mapping_state = (
                "MAPPED" if backend_id is not None else ("PENDING" if pending else "UNMAPPED")
            )
            effective_rtsp = str(values["rtsp_url"])
            connection.execute(
                "UPDATE cameras SET backend_camera_id=?,label=?,rtsp_url=?,"
                "normalized_stream_identity=?,space_id=?,mapping_state=?,decode_backend=?,"
                "floor_override=?,never_connected=?,last_probed_at=?,last_ok_at=?,"
                "revision=revision+1,updated_at=? WHERE camera_id=?",
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
            updated = get_camera(connection, camera_id, self._statuses)
            status = updates.get("status")
            if status in {"online", "offline", "starting", "unknown"}:
                self._statuses[camera_id] = status
            if updated is not None:
                updated["status"] = self._statuses.get(camera_id, "unknown")
            return updated

    def delete(self, camera_id: str) -> bool:
        with self._lock, camera_transaction(self._connection) as connection:
            cursor = connection.execute("DELETE FROM cameras WHERE camera_id=?", (camera_id,))
            if cursor.rowcount == 0:
                return False
            self._statuses.pop(camera_id, None)
            record_registry_mutation(connection)
            return True

    def get(self, camera_id: str) -> dict[str, object] | None:
        with self._lock:
            return get_camera(self._connection, camera_id, self._statuses)


__all__ = [
    "CameraRegistryStore",
    "CameraStatus",
    "DEFAULT_FLOOR",
    "DuplicateCameraError",
    "FLOOR_MAX",
    "FLOOR_MIN",
    "FLOOR_VALUES",
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
