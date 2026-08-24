"""Schema-18 camera registry and location authority."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.features.cameras.camera_values import (
    DEFAULT_FLOOR,
    FLOOR_MAX,
    FLOOR_MIN,
    FLOOR_VALUES,
    CameraRegistryData,
    CameraStatus,
    DuplicateCameraError,
    ProbeErrorClass,
    ProbeResult,
    floor_label,
    is_valid_floor,
    mask_rtsp_url,
    normalize_stream_identity,
    parse_legacy_floor,
    public_camera,
    registry_expected_cameras,
    status_from_probe,
)
from backend.app.features.cameras.topology import CameraTopologyStore, RegistryTopologySnapshot
from backend.app.edge_db.configuration import (
    ensure_edge_site,
    open_configuration_database,
    utc_now,
)

utc_now_iso = utc_now


class CameraRegistryStore:
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
            return self._read_unlocked()

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
        with self._lock, self._transaction_unlocked() as connection:
            duplicate = self._duplicate(connection, rtsp_url)
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
            self._record_mutation(connection)
            record = self._get_unlocked(connection, identifier)
            if record is None:
                raise sqlite3.DatabaseError("camera insert returned no row")
            self._statuses[identifier] = status
            record["status"] = status
            return record

    def update(self, camera_id: str, updates: dict[str, object]) -> dict[str, object] | None:
        with self._lock, self._transaction_unlocked() as connection:
            current = self._get_unlocked(connection, camera_id)
            if current is None:
                return None
            rtsp_url = updates.get("rtsp_url")
            if isinstance(rtsp_url, str):
                duplicate = self._duplicate(connection, rtsp_url, exclude_camera_id=camera_id)
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
            self._record_mutation(connection)
            updated = self._get_unlocked(connection, camera_id)
            status = updates.get("status")
            if status in {"online", "offline", "starting", "unknown"}:
                self._statuses[camera_id] = status
            if updated is not None:
                updated["status"] = self._statuses.get(camera_id, "unknown")
            return updated

    def delete(self, camera_id: str) -> bool:
        with self._lock, self._transaction_unlocked() as connection:
            cursor = connection.execute("DELETE FROM cameras WHERE camera_id=?", (camera_id,))
            if cursor.rowcount == 0:
                return False
            self._statuses.pop(camera_id, None)
            self._record_mutation(connection)
            return True

    def get(self, camera_id: str) -> dict[str, object] | None:
        with self._lock:
            return self._get_unlocked(self._connection, camera_id)

    def create_floor(self, *, edge_ref: str, name: str, order_index: int) -> None:
        with self._lock, self._transaction_unlocked() as connection:
            self._topology.create_floor(
                connection, edge_ref=edge_ref, name=name, order_index=order_index
            )
            self._record_mutation(connection)

    def update_floor(self, edge_ref: str, *, name: str, order_index: int) -> bool:
        with self._lock, self._transaction_unlocked() as connection:
            changed = self._topology.update_floor(
                connection, edge_ref, name=name, order_index=order_index
            )
            if changed:
                self._record_mutation(connection)
            return changed

    def delete_floor(self, edge_ref: str) -> bool:
        return self._location_mutation(self._topology.delete_floor, edge_ref)

    def create_room(
        self,
        *,
        edge_ref: str,
        floor_edge_ref: str,
        name: str,
        legacy_canonical_space_id: str | None = None,
    ) -> None:
        with self._lock, self._transaction_unlocked() as connection:
            self._topology.create_room(
                connection,
                edge_ref=edge_ref,
                floor_edge_ref=floor_edge_ref,
                name=name,
                legacy_canonical_space_id=legacy_canonical_space_id,
            )
            self._record_mutation(connection)

    def update_room(self, edge_ref: str, *, name: str) -> bool:
        with self._lock, self._transaction_unlocked() as connection:
            changed = self._topology.update_room(connection, edge_ref, name=name)
            if changed:
                self._record_mutation(connection)
            return changed

    def delete_room(self, edge_ref: str) -> bool:
        return self._location_mutation(self._topology.delete_room, edge_ref)

    def topology_snapshot(self) -> RegistryTopologySnapshot:
        with self._lock:
            data = self._read_unlocked()
            return self._topology.snapshot(
                self._connection,
                registry_version=data["registry_version"],
                camera_ids=tuple(str(record["id"]) for record in data["cameras"]),
            )

    def migrate_legacy_string_floors(self) -> list[dict[str, object]]:
        return []

    def _location_mutation(self, operation, edge_ref: str) -> bool:
        with self._lock, self._transaction_unlocked() as connection:
            changed = operation(connection, edge_ref)
            if changed:
                self._record_mutation(connection)
            return changed

    def _read_unlocked(self) -> CameraRegistryData:
        version_row = self._connection.execute(
            "SELECT registry_version FROM edge_site WHERE id=1"
        ).fetchone()
        rows = self._connection.execute(_CAMERA_SELECT + " ORDER BY camera_id").fetchall()
        cameras = [_camera_row(row) for row in rows]
        for camera in cameras:
            camera["status"] = self._statuses.get(str(camera["id"]), "unknown")
        return {
            "registry_version": 0 if version_row is None else int(version_row[0]),
            "cameras": cameras,
        }

    def _get_unlocked(
        self, connection: sqlite3.Connection, camera_id: str
    ) -> dict[str, object] | None:
        row = connection.execute(_CAMERA_SELECT + " WHERE camera_id=?", (camera_id,)).fetchone()
        if row is None:
            return None
        record = _camera_row(row)
        record["status"] = self._statuses.get(camera_id, "unknown")
        return record

    def _duplicate(
        self, connection: sqlite3.Connection, rtsp_url: str, *, exclude_camera_id: str | None = None
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
        return None if row is None else _camera_row(row)

    def _record_mutation(self, connection: sqlite3.Connection) -> None:
        ensure_edge_site(connection)
        now = utc_now()
        connection.execute(
            "UPDATE edge_site SET registry_version=registry_version+1,"
            "topology_dirty_registry_version=registry_version+1,"
            "topology_dirty_created_at=?,updated_at=? WHERE id=1",
            (now, now),
        )

    @contextmanager
    def _transaction_unlocked(self) -> Generator[sqlite3.Connection]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")


_CAMERA_SELECT = (
    "SELECT camera_id,label,rtsp_url,space_id,backend_camera_id,mapping_state,"
    "decode_backend,floor_override,created_at,last_probed_at,last_ok_at,never_connected,"
    "edge_ref,room_location_id FROM cameras"
)


def _camera_row(row: tuple[object, ...]) -> dict[str, object]:
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
    "CameraRegistryStore",
    "CameraStatus",
    "DEFAULT_FLOOR",
    "DuplicateCameraError",
    "FLOOR_MAX",
    "FLOOR_MIN",
    "FLOOR_VALUES",
    "ProbeErrorClass",
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
