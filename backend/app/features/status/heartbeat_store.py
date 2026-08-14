"""API-owned per-camera heartbeat liveness store.

The worker relays liveness facts to ml-api ``/relay/heartbeat``. ml-api records
the local ``received_at`` per camera right after relay-token auth -- before
registry binding is resolved and before any backend egress -- so ``/status``
reflects edge-local truth that is independent both of backend reachability and
of whether this camera has been onboarded onto the central backend's own roster
yet (see #183, #202).

This is NOT cross-process shared state with the worker: it is an api-local
snapshot built from relayed facts (one owner, fed by HTTP facts). The worker
owns its own runtime state; ml-api owns this relay-derived view.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import TypeAlias

from pydantic import TypeAdapter, ValidationError

from shared.edge_db.connection import RuntimeActor, open_runtime_database

# Default staleness window = heartbeat_interval (30s) x3. Configurable per-app.
DEFAULT_STALE_AFTER_SEC: float = 90.0

ONLINE = "online"
STALE = "stale"
NEVER_SEEN = "never_seen"

HeartbeatRow: TypeAlias = tuple[str, str, float, int | None]
_HEARTBEAT_ROW = TypeAdapter(HeartbeatRow)
JsonObject: TypeAlias = dict[str, object]


@dataclass(frozen=True, slots=True)
class CameraHeartbeat:
    camera_id: str
    facility_id: str
    received_at: float
    config_version: int | None = None


@dataclass(slots=True)
class HeartbeatStore:
    """Local, app-owned record of the most recent heartbeat per camera."""

    stale_after_sec: float = DEFAULT_STALE_AFTER_SEC
    database_path: Path | None = None
    _beats: dict[str, CameraHeartbeat] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.database_path is None:
            return
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            beats: dict[str, CameraHeartbeat] = {}
            for row in connection.execute(
                "SELECT camera_id,facility_id,received_at,config_version FROM control_heartbeats"
            ):
                parsed = _parse_heartbeat_row(row)
                if parsed is None:
                    continue
                beats[parsed.camera_id] = parsed
            self._beats = beats
        finally:
            connection.close()

    def record(
        self,
        camera_id: str,
        facility_id: str,
        *,
        received_at: float | None = None,
        config_version: int | None = None,
    ) -> None:
        """Record a relayed heartbeat. ``received_at`` is stamped by the caller
        right after relay-token auth, before camera binding and before any
        backend egress."""
        stamped = time() if received_at is None else received_at
        self._beats[camera_id] = CameraHeartbeat(camera_id, facility_id, stamped, config_version)
        if self.database_path is not None:
            connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO control_heartbeats VALUES (?,?,?,?) "
                    "ON CONFLICT(camera_id) DO UPDATE SET facility_id=excluded.facility_id, "
                    "received_at=excluded.received_at, config_version=excluded.config_version",
                    (camera_id, facility_id, stamped, config_version),
                )
                connection.commit()
            except sqlite3.Error:
                connection.rollback()
                raise
            finally:
                connection.close()

    def snapshot(
        self,
        expected_cameras: object = None,
        *,
        now: float | None = None,
    ) -> dict[str, object]:
        """Derive per-camera liveness over the union of expected + seen cameras.

        ``expected_cameras`` is the registry-derived id index (never env
        inventory). Local truth only: a camera is ``online`` while its last
        heartbeat age is within ``stale_after_sec``, ``stale`` once it exceeds
        it, and ``never_seen`` when it is expected but no heartbeat has arrived.
        """
        current = time() if now is None else now
        camera_ids: set[str] = set(self._beats)
        expected_index = _expected_camera_index(expected_cameras)
        if expected_index is not None:
            camera_ids |= set(expected_index)
        cameras: dict[str, JsonObject] = {}
        for camera_id in sorted(camera_ids):
            beat = self._beats.get(camera_id)
            expected = None if expected_index is None else expected_index.get(camera_id)
            expected_facility = None if expected is None else expected.get("facility_id")
            if beat is None:
                cameras[camera_id] = {
                    "camera_id": camera_id,
                    "facility_id": expected_facility,
                    "status": NEVER_SEEN,
                    "last_heartbeat_at": None,
                    "age_sec": None,
                    "config_version": None,
                }
                continue
            age = max(0.0, current - beat.received_at)
            cameras[camera_id] = {
                "camera_id": camera_id,
                "facility_id": beat.facility_id,
                "status": ONLINE if age <= self.stale_after_sec else STALE,
                "last_heartbeat_at": beat.received_at,
                "age_sec": age,
                "config_version": beat.config_version,
            }
        return {"cameras": cameras, "stale_after_sec": self.stale_after_sec}


def get_heartbeat_store(app: object) -> HeartbeatStore:
    """Return the app-owned heartbeat store, creating it on first use.

    Lets relay/status routes work under ``no_lifespan`` test apps without a
    cross-process dependency on the worker.
    """
    state = _require_app_state(app)
    store = getattr(state, "heartbeat_store", None)
    if not isinstance(store, HeartbeatStore):
        store = HeartbeatStore()
        _assign_state_attr(state, "heartbeat_store", store)
    return store


def _require_app_state(app: object) -> object:
    state = getattr(app, "state", None)
    if state is None:
        raise TypeError("app has no state")
    return state


def _assign_state_attr(state: object, name: str, value: object) -> None:
    """Write a dynamic app.state attribute without untyped attribute access.

    Starlette ``State`` stores values in ``_state``; plain test doubles use
    ``__dict__``. Both are exact dict writes at the boundary.
    """
    inner = getattr(state, "_state", None)
    if isinstance(inner, dict):
        inner[name] = value
        return
    namespace = getattr(state, "__dict__", None)
    if isinstance(namespace, dict):
        namespace[name] = value
        return
    raise TypeError(f"app state cannot store {name!r}")


def _parse_heartbeat_row(row: object) -> CameraHeartbeat | None:
    try:
        camera_id, facility_id, received_at, config_version = _HEARTBEAT_ROW.validate_python(row)
    except ValidationError:
        return None
    return CameraHeartbeat(camera_id, facility_id, received_at, config_version)


def _expected_camera_index(
    expected_cameras: object,
) -> dict[str, Mapping[str, object]] | None:
    if not isinstance(expected_cameras, dict):
        return None
    index: dict[str, Mapping[str, object]] = {}
    for camera_id, binding in expected_cameras.items():
        if not isinstance(camera_id, str):
            continue
        if isinstance(binding, Mapping):
            index[camera_id] = binding
    return index


__all__ = [
    "DEFAULT_STALE_AFTER_SEC",
    "NEVER_SEEN",
    "ONLINE",
    "STALE",
    "CameraHeartbeat",
    "HeartbeatStore",
    "get_heartbeat_store",
]
