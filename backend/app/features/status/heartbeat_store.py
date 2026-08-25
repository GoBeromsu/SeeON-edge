"""API-owned per-camera heartbeat liveness store.

The worker relays liveness facts to ml-api ``/relay/heartbeat``. ml-api records
the local ``received_at`` per camera right after relay-token auth -- before
registry binding is resolved and before any backend egress -- so ``/status``
reflects edge-local truth that is independent both of backend reachability and
of whether this camera has been onboarded onto the central backend's own roster
yet (see #183, #202).

Latest-only memory: a missing or stale observation is explicit. Process restart
forgets every beat. Nothing here opens SQLite.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import time
from typing import TypeAlias

# Default staleness window = heartbeat_interval (30s) x3. Configurable per-app.
DEFAULT_STALE_AFTER_SEC: float = 90.0
DEFAULT_MAX_CAMERAS: int = 256

ONLINE = "online"
STALE = "stale"
NEVER_SEEN = "never_seen"

JsonObject: TypeAlias = dict[str, object]
Clock: TypeAlias = Callable[[], float]


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
    retain_after_sec: float | None = None
    max_cameras: int = DEFAULT_MAX_CAMERAS
    clock: Clock = field(default=time)
    _beats: dict[str, CameraHeartbeat] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.retain_after_sec is None:
            self.retain_after_sec = max(self.stale_after_sec * 10.0, 3_600.0)

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
        with self._lock:
            now = self.clock()
            stamped = now if received_at is None else received_at
            if stamped > now:
                return
            previous = self._beats.get(camera_id)
            if previous is not None and previous.received_at > stamped:
                return
            self._beats[camera_id] = CameraHeartbeat(
                camera_id, facility_id, stamped, config_version
            )
            self._evict_overflow_locked()

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
        with self._lock:
            current = self.clock() if now is None else now
            self._evict_expired_locked(current)
            self._evict_overflow_locked()
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
                if current < beat.received_at:
                    cameras[camera_id] = {
                        "camera_id": camera_id,
                        "facility_id": beat.facility_id,
                        "status": STALE,
                        "last_heartbeat_at": beat.received_at,
                        "age_sec": None,
                        "config_version": beat.config_version,
                    }
                    continue
                age = current - beat.received_at
                cameras[camera_id] = {
                    "camera_id": camera_id,
                    "facility_id": beat.facility_id,
                    "status": ONLINE if age <= self.stale_after_sec else STALE,
                    "last_heartbeat_at": beat.received_at,
                    "age_sec": age,
                    "config_version": beat.config_version,
                }
            return {"cameras": cameras, "stale_after_sec": self.stale_after_sec}

    def _evict_expired_locked(self, now: float) -> None:
        retain_after = (
            self.stale_after_sec if self.retain_after_sec is None else self.retain_after_sec
        )
        for camera_id, beat in tuple(self._beats.items()):
            if now >= beat.received_at and now - beat.received_at > retain_after:
                del self._beats[camera_id]

    def _evict_overflow_locked(self) -> None:
        overflow = len(self._beats) - self.max_cameras
        if overflow <= 0:
            return
        oldest = sorted(self._beats.values(), key=lambda beat: beat.received_at)
        for beat in oldest[:overflow]:
            del self._beats[beat.camera_id]


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
    """Write a dynamic app.state attribute without untyped attribute access."""
    inner = getattr(state, "_state", None)
    if isinstance(inner, dict):
        inner[name] = value
        return
    namespace = getattr(state, "__dict__", None)
    if isinstance(namespace, dict):
        namespace[name] = value
        return
    raise TypeError(f"app state cannot store {name!r}")


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
    "DEFAULT_MAX_CAMERAS",
    "DEFAULT_STALE_AFTER_SEC",
    "NEVER_SEEN",
    "ONLINE",
    "STALE",
    "CameraHeartbeat",
    "HeartbeatStore",
    "get_heartbeat_store",
]
