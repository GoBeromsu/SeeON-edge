"""API-owned per-camera heartbeat liveness store.

The worker relays liveness facts to ml-api ``/relay/heartbeat``. ml-api records
the local ``received_at`` per camera right after relay-token auth -- before
camera_inventory/registry binding is resolved and before any backend egress --
so ``/status`` reflects edge-local truth that is independent both of backend
reachability and of whether this camera has been onboarded onto the central
backend's own roster yet (see #183, #202).

This is NOT cross-process shared state with the worker: it is an api-local
snapshot built from relayed facts (one owner, fed by HTTP facts). The worker
owns its own runtime state; ml-api owns this relay-derived view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any

# Default staleness window = heartbeat_interval (30s) x3. Configurable per-app.
DEFAULT_STALE_AFTER_SEC: float = 90.0

ONLINE = "online"
STALE = "stale"
NEVER_SEEN = "never_seen"


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
    _beats: dict[str, CameraHeartbeat] = field(default_factory=dict)

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
        self._beats[camera_id] = CameraHeartbeat(
            camera_id,
            facility_id,
            stamped,
            config_version,
        )

    def snapshot(
        self,
        inventory: object = None,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Derive per-camera liveness over the union of inventory + seen cameras.

        Local truth only: a camera is ``online`` while its last heartbeat age is
        within ``stale_after_sec``, ``stale`` once it exceeds it, and
        ``never_seen`` when it is in inventory but no heartbeat has arrived.
        """
        current = time() if now is None else now
        camera_ids: set[str] = set(self._beats)
        if isinstance(inventory, dict):
            camera_ids |= set(inventory)
        cameras: dict[str, dict[str, Any]] = {}
        for camera_id in sorted(camera_ids):
            beat = self._beats.get(camera_id)
            inv = inventory.get(camera_id) if isinstance(inventory, dict) else None
            inv_facility = inv.get("facility_id") if isinstance(inv, dict) else None
            if beat is None:
                cameras[camera_id] = {
                    "camera_id": camera_id,
                    "facility_id": inv_facility,
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
    state = app.state  # type: ignore[attr-defined]
    store = getattr(state, "heartbeat_store", None)
    if not isinstance(store, HeartbeatStore):
        store = HeartbeatStore()
        state.heartbeat_store = store
    return store


__all__ = [
    "DEFAULT_STALE_AFTER_SEC",
    "NEVER_SEEN",
    "ONLINE",
    "STALE",
    "CameraHeartbeat",
    "HeartbeatStore",
    "get_heartbeat_store",
]
