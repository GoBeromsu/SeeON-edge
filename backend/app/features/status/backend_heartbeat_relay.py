"""Per-camera heartbeat relay to the external backend (story G005).

The external eldercare-fall-ai dashboard only shows a camera as alive once it
receives ``POST /v1/events/heartbeat`` with ``{camera_id}``. Workers already
heartbeat ml-api internally (``POST /api/v1/relay/heartbeat`` ->
``HeartbeatStore``); this module forwards that internal, edge-local liveness
out to the external backend on a timer owned by ``backend/app/lifespan.py``
(``_backend_heartbeat_relay_loop``).

Design notes:
- Only ``ONLINE`` cameras are relayed. ``stale``/``never_seen`` cameras must
  stay silent -- that silence is exactly what lets the external card go gray
  once ml-api itself stops hearing from a camera.
- A failing/unreachable backend must never affect anything local: per-camera
  failures are counted, never raised, and a stretch of all-fail ticks widens
  the relay loop's wait via ``HeartbeatRelayState`` (state lives on
  ``app.state``, not module globals, so it is per-app like every other store
  here).
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.features.status.heartbeat_store import ONLINE, get_heartbeat_store

# Consecutive all-fail ticks widen the effective relay interval up to this
# multiplier of the configured base interval (e.g. 30s base -> 240s ceiling).
MAX_BACKOFF_MULTIPLIER = 8


@dataclass(slots=True)
class RelayTickResult:
    """Outcome of a single ``relay_heartbeats_once`` call."""

    attempted: int = 0
    sent: int = 0
    failed: int = 0
    # Set (and attempted/sent/failed left at 0) when the tick did no work at
    # all: no configured client, or no ONLINE cameras to relay this tick.
    skipped_reason: str | None = None


@dataclass(slots=True)
class HeartbeatRelayState:
    """Backoff bookkeeping for the relay loop, tracked on ``app.state``."""

    consecutive_all_fail_ticks: int = 0
    backoff_multiplier: int = 1


def get_heartbeat_relay_state(app: object) -> HeartbeatRelayState:
    """Return the app-owned relay backoff state, creating it on first use."""
    state = app.state  # type: ignore[attr-defined]
    relay_state = getattr(state, "backend_heartbeat_relay_state", None)
    if not isinstance(relay_state, HeartbeatRelayState):
        relay_state = HeartbeatRelayState()
        state.backend_heartbeat_relay_state = relay_state
    return relay_state


def effective_relay_interval_sec(
    base_interval_sec: float, relay_state: HeartbeatRelayState
) -> float:
    """Widen the base interval by the current backoff multiplier."""
    return base_interval_sec * relay_state.backoff_multiplier


def relay_heartbeats_once(app: object, now: float | None = None) -> RelayTickResult:
    """Forward every ONLINE camera's liveness to the external backend once.

    Pure-ish and independent of the asyncio loop so it is directly testable.
    Never raises: a misconfigured or unreachable backend must never affect
    anything local. Callers wanting a no-op tick just need to leave
    ``app.state.backend_ingest_client`` unset -- the exact state a freshly
    booted app without connection settings, or a test app, is already in.
    """
    client = getattr(app.state, "backend_ingest_client", None)
    if client is None or not hasattr(client, "for_camera"):
        return RelayTickResult(skipped_reason="no_client")

    inventory = getattr(app.state, "camera_inventory", None)
    snapshot = get_heartbeat_store(app).snapshot(inventory, now=now)
    online_camera_ids = [
        camera_id
        for camera_id, info in snapshot["cameras"].items()
        if info["status"] == ONLINE
    ]
    if not online_camera_ids:
        return RelayTickResult(skipped_reason="no_online_cameras")

    result = RelayTickResult(attempted=len(online_camera_ids))
    # Sequential POSTs, one per online camera, each bounded by the ingest
    # client's own timeout (default 0.5s, env API_BACKEND_INGEST_TIMEOUT_SEC).
    # Worst case a tick takes N * timeout_sec wall time; fine at edge scale
    # (single-digit cameras per facility) but would need fan-out if that
    # roster ever grows large.
    for camera_id in online_camera_ids:
        if _send_one(client, camera_id):
            result.sent += 1
        else:
            result.failed += 1

    _update_backoff(get_heartbeat_relay_state(app), result)
    return result


def _send_one(client: object, camera_id: str) -> bool:
    try:
        camera_client = client.for_camera(camera_id)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - backend egress must never crash the loop
        return False
    sender = getattr(camera_client, "send_heartbeat", None)
    if not callable(sender):
        return False
    try:
        return bool(sender())
    except Exception:  # noqa: BLE001 - ditto
        return False


def _update_backoff(relay_state: HeartbeatRelayState, result: RelayTickResult) -> None:
    if result.attempted == 0:
        # Skipped tick (no client / no online cameras): not a delivery
        # attempt, so it neither trips nor resets the backoff.
        return
    if result.failed == result.attempted:
        relay_state.consecutive_all_fail_ticks += 1
        relay_state.backoff_multiplier = min(
            MAX_BACKOFF_MULTIPLIER, relay_state.backoff_multiplier * 2
        )
    else:
        relay_state.consecutive_all_fail_ticks = 0
        relay_state.backoff_multiplier = 1


__all__ = [
    "MAX_BACKOFF_MULTIPLIER",
    "HeartbeatRelayState",
    "RelayTickResult",
    "effective_relay_interval_sec",
    "get_heartbeat_relay_state",
    "relay_heartbeats_once",
]
