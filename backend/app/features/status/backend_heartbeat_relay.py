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
- HeartbeatStore records liveness under whatever raw id the worker sends
  (typically the local registry id -- see ``relay/router.py``'s
  ``relay_heartbeat``, which stamps liveness before resolving any backend
  mapping). The external backend only knows its own camera ids, so every id
  must be canonicalized through the camera registry's ``backend_camera_id``
  mapping before being pushed -- mirroring the one-shot alert/heartbeat path
  in ``relay/router.py``'s ``_camera_binding_from_registry``. Unlike that
  one-shot path (which falls back to the local id when no backend mapping
  exists yet, because the registry lookup there also gates local admission),
  this periodic tick has nothing to gate: an unmapped camera is simply
  skipped-and-logged rather than sent as a guaranteed-404 request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from backend.app.features.status.heartbeat_store import ONLINE, get_heartbeat_store
from backend.app.shared.backend_client_bundle import backend_client_bundle

logger = logging.getLogger(__name__)

# Consecutive all-fail ticks widen the effective relay interval up to this
# multiplier of the configured base interval (e.g. 30s base -> 240s ceiling).
MAX_BACKOFF_MULTIPLIER = 8

# Relative severity of a tick's classified failures -- when a tick mixes
# error classes across cameras, the most severe one wins as the tick's
# (and then the relay state's) reported error_class.
_ERROR_CLASS_SEVERITY: dict[str, int] = {"auth": 3, "timeout": 2, "unreachable": 1}


@dataclass(slots=True)
class RelayTickResult:
    """Outcome of a single ``relay_heartbeats_once`` call."""

    attempted: int = 0
    sent: int = 0
    failed: int = 0
    # Set (and attempted/sent/failed left at 0) when the tick did no work at
    # all: no configured client, or no ONLINE cameras to relay this tick.
    skipped_reason: str | None = None
    # Most severe classified failure this tick (auth > timeout > unreachable),
    # or None when nothing failed with a known class -- including plain-bool
    # fakes/clients that predate send_heartbeat_result().
    error_class: str | None = None


@dataclass(slots=True)
class HeartbeatRelayState:
    """Backoff + last-known-status bookkeeping for the relay loop, tracked on
    ``app.state``."""

    consecutive_all_fail_ticks: int = 0
    backoff_multiplier: int = 1
    last_error_class: str | None = None
    last_success_at: str | None = None


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
    bundle = backend_client_bundle(app)
    client = (
        bundle.ingest_client
        if bundle is not None
        else getattr(app.state, "backend_ingest_client", None)
    )
    if client is None or not hasattr(client, "for_camera"):
        return RelayTickResult(skipped_reason="no_client")

    from backend.app.features.cameras.store import CameraRegistryStore, registry_expected_cameras

    registry = getattr(app.state, "camera_registry", None)
    registry_store = registry if isinstance(registry, CameraRegistryStore) else None
    expected = registry_expected_cameras(registry_store)
    snapshot = get_heartbeat_store(app).snapshot(expected, now=now)
    online_camera_ids = [
        camera_id for camera_id, info in snapshot["cameras"].items() if info["status"] == ONLINE
    ]
    if not online_camera_ids:
        return RelayTickResult(skipped_reason="no_online_cameras")

    # HeartbeatStore ids are worker-local (see module docstring); canonicalize
    # to the backend's own camera id before pushing, and skip -- rather than
    # send a guaranteed-404 -- any camera the registry has no backend mapping
    # for yet.
    canonical_camera_ids: list[str] = []
    for camera_id in online_camera_ids:
        canonical_id = _canonical_backend_camera_id(registry_store, camera_id)
        if canonical_id is None:
            logger.info(
                "backend heartbeat relay: skipping camera_id=%s, no backend mapping yet",
                camera_id,
            )
            continue
        canonical_camera_ids.append(canonical_id)

    if not canonical_camera_ids:
        return RelayTickResult(skipped_reason="no_mapped_cameras")

    result = RelayTickResult(attempted=len(canonical_camera_ids))
    # Sequential POSTs, one per online camera, each bounded by the ingest
    # client's own timeout (default 0.5s, env API_BACKEND_INGEST_TIMEOUT_SEC).
    # Worst case a tick takes N * timeout_sec wall time; fine at edge scale
    # (single-digit cameras per facility) but would need fan-out if that
    # roster ever grows large.
    for camera_id in canonical_camera_ids:
        ok, error_class = _send_one(client, camera_id)
        if ok:
            result.sent += 1
        else:
            result.failed += 1
            result.error_class = _more_severe(result.error_class, error_class)

    relay_state = get_heartbeat_relay_state(app)
    _update_backoff(relay_state, result)
    _update_error_state(relay_state, result)
    return result


def _canonical_backend_camera_id(registry: object | None, camera_id: str) -> str | None:
    """Resolve a HeartbeatStore camera_id (worker-local or already-canonical)
    to the external backend's own camera id, or ``None`` if the registry has
    no explicit ``backend_camera_id`` mapping for it yet.

    Mirrors ``relay/router.py``'s ``_camera_binding_from_registry`` matching
    (by local id OR backend_camera_id), but deliberately does NOT fall back
    to the local id when unmapped -- that one-shot path's fallback exists to
    gate local admission, which this periodic tick has no equivalent of; here
    an unmapped camera must be skipped, not sent under an id the backend will
    reject.
    """
    if registry is None:
        return None
    snapshot = registry.snapshot()  # type: ignore[attr-defined]
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, list):
        return None
    for record in cameras:
        if not isinstance(record, dict):
            continue
        local_id = record.get("id")
        backend_id = record.get("backend_camera_id")
        if camera_id in (local_id, backend_id):
            return backend_id if isinstance(backend_id, str) and backend_id else None
    return None


def _send_one(client: object, camera_id: str) -> tuple[bool, str | None]:
    try:
        camera_client = client.for_camera(camera_id)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - backend egress must never crash the loop
        return False, None
    # Prefer the classified result path (send_heartbeat_result) when the
    # client offers one; fall back to the plain bool send_heartbeat() for
    # older/fake clients that predate it -- their failures are reported with
    # error_class=None, same as before this classification existed.
    result_sender = getattr(camera_client, "send_heartbeat_result", None)
    if callable(result_sender):
        try:
            sent = result_sender()
        except Exception:  # noqa: BLE001 - ditto
            return False, None
        return bool(getattr(sent, "ok", False)), getattr(sent, "error_class", None)
    sender = getattr(camera_client, "send_heartbeat", None)
    if not callable(sender):
        return False, None
    try:
        return bool(sender()), None
    except Exception:  # noqa: BLE001 - ditto
        return False, None


def _more_severe(current: str | None, candidate: str | None) -> str | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    if _ERROR_CLASS_SEVERITY.get(candidate, 0) > _ERROR_CLASS_SEVERITY.get(current, 0):
        return candidate
    return current


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


def _update_error_state(relay_state: HeartbeatRelayState, result: RelayTickResult) -> None:
    if result.sent > 0:
        relay_state.last_success_at = _utc_now_iso()
    if result.attempted == 0:
        # Skipped tick did no delivery attempt -- leave the last-known error
        # class alone rather than falsely clearing it.
        return
    previous = relay_state.last_error_class
    relay_state.last_error_class = result.error_class
    if result.error_class != previous:
        logger.warning(
            "backend heartbeat relay error_class transition: %s -> %s",
            previous,
            result.error_class,
        )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "MAX_BACKOFF_MULTIPLIER",
    "HeartbeatRelayState",
    "RelayTickResult",
    "effective_relay_interval_sec",
    "get_heartbeat_relay_state",
    "relay_heartbeats_once",
]
