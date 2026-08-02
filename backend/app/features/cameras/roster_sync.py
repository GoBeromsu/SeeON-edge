"""Camera roster sync to the external eldercare-fall-ai backend (story G004).

Pushes the local camera registry to the external backend via
``PUT /v1/edge/cameras`` (``BackendCameraMapper.put_roster``, reused from
``shared/backend_mapping.py`` -- see that module's docstring) so the
facility's roster appears there, and tracks per-camera + overall sync state
on ``app.state`` (same in-memory-store-on-app.state precedent as
``runtime_status_store``/``heartbeat_store``).

Event-driven, not periodic: ``sync_camera_roster`` is meant to be called by
routers right after a mutation (camera create/update/delete, connection
settings save) or the explicit ``POST /connection/sync-cameras`` endpoint.
A failed attempt records ``next_retry_at`` with exponential, capped backoff;
a call made before that deadline is a no-op that returns the last known
result unchanged. No background thread or lifespan wiring lives here -- the
periodic lane (if one is ever added) belongs to whoever owns the lifespan
refresh loop.

Endpoint/token resolution deliberately routes through
``ConnectionSettingsStore`` effective settings (lazy import, matching the
trick ``connection/router.py`` and ``lifespan.py`` already use to avoid a
circular import) rather than ``BackendCameraMapper.from_env()`` -- see
``_build_mapper`` and the G004 note in ``shared/backend_mapping.py``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Literal

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.shared.backend_mapping import BackendCameraMapper, RosterErrorClass

SyncStatus = Literal["pending", "synced", "failed", "disabled"]

# Attempt-time backoff after a failed push: base delay, doubled per
# consecutive failure, capped so a persistently broken backend never pushes
# the next allowed attempt out indefinitely.
BASE_BACKOFF_SEC = 5.0
MAX_BACKOFF_SEC = 300.0

_UNCONFIGURED_DETAIL = "백엔드 연결이 설정되지 않았습니다. 연결 설정을 저장한 뒤 다시 시도하세요."
_AUTH_DETAIL = "인증에 실패했습니다. 시설 토큰을 다시 확인하세요."
_TIMEOUT_DETAIL = "백엔드 응답 시간이 초과되었습니다. 네트워크 상태를 확인하세요."
_UNREACHABLE_DETAIL = "백엔드에 연결할 수 없습니다. 주소와 네트워크 연결을 확인하세요."
# PUT /v1/edge/cameras requires spaceId (EdgeCameraMappingRequestDto) -- a
# camera with no space_id yet cannot be pushed at all (it would 400), so it
# reads as "pending" rather than "failed": there is nothing wrong with the
# connection, the camera just hasn't been assigned a room yet.
_SPACE_PENDING_DETAIL = (
    "공간(방) 배정 대기 중입니다. 외부 서비스에서 카메라에 공간이 배정되면 자동으로 동기화됩니다."
)

_DETAIL_BY_ERROR_CLASS: dict[RosterErrorClass, str] = {
    "unconfigured": _UNCONFIGURED_DETAIL,
    "auth": _AUTH_DETAIL,
    "timeout": _TIMEOUT_DETAIL,
    "unreachable": _UNREACHABLE_DETAIL,
}


@dataclass(slots=True)
class CameraSyncState:
    """Last known sync outcome for one camera_id."""

    status: SyncStatus = "pending"
    error_class: RosterErrorClass | None = None
    detail: str | None = None
    last_attempt_at: str | None = None
    last_ok_at: str | None = None


@dataclass(slots=True)
class RosterSyncState:
    """App-owned roster-sync bookkeeping (per-camera + overall), lives on
    ``app.state.roster_sync_state`` -- same precedent as
    ``HeartbeatRelayState``/``RuntimeStatusStore``."""

    cameras: dict[str, CameraSyncState] = field(default_factory=dict)
    overall_status: SyncStatus = "pending"
    overall_error_class: RosterErrorClass | None = None
    overall_detail: str | None = None
    last_attempt_at: str | None = None
    last_ok_at: str | None = None
    consecutive_failures: int = 0
    next_retry_at: str | None = None
    _next_retry_monotonic: float | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)


def get_roster_sync_state(app: object) -> RosterSyncState:
    """Return the app-owned roster-sync state, creating it on first use."""
    state = app.state  # type: ignore[attr-defined]
    sync_state = getattr(state, "roster_sync_state", None)
    if not isinstance(sync_state, RosterSyncState):
        sync_state = RosterSyncState()
        state.roster_sync_state = sync_state
    return sync_state


@dataclass(slots=True)
class RosterSyncResult:
    """Outcome of a single ``sync_camera_roster`` call."""

    attempted: bool  # False when skipped: disabled, or inside the backoff window
    status: SyncStatus
    error_class: RosterErrorClass | None
    detail: str | None
    last_ok_at: str | None
    next_retry_at: str | None
    camera_count: int


def sync_camera_roster(app: object, *, _now: float | None = None) -> RosterSyncResult:
    """Best-effort push of the local camera registry to the external backend.

    Never raises: any resolution/network failure is captured into the
    returned result and the persisted state, never propagated. Callers in
    request paths (camera CRUD, connection settings save) must treat this as
    fire-and-forget; only the explicit ``POST /connection/sync-cameras``
    entrypoint should call it synchronously and surface the result directly.
    """
    sync_state = get_roster_sync_state(app)
    now_monotonic = monotonic() if _now is None else _now
    with sync_state._lock:
        next_retry = sync_state._next_retry_monotonic
        if next_retry is not None and now_monotonic < next_retry:
            return _result_from_state(sync_state, attempted=False)

    records = _snapshot_camera_records(_camera_store(app))
    camera_ids = [str(record["id"]) for record in records if _is_nonempty_str(record.get("id"))]

    mapper = _build_mapper(app)
    if not mapper.configured:
        _apply_disabled(sync_state, camera_ids)
        return _result_from_state(sync_state, attempted=False)

    payload, eligible_ids, unassigned_ids = _split_roster_payload(records)
    push_result = mapper.put_roster(payload)
    stamp = _utc_now_iso()
    with sync_state._lock:
        sync_state.last_attempt_at = stamp
        for camera_id in camera_ids:
            sync_state.cameras.setdefault(camera_id, CameraSyncState())

        # Cameras with no space_id yet are never pushed (the external DTO
        # requires spaceId) -- they read as "pending" with an explanatory
        # detail, independent of whether the eligible cameras below succeed.
        for camera_id in unassigned_ids:
            cam = sync_state.cameras[camera_id]
            cam.status = "pending"
            cam.error_class = None
            cam.detail = _SPACE_PENDING_DETAIL
            cam.last_attempt_at = stamp

        any_failed = False
        first_failed_error_class: RosterErrorClass | None = None
        for camera_id in eligible_ids:
            camera_result = push_result.cameras.get(camera_id)
            cam = sync_state.cameras[camera_id]
            cam.last_attempt_at = stamp
            if camera_result is not None and camera_result.ok:
                cam.status = "synced"
                cam.error_class = None
                cam.detail = None
                cam.last_ok_at = stamp
            else:
                any_failed = True
                error_class = (
                    camera_result.error_class if camera_result is not None else None
                ) or "unreachable"
                first_failed_error_class = first_failed_error_class or error_class
                cam.status = "failed"
                cam.error_class = error_class
                cam.detail = _DETAIL_BY_ERROR_CLASS[error_class]

        if any_failed:
            sync_state.consecutive_failures += 1
            backoff = min(
                BASE_BACKOFF_SEC * (2 ** (sync_state.consecutive_failures - 1)),
                MAX_BACKOFF_SEC,
            )
            sync_state._next_retry_monotonic = now_monotonic + backoff
            sync_state.next_retry_at = _utc_iso_offset(backoff)
            sync_state.overall_status = "failed"
            sync_state.overall_error_class = first_failed_error_class
            sync_state.overall_detail = (
                _DETAIL_BY_ERROR_CLASS[first_failed_error_class]
                if first_failed_error_class is not None
                else None
            )
        else:
            sync_state.consecutive_failures = 0
            sync_state.next_retry_at = None
            sync_state._next_retry_monotonic = None
            sync_state.overall_error_class = None
            sync_state.overall_detail = None
            if eligible_ids:
                # At least one camera actually pushed and none failed.
                sync_state.overall_status = "synced"
                sync_state.last_ok_at = stamp
            else:
                # Nothing was eligible to push this attempt (every camera is
                # still awaiting a space assignment) -- nothing failed, but
                # nothing was confirmed synced either, so overall stays
                # "pending" rather than falsely claiming success.
                sync_state.overall_status = "pending"
        return _result_from_state(sync_state, attempted=True)


def camera_sync_view(app: object, camera_id: str) -> dict[str, object]:
    """Per-camera ``sync`` payload for GET /cameras (deliverable 3).

    ``disabled`` (no backend configured right now) always wins over any
    historical per-camera state, matching the response-shape spec: a camera
    that synced successfully before the operator cleared the connection
    settings should read as disabled, not stale-synced.
    """
    configured = _build_mapper(app).configured
    sync_state = get_roster_sync_state(app)
    with sync_state._lock:
        cam = sync_state.cameras.get(camera_id)
        cam_snapshot = None if cam is None else replace(cam)
    if not configured:
        return {
            "status": "disabled",
            "error_class": "unconfigured",
            "detail": _UNCONFIGURED_DETAIL,
            "last_ok_at": cam_snapshot.last_ok_at if cam_snapshot is not None else None,
        }
    if cam_snapshot is None:
        return {"status": "pending", "error_class": None, "detail": None, "last_ok_at": None}
    return {
        "status": cam_snapshot.status,
        "error_class": cam_snapshot.error_class,
        "detail": cam_snapshot.detail,
        "last_ok_at": cam_snapshot.last_ok_at,
    }


def _apply_disabled(sync_state: RosterSyncState, camera_ids: list[str]) -> None:
    with sync_state._lock:
        sync_state.overall_status = "disabled"
        sync_state.overall_error_class = "unconfigured"
        sync_state.overall_detail = _UNCONFIGURED_DETAIL
        for camera_id in camera_ids:
            sync_state.cameras.setdefault(camera_id, CameraSyncState())


def _result_from_state(sync_state: RosterSyncState, *, attempted: bool) -> RosterSyncResult:
    return RosterSyncResult(
        attempted=attempted,
        status=sync_state.overall_status,
        error_class=sync_state.overall_error_class,
        detail=sync_state.overall_detail,
        last_ok_at=sync_state.last_ok_at,
        next_retry_at=sync_state.next_retry_at,
        camera_count=len(sync_state.cameras),
    )


def _build_mapper(app: object) -> BackendCameraMapper:
    """Resolve endpoint/token/facility_id from ``ConnectionSettingsStore``
    effective settings (file-over-env), not raw env directly -- lazy import
    to avoid a circular import (store.py imports lifespan.py's env-name
    constants at module load, same trick connection/router.py uses).
    """
    import os

    from backend.app.features.connection.store import ConnectionSettingsStore
    from backend.app.shared.backend_mapping import (
        API_BACKEND_CAMERA_MAPPING_TIMEOUT_SEC_ENV,
        API_BACKEND_EDGE_CAMERAS_URL_ENV,
        API_BACKEND_URL_ENV,
        derive_edge_cameras_endpoint,
    )

    settings = ConnectionSettingsStore.from_env().load()
    explicit = (os.environ.get(API_BACKEND_EDGE_CAMERAS_URL_ENV) or "").strip() or None
    base = (os.environ.get(API_BACKEND_URL_ENV) or "").strip() or None
    if explicit:
        endpoint = explicit
    elif base:
        endpoint = f"{base.rstrip('/')}/api/v1/edge/cameras"
    else:
        endpoint = derive_edge_cameras_endpoint(settings.events_url)
    raw_timeout = os.environ.get(API_BACKEND_CAMERA_MAPPING_TIMEOUT_SEC_ENV)
    timeout_sec = 0.5 if raw_timeout is None else float(raw_timeout)
    return BackendCameraMapper(
        endpoint=endpoint,
        token=settings.facility_token,
        facility_id=settings.facility_id,
        timeout_sec=timeout_sec,
    )


def _split_roster_payload(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    """Split local registry records into (a) eligible cameras -- those with a
    non-empty ``space_id`` -- in the exact single-object shape
    ``PUT /v1/edge/cameras`` requires (``edge_camera_ref``/``label``/``spaceId``,
    all required -- see the G004 ADR) and (b) camera_ids still awaiting a space
    assignment, which must never be pushed (the external DTO requires
    ``spaceId``; sending one without it is a guaranteed 400). Mirrors the
    same space_id-required precedent already applied at
    ``cameras/router.py``'s ``retry_pending_backend_mappings``.

    Returns ``(payload, eligible_ids, unassigned_ids)``.
    """
    payload: list[dict[str, object]] = []
    eligible_ids: list[str] = []
    unassigned_ids: list[str] = []
    for record in records:
        camera_id = record.get("id")
        label = record.get("label")
        if not _is_nonempty_str(camera_id) or not _is_nonempty_str(label):
            continue
        space_id = record.get("space_id")
        if _is_nonempty_str(space_id):
            payload.append({"edge_camera_ref": camera_id, "label": label, "spaceId": space_id})
            eligible_ids.append(str(camera_id))
        else:
            unassigned_ids.append(str(camera_id))
    return payload, eligible_ids, unassigned_ids


def _camera_store(app: object) -> CameraRegistryStore:
    state = app.state  # type: ignore[attr-defined]
    store = getattr(state, "camera_registry", None)
    if not isinstance(store, CameraRegistryStore):
        store = CameraRegistryStore.from_env()
        state.camera_registry = store
    return store


def _snapshot_camera_records(store: CameraRegistryStore) -> list[dict[str, object]]:
    cameras = store.snapshot().get("cameras")
    if not isinstance(cameras, list):
        return []
    return [record for record in cameras if isinstance(record, dict)]


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _utc_iso_offset(seconds: float) -> str:
    return (
        (datetime.now(UTC) + timedelta(seconds=seconds))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


__all__ = [
    "BASE_BACKOFF_SEC",
    "MAX_BACKOFF_SEC",
    "CameraSyncState",
    "RosterSyncResult",
    "RosterSyncState",
    "camera_sync_view",
    "get_roster_sync_state",
    "sync_camera_roster",
]
