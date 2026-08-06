"""Camera registry routes."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import get_settings
from backend.app.features.cameras.bed_zone_router import BedZonePayload
from backend.app.features.cameras.bed_zone_store import BedZone, BedZoneStore
from backend.app.features.cameras.roster_sync import camera_sync_view, sync_camera_roster
from backend.app.features.cameras.store import (
    CameraRegistryStore,
    DuplicateCameraError,
    ProbeErrorClass,
    ProbeResult,
    is_valid_floor,
    public_camera,
    status_from_probe,
    utc_now_iso,
)
from backend.app.features.clips.storage_location_store import ClipStorageLocationStore
from backend.app.features.detection_settings.store import DetectionSettingsStore
from backend.app.features.status.heartbeat_store import ONLINE, get_heartbeat_store
from backend.app.lifespan import API_FACILITY_ID_ENV
from backend.app.shared.backend_mapping import (
    BackendCameraMapper,
    MappingResult,
    mark_backend_status,
)
from backend.app.shared.dashboard_auth import authorize_dashboard
from contracts.worker_config import PulledWorkerConfig

RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"

router = APIRouter(prefix="/cameras", tags=["cameras"])


class CameraSyncStatus(BaseModel):
    """Per-camera roster-sync state (story G004): whether this camera's last
    push to the external backend (``PUT /v1/edge/cameras``) succeeded, is
    still pending, or failed, and why. See
    ``backend/app/features/cameras/roster_sync.py``.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["synced", "pending", "failed", "disabled"]
    error_class: Literal["unreachable", "timeout", "auth", "unconfigured"] | None = None
    detail: str | None = None
    last_ok_at: str | None = None


class CameraResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    rtsp_url_masked: str = Field(min_length=1)
    space_id: str | None = None
    backend_camera_id: str | None = None
    mapping_pending: bool = False
    status: Literal["online", "offline", "starting", "unknown"]
    decode_backend: str | None = None
    fps: float | None = None
    created_at: str | None = None
    space_name: str | None = None
    # Space-sync-owned floor name (external roster pull, read-only from here --
    # see _public_snapshot). None for a camera with no backend space mapping.
    floor_name: str | None = None
    # User-set floor override (issue #85 design handoff: floor selector; a
    # fixed integer catalog since issue #155 -- B1 = -1 .. 10층 = 10, see
    # store.py's FLOOR_VALUES/floor_label). Persisted on the local registry
    # record, so it survives every space-sync roster re-sync untouched (that
    # merge only ever assigns space_name/floor_name -- see
    # CameraRegistryStore.public_camera). Display precedence is user-set
    # floor first, falling back to the space-sync floor_name:
    # `camera.floor ?? camera.floor_name`.
    floor: int | None = None
    # Roster-sync state (story G004): populated only by GET /cameras (see
    # _public_snapshot); create/update/delete/test responses leave this None
    # rather than racing the fire-and-forget background sync they trigger.
    sync: CameraSyncStatus | None = None
    # Live heartbeat freshness (see _heartbeat_camera_fields); only populated
    # by GET /cameras, which is the only route that joins heartbeat liveness.
    last_heartbeat_at: float | None = None
    heartbeat_age_sec: float | None = None
    # Probe-history fields persisted on the registry record (see
    # CameraRegistryStore.create/update); None for backend-only roster
    # cameras that have no local registry record at all.
    never_connected: bool | None = None
    last_ok_at: str | None = None
    last_probed_at: str | None = None
    # Persisted on-demand bed-zone recognition result (see bed_zone_router.py
    # and BedZoneStore); None when never recognized for this camera.
    bed_zone: BedZonePayload | None = None


class ListCamerasResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry_version: int = Field(ge=0)
    cameras: list[CameraResponse]


class CreateCameraRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    rtsp_url: str = Field(min_length=1)
    space_id: str | None = None
    decode_backend: str | None = None
    fps: float | None = None
    floor: int | None = None
    # 더 이상 아무것도 하지 않는다. 등록이 probe 통과를 요구하던 시절, 그
    # 게이트를 건너뛰는 탈출구였다 (`create_camera` 참고). 지금은 등록이
    # 항상 저장하므로 값과 무관하게 결과가 같다. 기존 클라이언트가 계속
    # 보내고 있어(`extra="forbid"`라 지우면 422가 된다) 필드만 남겨 둔다 --
    # 프론트의 "강제 등록" 흐름까지 걷어낼 때 같이 제거한다.
    force_register: bool = False


class UpdateCameraRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(default=None, min_length=1)
    rtsp_url: str | None = Field(default=None, min_length=1)
    space_id: str | None = None
    decode_backend: str | None = None
    fps: float | None = None
    floor: int | None = None


class TestCameraRequest(BaseModel):
    """연결 테스트 요청.

    ``rtsp_url``을 주면 저장된 값 대신 그 값을 검사한다. 수정 화면에서
    기사님이 방금 입력한 URL을 검사해야 하기 때문이다. 예전에는 저장된
    URL만 검사해서, 오타를 넣어도 "연결 성공"이 뜬 뒤 그 오타가 저장됐다.
    """

    model_config = ConfigDict(extra="forbid")

    # UpdateCameraRequest.rtsp_url과 같은 제약을 쓴다 — 빈 문자열을 보내
    # 저장된 URL 검사로 조용히 되돌아가는 경로를 막는다.
    rtsp_url: str | None = Field(default=None, min_length=1)


class TestCameraResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    error_class: Literal["timeout", "decode", "auth"] | None = None
    # worker의 /probe에 닿지 못해 검사 자체를 못 했다는 뜻이다 (True일 때만
    # 응답에 실린다 -- response_model_exclude_none이 없애지 못하는 bool
    # 기본값 노출을 피하려고 Optional로 선언했다). error_class는 이 경우
    # 항상 None이다: worker가 응답하지 못했으니 timeout/decode/auth 중
    # 어느 것도 아니다. 이슈 #151.
    probe_unavailable: bool | None = None
    width: int | None = None
    height: int | None = None


class WorkerCameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1)
    facility_id: str = Field(min_length=1)
    rtsp_url: str = Field(min_length=1)
    fps: float | None = Field(default=None, gt=0)
    frame_stride: int | None = Field(default=None, gt=0)
    decode_backend: str | None = Field(default=None)
    domains: list[str] | None = None
    # Persisted bed-zone recognition (see BedZoneStore): threaded through so
    # the worker's _CameraPayload (worker/runtime/config/pull_models.py) can
    # seed SceneState.persisted_bed_regions, making it the authoritative bed
    # region for bed-exit instead of live per-frame segmentation.
    bed_zone_polygon: list[list[int]] | None = None
    bed_zone_image_width: int | None = Field(default=None, gt=0)
    bed_zone_image_height: int | None = Field(default=None, gt=0)


class WorkerConfigResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry_version: int = Field(ge=0)
    cameras: list[WorkerCameraConfig]
    config_version: int | None = Field(default=None, ge=0)
    restart_epoch: int | None = Field(default=None, ge=0)
    # Deprecated alias for detection_windows["bed_exit"]; kept for old workers.
    night_window: dict[str, object] | None = None
    detection_windows: dict[str, dict[str, object]] | None = None
    # Local per-domain enable/disable overrides (see detection_settings slice,
    # PUT /api/v1/detection-settings): populated only once an operator has
    # saved settings at least once, and takes precedence over whatever was
    # externally pulled above. Consumed by the worker's
    # BackendWorkerConfigPayload.domains (worker/runtime/config/pull_models.py).
    domains: dict[str, dict[str, object]] | None = None
    # Selected clip storage subdirectory (see clips/storage_router.py),
    # relative to the worker's CLIP_STORE_DIR mount; omitted/None means the
    # mount root (the pre-existing default, unchanged). Consumed by
    # BackendWorkerConfigPayload.clip_store_subdir.
    clip_store_subdir: str | None = None


@router.get("", response_model=ListCamerasResponse)
def list_cameras(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, relay_token, authorization)
    heartbeats = get_heartbeat_store(request.app).snapshot()
    return _public_snapshot(
        request.app,
        _store(request.app).snapshot(),
        getattr(request.app.state, "pulled_config", None),
        heartbeats,
    )


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CameraResponse)
def create_camera(
    payload: CreateCameraRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, relay_token, authorization)
    decode_backend = _normalize_decode_backend(payload.decode_backend)
    fps = _normalize_fps(payload.fps)
    floor = _normalize_floor(payload.floor)
    # 등록은 저장이다. probe는 상태 표시(online/offline, never_connected)에만
    # 쓰고 등록을 막지 않는다.
    #
    # 예전에는 probe 실패 시 422로 거절했는데(force_register만 예외), 그러면
    # 최초 등록이 구조적으로 불가능했다: probe는 worker의 `/probe`에 위임되고
    # (`_probe_rtsp_url`), worker는 카메라가 한 대 이상일 때만 부팅하므로
    # (refuse-to-start), 첫 카메라를 넣으려면 아직 존재하지 않는 worker의
    # 판정을 통과해야 했다. 게다가 worker에 닿지 못한 경우까지 전부
    # `error_class="decode"`로 뭉개져서, 실제 원인(예: RTSP 401)이 "디코드
    # 실패"로 잘못 표시됐다. 죽은 카메라는 offline/never_connected로 목록에
    # 그대로 보이고, 연결 여부는 worker의 첫 heartbeat이 확정한다.
    probe = _probe_rtsp_url(request, payload.rtsp_url)
    provisional_id = str(uuid.uuid4())
    mapping = _map_backend(
        request.app,
        camera_id=provisional_id,
        label=payload.label,
        space_id=payload.space_id,
    )
    camera_id = mapping.backend_camera_id or provisional_id
    now = utc_now_iso()
    try:
        record = _store(request.app).create(
            camera_id=camera_id,
            label=payload.label,
            rtsp_url=payload.rtsp_url,
            space_id=payload.space_id,
            status=status_from_probe(probe),
            backend_camera_id=mapping.backend_camera_id,
            mapping_pending=mapping.pending,
            decode_backend=decode_backend,
            fps=fps,
            floor=floor,
            last_probed_at=now,
            last_ok_at=now if probe.ok else None,
            never_connected=not probe.ok,
        )
    except DuplicateCameraError as exc:
        raise _duplicate_camera_error(exc) from exc
    background_tasks.add_task(_trigger_roster_sync, request.app)
    return public_camera(record)


@router.post(
    "/{camera_id}/test",
    response_model=TestCameraResponse,
    response_model_exclude_none=True,
)
def test_camera(
    camera_id: str,
    request: Request,
    payload: TestCameraRequest | None = None,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, relay_token, authorization)
    record = _store(request.app).get(camera_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")

    stored_url = str(record.get("rtsp_url", ""))
    draft_url = (payload.rtsp_url or "").strip() if payload is not None else ""
    # 기사님이 방금 입력한 값이 있으면 그것을 검사한다. 없으면 저장된 값.
    target_url = draft_url or stored_url
    probe = _probe_rtsp_url(request, target_url)

    # 저장된 URL을 실제로 검사했을 때만 카메라 상태를 갱신한다. draft를
    # 검사해놓고 저장된 카메라를 "정상"으로 표시하면 거짓 신호가 된다.
    if target_url == stored_url:
        now = utc_now_iso()
        updates: dict[str, object] = {"last_probed_at": now}
        if probe.ok:
            updates["last_ok_at"] = now
            updates["never_connected"] = False
        _store(request.app).update(camera_id, updates)
    return _probe_response(probe)


@router.patch("/{camera_id}", response_model=CameraResponse)
def update_camera(
    camera_id: str,
    payload: UpdateCameraRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, relay_token, authorization)
    current = _store(request.app).get(camera_id)
    if current is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    if not payload.model_fields_set:
        return public_camera(current)
    updates: dict[str, object] = {}
    next_label = str(current.get("label", ""))
    next_space_id = current.get("space_id") if current.get("space_id") is not None else None
    if "label" in payload.model_fields_set and payload.label is not None:
        updates["label"] = payload.label
        next_label = payload.label
    if "rtsp_url" in payload.model_fields_set and payload.rtsp_url is not None:
        probe = _probe_rtsp_url(request, payload.rtsp_url)
        updates["rtsp_url"] = payload.rtsp_url
        updates["status"] = status_from_probe(probe)
        now = utc_now_iso()
        updates["last_probed_at"] = now
        if probe.ok:
            updates["last_ok_at"] = now
            updates["never_connected"] = False
    if "space_id" in payload.model_fields_set:
        updates["space_id"] = payload.space_id
        next_space_id = payload.space_id
    if "decode_backend" in payload.model_fields_set:
        updates["decode_backend"] = _normalize_decode_backend(payload.decode_backend)
    if "fps" in payload.model_fields_set:
        updates["fps"] = _normalize_fps(payload.fps)
    if "floor" in payload.model_fields_set:
        updates["floor"] = _normalize_floor(payload.floor)

    if "space_id" in payload.model_fields_set or "label" in payload.model_fields_set:
        mapping = _map_backend(
        request.app,
            camera_id=camera_id,
            label=next_label,
            space_id=next_space_id if isinstance(next_space_id, str) else None,
        )
        if mapping.backend_camera_id is not None:
            updates["backend_camera_id"] = mapping.backend_camera_id
        elif current.get("backend_camera_id") is None:
            updates["backend_camera_id"] = None
        updates["mapping_pending"] = mapping.pending

    try:
        updated = _store(request.app).update(camera_id, updates)
    except DuplicateCameraError as exc:
        raise _duplicate_camera_error(exc) from exc
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    background_tasks.add_task(_trigger_roster_sync, request.app)
    return public_camera(updated)


def _trigger_roster_sync(app: FastAPI) -> None:
    """Best-effort roster push, run as a BackgroundTask after create/update/
    delete so it never adds latency to -- or can fail -- the CRUD response.

    BackgroundTasks run after the response body has already been sent (see
    Starlette's Response.__call__), so by the time a caller's next request
    lands the sync has already been attempted; sync_camera_roster() itself
    never raises, but this still guards against a future change there.
    """
    try:
        sync_camera_roster(app)
    except Exception:  # noqa: BLE001, S110 - a roster-sync bug must never surface here
        pass


def _duplicate_camera_error(exc: DuplicateCameraError) -> HTTPException:
    existing = exc.existing_record
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "error": "duplicate_camera",
            "existing_camera_id": existing.get("id"),
            "existing_label": existing.get("label"),
        },
    )


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> Response:
    _authorize(request, relay_token, authorization)
    existing = _store(request.app).get(camera_id)
    if existing is None or not _store(request.app).delete(camera_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    # A bed zone may be keyed by either the local registry id or the
    # canonical backend_camera_id (see _lookup_bed_zone) depending on which
    # id was canonical when it was recognized -- delete both so a re-created
    # camera with the same id never inherits a stale polygon.
    canonical_id = existing.get("backend_camera_id")
    _bed_zone_store(request.app).delete(camera_id)
    if isinstance(canonical_id, str) and canonical_id and canonical_id != camera_id:
        _bed_zone_store(request.app).delete(canonical_id)
    background_tasks.add_task(_trigger_roster_sync, request.app)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/worker-config", response_model=WorkerConfigResponse, response_model_exclude_none=True)
def worker_config(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize_worker(request, relay_token or _bearer_token(authorization))
    return worker_config_snapshot(request)


def worker_config_snapshot(
    request: Request, *, require_available: bool = False
) -> dict[str, object]:
    snapshot = _store(request.app).snapshot()
    facility_id = _facility_id()
    bed_zones = _bed_zone_store(request.app).get_all()
    cameras = []
    for record in _snapshot_camera_records(snapshot):
        rtsp_url = record.get("rtsp_url")
        if not isinstance(rtsp_url, str) or not rtsp_url.strip():
            continue
        canonical_id = str(record.get("backend_camera_id") or record.get("id", ""))
        camera: dict[str, object] = {
            "camera_id": canonical_id,
            "facility_id": facility_id,
            "rtsp_url": rtsp_url,
        }
        fps = record.get("fps") or _default_camera_fps()
        if fps is not None:
            camera["fps"] = fps
        stride = _default_frame_stride()
        if stride is not None:
            camera["frame_stride"] = stride
        decode_backend = record.get("decode_backend") or _default_decode_backend()
        if decode_backend is not None:
            camera["decode_backend"] = decode_backend
        bed_zone = _lookup_bed_zone(bed_zones, canonical_id, record.get("id"))
        if bed_zone is not None:
            camera["bed_zone_polygon"] = [[x, y] for x, y in bed_zone.polygon]
            camera["bed_zone_image_width"] = bed_zone.image_width
            camera["bed_zone_image_height"] = bed_zone.image_height
        cameras.append(camera)
    pulled = getattr(request.app.state, "pulled_config", None)
    if require_available and not cameras:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker config unavailable",
        )
    response: dict[str, object] = {
        "registry_version": snapshot["registry_version"],
        "cameras": cameras,
    }
    live_pulled: PulledWorkerConfig | None = None
    if isinstance(pulled, PulledWorkerConfig):
        live_pulled = _live_pulled_config(request, pulled)
        response["config_version"] = live_pulled.config_version
        response["restart_epoch"] = live_pulled.restart_epoch
        if live_pulled.night_window is not None:
            response["night_window"] = live_pulled.night_window.as_dict()
        if live_pulled.detection_windows:
            response["detection_windows"] = {
                domain: window.as_dict()
                for domain, window in live_pulled.detection_windows.items()
            }
    # Local overrides run unconditionally (not only when an external pull
    # exists): an operator can save detection settings or a clip storage
    # location before the backend has ever successfully pulled anything.
    _apply_local_detection_overrides(request.app, response, live_pulled)
    _apply_clip_storage_override(request.app, response)
    return response


def _apply_local_detection_overrides(
    app: FastAPI, response: dict[str, object], live_pulled: PulledWorkerConfig | None
) -> None:
    """Merge operator-saved per-domain detection settings (see
    ``detection_settings/store.py``) into the worker-config response,
    overriding whatever was externally pulled above.

    A no-op until an operator has saved at least once via
    ``PUT /api/v1/detection-settings`` -- ``app.state.pulled_config`` is
    never mutated by this (it is memory-only and gets clobbered every ~30s by
    ``_apply_backend_config``), so this merge is redone fresh on every
    worker-config response instead of being applied once to stored state.
    """
    stored = _detection_settings_store(app).get_all()
    if not stored:
        return
    domains: dict[str, dict[str, object]] = {}
    detection_windows = _as_window_dict_map(response.get("detection_windows"))
    for domain, setting in stored.items():
        domains[domain] = {"enabled": setting.on}
        if not setting.on or setting.mode == "always":
            # Domain off, or on with no window restriction: no detection
            # window applies (worker ambient default is 24/7 when a domain
            # has no window entry at all).
            detection_windows.pop(domain, None)
            continue
        detection_windows[domain] = {
            "start": setting.start,
            "end": setting.end,
            "tz": _resolved_tz(live_pulled, domain),
        }
    response["domains"] = domains
    if detection_windows:
        response["detection_windows"] = detection_windows
    else:
        response.pop("detection_windows", None)
    bed_exit_window = detection_windows.get("bed_exit")
    if bed_exit_window is not None:
        response["night_window"] = bed_exit_window
    elif "bed_exit" in stored:
        # An explicit local bed_exit setting (off, or on/always) means no
        # window is in effect -- drop the deprecated alias too rather than
        # leaving it pointing at a stale externally-pulled window.
        response.pop("night_window", None)
    pulled_version = 0 if live_pulled is None else live_pulled.config_version
    response["config_version"] = _local_config_version(
        pulled_version, domains, detection_windows, response.get("night_window")
    )


def _local_config_version(
    pulled_version: int,
    domains: dict[str, dict[str, object]],
    detection_windows: dict[str, dict[str, object]],
    night_window: object,
) -> int:
    """Deterministically derive a ``config_version`` for a response carrying
    local detection-setting overrides.

    The worker's restart poll (``worker/runtime/config/restart.py``,
    ``make_restart_check``) only compares ``(restart_epoch, config_version)``
    on each ~60s pull. Before this, a local-only edit via
    ``PUT /api/v1/detection-settings`` never moved either value, so the
    override was saved to the DB but the running worker never learned about
    it (issue #190).

    This is a pure hash of the override content that ends up in the response
    right next to it -- deliberately NOT a timestamp or a monotonic counter:
    the endpoint can be polled indefinitely between edits and must be stable
    (same override content -> same version) or the worker would restart on
    every poll. It only needs to change when the *effective* override content
    changes.
    """
    payload = json.dumps(
        {"domains": domains, "detection_windows": detection_windows, "night_window": night_window},
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    offset = 1 + (int(digest[:8], 16) % 1_000_000)
    return pulled_version + offset


def _as_window_dict_map(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        return {}
    return {
        domain: window
        for domain, window in value.items()
        if isinstance(domain, str) and isinstance(window, dict)
    }


def _resolved_tz(live_pulled: PulledWorkerConfig | None, domain: str) -> str:
    """The IANA tz to stamp on a locally-configured window.

    No facility-timezone setting exists anywhere else in this codebase, so
    this reuses whatever tz the live externally-pulled window for the same
    domain (or, for bed_exit, the deprecated night_window alias) is already
    using when one is available, and otherwise falls back to
    ``ML_API_DETECTION_TZ`` (default ``"UTC"``).
    """
    if live_pulled is not None:
        window = live_pulled.detection_windows.get(domain)
        if window is None and domain == "bed_exit":
            window = live_pulled.night_window
        if window is not None:
            return window.tz
    return os.environ.get("ML_API_DETECTION_TZ", "UTC").strip() or "UTC"


def _apply_clip_storage_override(app: FastAPI, response: dict[str, object]) -> None:
    """Thread the operator-selected clip storage subdirectory (see
    ``clips/storage_router.py``) through to the worker. ``""`` (mount root,
    the default) is omitted so old and new workers alike keep recording at
    the mount root unless a real subdirectory was explicitly chosen.
    """
    selected = _clip_storage_location_store(app).get()
    if selected:
        response["clip_store_subdir"] = selected


def _detection_settings_store(app: FastAPI) -> DetectionSettingsStore:
    store = getattr(app.state, "detection_settings_store", None)
    if not isinstance(store, DetectionSettingsStore):
        store = DetectionSettingsStore.from_env()
        app.state.detection_settings_store = store
    return store


def _clip_storage_location_store(app: FastAPI) -> ClipStorageLocationStore:
    store = getattr(app.state, "clip_storage_location_store", None)
    if not isinstance(store, ClipStorageLocationStore):
        store = ClipStorageLocationStore.from_env()
        app.state.clip_storage_location_store = store
    return store


def _live_pulled_config(request: Request, pulled: PulledWorkerConfig) -> PulledWorkerConfig:
    return PulledWorkerConfig(
        config_version=int(getattr(request.app.state, "config_version", 0)),
        restart_epoch=int(getattr(request.app.state, "restart_epoch", 0)),
        night_window=pulled.night_window,
        cameras=pulled.cameras,
        detection_windows=pulled.detection_windows,
    )

def retry_pending_backend_mappings(app: FastAPI) -> int:
    """Resolve explicit backend mappings for registry records still pending.

    Called by the roster-refresh owner after a successful backend pull so a
    camera created/edited while the backend was unreachable (or before the
    facility token was wired) converges to its canonical backend identity.
    """
    store = _store(app)
    retried = 0
    for record in _snapshot_camera_records(store.snapshot()):
        if not record.get("mapping_pending"):
            continue
        space_id = record.get("space_id")
        label = record.get("label")
        camera_id = record.get("id")
        if not isinstance(space_id, str) or not space_id.strip():
            continue
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(camera_id, str) or not camera_id.strip():
            continue
        mapping = _map_backend(app, camera_id=camera_id, label=label, space_id=space_id)
        if mapping.backend_camera_id is None:
            continue
        store.update(
            camera_id,
            {"backend_camera_id": mapping.backend_camera_id, "mapping_pending": mapping.pending},
        )
        retried += 1
    return retried



def _public_snapshot(
    app: FastAPI, snapshot: dict[str, object], pulled: object, heartbeats: object = None
) -> dict[str, object]:
    records = _snapshot_camera_records(snapshot)
    bed_zones = _bed_zone_store(app).get_all()
    roster = (
        {camera.camera_id: camera for camera in pulled.cameras}
        if isinstance(pulled, PulledWorkerConfig)
        else {}
    )
    # Pass 1: reserve every explicit id match (backend_camera_id, or a local id
    # that equals a roster camera id) before any space-based fallback so an
    # unmapped sibling can never consume a roster row that an explicitly mapped
    # local registration owns.
    explicit_backend_by_record_index: dict[int, str] = {}
    claimed_roster_ids: set[str] = set()
    for index, record in enumerate(records):
        backend_camera_id = record.get("backend_camera_id")
        local_camera_id = record.get("id")
        join_key = (
            backend_camera_id
            if isinstance(backend_camera_id, str) and backend_camera_id
            else local_camera_id
        )
        if isinstance(join_key, str) and join_key in roster and join_key not in claimed_roster_ids:
            explicit_backend_by_record_index[index] = join_key
            claimed_roster_ids.add(join_key)

    # Pass 2: space fallback is allowed only when, after explicit reservations,
    # a space holds exactly one unmatched local registration and exactly one
    # unclaimed roster row. Cameras never move between rooms (spec R17), but
    # multiple registrations make matching by room unsafe.
    unclaimed_roster_ids_by_space: dict[str, list[str]] = {}
    for backend_camera in roster.values():
        if backend_camera.camera_id in claimed_roster_ids:
            continue
        if isinstance(backend_camera.space_id, str) and backend_camera.space_id:
            unclaimed_roster_ids_by_space.setdefault(backend_camera.space_id, []).append(
                backend_camera.camera_id
            )
    unmatched_record_indexes_by_space: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if index in explicit_backend_by_record_index:
            continue
        space_id = record.get("space_id")
        if isinstance(space_id, str) and space_id:
            unmatched_record_indexes_by_space.setdefault(space_id, []).append(index)
    fallback_backend_by_record_index: dict[int, str] = {}
    for space_id, record_indexes in unmatched_record_indexes_by_space.items():
        roster_ids = unclaimed_roster_ids_by_space.get(space_id, [])
        if len(record_indexes) == 1 and len(roster_ids) == 1:
            fallback_backend_by_record_index[record_indexes[0]] = roster_ids[0]
            claimed_roster_ids.add(roster_ids[0])

    cameras: list[dict[str, object]] = []
    for index, record in enumerate(records):
        camera = public_camera(record)
        # Derive live status from heartbeat freshness instead of the
        # persisted registry snapshot, which is written once at
        # registration/probe time and never reflects a camera going offline
        # afterward. Same canonical id used everywhere else in this function.
        canonical_id = record.get("backend_camera_id") or record.get("id")
        local_id = record.get("id")
        # The worker may still be configured to heartbeat under the local
        # registry id even after a backend mapping assigns a distinct
        # backend_camera_id (relay_heartbeat records under whatever raw id
        # the worker sends -- see _camera_binding_from_registry, which
        # accepts either). Try the canonical id first, then the local id, so
        # a real, freshly-heartbeating camera never reads as offline purely
        # because of which id happened to be recorded under.
        candidate_ids: tuple[str, ...] = tuple(
            dict.fromkeys(
                candidate
                for candidate in (canonical_id, local_id)
                if isinstance(candidate, str)
            )
        )
        camera["status"], camera["last_heartbeat_at"], camera["heartbeat_age_sec"] = (
            _heartbeat_camera_fields(heartbeats, candidate_ids)
        )
        if isinstance(local_id, str) and local_id:
            camera["sync"] = camera_sync_view(app, local_id)
        bed_zone = _lookup_bed_zone(bed_zones, canonical_id, local_id)
        camera["bed_zone"] = bed_zone.as_dict() if bed_zone is not None else None
        backend_id = explicit_backend_by_record_index.get(
            index, fallback_backend_by_record_index.get(index)
        )
        backend_camera = roster.pop(backend_id, None) if backend_id is not None else None
        if backend_camera is not None:
            camera.update(
                {
                    "label": backend_camera.label,
                    "space_id": backend_camera.space_id,
                    "backend_camera_id": backend_camera.camera_id,
                    "mapping_pending": False,
                    "created_at": backend_camera.created_at or camera["created_at"],
                    "space_name": backend_camera.space_name,
                    "floor_name": backend_camera.floor_name,
                }
            )
        else:
            camera.update(
                {
                    "mapping_pending": bool(record.get("mapping_pending", False)),
                    "space_name": None,
                    "floor_name": None,
                }
            )
        cameras.append(camera)

    for backend_camera in roster.values():
        roster_bed_zone = bed_zones.get(backend_camera.camera_id)
        cameras.append(
            {
                "id": backend_camera.camera_id,
                "label": backend_camera.label,
                "rtsp_url_masked": "rtsp://***",
                "space_id": backend_camera.space_id,
                "backend_camera_id": backend_camera.camera_id,
                "mapping_pending": True,
                "status": "unknown",
                "decode_backend": None,
                "fps": None,
                "created_at": backend_camera.created_at,
                "space_name": backend_camera.space_name,
                "floor_name": backend_camera.floor_name,
                "bed_zone": roster_bed_zone.as_dict() if roster_bed_zone is not None else None,
            }
        )
    return {
        "registry_version": snapshot["registry_version"],
        "cameras": cameras,
    }


def _heartbeat_camera_fields(
    heartbeats: object, candidate_ids: tuple[str, ...]
) -> tuple[Literal["online", "offline"], float | None, float | None]:
    """Map a HeartbeatStore.snapshot() entry to (status, last_heartbeat_at,
    heartbeat_age_sec) for the public GET /cameras response.

    Tries each id in ``candidate_ids`` in order (canonical id, then local
    registry id) and uses the first entry found: relay_heartbeat records
    under whichever raw id the worker sends, which may be either one (see
    _camera_binding_from_registry), independent of which id GET /cameras
    otherwise treats as canonical.

    Fail-closed: no candidate ids, no matching entry for any of them, a
    missing snapshot, or any heartbeat state other than ONLINE (i.e. STALE or
    NEVER_SEEN) resolves to "offline" -- never "online" on uncertainty.
    last_heartbeat_at/age_sec are still surfaced when known (even while
    offline) so the UI can render a "last seen Ns ago" style text for a stale
    camera.
    """
    if not candidate_ids or not isinstance(heartbeats, dict):
        return "offline", None, None
    cameras = heartbeats.get("cameras")
    if not isinstance(cameras, dict):
        return "offline", None, None
    entry: object = None
    for candidate_id in candidate_ids:
        entry = cameras.get(candidate_id)
        if isinstance(entry, dict):
            break
    else:
        entry = None
    if not isinstance(entry, dict):
        return "offline", None, None
    live_status: Literal["online", "offline"] = (
        "online" if entry.get("status") == ONLINE else "offline"
    )
    last_heartbeat_at = entry.get("last_heartbeat_at")
    age_sec = entry.get("age_sec")
    return (
        live_status,
        last_heartbeat_at if isinstance(last_heartbeat_at, (int, float)) else None,
        age_sec if isinstance(age_sec, (int, float)) else None,
    )


def _snapshot_camera_records(snapshot: dict[str, object]) -> list[dict[str, object]]:
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, list):
        return []
    return [record for record in cameras if isinstance(record, dict)]


def _store(app: FastAPI) -> CameraRegistryStore:
    store = getattr(app.state, "camera_registry", None)
    if not isinstance(store, CameraRegistryStore):
        store = CameraRegistryStore.from_env()
        app.state.camera_registry = store
    return store


def _bed_zone_store(app: FastAPI) -> BedZoneStore:
    store = getattr(app.state, "bed_zone_store", None)
    if not isinstance(store, BedZoneStore):
        store = BedZoneStore.from_env()
        app.state.bed_zone_store = store
    return store


def _lookup_bed_zone(
    bed_zones: dict[str, BedZone], canonical_id: object, local_id: object
) -> BedZone | None:
    """Match a persisted bed zone by canonical id first, then local registry
    id, mirroring _heartbeat_camera_fields' candidate-id fallback: the
    recognize endpoint may have been called with either id historically, and
    a backend mapping can change which id is canonical after the fact."""
    for candidate in (canonical_id, local_id):
        if isinstance(candidate, str) and candidate in bed_zones:
            return bed_zones[candidate]
    return None


def _mapper(app: FastAPI) -> BackendCameraMapper:
    mapper = getattr(app.state, "backend_camera_mapper", None)
    if not isinstance(mapper, BackendCameraMapper):
        mapper = BackendCameraMapper.from_env()
        app.state.backend_camera_mapper = mapper
    return mapper


def _map_backend(
    app: FastAPI,
    *,
    camera_id: str,
    label: str,
    space_id: str | None,
) -> MappingResult:
    mapper = _mapper(app)
    if space_id is None:
        return MappingResult(backend_camera_id=None, pending=False, reachable=None)
    result = mapper.put_mapping(edge_camera_ref=camera_id, label=label, space_id=space_id)
    mark_backend_status(app.state, result.reachable)
    return result


def _authorize(request: Request, relay_token: str | None, authorization: str | None) -> None:
    bearer = _bearer_token(authorization)
    authorize_dashboard(request, legacy_token=relay_token or bearer)


def _authorize_worker(request: Request, relay_token: str | None) -> None:
    expected = _expected_relay_token(request)
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="relay token is not configured",
        )
    if relay_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="relay token required",
        )
    # Encode to UTF-8 bytes before comparing: hmac.compare_digest raises
    # TypeError for non-ASCII str arguments, and a relay token sourced from an
    # env var or config file is not guaranteed to be ASCII-only.
    if not hmac.compare_digest(relay_token.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="relay token mismatch",
        )


def _bearer_token(value: str | None) -> str | None:
    if value is None:
        return None
    prefix = "Bearer "
    if not value.startswith(prefix):
        return None
    token = value[len(prefix) :].strip()
    return token or None


def _facility_id() -> str:
    return os.environ.get(API_FACILITY_ID_ENV, "local-facility").strip() or "local-facility"

def _default_camera_fps() -> float | None:
    """Facility-wide processed FPS for live camera streams (worker default 5.0).

    Set ML_DEFAULT_CAMERA_FPS to smooth the live MJPEG/overlay view (detection
    runs at this rate). Unset -> worker keeps its 5.0 default. GPU headroom
    permitting, 12-15 gives a noticeably smoother wall without corruption.
    """
    raw = os.environ.get("ML_DEFAULT_CAMERA_FPS", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _default_frame_stride() -> int | None:
    """Facility-wide detection cadence divisor (worker default 1 -- every frame).

    Set ML_DEFAULT_FRAME_STRIDE to decouple detection cadence from live view:
    the worker still decodes/serves the live MJPEG view at fps (see
    ML_DEFAULT_CAMERA_FPS), but only runs pose+person inference every Nth
    decoded frame. Unset -> worker keeps its stride-1 default (detect every
    frame). Lets deployments raise fps for a smoother live wall without
    overloading inference-bound hardware.
    """
    raw = os.environ.get("ML_DEFAULT_FRAME_STRIDE", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _normalize_floor(value: object) -> int | None:
    """Validate a user-set floor override (issue #155).

    Fixed catalog (not free-text like it was pre-#155): B1 through 10층,
    encoded as an integer (basement negative, e.g. B1 = -1) so display
    strings never drift and floors sort numerically instead of
    lexicographically. None passes through untouched (not set / clear an
    existing override, falling back to the space-sync floor_name). Anything
    else is a 400, matching _normalize_decode_backend's shape -- unlike
    parse_legacy_floor (used only for self-healing pre-#155 stored data),
    a fresh write is never silently coerced to a default.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not is_valid_floor(value):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid floor")
    return value


DECODE_BACKENDS = {"auto", "nvdec", "opencv", "cpu"}


def _normalize_decode_backend(value: object) -> str | None:
    """Validate a per-camera decode backend selector.

    None passes through untouched (not set / clear). A string must match one of
    auto|nvdec|opencv|cpu case-insensitively, mirroring the worker's
    CameraRuntimeConfig.decode_backend validator; anything else is a 400.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid decode_backend"
        )
    normalized = value.strip().lower()
    if normalized not in DECODE_BACKENDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid decode_backend"
        )
    return normalized


def _normalize_fps(value: object) -> float | None:
    """Validate a per-camera processed-fps override.

    None passes through untouched (not set / clear). A number must be > 0,
    mirroring the worker's CameraRuntimeConfig.fps validator (Field(gt=0));
    anything else is a 400, matching _normalize_decode_backend's shape.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid fps")
    fps = float(value)
    if fps <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid fps")
    return fps


def _default_decode_backend() -> str | None:
    """Facility-wide default decode backend (worker default: auto = NVDEC->CPU fallback).

    Set ML_DEFAULT_DECODE_BACKEND to auto|nvdec|opencv|cpu to steer cameras that
    do not set a per-camera decode_backend. Unset or invalid -> None (worker
    keeps its own "auto" default).
    """
    raw = os.environ.get("ML_DEFAULT_DECODE_BACKEND", "").strip().lower()
    if not raw:
        return None
    return raw if raw in DECODE_BACKENDS else None


def _probe_rtsp_url(request: Request, rtsp_url: str) -> ProbeResult:
    settings = get_settings()
    origin = settings.worker_probe_origin.strip().rstrip("/")
    if not origin:
        # ML_API_WORKER_PROBE_ORIGIN 자체가 미설정 -- worker에 요청을 보낼
        # 주소가 없다. worker가 살아서 "디코드 실패"라고 답한 것과 전혀
        # 다른 상황이므로 error_class를 채우지 않는다 (이슈 #151).
        return ProbeResult(ok=False, probe_unavailable=True)
    token = _expected_relay_token(request)
    if token is None:
        # relay 토큰 미설정 -- 마찬가지로 검사 요청 자체를 보낼 수 없다.
        return ProbeResult(ok=False, probe_unavailable=True)
    body = json.dumps({"rtsp_url": rtsp_url}, separators=(",", ":")).encode("utf-8")
    probe_request = urllib.request.Request(
        f"{origin}/probe",
        data=body,
        headers={
            "Content-Type": "application/json",
            RELAY_TOKEN_HEADER: token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            probe_request,
            timeout=settings.worker_probe_timeout_s,
        ) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except TimeoutError:
        # worker에 보낸 HTTP 요청이 timeout한 것이지 RTSP가 timeout한 게
        # 아니다 -- RTSP 타임아웃은 worker가 살아서 payload로 알려주고
        # (_probe_result_from_worker), 그쪽에서 error_class="timeout"이
        # 채워진다. 여기까지 왔다는 건 worker가 제때 답을 못 했다는 뜻이므로
        # "검사 불가"로 분류한다 (이슈 #151).
        return ProbeResult(ok=False, probe_unavailable=True)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        # 연결 거부/DNS 실패/HTTP 레벨 오류 등 -- worker에 닿지 못했거나
        # 응답을 아예 받지 못한 경우다. worker가 실제로 응답했는데 그
        # 내용이 디코드 실패였던 것(_probe_result_from_worker 경로)과는
        # 구분해야 한다 (이슈 #151: RTSP 401을 "디코드 실패"로 오진했던
        # 원인 중 하나가 이 catch-all이었다).
        return ProbeResult(ok=False, probe_unavailable=True)
    if not isinstance(payload, dict):
        return ProbeResult(ok=False, error_class="decode")
    return _probe_result_from_worker(payload)


def _probe_result_from_worker(payload: dict[object, object]) -> ProbeResult:
    raw_error_class = payload.get("error_class")
    error_class: ProbeErrorClass | None
    if raw_error_class == "timeout":
        error_class = "timeout"
    elif raw_error_class == "decode":
        error_class = "decode"
    elif raw_error_class == "auth":
        error_class = "auth"
    else:
        error_class = None
    width = _optional_positive_int(payload.get("width"))
    height = _optional_positive_int(payload.get("height"))
    return ProbeResult(
        ok=payload.get("ok") is True,
        error_class=error_class,
        width=width,
        height=height,
    )


def _optional_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _expected_relay_token(request: Request) -> str | None:
    # `app.state.edge_relay_token`이 유일한 출처다. 부팅 때
    # `lifespan._configure_backend_ingest`가 `API_EDGE_RELAY_TOKEN`에서 한 번
    # 채운다(`lifespan.py:105`가 반드시 부른다). 여기서 env를 또 읽으면
    # state에 `None`을 명시적으로 넣어 미설정을 재현하려는 경우까지
    # env가 조용히 덮어써서, 실제로 무엇이 유효한지 두 곳을 봐야 한다.
    expected = getattr(request.app.state, "edge_relay_token", None)
    return expected if isinstance(expected, str) and expected else None


def _probe_response(probe: ProbeResult) -> dict[str, object]:
    response: dict[str, object] = {"ok": probe.ok}
    if probe.error_class is not None:
        response["error_class"] = probe.error_class
    if probe.probe_unavailable:
        response["probe_unavailable"] = True
    if probe.width is not None:
        response["width"] = probe.width
    if probe.height is not None:
        response["height"] = probe.height
    return response


__all__ = ["retry_pending_backend_mappings", "router", "worker_config_snapshot"]
