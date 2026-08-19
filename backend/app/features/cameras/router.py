"""Camera registry routes."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
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
from backend.app.features.cameras.roster_sync import (
    camera_sync_view,
    sync_camera_roster,
)
from backend.app.features.cameras.store import (
    CameraRegistryData,
    CameraRegistryStore,
    DuplicateCameraError,
    ProbeErrorClass,
    ProbeResult,
    is_valid_floor,
    public_camera,
    status_from_probe,
    utc_now_iso,
)
from backend.app.features.cameras.topology import (
    RegistryTopologySnapshot,
    TopologyConflictError,
)
from backend.app.features.clips.storage_location_store import ClipStorageLocationStore
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.detection_settings.policy_store import (
    DetectionPolicyStore,
    PolicyActivationRefused,
    PolicyCameraIdentity,
)
from backend.app.features.detection_settings.store import DetectionSettingsStore
from backend.app.features.runtime_settings.store import get_runtime_settings_store
from backend.app.features.status.heartbeat_store import ONLINE, get_heartbeat_store
from backend.app.shared.dashboard_auth import authorize_dashboard
from contracts.edge_provisioning_models import EdgeErrorCode, TopologyFloor, TopologyRoom
from contracts.worker_config import PulledWorkerConfig
from shared.rtsp_url_policy import assert_rtsp_endpoint_allowed

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
    # Must mirror TopologySyncErrorClass (backend/app/features/connection/
    # topology_retry_result.py), the actual source of these values -- a
    # conflict-paused topology sync reports error_class="conflict" and this
    # literal previously omitted it, 500ing every GET /cameras while paused.
    error_class: Literal["unreachable", "timeout", "auth", "unconfigured", "conflict"] | None = None
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
    # Explicit Hub-mapping state so an operator can distinguish "registered and
    # waiting on Hub sync" (pending -- normal right after a technician adds a
    # camera) from "no mapping and none in flight" (unmapped). Cameras in either
    # state are deliberately omitted from worker-config, because emitting the
    # edge-local id where a Hub-issued id belongs is what the Hub rejects with
    # FACILITY_BINDING_MISMATCH (issue #308). This is an edge-local dashboard
    # field only; the Hub provisioning contract in contracts/ is untouched.
    mapping_state: Literal["mapped", "pending", "unmapped"] = "unmapped"
    status: Literal["online", "offline", "starting", "unknown"]
    decode_backend: str | None = None
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
    edge_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)
    room_edge_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)


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
    floor: int | None = None
    edge_ref: str | None = Field(default=None, min_length=1, max_length=64)
    room_edge_ref: str | None = Field(default=None, min_length=1, max_length=64)
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
    floor: int | None = None
    edge_ref: str | None = Field(default=None, min_length=1, max_length=64)
    room_edge_ref: str | None = Field(default=None, min_length=1, max_length=64)


class CreateTopologyFloorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_ref: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    order_index: int = Field(ge=0)


class UpdateTopologyFloorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    order_index: int = Field(ge=0)


class CreateTopologyRoomRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_ref: str = Field(min_length=1, max_length=64)
    floor_edge_ref: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    legacy_canonical_space_id: str | None = Field(default=None, max_length=36)


class UpdateTopologyRoomRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)


class TopologyCameraResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_ref: str
    label: str


class TopologyRoomResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_ref: str
    name: str
    room_type: Literal["ROOM"]
    capacity: int
    legacy_canonical_space_id: str | None
    cameras: list[TopologyCameraResponse]


class TopologyFloorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_ref: str
    name: str
    order_index: int
    rooms: list[TopologyRoomResponse]


class CameraTopologyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry_version: int = Field(ge=0)
    dirty_registry_version: int | None = Field(default=None, ge=1)
    readiness_error: EdgeErrorCode | None
    unmapped_camera_ids: list[str]
    floors: list[TopologyFloorResponse]


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
    error_class: Literal["timeout", "decode", "auth", "unsupported"] | None = None
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
    # Optional: site facility is not stamped here. Worker defaults missing
    # facility_id to the local wire placeholder "local".
    facility_id: str | None = Field(default=None, min_length=1)
    space_id: str | None = Field(default=None, min_length=1)
    rtsp_url: str = Field(min_length=1)
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
    # Closed typed numeric-policy bundle. The worker parses this as one unit and
    # refuses unknown module/version/schema/fields rather than partially applying it.
    detection_policies: dict[str, object] | None = None
    # Selected clip storage subdirectory (see clips/storage_router.py),
    # relative to the worker's CLIP_STORE_DIR mount; omitted/None means the
    # mount root (the pre-existing default, unchanged). Consumed by
    # BackendWorkerConfigPayload.clip_store_subdir.
    clip_store_subdir: str | None = None
    clip_export_enabled: bool = False
    clip_export_version: int = Field(default=0, ge=0)


@router.get("", response_model=ListCamerasResponse)
def list_cameras(request: Request) -> dict[str, object]:
    _authorize(request)
    heartbeats = get_heartbeat_store(request.app).snapshot()
    return _public_snapshot(
        request.app,
        _store(request.app).snapshot(),
        getattr(request.app.state, "pulled_config", None),
        heartbeats,
    )


@router.get("/topology", response_model=CameraTopologyResponse)
def get_camera_topology(request: Request) -> dict[str, object]:
    _authorize(request)
    return _topology_response(_store(request.app).topology_snapshot())


@router.post("/topology/floors", status_code=status.HTTP_201_CREATED)
def create_topology_floor(
    payload: CreateTopologyFloorRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    _authorize(request)
    try:
        _store(request.app).create_floor(
            edge_ref=payload.edge_ref, name=payload.name, order_index=payload.order_index
        )
    except TopologyConflictError as error:
        raise _topology_conflict(error) from error
    background_tasks.add_task(_trigger_roster_sync, request.app)
    return payload.model_dump()


@router.patch("/topology/floors/{edge_ref}")
def update_topology_floor(
    edge_ref: str,
    payload: UpdateTopologyFloorRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    _authorize(request)
    if not _store(request.app).update_floor(
        edge_ref, name=payload.name, order_index=payload.order_index
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="floor not found")
    background_tasks.add_task(_trigger_roster_sync, request.app)
    return {"edge_ref": edge_ref, **payload.model_dump()}


@router.delete("/topology/floors/{edge_ref}", status_code=status.HTTP_204_NO_CONTENT)
def delete_topology_floor(
    edge_ref: str, request: Request, background_tasks: BackgroundTasks
) -> Response:
    _authorize(request)
    try:
        changed = _store(request.app).delete_floor(edge_ref)
    except TopologyConflictError as error:
        raise _topology_conflict(error) from error
    if not changed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="floor not found")
    background_tasks.add_task(_trigger_roster_sync, request.app)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/topology/rooms", status_code=status.HTTP_201_CREATED)
def create_topology_room(
    payload: CreateTopologyRoomRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    _authorize(request)
    try:
        _store(request.app).create_room(
            edge_ref=payload.edge_ref,
            floor_edge_ref=payload.floor_edge_ref,
            name=payload.name,
            legacy_canonical_space_id=payload.legacy_canonical_space_id,
        )
    except TopologyConflictError as error:
        raise _topology_conflict(error) from error
    background_tasks.add_task(_trigger_roster_sync, request.app)
    return payload.model_dump()


@router.patch("/topology/rooms/{edge_ref}")
def update_topology_room(
    edge_ref: str,
    payload: UpdateTopologyRoomRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    _authorize(request)
    if not _store(request.app).update_room(edge_ref, name=payload.name):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="room not found")
    background_tasks.add_task(_trigger_roster_sync, request.app)
    return {"edge_ref": edge_ref, "name": payload.name}


@router.delete("/topology/rooms/{edge_ref}", status_code=status.HTTP_204_NO_CONTENT)
def delete_topology_room(
    edge_ref: str, request: Request, background_tasks: BackgroundTasks
) -> Response:
    _authorize(request)
    try:
        changed = _store(request.app).delete_room(edge_ref)
    except TopologyConflictError as error:
        raise _topology_conflict(error) from error
    if not changed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="room not found")
    background_tasks.add_task(_trigger_roster_sync, request.app)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CameraResponse)
def create_camera(
    payload: CreateCameraRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    _authorize(request)
    rtsp_url = _validated_rtsp_url(payload.rtsp_url)
    decode_backend = _normalize_decode_backend(payload.decode_backend)
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
    probe = _probe_rtsp_url(request, rtsp_url)
    provisional_id = str(uuid.uuid4())
    now = utc_now_iso()
    try:
        record = _store(request.app).create(
            camera_id=provisional_id,
            label=payload.label,
            rtsp_url=rtsp_url,
            space_id=payload.space_id,
            status=status_from_probe(probe),
            backend_camera_id=None,
            mapping_pending=False,
            decode_backend=decode_backend,
            floor=floor,
            last_probed_at=now,
            last_ok_at=now if probe.ok else None,
            never_connected=not probe.ok,
            edge_ref=payload.edge_ref,
            room_edge_ref=payload.room_edge_ref,
        )
    except DuplicateCameraError as exc:
        raise _duplicate_camera_error(exc) from exc
    except TopologyConflictError as error:
        raise _topology_conflict(error) from error
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
) -> dict[str, object]:
    _authorize(request)
    record = _store(request.app).get(camera_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")

    stored_url = str(record.get("rtsp_url", ""))
    draft_url = (payload.rtsp_url or "").strip() if payload is not None else ""
    # 기사님이 방금 입력한 값이 있으면 그것을 검사한다. 없으면 저장된 값.
    target_url = _validated_rtsp_url(draft_url or stored_url)
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
) -> dict[str, object]:
    _authorize(request)
    current = _store(request.app).get(camera_id)
    if current is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    if not payload.model_fields_set:
        return public_camera(current)
    updates: dict[str, object] = {}
    if "label" in payload.model_fields_set and payload.label is not None:
        updates["label"] = payload.label
    if "rtsp_url" in payload.model_fields_set and payload.rtsp_url is not None:
        rtsp_url = _validated_rtsp_url(payload.rtsp_url)
        probe = _probe_rtsp_url(request, rtsp_url)
        updates["rtsp_url"] = rtsp_url
        updates["status"] = status_from_probe(probe)
        now = utc_now_iso()
        updates["last_probed_at"] = now
        if probe.ok:
            updates["last_ok_at"] = now
            updates["never_connected"] = False
    if "space_id" in payload.model_fields_set:
        updates["space_id"] = payload.space_id
    if "decode_backend" in payload.model_fields_set:
        updates["decode_backend"] = _normalize_decode_backend(payload.decode_backend)
    if "floor" in payload.model_fields_set:
        updates["floor"] = _normalize_floor(payload.floor)
    if "edge_ref" in payload.model_fields_set:
        updates["edge_ref"] = payload.edge_ref
    if "room_edge_ref" in payload.model_fields_set:
        updates["room_edge_ref"] = payload.room_edge_ref

    try:
        updated = _store(request.app).update(camera_id, updates)
    except DuplicateCameraError as exc:
        raise _duplicate_camera_error(exc) from exc
    except TopologyConflictError as error:
        raise _topology_conflict(error) from error
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


def _topology_conflict(error: TopologyConflictError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": error.code.value, "edge_ref": error.edge_ref},
    )


def _topology_response(snapshot: RegistryTopologySnapshot) -> dict[str, object]:
    return {
        "registry_version": snapshot.registry_version,
        "dirty_registry_version": (
            None if snapshot.dirty is None else snapshot.dirty.registry_version
        ),
        "readiness_error": snapshot.readiness_error,
        "unmapped_camera_ids": list(snapshot.unmapped_camera_ids),
        "floors": [_topology_floor_response(floor) for floor in snapshot.floors],
    }


def _topology_floor_response(floor: TopologyFloor) -> dict[str, object]:
    return {
        "edge_ref": floor.edge_ref,
        "name": floor.name,
        "order_index": floor.order_index,
        "rooms": [_topology_room_response(room) for room in floor.rooms],
    }


def _topology_room_response(room: TopologyRoom) -> dict[str, object]:
    return {
        "edge_ref": room.edge_ref,
        "name": room.name,
        "room_type": room.room_type,
        "capacity": room.capacity,
        "legacy_canonical_space_id": room.legacy_canonical_space_id,
        "cameras": [
            {"edge_ref": camera.edge_ref, "label": camera.label} for camera in room.cameras
        ],
    }


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> Response:
    _authorize(request)
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


_LOGGER = logging.getLogger(__name__)


def _mapping_state(record: dict[str, object]) -> str:
    """Classify a registry record's Hub mapping into a three-way state.

    ``mapped`` means the Hub issued a canonical id for this camera.
    ``pending`` means the mapping call has not resolved yet (normal right after
    registration, before topology sync lands).
    ``unmapped`` means there is no mapping and none is in flight.

    The edge-local primary key is never a substitute for the Hub-issued id:
    the Hub rejects an id it never issued with FACILITY_BINDING_MISMATCH, which
    surfaces on the edge as an opaque relay 502 and gets misread as an auth
    failure. Absence is reported as absence instead (issue #308).
    """
    backend_camera_id = record.get("backend_camera_id")
    if isinstance(backend_camera_id, str) and backend_camera_id.strip():
        return "mapped"
    if bool(record.get("mapping_pending", False)):
        return "pending"
    return "unmapped"


def _hub_canonical_id(record: dict[str, object]) -> str | None:
    """Return the Hub-issued canonical id, or None when the record is unmapped.

    Callers that emit an outbound payload MUST treat None as "omit this camera".
    Inbound lookup paths are free to accept either id (see the heartbeat index
    in store.py and the offline probe in this module); accepting both on the way
    in is deliberate tolerance, while emitting the local id on the way out is
    the defect.
    """
    backend_camera_id = record.get("backend_camera_id")
    if isinstance(backend_camera_id, str) and backend_camera_id.strip():
        return backend_camera_id
    return None


def worker_config_snapshot(
    request: Request, *, require_available: bool = False
) -> dict[str, object]:
    snapshot = _store(request.app).snapshot()
    bed_zones = _bed_zone_store(request.app).get_all()
    facility_id = _connection_settings_store(request.app).load().facility_id
    cameras = []
    policy_cameras: list[PolicyCameraIdentity] = []
    for record in _snapshot_camera_records(snapshot):
        rtsp_url = record.get("rtsp_url")
        if not isinstance(rtsp_url, str) or not rtsp_url.strip():
            continue
        # DO NOT exclude unmapped cameras here. worker-config.cameras is the
        # exact set the worker ingests (worker/runtime/worker.py:1978 feeds it to
        # build_camera_source_registry), so dropping a camera stops fall
        # detection for that room entirely. On a live nursing-home edge that is
        # strictly worse than the issue #308 symptom it was meant to fix, where
        # the camera is still watched and only the upstream submission is
        # rejected. The Hub-boundary fix belongs at the relay/report path, not
        # here -- tracked as a review blocker on this goal.
        canonical_id = str(record.get("backend_camera_id") or record.get("id", ""))
        if _hub_canonical_id(record) is None:
            _LOGGER.warning(
                "worker-config emitting camera %s without a Hub mapping (state=%s)",
                record.get("id"),
                _mapping_state(record),
                extra={
                    "local_camera_id": record.get("id"),
                    "mapping_state": _mapping_state(record),
                },
            )
        # No site facility stamp: worker defaults missing facility_id to the
        # local wire placeholder "local". space_id is optional registry metadata.
        camera: dict[str, object] = {
            "camera_id": canonical_id,
            "rtsp_url": rtsp_url,
        }
        policy_cameras.append(PolicyCameraIdentity(camera_id=canonical_id))
        space_id = record.get("space_id")
        if isinstance(space_id, str) and space_id.strip():
            camera["space_id"] = space_id
        # frame_stride/decode_backend are per-camera registry values only.
        # The facility-wide ML_DEFAULT_* environment fallbacks were retired
        # (see core.config._RETIRED_BACKEND_ENV, which fails boot on them);
        # the registry is the sole authority, so an unset value is simply
        # omitted and the worker keeps its own default.
        #
        # fps is deliberately NOT emitted: the worker's TemporalProfile owns
        # ingest pacing (design B), so a relay-declared per-camera fps was a
        # dead control that saved successfully and changed nothing.
        decode_backend = record.get("decode_backend")
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
                domain: window.as_dict() for domain, window in live_pulled.detection_windows.items()
            }
    # Local overrides run unconditionally (not only when an external pull
    # exists): an operator can save detection settings or a clip storage
    # location before the backend has ever successfully pulled anything.
    _apply_local_detection_overrides(request.app, response, live_pulled)
    _apply_clip_storage_override(request.app, response)
    _apply_numeric_detection_policies(
        request.app,
        response,
        facility_id=facility_id,
        cameras=tuple(policy_cameras),
    )
    runtime_setting = get_runtime_settings_store(request.app).get()
    response["clip_export_enabled"] = runtime_setting.clip_export_enabled
    response["clip_export_version"] = runtime_setting.version
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
    using when one is available, and otherwise falls back to a fixed
    ``"UTC"``. The former ``ML_API_DETECTION_TZ`` override is retired
    (``core.config._RETIRED_BACKEND_ENV`` fails boot on it), so there is no
    environment knob here.
    """
    if live_pulled is not None:
        window = live_pulled.detection_windows.get(domain)
        if window is None and domain == "bed_exit":
            window = live_pulled.night_window
        if window is not None:
            return window.tz
    return "UTC"


def _apply_clip_storage_override(app: FastAPI, response: dict[str, object]) -> None:
    """Thread the operator-selected clip storage subdirectory (see
    ``clips/storage_router.py``) through to the worker. ``""`` (mount root,
    the default) is omitted so old and new workers alike keep recording at
    the mount root unless a real subdirectory was explicitly chosen.
    """
    selected = _clip_storage_location_store(app).get()
    if selected:
        response["clip_store_subdir"] = selected


def acknowledge_applied_detection_policies(
    request: Request, *, facility_id: str, config_version: int | None
) -> None:
    """Move pending activations to applied only after a restarted worker heartbeats."""
    enrolled_facility = _connection_settings_store(request.app).load().facility_id
    if enrolled_facility != facility_id or config_version is None:
        return
    expected = worker_config_snapshot(request).get("config_version")
    if expected != config_version:
        return
    store = _detection_policy_store(request.app)
    store.mark_applied(facility_id, store.generation(facility_id))


def _apply_numeric_detection_policies(
    app: FastAPI,
    response: dict[str, object],
    *,
    facility_id: str | None,
    cameras: tuple[PolicyCameraIdentity, ...],
) -> None:
    """Resolve immutable policy revisions at the worker-restart boundary.

    Running camera modules retain their boot-time ``WorkerConfig``. A policy
    activation increments the persisted generation below; restart polling sees
    that generation and only the next worker process receives this bundle.
    """
    store = _detection_policy_store(app)
    generation = store.generation(facility_id)
    # Before the first explicit Apply there is no desired policy revision to
    # publish. New workers parse absence as the typed image-default bundle;
    # omitting it preserves the established old-worker/setup wire contract.
    if generation == 0:
        return
    try:
        bundle = store.resolve_bundle(facility_id, cameras)
    except PolicyActivationRefused as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    response["detection_policies"] = bundle.as_dict()
    if facility_id is not None:
        response_cameras = response.get("cameras")
        if isinstance(response_cameras, list):
            for camera in response_cameras:
                if isinstance(camera, dict):
                    camera["facility_id"] = facility_id
    raw_base_version = response.get("config_version", 0)
    base_version = raw_base_version if isinstance(raw_base_version, int) else 0
    policy_hash_part = int(bundle.content_sha256[:8], 16) % 1_000_000_000
    response["config_version"] = base_version * 1_000_000_000 + policy_hash_part
    raw_restart_epoch = response.get("restart_epoch", 0)
    restart_epoch = raw_restart_epoch if isinstance(raw_restart_epoch, int) else 0
    response["restart_epoch"] = restart_epoch + generation


def _detection_policy_store(app: FastAPI) -> DetectionPolicyStore:
    store = getattr(app.state, "detection_policy_store", None)
    if not isinstance(store, DetectionPolicyStore):
        store = DetectionPolicyStore.from_env()
        app.state.detection_policy_store = store
    return store


def _connection_settings_store(app: FastAPI) -> ConnectionSettingsStore:
    store = getattr(app.state, "connection_settings_store", None)
    if isinstance(store, ConnectionSettingsStore):
        return store
    return ConnectionSettingsStore.from_env()


def _detection_settings_store(app: FastAPI) -> DetectionSettingsStore:
    store = getattr(app.state, "detection_settings_store", None)
    if not isinstance(store, DetectionSettingsStore):
        store = DetectionSettingsStore(_store(app).path)
        app.state.detection_settings_store = store
    return store


def _clip_storage_location_store(app: FastAPI) -> ClipStorageLocationStore:
    store = getattr(app.state, "clip_storage_location_store", None)
    if not isinstance(store, ClipStorageLocationStore):
        store = ClipStorageLocationStore(_store(app).path)
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


def _public_snapshot(
    app: FastAPI,
    snapshot: CameraRegistryData | dict[str, object],
    pulled: object,
    heartbeats: object = None,
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
                candidate for candidate in (canonical_id, local_id) if isinstance(candidate, str)
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
                    "mapping_state": "mapped",
                    "created_at": backend_camera.created_at or camera["created_at"],
                    "space_name": backend_camera.space_name,
                    "floor_name": backend_camera.floor_name,
                }
            )
        else:
            camera.update(
                {
                    "mapping_pending": bool(record.get("mapping_pending", False)),
                    # Explicit three-way state so an operator can tell "waiting on
                    # Hub sync" (pending, normal right after registration) apart
                    # from "no mapping and none in flight" (unmapped). A camera in
                    # either state is omitted from worker-config on purpose.
                    "mapping_state": _mapping_state(record),
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
                "mapping_state": "mapped",
                "status": "unknown",
                "decode_backend": None,
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


def _snapshot_camera_records(
    snapshot: CameraRegistryData | dict[str, object],
) -> list[dict[str, object]]:
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
        store = BedZoneStore(_store(app).path)
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


def _authorize(request: Request) -> None:
    authorize_dashboard(request)


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


def _validated_rtsp_url(rtsp_url: str) -> str:
    """Admit only policy-allowed RTSP destinations before store or probe.

    Resolves hostnames and rejects when any A/AAAA answer violates policy.
    The original (hostname) URL is stored; workers pin the IP at open/probe.
    """

    try:
        endpoint = assert_rtsp_endpoint_allowed(rtsp_url)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return endpoint.original_url


def _probe_rtsp_url(request: Request, rtsp_url: str) -> ProbeResult:
    # Re-check (including DNS answers) at the probe boundary so a future
    # caller cannot bypass create/update admission.
    try:
        endpoint = assert_rtsp_endpoint_allowed(rtsp_url)
    except ValueError:
        return ProbeResult(ok=False, error_class="unsupported")
    rtsp_url = endpoint.original_url
    settings = get_settings()
    origin = settings.worker_probe_origin.strip().rstrip("/")
    if not origin:
        # worker probe origin(Settings.worker_probe_origin)이 비어 있다 --
        # worker에 요청을 보낼 주소가 없다. 옛 ML_API_WORKER_PROBE_ORIGIN
        # 환경변수는 폐기되어(core.config._RETIRED_BACKEND_ENV) 더는 이 값을
        # 주입하지 못한다. worker가 살아서 "디코드 실패"라고 답한 것과 전혀
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
    elif raw_error_class == "unsupported":
        error_class = "unsupported"
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


__all__ = [
    "acknowledge_applied_detection_policies",
    "router",
    "worker_config_snapshot",
]
