from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks, FastAPI, Request
from fastapi.exceptions import HTTPException
from pydantic import UUID4, BaseModel, ConfigDict, Field

from backend.app.core.config import get_settings
from backend.app.features.cameras.roster_sync import sync_camera_roster
from backend.app.features.cameras.router import _authorize
from backend.app.features.connection.enrollment import (
    EnrollmentCredentials,
    EnrollmentErrorClass,
    EnrollmentVerificationFailure,
    verify_enrollment,
)
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.connection.topology_retry_coordinator import (
    TopologySyncErrorClass,
    TopologySyncStatus,
)
from backend.app.features.status.backend_heartbeat_relay import HeartbeatRelayState
from backend.app.lifespan import apply_connection_settings, refresh_backend_config

router = APIRouter(prefix="/connection", tags=["connection"])

_HEARTBEAT_DETAIL: dict[str, str] = {
    "auth": "외부 백엔드 인증에 실패했습니다. 시설 토큰을 확인해 주세요.",
    "timeout": "외부 백엔드 응답이 지연되고 있습니다.",
    "unreachable": "외부 백엔드에 연결할 수 없습니다.",
}
_ENROLLMENT_DETAIL: dict[EnrollmentErrorClass, str] = {
    "auth": "등록 코드 또는 토큰을 확인해 주세요.",
    "conflict": "이 설치는 다른 장치에 연결되어 있습니다.",
    "timeout": "외부 백엔드 응답 시간이 초과되었습니다.",
    "unreachable": "외부 백엔드에 연결할 수 없습니다.",
    "invalid_response": "외부 백엔드의 등록 응답이 올바르지 않습니다.",
}


class HeartbeatRelayStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    last_success_at: str | None = None
    last_error_class: str | None = None
    detail: str | None = None


class ConnectionStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    events_url: str | None = None
    config_url: str | None = None
    facility_code: str | None = None
    client_installation_ref: str | None = None
    facility_id: str | None = None
    edge_installation_id: str | None = None
    enrollment_generation: int | None = None
    facility_token_set: bool = False
    facility_token_masked: str | None = None
    enrolled: bool = False
    configured: bool = False
    reachable: bool | None = None
    last_ok_at: str | None = None
    updated_at: str | None = None
    heartbeat_relay: HeartbeatRelayStatus


class ConnectionEnrollmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_code: str = Field(pattern=r"^NH-[0-9A-HJKMNP-TV-Z]{10}$")
    facility_token: str = Field(min_length=1)
    client_installation_ref: UUID4


class ConnectionTestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    error_class: EnrollmentErrorClass | None = None
    detail: str
    facility_id: str | None = None
    edge_installation_id: str | None = None
    enrollment_generation: int | None = None


class CameraRosterSyncResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TopologySyncStatus
    error_class: TopologySyncErrorClass | None = None
    detail: str | None = None
    last_ok_at: str | None = None
    next_retry_at: str | None = None
    camera_count: int


@router.get("", response_model=ConnectionStatusResponse)
def get_connection(request: Request) -> dict[str, object]:
    _authorize(request)
    return _status_response(request.app)


@router.put("", response_model=ConnectionStatusResponse)
def put_connection(
    payload: ConnectionEnrollmentRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    _authorize(request)
    store = ConnectionSettingsStore.from_env()
    credentials = _credentials(payload)
    try:
        verified = verify_enrollment(
            store.load().events_url,
            credentials,
            timeout_sec=get_settings().connection_test_timeout_s,
        )
    except EnrollmentVerificationFailure as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=_ENROLLMENT_DETAIL[exc.error_class],
        ) from None
    store.save(
        {
            "facility_code": credentials.facility_code,
            "client_installation_ref": credentials.client_installation_ref,
            "facility_token": credentials.facility_token,
            "facility_id": verified.facility.facility_id,
            "edge_installation_id": verified.principal.edge_installation_id,
            "enrollment_generation": verified.principal.enrollment_generation,
        }
    )
    apply_connection_settings(request.app)
    _kick_backend_config_refresh(request.app)
    background_tasks.add_task(_trigger_roster_sync, request.app)
    return _status_response(request.app)


@router.post("/test", response_model=ConnectionTestResponse)
def test_connection(
    payload: ConnectionEnrollmentRequest,
    request: Request,
) -> dict[str, object]:
    _authorize(request)
    try:
        verified = verify_enrollment(
            ConnectionSettingsStore.from_env().load().events_url,
            _credentials(payload),
            timeout_sec=get_settings().connection_test_timeout_s,
        )
    except EnrollmentVerificationFailure as exc:
        return {
            "ok": False,
            "error_class": exc.error_class,
            "detail": _ENROLLMENT_DETAIL[exc.error_class],
        }
    return {
        "ok": True,
        "error_class": None,
        "detail": "등록 정보를 확인했습니다.",
        "facility_id": verified.facility.facility_id,
        "edge_installation_id": verified.principal.edge_installation_id,
        "enrollment_generation": verified.principal.enrollment_generation,
    }


@router.post("/sync-cameras", response_model=CameraRosterSyncResponse)
def sync_cameras(request: Request) -> dict[str, object]:
    _authorize(request)
    result = sync_camera_roster(request.app)
    return {
        "status": result.status,
        "error_class": result.error_class,
        "detail": result.detail,
        "last_ok_at": result.last_ok_at,
        "next_retry_at": result.next_retry_at,
        "camera_count": result.camera_count,
    }


def _credentials(payload: ConnectionEnrollmentRequest) -> EnrollmentCredentials:
    return EnrollmentCredentials(
        payload.facility_code,
        str(payload.client_installation_ref),
        payload.facility_token,
    )


def _status_response(app: FastAPI) -> dict[str, object]:
    settings = ConnectionSettingsStore.from_env().load()
    enrolled = all(
        value is not None
        for value in (
            settings.facility_code,
            settings.client_installation_ref,
            settings.facility_id,
            settings.facility_token,
            settings.edge_installation_id,
            settings.enrollment_generation,
        )
    )
    return {
        "events_url": settings.events_url,
        "config_url": settings.config_url,
        "facility_code": settings.facility_code,
        "client_installation_ref": settings.client_installation_ref,
        "facility_id": settings.facility_id,
        "edge_installation_id": settings.edge_installation_id,
        "enrollment_generation": settings.enrollment_generation,
        "facility_token_set": settings.facility_token is not None,
        "facility_token_masked": (
            None if settings.facility_token is None else f"****{settings.facility_token[-4:]}"
        ),
        # "enrolled" is the strict full facility-code + enrollment-token
        # provisioning contract; "configured" is the looser legacy
        # events_url/facility_id/token check so a technician who only used
        # the dashboard settings form (no enrollment flow) still reads as
        # configured.
        "enrolled": enrolled,
        "configured": bool(
            settings.events_url and settings.facility_id and settings.facility_token
        ),
        # Reuses the flags mark_backend_status() already maintains on
        # app.state (shared/backend_mapping.py) instead of tracking a
        # parallel copy of backend reachability here.
        "reachable": getattr(app.state, "backend_reachable", None),
        "last_ok_at": getattr(app.state, "backend_last_ok_at", None),
        "updated_at": settings.updated_at,
        "heartbeat_relay": _heartbeat_relay_view(app),
    }


def _heartbeat_relay_view(app: FastAPI) -> dict[str, object]:
    relay_state = getattr(app.state, "backend_heartbeat_relay_state", None)
    if not isinstance(relay_state, HeartbeatRelayState):
        return {"enabled": False, "last_success_at": None, "last_error_class": None, "detail": None}
    error_class = relay_state.last_error_class
    return {
        "enabled": getattr(app.state, "backend_heartbeat_relay_task", None) is not None,
        "last_success_at": relay_state.last_success_at,
        "last_error_class": error_class,
        "detail": _HEARTBEAT_DETAIL.get(error_class) if error_class else None,
    }


def _trigger_roster_sync(app: FastAPI) -> None:
    try:
        sync_camera_roster(app, _force=True, _refresh=True)
    except Exception:  # noqa: BLE001, S110
        pass


def _kick_backend_config_refresh(app: FastAPI) -> None:
    executor = getattr(app.state, "backend_config_refresh_executor", None)
    if not isinstance(executor, ThreadPoolExecutor):
        return
    try:
        executor.submit(refresh_backend_config, app)
    except RuntimeError:
        pass


__all__ = ["router"]
