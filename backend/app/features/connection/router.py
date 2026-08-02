"""Dashboard-facing HTTP API for the external-backend connection settings.

Story G003 of the edge-onboarding plan: exposes ``ConnectionSettingsStore``
(G001) and the runtime relink entrypoint ``apply_connection_settings``
(G002) as endpoints a non-developer technician can drive from the dashboard
to configure, live-test, and check the status of the ml-api -> backend link,
without editing env files or restarting the process.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, FastAPI, Header, Request, status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ConfigDict

from backend.app.core.config import get_settings
from backend.app.features.cameras.roster_sync import SyncStatus, sync_camera_roster
from backend.app.features.cameras.router import RELAY_TOKEN_HEADER, _authorize
from backend.app.features.connection.store import ConnectionSettings, ConnectionSettingsStore
from backend.app.lifespan import apply_connection_settings, refresh_backend_config
from backend.app.shared.backend_mapping import RosterErrorClass

router = APIRouter(prefix="/connection", tags=["connection"])

_FIELDS = ("events_url", "config_url", "facility_id", "facility_token")
ErrorClass = Literal["unconfigured", "invalid_url", "unreachable", "timeout", "auth"]


class ConnectionStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events_url: str | None = None
    config_url: str | None = None
    facility_id: str | None = None
    facility_token_set: bool = False
    facility_token_masked: str | None = None
    configured: bool = False
    reachable: bool | None = None
    last_ok_at: str | None = None
    updated_at: str | None = None


class ConnectionSettingsUpdateRequest(BaseModel):
    """Partial update: a field left unset is untouched; an explicit JSON
    `null` clears the saved value back to its env-seed (see
    ``ConnectionSettingsStore.save``). ``model_fields_set`` is how the
    route tells "omitted" from "explicitly set to null" apart -- pydantic
    records a field there whenever it appeared in the payload, whatever its
    value.
    """

    model_config = ConfigDict(extra="forbid")

    events_url: str | None = None
    config_url: str | None = None
    facility_id: str | None = None
    facility_token: str | None = None


class ConnectionTestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    error_class: ErrorClass | None = None
    detail: str
    probed_url: str | None = None


class CameraRosterSyncResponse(BaseModel):
    """Fresh roster-sync state (story G004), returned by both the explicit
    POST /connection/sync-cameras trigger below and usable to poll the same
    state GET /cameras surfaces per-camera.
    """

    model_config = ConfigDict(extra="forbid")

    status: SyncStatus
    error_class: RosterErrorClass | None = None
    detail: str | None = None
    last_ok_at: str | None = None
    next_retry_at: str | None = None
    camera_count: int


@router.get("", response_model=ConnectionStatusResponse)
def get_connection(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, relay_token, authorization)
    return _status_response(request.app)


@router.put("", response_model=ConnectionStatusResponse)
def put_connection(
    payload: ConnectionSettingsUpdateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, relay_token, authorization)
    updates = _explicit_fields(payload)
    _reject_invalid_urls(updates)
    ConnectionSettingsStore.from_env().save(updates)
    apply_connection_settings(request.app)  # immediate relink, no restart
    _kick_backend_config_refresh(request.app)
    # Best-effort, non-blocking: new/changed connection settings may make the
    # roster reachable (or newly unreachable) -- re-push it so per-camera
    # sync state converges without waiting for the next camera CRUD.
    background_tasks.add_task(_trigger_roster_sync, request.app)
    return _status_response(request.app)


@router.post("/sync-cameras", response_model=CameraRosterSyncResponse)
def sync_cameras(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    """Explicit, synchronous roster push -- unlike the CRUD/PUT triggers
    (fire-and-forget background tasks), this endpoint's whole point is to
    return the fresh result, so it calls sync_camera_roster() directly.
    """
    _authorize(request, relay_token, authorization)
    result = sync_camera_roster(request.app)
    return {
        "status": result.status,
        "error_class": result.error_class,
        "detail": result.detail,
        "last_ok_at": result.last_ok_at,
        "next_retry_at": result.next_retry_at,
        "camera_count": result.camera_count,
    }


@router.post("/test", response_model=ConnectionTestResponse)
def test_connection(
    request: Request,
    payload: ConnectionSettingsUpdateRequest | None = None,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, relay_token, authorization)
    overrides = _explicit_fields(payload) if payload is not None else {}
    _reject_invalid_urls(overrides)
    settings = ConnectionSettingsStore.from_env().load()  # not persisted: overrides are probe-only
    return _probe(
        config_url=_effective(overrides, settings, "config_url"),
        facility_id=_effective(overrides, settings, "facility_id"),
        facility_token=_effective(overrides, settings, "facility_token"),
    )


def _status_response(app: FastAPI) -> dict[str, object]:
    masked = ConnectionSettingsStore.from_env().masked()
    return {
        "events_url": masked["events_url"],
        "config_url": masked["config_url"],
        "facility_id": masked["facility_id"],
        "facility_token_set": masked["facility_token_set"],
        "facility_token_masked": masked["facility_token_masked"],
        "configured": bool(masked["events_url"]),
        # Reuses the flags mark_backend_status() already maintains on
        # app.state (shared/backend_mapping.py) instead of tracking a
        # parallel copy of backend reachability here.
        "reachable": getattr(app.state, "backend_reachable", None),
        "last_ok_at": getattr(app.state, "backend_last_ok_at", None),
        "updated_at": masked["updated_at"],
    }


def _explicit_fields(payload: ConnectionSettingsUpdateRequest) -> dict[str, str | None]:
    data = payload.model_dump()
    return {field: data[field] for field in payload.model_fields_set if field in _FIELDS}


def _reject_invalid_urls(fields: dict[str, str | None]) -> None:
    for name in ("events_url", "config_url"):
        value = fields.get(name)
        if value is not None and not _is_http_url(value):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"{name} must be an absolute HTTP(S) URL: {value}",
            )


def _is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.netloc != ""


def _effective(
    overrides: dict[str, str | None], settings: ConnectionSettings, field: str
) -> str | None:
    if field in overrides:
        return overrides[field]
    return getattr(settings, field)


def _trigger_roster_sync(app: FastAPI) -> None:
    """Same fire-and-forget BackgroundTask pattern as the cameras router's
    own trigger (backend/app/features/cameras/router.py::_trigger_roster_sync)
    -- duplicated rather than imported across routers to keep each router's
    background-task helper trivially local; sync_camera_roster() itself
    never raises, but this still guards against a future change there.
    """
    try:
        sync_camera_roster(app)
    except Exception:  # noqa: BLE001, S110 - a roster-sync bug must never surface here
        pass


def _kick_backend_config_refresh(app: FastAPI) -> None:
    """Best-effort, non-blocking nudge so the camera roster repulls promptly
    after a settings save. Reuses the single-flight executor/lock lifespan.py
    already owns (``refresh_backend_config`` acquires
    ``backend_config_refresh_lock`` non-blocking and no-ops if a refresh is
    already in flight) rather than adding a second refresh path.

    A no-op when the executor isn't present -- e.g. the app was booted via
    ``no_lifespan`` in tests, or the app is mid-shutdown. Either way the
    periodic refresh loop (or the next natural pull) still picks up the new
    settings; this is purely a latency optimization, never required for
    correctness.
    """
    executor = getattr(app.state, "backend_config_refresh_executor", None)
    if not isinstance(executor, ThreadPoolExecutor):
        return
    try:
        executor.submit(refresh_backend_config, app)
    except RuntimeError:
        pass  # executor already shutting down


def _probe(
    *, config_url: str | None, facility_id: str | None, facility_token: str | None
) -> dict[str, object]:
    if not config_url or not facility_id:
        return {
            "ok": False,
            "error_class": "unconfigured",
            "detail": (
                "설정 조회 URL 또는 시설 ID가 설정되지 않았습니다. "
                "연결 설정을 저장한 뒤 다시 시도하세요."
            ),
            "probed_url": None,
        }
    if not _is_http_url(config_url):
        return {
            "ok": False,
            "error_class": "invalid_url",
            "detail": (
                "설정된 URL 형식이 올바르지 않습니다. "
                "http:// 또는 https://로 시작하는 주소를 입력하세요."
            ),
            "probed_url": None,
        }

    probed_url = f"{config_url.rstrip('/')}/{facility_id}"
    headers = {"Accept": "application/json"}
    if facility_token:
        headers["Authorization"] = f"Bearer {facility_token}"
    timeout_sec = get_settings().connection_test_timeout_s
    probe_request = urllib.request.Request(probed_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(probe_request, timeout=timeout_sec) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return {
                "ok": False,
                "error_class": "auth",
                "detail": "인증에 실패했습니다. 시설 토큰을 다시 확인하세요.",
                "probed_url": probed_url,
            }
        return {
            "ok": False,
            "error_class": "unreachable",
            "detail": f"백엔드가 오류를 반환했습니다 (HTTP {exc.code}). 설정을 확인하세요.",
            "probed_url": probed_url,
        }
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return {
                "ok": False,
                "error_class": "timeout",
                "detail": "백엔드 응답 시간이 초과되었습니다. 네트워크 상태를 확인하세요.",
                "probed_url": probed_url,
            }
        return {
            "ok": False,
            "error_class": "unreachable",
            "detail": "백엔드에 연결할 수 없습니다. 주소와 네트워크 연결을 확인하세요.",
            "probed_url": probed_url,
        }
    except TimeoutError:
        return {
            "ok": False,
            "error_class": "timeout",
            "detail": "백엔드 응답 시간이 초과되었습니다. 네트워크 상태를 확인하세요.",
            "probed_url": probed_url,
        }
    return {
        "ok": True,
        "error_class": None,
        "detail": "백엔드 연결에 성공했습니다.",
        "probed_url": probed_url,
    }


__all__ = ["router"]
