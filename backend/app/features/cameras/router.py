"""Camera registry routes."""

from __future__ import annotations

import hmac
import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import get_settings
from backend.app.features.cameras.store import (
    CameraRegistryStore,
    DuplicateCameraError,
    ProbeErrorClass,
    ProbeResult,
    public_camera,
    status_from_probe,
    utc_now_iso,
)
from backend.app.features.status.heartbeat_store import ONLINE, get_heartbeat_store
from backend.app.lifespan import API_EDGE_RELAY_TOKEN_ENV, API_FACILITY_ID_ENV
from backend.app.shared.backend_mapping import (
    BackendCameraMapper,
    MappingResult,
    mark_backend_status,
)
from backend.app.shared.dashboard_auth import authorize_dashboard
from contracts.worker_config import PulledWorkerConfig

RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"

router = APIRouter(prefix="/cameras", tags=["cameras"])


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
    created_at: str | None = None
    space_name: str | None = None
    floor_name: str | None = None
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
    # Default False: registration probes the stream first and writes nothing
    # on failure (see create_camera). True bypasses that gate for cameras
    # that are known offline but should still be registered.
    force_register: bool = False


class UpdateCameraRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(default=None, min_length=1)
    rtsp_url: str | None = Field(default=None, min_length=1)
    space_id: str | None = None
    decode_backend: str | None = None


class TestCameraResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    error_class: Literal["timeout", "decode", "auth"] | None = None
    width: int | None = None
    height: int | None = None


class WorkerCameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1)
    facility_id: str = Field(min_length=1)
    rtsp_url: str = Field(min_length=1)
    fps: float | None = Field(default=None, gt=0)
    decode_backend: str | None = Field(default=None)
    domains: list[str] | None = None


class WorkerConfigResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry_version: int = Field(ge=0)
    cameras: list[WorkerCameraConfig]
    config_version: int | None = Field(default=None, ge=0)
    restart_epoch: int | None = Field(default=None, ge=0)
    night_window: dict[str, object] | None = None


@router.get("", response_model=ListCamerasResponse)
def list_cameras(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, relay_token, authorization)
    heartbeats = get_heartbeat_store(request.app).snapshot()
    return _public_snapshot(
        _store(request.app).snapshot(),
        getattr(request.app.state, "pulled_config", None),
        heartbeats,
    )


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CameraResponse)
def create_camera(
    payload: CreateCameraRequest,
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, relay_token, authorization)
    decode_backend = _normalize_decode_backend(payload.decode_backend)
    probe = _probe_rtsp_url(request, payload.rtsp_url)
    if not probe.ok and not payload.force_register:
        # Registration gating: a dead camera is not persisted unless the
        # caller explicitly opts in via force_register. Nothing is written.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "probe_failed",
                "error_class": probe.error_class or "decode",
            },
        )
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
            last_probed_at=now,
            last_ok_at=now if probe.ok else None,
            never_connected=not probe.ok,
        )
    except DuplicateCameraError as exc:
        raise _duplicate_camera_error(exc) from exc
    return public_camera(record)


@router.post(
    "/{camera_id}/test",
    response_model=TestCameraResponse,
    response_model_exclude_none=True,
)
def test_camera(
    camera_id: str,
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, relay_token, authorization)
    record = _store(request.app).get(camera_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    probe = _probe_rtsp_url(request, str(record.get("rtsp_url", "")))
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
    return public_camera(updated)


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
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> Response:
    _authorize(request, relay_token, authorization)
    if not _store(request.app).delete(camera_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
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
    cameras = []
    for record in _snapshot_camera_records(snapshot):
        rtsp_url = record.get("rtsp_url")
        if not isinstance(rtsp_url, str) or not rtsp_url.strip():
            continue
        canonical_id = record.get("backend_camera_id") or record.get("id", "")
        camera: dict[str, object] = {
            "camera_id": str(canonical_id),
            "facility_id": facility_id,
            "rtsp_url": rtsp_url,
        }
        fps = _default_camera_fps()
        if fps is not None:
            camera["fps"] = fps
        decode_backend = record.get("decode_backend") or _default_decode_backend()
        if decode_backend is not None:
            camera["decode_backend"] = decode_backend
        cameras.append(camera)
    pulled = getattr(request.app.state, "pulled_config", None)
    if not cameras and isinstance(pulled, PulledWorkerConfig):
        cameras = _worker_cameras_from_pulled_config(pulled, facility_id)
    if require_available and not cameras:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker config unavailable",
        )
    response: dict[str, object] = {
        "registry_version": snapshot["registry_version"],
        "cameras": cameras,
    }
    if isinstance(pulled, PulledWorkerConfig):
        live_pulled = _live_pulled_config(request, pulled)
        response["config_version"] = live_pulled.config_version
        response["restart_epoch"] = live_pulled.restart_epoch
        if live_pulled.night_window is not None:
            response["night_window"] = live_pulled.night_window.as_dict()
    return response


def _worker_cameras_from_pulled_config(
    pulled: PulledWorkerConfig, facility_id: str
) -> list[dict[str, object]]:
    cameras: list[dict[str, object]] = []
    for camera in pulled.cameras:
        if camera.rtsp_url is None or not camera.rtsp_url.strip():
            continue
        worker_camera: dict[str, object] = {
            "camera_id": camera.camera_id,
            "facility_id": facility_id,
            "rtsp_url": camera.rtsp_url,
        }
        fps = _default_camera_fps()
        if fps is not None:
            worker_camera["fps"] = fps
        decode_backend = _default_decode_backend()
        if decode_backend is not None:
            worker_camera["decode_backend"] = decode_backend
        cameras.append(worker_camera)
    return cameras

def _live_pulled_config(request: Request, pulled: PulledWorkerConfig) -> PulledWorkerConfig:
    return PulledWorkerConfig(
        config_version=int(getattr(request.app.state, "config_version", 0)),
        restart_epoch=int(getattr(request.app.state, "restart_epoch", 0)),
        night_window=pulled.night_window,
        cameras=pulled.cameras,
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
    snapshot: dict[str, object], pulled: object, heartbeats: object = None
) -> dict[str, object]:
    records = _snapshot_camera_records(snapshot)
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
                "created_at": backend_camera.created_at,
                "space_name": backend_camera.space_name,
                "floor_name": backend_camera.floor_name,
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
    if not hmac.compare_digest(relay_token, expected):
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
        return ProbeResult(ok=False, error_class="decode")
    token = _expected_relay_token(request)
    if token is None:
        return ProbeResult(ok=False, error_class="decode")
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
        return ProbeResult(ok=False, error_class="timeout")
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return ProbeResult(ok=False, error_class="decode")
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
    expected = getattr(request.app.state, "edge_relay_token", None) or os.environ.get(
        API_EDGE_RELAY_TOKEN_ENV
    )
    return expected if isinstance(expected, str) and expected else None


def _probe_response(probe: ProbeResult) -> dict[str, object]:
    response: dict[str, object] = {"ok": probe.ok}
    if probe.error_class is not None:
        response["error_class"] = probe.error_class
    if probe.width is not None:
        response["width"] = probe.width
    if probe.height is not None:
        response["height"] = probe.height
    return response


__all__ = ["retry_pending_backend_mappings", "router", "worker_config_snapshot"]
