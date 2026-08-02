"""Serving application lifespan assembly."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from threading import Lock

from fastapi import FastAPI

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.catalog import CatalogStore
from backend.app.features.status.heartbeat_store import DEFAULT_STALE_AFTER_SEC, HeartbeatStore
from backend.app.features.status.runtime_status_store import RuntimeStatusStore
from backend.app.shared.backend_mapping import (
    BackendCameraMapper,
    backend_status_from_env,
    mark_backend_status,
)
from contracts.worker_config import (
    PulledCameraConfig,
    PulledNightWindow,
    PulledWorkerConfig,
    detection_window_validation_error,
)
from shared.events.edge_ingest_client import (
    DEFAULT_TIMEOUT_SEC,
    BackendEvidenceClient,
    EdgeIngestClient,
)

API_BACKEND_EVENTS_URL_ENV = "API_BACKEND_EVENTS_URL"
API_EDGE_RELAY_TOKEN_ENV = "API_EDGE_RELAY_TOKEN"
# Shared secret for the ml-api -> backend Event API bearer auth (issue #552).
# Name matches the backend's EdgeFacilityTokenGuard config key exactly so the
# same value can be copied verbatim across the edge and host env files.
EDGE_FACILITY_TOKEN_ENV = "EDGE_FACILITY_TOKEN"
API_CAMERA_INVENTORY_ENV = "API_CAMERA_INVENTORY"
API_FACILITY_ID_ENV = "API_FACILITY_ID"
API_BACKEND_CONFIG_URL_ENV = "API_BACKEND_CONFIG_URL"
API_BACKEND_INGEST_TIMEOUT_SEC_ENV = "API_BACKEND_INGEST_TIMEOUT_SEC"
API_HEARTBEAT_STALE_AFTER_SEC_ENV = "API_HEARTBEAT_STALE_AFTER_SEC"
API_BACKEND_CONFIG_REFRESH_SEC_ENV = "API_BACKEND_CONFIG_REFRESH_SEC"

BACKEND_CONFIG_SHUTDOWN_WAIT_SEC = 1.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Boot ml-api as a thin backend gateway (ADR)."""
    _load_config(app)

    if not isinstance(getattr(app.state, "heartbeat_store", None), HeartbeatStore):
        app.state.heartbeat_store = HeartbeatStore(stale_after_sec=_heartbeat_stale_after_sec())
    if not isinstance(getattr(app.state, "runtime_status_store", None), RuntimeStatusStore):
        app.state.runtime_status_store = RuntimeStatusStore()

    if not isinstance(getattr(app.state, "camera_registry", None), CameraRegistryStore):
        app.state.camera_registry = CameraRegistryStore.from_env()
    if not isinstance(getattr(app.state, "backend_camera_mapper", None), BackendCameraMapper):
        app.state.backend_camera_mapper = BackendCameraMapper.from_env()
    backend_status = backend_status_from_env()
    app.state.backend_configured = backend_status["configured"]
    app.state.backend_reachable = getattr(
        app.state, "backend_reachable", backend_status["reachable"]
    )
    app.state.backend_last_ok_at = getattr(
        app.state, "backend_last_ok_at", backend_status["last_ok_at"]
    )

    app.state.restart_epoch = getattr(app.state, "restart_epoch", 0)
    app.state.config_version = getattr(app.state, "config_version", 0)
    app.state.pulled_config = getattr(app.state, "pulled_config", None)
    app.state.backend_config_refresh_lock = getattr(
        app.state, "backend_config_refresh_lock", Lock()
    )
    refresh_stop = getattr(app.state, "backend_config_refresh_stop", None)
    if not isinstance(refresh_stop, asyncio.Event) or refresh_stop.is_set():
        refresh_stop = asyncio.Event()
        app.state.backend_config_refresh_stop = refresh_stop
    refresh_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="backend-config-refresh"
    )
    app.state.backend_config_refresh_executor = refresh_executor

    _configure_backend_ingest(app)
    await asyncio.get_running_loop().run_in_executor(
        refresh_executor, _pull_backend_config, app, refresh_stop
    )
    app.state.backend_config_refresh_task = asyncio.create_task(
        _backend_config_refresh_loop(app, refresh_stop, refresh_executor),
        name="backend-config-refresh",
    )
    app.state.readiness = {"ready": True, "status": "ready"}
    try:
        yield
    finally:
        refresh_stop.set()
        refresh_task = app.state.backend_config_refresh_task
        try:
            await asyncio.wait_for(
                asyncio.shield(refresh_task), timeout=BACKEND_CONFIG_SHUTDOWN_WAIT_SEC
            )
        except TimeoutError:
            refresh_task.cancel()
        refresh_executor.shutdown(wait=False, cancel_futures=True)
        app.state.backend_config_refresh_executor = None
        app.state.backend_config_refresh_task = None
        catalog_store = getattr(app.state, "catalog_store", None)
        if isinstance(catalog_store, CatalogStore):
            catalog_store.close()


def _configure_backend_ingest(app: FastAPI) -> None:
    if not hasattr(app.state, "edge_relay_token"):
        app.state.edge_relay_token = os.environ.get(API_EDGE_RELAY_TOKEN_ENV)
    if not hasattr(app.state, "camera_inventory"):
        app.state.camera_inventory = _camera_inventory_from_env_or_state(app)
    if hasattr(app.state, "backend_ingest_client"):
        if not hasattr(app.state, "backend_evidence_client"):
            existing = app.state.backend_ingest_client
            if isinstance(existing, EdgeIngestClient):
                app.state.backend_evidence_client = BackendEvidenceClient(
                    existing.events_url, existing.bearer_token, existing.timeout_sec
                )
        return

    events_url = os.environ.get(API_BACKEND_EVENTS_URL_ENV)
    if not events_url:
        return

    first_camera = next(iter(app.state.camera_inventory.values()), {})
    app.state.backend_ingest_client = EdgeIngestClient(
        events_url=events_url,
        camera_id=str(first_camera.get("camera_id", "api-relay")),
        timeout_sec=_backend_ingest_timeout_sec(),
        bearer_token=os.environ.get(EDGE_FACILITY_TOKEN_ENV),
    )
    app.state.backend_evidence_client = BackendEvidenceClient(
        events_url=events_url,
        bearer_token=os.environ.get(EDGE_FACILITY_TOKEN_ENV),
        timeout_sec=_backend_ingest_timeout_sec(),
    )


def _pull_backend_config(app: FastAPI, stop_token: asyncio.Event) -> None:
    """Boot-time backend config pull."""
    refresh_backend_config(app, stop_token)


def refresh_backend_config(app: FastAPI, stop_token: asyncio.Event | None = None) -> bool:
    """Refresh the cached backend roster without discarding last-known-good data."""
    if stop_token is None:
        candidate = getattr(app.state, "backend_config_refresh_stop", None)
        stop_token = candidate if isinstance(candidate, asyncio.Event) else None
    refresh_lock = getattr(app.state, "backend_config_refresh_lock", None)
    if not hasattr(refresh_lock, "acquire"):
        refresh_lock = Lock()
        app.state.backend_config_refresh_lock = refresh_lock
    if not refresh_lock.acquire(blocking=False):
        return False
    try:
        if not _backend_config_refresh_is_current(app, stop_token):
            return False
        restart_epoch = int(getattr(app.state, "restart_epoch", 0))
        cfg = _fetch_backend_config(restart_epoch)
        if not _backend_config_refresh_is_current(app, stop_token):
            return False
        if cfg is None:
            mark_backend_status(app.state, False)
            _mark_backend_roster_stale(app)
            return False
        mark_backend_status(app.state, True)
        _apply_backend_config(app, cfg)
        # A reachable backend is the moment pending explicit camera mappings
        # can converge; the refresh owner is single-flight, so this stays
        # bounded and never runs concurrently with itself. Imported lazily:
        # api.routes.cameras imports lifespan env-name constants at module load.
        from backend.app.features.cameras.router import retry_pending_backend_mappings

        retry_pending_backend_mappings(app)
        return True
    finally:
        refresh_lock.release()


async def _backend_config_refresh_loop(
    app: FastAPI, stop_event: asyncio.Event, executor: ThreadPoolExecutor
) -> None:
    """Own the single periodic backend pull for this application instance."""
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_backend_config_refresh_sec())
        except TimeoutError:
            pass
        if stop_event.is_set():
            break
        await asyncio.get_running_loop().run_in_executor(
            executor, refresh_backend_config, app, stop_event
        )


def _backend_config_refresh_is_current(app: FastAPI, stop_token: asyncio.Event | None) -> bool:
    if stop_token is None:
        return True
    return (
        getattr(app.state, "backend_config_refresh_stop", None) is stop_token
        and not stop_token.is_set()
    )


def _fetch_backend_config(restart_epoch: int) -> PulledWorkerConfig | None:
    facility_id = os.environ.get(API_FACILITY_ID_ENV)
    base_url = os.environ.get(API_BACKEND_CONFIG_URL_ENV)
    if not facility_id or not base_url:
        return None
    try:
        url = f"{base_url.rstrip('/')}/{facility_id}"
        # The production backend guards the RTSP-bearing ml-config read with the
        # same shared edge bearer the Event API ingest already sends.
        headers: dict[str, str] = {"Accept": "application/json"}
        bearer = os.environ.get(EDGE_FACILITY_TOKEN_ENV)
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        # urlopen applies this timeout to both the connect and socket reads.
        # Keep it within the lifespan shutdown wait bound.
        with urllib.request.urlopen(request, timeout=_backend_config_timeout_sec()) as response:
            parsed = _as_mapping(json.loads(response.read().decode("utf-8")))
        detection_windows = _pulled_detection_windows(parsed)
        return PulledWorkerConfig(
            config_version=_backend_config_version(parsed),
            restart_epoch=restart_epoch,
            night_window=detection_windows.get("bed_exit"),
            cameras=_pulled_cameras(parsed.get("cameras")),
            detection_windows=detection_windows,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort pull must never crash boot/serve
        print(f"failed to pull backend ml config: {exc}", file=sys.stderr)
        return None


def _apply_backend_config(app: FastAPI, cfg: PulledWorkerConfig) -> None:
    facility_id = os.environ.get(API_FACILITY_ID_ENV)
    app.state.pulled_config = cfg
    app.state.config_version = cfg.config_version
    app.state.backend_roster = {
        "config_version": cfg.config_version,
        "received_at": _utc_now(),
        "stale": False,
    }
    app.state.camera_inventory = {
        camera.camera_id: {
            "camera_id": camera.camera_id,
            "facility_id": facility_id,
            "resident_id": None,
        }
        for camera in cfg.cameras
    }


def _mark_backend_roster_stale(app: FastAPI) -> None:
    previous = getattr(app.state, "backend_roster", {})
    received_at = previous.get("received_at") if isinstance(previous, dict) else None
    app.state.backend_roster = {
        "config_version": int(getattr(app.state, "config_version", 0)),
        "received_at": received_at,
        "stale": True,
    }


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("backend config response must be an object")
    return value


def _backend_config_version(data: dict[str, object]) -> int:
    value = data.get("configVersion")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("configVersion must be an integer")
    return value


def _pulled_night_window(domain: str, value: object) -> PulledNightWindow | None:
    """Parse and validate one raw ``{start,end,tz}`` window, failing open
    (returning ``None``, i.e. ALWAYS/24-7 for that domain) on any structural
    or semantic problem rather than raising -- a malformed value from a
    single domain must never crash the whole backend-config pull."""
    if value is None:
        return None
    if not isinstance(value, dict):
        _log_invalid_window(domain, value, "must be an object or null")
        return None
    try:
        window = PulledNightWindow(
            start=_require_text(value, "start"),
            end=_require_text(value, "end"),
            tz=_require_text(value, "tz"),
        )
    except (ValueError, TypeError) as exc:
        _log_invalid_window(domain, value, str(exc))
        return None
    reason = detection_window_validation_error(window.start, window.end, window.tz)
    if reason is not None:
        _log_invalid_window(domain, value, reason)
        return None
    return window


def _log_invalid_window(domain: str, value: object, reason: str) -> None:
    print(
        f"detection window for domain {domain!r} is invalid ({reason}): {value!r}; "
        "falling open to ALWAYS/24-7 detection for this domain",
        file=sys.stderr,
    )


def _pulled_detection_windows(parsed: dict[str, object]) -> dict[str, PulledNightWindow]:
    """Parse backend ``detectionWindows`` (domain -> {start,end,tz} | null).

    If ``detectionWindows`` is present at all, that map is the sole
    authority for every domain and the legacy ``nightWindow`` field is
    ignored entirely -- this is what makes a window an operator clears in
    the dashboard stay cleared instead of being resurrected by a stale
    ``nightWindow``. Only when ``detectionWindows`` is absent does the
    legacy single ``nightWindow`` field fall back to the "bed_exit" domain.
    Unknown domain names are kept in the map (forward-compatible) rather
    than rejected.
    """
    raw_map = parsed.get("detectionWindows")
    if raw_map is not None:
        if not isinstance(raw_map, dict):
            print(
                "detectionWindows must be an object or null; ignoring and "
                "falling back to legacy nightWindow",
                file=sys.stderr,
            )
        else:
            windows: dict[str, PulledNightWindow] = {}
            for domain, value in raw_map.items():
                if not isinstance(domain, str):
                    continue
                window = _pulled_night_window(domain, value)
                if window is not None:
                    windows[domain] = window
            return windows
    legacy_window = _pulled_night_window("bed_exit", parsed.get("nightWindow"))
    return {} if legacy_window is None else {"bed_exit": legacy_window}


def _pulled_cameras(value: object) -> tuple[PulledCameraConfig, ...]:
    if not isinstance(value, list):
        raise TypeError("cameras must be a list")
    cameras: list[PulledCameraConfig] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("each camera must be an object")
        cameras.append(_pulled_camera(item))
    return tuple(cameras)


def _pulled_camera(data: dict[str, object]) -> PulledCameraConfig:
    return PulledCameraConfig(
        camera_id=_require_text(data, "id"),
        space_id=_require_text(data, "spaceId"),
        label=_require_text(data, "label"),
        rtsp_url=_optional_text(data, "rtspUrl"),
        online=bool(data.get("online", False)),
        space_name=_optional_text(data, "spaceName"),
        floor_name=_optional_text(data, "floorName"),
        created_at=_optional_text(data, "createdAt"),
    )


def _require_text(data: dict[str, object], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(data: dict[str, object], name: str) -> str | None:
    value = data.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or null")
    return value


def _camera_inventory_from_env_or_state(app: FastAPI) -> dict[str, dict[str, str | None]]:
    raw_inventory = os.environ.get(API_CAMERA_INVENTORY_ENV)
    if raw_inventory:
        parsed = json.loads(raw_inventory)
        if not isinstance(parsed, list):
            raise ValueError(f"{API_CAMERA_INVENTORY_ENV} must be a JSON list")
        return _camera_inventory_from_items(parsed)

    camera_configs = getattr(app.state, "camera_configs", ())
    return _camera_inventory_from_items(camera_configs)


def _camera_inventory_from_items(items: object) -> dict[str, dict[str, str | None]]:
    inventory: dict[str, dict[str, str | None]] = {}
    for item in items:
        camera_id = _item_text(item, "camera_id")
        facility_id = _item_text(item, "facility_id")
        if camera_id is None or facility_id is None:
            continue
        inventory[camera_id] = {
            "camera_id": camera_id,
            "facility_id": facility_id,
            "resident_id": _item_text(item, "resident_id"),
        }
    return inventory


def _item_text(item: object, name: str) -> str | None:
    value = item.get(name) if isinstance(item, dict) else getattr(item, name, None)
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _backend_ingest_timeout_sec() -> float:
    raw = os.environ.get(API_BACKEND_INGEST_TIMEOUT_SEC_ENV)
    if raw is None:
        return DEFAULT_TIMEOUT_SEC
    return float(raw)


def _backend_config_timeout_sec() -> float:
    return min(_backend_ingest_timeout_sec(), BACKEND_CONFIG_SHUTDOWN_WAIT_SEC)


def _backend_config_refresh_sec() -> float:
    raw = os.environ.get(API_BACKEND_CONFIG_REFRESH_SEC_ENV)
    if raw is None:
        return 30.0
    try:
        return min(max(float(raw), 1.0), 3600.0)
    except ValueError:
        return 30.0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _heartbeat_stale_after_sec() -> float:
    raw = os.environ.get(API_HEARTBEAT_STALE_AFTER_SEC_ENV)
    if raw is None:
        return DEFAULT_STALE_AFTER_SEC
    return float(raw)


def _load_config(app: FastAPI) -> None:
    loader = getattr(app.state, "config_loader", None)
    if callable(loader):
        app.state.config = loader()
    validator = getattr(app.state, "config_validator", None)
    if callable(validator):
        validator(getattr(app.state, "config", None))
