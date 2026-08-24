"""Serving application lifespan assembly."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import urllib.request
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Protocol, TypeGuard

from fastapi import FastAPI

from backend.app.core.config import reject_retired_backend_environment
from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.features.audit.startup import configure_audit_readiness
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.catalog import CatalogStore
from backend.app.features.clips.listing_runtime import maintain_clip_listing
from backend.app.features.status.backend_heartbeat_relay import (
    effective_relay_interval_sec,
    get_heartbeat_relay_state,
    relay_heartbeats_once,
)
from backend.app.features.status.heartbeat_store import DEFAULT_STALE_AFTER_SEC, HeartbeatStore
from backend.app.features.status.runtime_status_store import RuntimeStatusStore
from backend.app.shared.backend_client_bundle import (
    BackendClientBundle,
    backend_client_bundle,
)
from backend.app.shared.backend_mapping import (
    BackendCameraMapper,
    derive_edge_cameras_endpoint,
    mark_backend_status,
)
from backend.app.shared.state_dir import resolve_state_dir
from contracts.worker_config import (
    PulledCameraConfig,
    PulledNightWindow,
    PulledWorkerConfig,
    detection_window_validation_error,
)
from shared.events.edge_ingest_client import (
    BackendEvidenceClient,
    EdgeIngestClient,
)

API_BACKEND_EVENTS_URL_ENV = "API_BACKEND_EVENTS_URL"
API_EDGE_RELAY_TOKEN_ENV = "API_EDGE_RELAY_TOKEN"
# Shared secret for the ml-api -> backend Event API bearer auth (issue #552).
# Name matches the backend's EdgeFacilityTokenGuard config key exactly so the
# same value can be copied verbatim across the edge and host env files.
EDGE_FACILITY_TOKEN_ENV = "EDGE_FACILITY_TOKEN"  # scope-fidelity: name-only
# Retained as a name-only constant for tests/docs that assert the env key is
# no longer an admission authority. Production code must not read this env.
API_FACILITY_ID_ENV = "API_FACILITY_ID"  # scope-fidelity: name-only
# Same name-only retention as the two constants above: the retired ml-config
# URL is asserted by tests/docs as no longer being an authority, and no
# production code reads it.
API_BACKEND_CONFIG_URL_ENV = "API_BACKEND_CONFIG_URL"  # scope-fidelity: name-only
API_BACKEND_INGEST_TIMEOUT_SEC_ENV = "API_BACKEND_INGEST_TIMEOUT_SEC"
API_HEARTBEAT_STALE_AFTER_SEC_ENV = "API_HEARTBEAT_STALE_AFTER_SEC"
API_BACKEND_CONFIG_REFRESH_SEC_ENV = "API_BACKEND_CONFIG_REFRESH_SEC"
API_BACKEND_HEARTBEAT_RELAY_SEC_ENV = "API_BACKEND_HEARTBEAT_RELAY_SEC"

BACKEND_CONFIG_SHUTDOWN_WAIT_SEC = 1.0
BACKEND_HEARTBEAT_RELAY_SHUTDOWN_WAIT_SEC = 1.0


class InvalidBackendIngestTimeoutError(ValueError):
    """The public ingest timeout is malformed or outside the finite positive domain."""


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Boot ml-api as a thin backend gateway (ADR)."""
    reject_retired_backend_environment(os.environ)
    logger.info("ml-api state directory resolved to %s", resolve_state_dir("ml-api"))
    _load_config(app)

    audit_healthy = configure_audit_readiness(app, EDGE_DATABASE_PATH)

    if not isinstance(getattr(app.state, "heartbeat_store", None), HeartbeatStore):
        app.state.heartbeat_store = HeartbeatStore(
            stale_after_sec=_heartbeat_stale_after_sec(), database_path=EDGE_DATABASE_PATH
        )
    if not isinstance(getattr(app.state, "runtime_status_store", None), RuntimeStatusStore):
        app.state.runtime_status_store = RuntimeStatusStore()

    if not isinstance(getattr(app.state, "camera_registry", None), CameraRegistryStore):
        app.state.camera_registry = CameraRegistryStore.from_env()
    _configure_backend_ingest(app)
    bundle = backend_client_bundle(app)
    app.state.backend_configured = bundle is not None
    app.state.backend_reachable = getattr(app.state, "backend_reachable", None)
    app.state.backend_last_ok_at = getattr(app.state, "backend_last_ok_at", None)

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

    from backend.app.features.cameras.roster_sync import recover_camera_roster_on_boot

    await asyncio.get_running_loop().run_in_executor(
        refresh_executor, recover_camera_roster_on_boot, app
    )
    await asyncio.get_running_loop().run_in_executor(
        refresh_executor, _pull_backend_config, app, refresh_stop
    )
    app.state.backend_config_refresh_task = asyncio.create_task(
        _backend_config_refresh_loop(app, refresh_stop, refresh_executor),
        name="backend-config-refresh",
    )

    relay_interval_sec = _backend_heartbeat_relay_sec()
    relay_stop = getattr(app.state, "backend_heartbeat_relay_stop", None)
    if not isinstance(relay_stop, asyncio.Event) or relay_stop.is_set():
        relay_stop = asyncio.Event()
        app.state.backend_heartbeat_relay_stop = relay_stop
    if relay_interval_sec > 0:
        # Dedicated 1-worker executor, mirroring refresh_executor above,
        # rather than sharing it: the relay tick and a backend-config pull
        # are independent concerns and neither should be able to make the
        # other wait behind it in the same executor's single worker thread.
        relay_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="backend-heartbeat-relay"
        )
        app.state.backend_heartbeat_relay_executor = relay_executor
        app.state.backend_heartbeat_relay_task = asyncio.create_task(
            _backend_heartbeat_relay_loop(app, relay_stop, relay_executor, relay_interval_sec),
            name="backend-heartbeat-relay",
        )
    else:
        # env=0 or unset-invalid -- relay disabled (kill-switch).
        app.state.backend_heartbeat_relay_executor = None
        app.state.backend_heartbeat_relay_task = None

    clip_listing_stack = AsyncExitStack()
    await clip_listing_stack.enter_async_context(maintain_clip_listing(app))
    app.state.readiness = (
        {"ready": True, "status": "ready"}
        if audit_healthy
        else {"ready": False, "status": "degraded", "reason": "audit unavailable"}
    )
    try:
        yield
    finally:
        await clip_listing_stack.aclose()
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

        relay_stop.set()
        relay_task = app.state.backend_heartbeat_relay_task
        if relay_task is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(relay_task), timeout=BACKEND_HEARTBEAT_RELAY_SHUTDOWN_WAIT_SEC
                )
            except TimeoutError:
                relay_task.cancel()
            relay_executor = app.state.backend_heartbeat_relay_executor
            if relay_executor is not None:
                relay_executor.shutdown(wait=False, cancel_futures=True)
            app.state.backend_heartbeat_relay_executor = None
            app.state.backend_heartbeat_relay_task = None

        catalog_store = getattr(app.state, "catalog_store", None)
        if isinstance(catalog_store, CatalogStore):
            catalog_store.close()


def _configure_backend_ingest(app: FastAPI) -> None:
    if not hasattr(app.state, "edge_relay_token"):
        app.state.edge_relay_token = os.environ.get(API_EDGE_RELAY_TOKEN_ENV)
    if hasattr(app.state, "backend_ingest_client"):
        if not hasattr(app.state, "backend_evidence_client"):
            existing = app.state.backend_ingest_client
            if isinstance(existing, EdgeIngestClient):
                app.state.backend_evidence_client = BackendEvidenceClient(
                    existing.events_url, existing.bearer_token, existing.timeout_sec
                )
        # A test/caller already assigned backend_ingest_client before lifespan
        # ran (fixture-injection pattern, e.g. tests/test_api_ingest_relay.py).
        # Leave it alone at boot -- runtime rebuilds only ever happen via an
        # explicit apply_connection_settings(app) call (G003's settings-save
        # route will be the caller), never from this boot path or the
        # periodic refresh loop.
        return

    apply_connection_settings(app)


def apply_connection_settings(app: FastAPI) -> None:
    """Atomically publish all cloud clients for one persisted enrollment generation."""
    from backend.app.features.connection.store import ConnectionSettingsStore

    settings = ConnectionSettingsStore.from_env().load()
    required = (
        settings.events_url,
        settings.config_url,
        settings.facility_code,
        settings.client_installation_ref,
        settings.facility_id,
        settings.facility_token,
        settings.edge_installation_id,
        settings.enrollment_generation,
    )
    if any(value is None for value in required):
        for attribute in (
            "backend_client_bundle",
            "backend_ingest_client",
            "backend_evidence_client",
            "backend_camera_mapper",
        ):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)
        app.state.backend_configured = False
        return
    assert settings.events_url is not None
    assert settings.config_url is not None
    assert settings.facility_code is not None
    assert settings.client_installation_ref is not None
    assert settings.facility_id is not None
    assert settings.facility_token is not None
    assert settings.edge_installation_id is not None
    assert settings.enrollment_generation is not None
    # camera_id fallback identity for the rebuilt EdgeIngestClient. Every real
    # caller reaches the client through `.for_camera()` (relay/evidence
    # routers), which overrides camera_id per request, so this default is
    # currently inert -- kept only for EdgeIngestClient's required constructor
    # field. The camera registry is the sole camera SSOT; nothing here reads
    # a locally cached camera roster.
    timeout_sec = _backend_ingest_timeout_sec()
    ingest_client = EdgeIngestClient(
        events_url=settings.events_url,
        camera_id="api-relay",
        timeout_sec=timeout_sec,
        bearer_token=settings.facility_token,
    )
    evidence_client = BackendEvidenceClient(
        events_url=settings.events_url,
        bearer_token=settings.facility_token,
        timeout_sec=timeout_sec,
    )
    camera_mapper = BackendCameraMapper(
        endpoint=derive_edge_cameras_endpoint(settings.events_url),
        token=settings.facility_token,
        timeout_sec=timeout_sec,
    )
    bundle = BackendClientBundle(
        facility_code=settings.facility_code,
        client_installation_ref=settings.client_installation_ref,
        facility_id=settings.facility_id,
        edge_installation_id=settings.edge_installation_id,
        enrollment_generation=settings.enrollment_generation,
        facility_token=settings.facility_token,
        events_url=settings.events_url,
        config_url=settings.config_url,
        ingest_client=ingest_client,
        evidence_client=evidence_client,
        camera_mapper=camera_mapper,
    )
    app.state.backend_client_bundle = bundle
    app.state.backend_ingest_client = bundle.ingest_client
    app.state.backend_evidence_client = bundle.evidence_client
    app.state.backend_camera_mapper = bundle.camera_mapper
    app.state.backend_configured = True


def _pull_backend_config(app: FastAPI, stop_token: asyncio.Event) -> None:
    """Boot-time backend config pull."""
    refresh_backend_config(app, stop_token)


def refresh_backend_config(app: FastAPI, stop_token: asyncio.Event | None = None) -> bool:
    """Refresh the cached backend roster without discarding last-known-good data."""
    if stop_token is None:
        candidate = getattr(app.state, "backend_config_refresh_stop", None)
        stop_token = candidate if isinstance(candidate, asyncio.Event) else None
    refresh_lock = _backend_config_refresh_lock(app)
    if not refresh_lock.acquire(blocking=False):
        return False
    try:
        if not _backend_config_refresh_is_current(app, stop_token):
            return False
        bundle = backend_client_bundle(app)
        if bundle is None:
            _mark_app_backend_status(app, None)
            _mark_backend_roster_stale(app)
            return False
        restart_epoch = _as_int(getattr(app.state, "restart_epoch", 0), default=0)
        cfg = _fetch_backend_config(bundle, restart_epoch)
        if not _backend_config_refresh_is_current(app, stop_token):
            return False
        if cfg is None or backend_client_bundle(app) is not bundle:
            _mark_app_backend_status(app, False)
            _mark_backend_roster_stale(app)
            return False
        was_reachable = getattr(app.state, "backend_reachable", None)
        _mark_app_backend_status(app, True)
        _apply_backend_config(app, cfg)
        if was_reachable is not True:
            from backend.app.features.cameras.roster_sync import (
                resume_camera_roster_after_connectivity,
            )

            resume_camera_roster_after_connectivity(app)
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


async def _backend_heartbeat_relay_loop(
    app: FastAPI,
    stop_event: asyncio.Event,
    executor: ThreadPoolExecutor,
    base_interval_sec: float,
) -> None:
    """Own the periodic per-camera heartbeat relay to the external backend.

    Mirrors ``_backend_config_refresh_loop``'s wait/tick/repeat shape. The
    wait between ticks widens via ``relay_heartbeats_once``'s backoff state
    after consecutive all-fail ticks (external backend down/unreachable) and
    snaps back to ``base_interval_sec`` the moment any send succeeds.
    """
    while not stop_event.is_set():
        relay_state = get_heartbeat_relay_state(app)
        wait_sec = effective_relay_interval_sec(base_interval_sec, relay_state)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=wait_sec)
        except TimeoutError:
            pass
        if stop_event.is_set():
            break
        await asyncio.get_running_loop().run_in_executor(executor, relay_heartbeats_once, app)


def _backend_config_refresh_is_current(app: FastAPI, stop_token: asyncio.Event | None) -> bool:
    if stop_token is None:
        return True
    return (
        getattr(app.state, "backend_config_refresh_stop", None) is stop_token
        and not stop_token.is_set()
    )


def _fetch_backend_config(
    bundle: BackendClientBundle, restart_epoch: int
) -> PulledWorkerConfig | None:
    try:
        url = f"{bundle.config_url.rstrip('/')}/{bundle.facility_id}"
        # The production backend guards the RTSP-bearing ml-config read with the
        # same shared edge bearer the Event API ingest already sends.
        headers: dict[str, str] = {"Accept": "application/json"}
        headers["Authorization"] = f"Bearer {bundle.facility_token}"
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
    """Apply non-camera ml-config metadata only.

    Detection windows / config_version land on app.state for worker-config
    merge. Pulled ``cameras`` are intentionally ignored as a local admission
    or enumeration authority -- the dashboard camera registry is the sole
    camera SSOT (see AGENTS.md anti-pattern: no pre-provisioned camera rosters).
    """
    app.state.pulled_config = cfg
    app.state.config_version = cfg.config_version
    app.state.backend_roster = {
        "config_version": cfg.config_version,
        "received_at": _utc_now(),
        "stale": False,
    }


def _mark_backend_roster_stale(app: FastAPI) -> None:
    previous = getattr(app.state, "backend_roster", {})
    received_at = previous.get("received_at") if isinstance(previous, dict) else None
    app.state.backend_roster = {
        "config_version": _as_int(getattr(app.state, "config_version", 0), default=0),
        "received_at": received_at,
        "stale": True,
    }


class _RefreshLock(Protocol):
    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool: ...

    def release(self) -> None: ...


def _is_refresh_lock(value: object) -> TypeGuard[_RefreshLock]:
    return callable(getattr(value, "acquire", None)) and callable(getattr(value, "release", None))


def _backend_config_refresh_lock(app: FastAPI) -> _RefreshLock:
    candidate = getattr(app.state, "backend_config_refresh_lock", None)
    if _is_refresh_lock(candidate):
        return candidate
    refresh_lock = Lock()
    app.state.backend_config_refresh_lock = refresh_lock
    return refresh_lock


@dataclass(slots=True)
class _BackendStatusBuffer:
    backend_reachable: bool | None
    backend_last_ok_at: str | None


def _mark_app_backend_status(app: FastAPI, reachable: bool | None) -> None:
    """Copy FastAPI dynamic state into a typed buffer, mutate, write back."""
    reachable_raw = getattr(app.state, "backend_reachable", None)
    last_ok_raw = getattr(app.state, "backend_last_ok_at", None)
    buffer = _BackendStatusBuffer(
        backend_reachable=reachable_raw if isinstance(reachable_raw, bool) else None,
        backend_last_ok_at=last_ok_raw if isinstance(last_ok_raw, str) else None,
    )
    mark_backend_status(buffer, reachable)
    app.state.backend_reachable = buffer.backend_reachable
    app.state.backend_last_ok_at = buffer.backend_last_ok_at


def _as_int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("backend config response must be an object")
    return {str(key): item for key, item in value.items()}


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


def _backend_ingest_timeout_sec() -> float:
    raw = os.environ.get(API_BACKEND_INGEST_TIMEOUT_SEC_ENV)
    if raw is None:
        return 10.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise InvalidBackendIngestTimeoutError(
            f"{API_BACKEND_INGEST_TIMEOUT_SEC_ENV} must be a finite positive number, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise InvalidBackendIngestTimeoutError(
            f"{API_BACKEND_INGEST_TIMEOUT_SEC_ENV} must be a finite positive number, got {raw!r}"
        )
    return value


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


def _backend_heartbeat_relay_sec() -> float:
    """Relay interval in seconds; ``<= 0`` or unparseable disables the relay.

    Unlike ``_backend_config_refresh_sec``, a malformed value here does NOT
    fall back to the 30s default -- it fails safe to disabled, since a typo'd
    env var should never silently start egress to an external backend.
    """
    raw = os.environ.get(API_BACKEND_HEARTBEAT_RELAY_SEC_ENV)
    if raw is None:
        return 30.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return value if value > 0 else 0.0


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
