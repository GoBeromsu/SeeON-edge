from __future__ import annotations

import sys
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import TypeAdapter, ValidationError

from contracts.worker_config import WORKER_CONFIG_PATH, PulledWorkerConfig
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.http_transport import UrlOpen, stdlib_urlopen
from worker.runtime.config.lkg_store import (
    JsonObject,
    StoredConfigPayload,
    WorkerConfigLkgStore,
)
from worker.runtime.config.local_env import resolve_local_overrides
from worker.runtime.config.pull_models import BackendWorkerConfigPayload
from worker.runtime.config.restart import RestartDirective
from worker.runtime.config.worker_models import (
    ClipRecordingConfig,
    DevMjpegConfig,
    WorkerConfig,
    WorkerModelsConfig,
)

RELAY_URL_ENV: Final = "RELAY_URL"
RELAY_TOKEN_ENV: Final = "RELAY_TOKEN"
API_FACILITY_ID_ENV: Final = "API_FACILITY_ID"

_JSON_OBJECT_ADAPTER: Final = TypeAdapter(JsonObject)


class ConfigSource(StrEnum):
    PULLED = "pulled"
    LKG = "lkg"
    YAML = "yaml"


@dataclass(frozen=True, slots=True)
class ConfigSnapshot:
    config: WorkerConfig
    registry_version: int
    directive: RestartDirective
    source: ConfigSource
    stale: bool


def pull_worker_config(
    relay_url: str,
    relay_token: str | None,
    *,
    timeout_sec: float = 5.0,
    urlopen: UrlOpen | None = None,
) -> PulledWorkerConfig | None:
    payload = _pull_payload(
        relay_url,
        relay_token,
        timeout_sec=timeout_sec,
        urlopen=urlopen,
    )
    if payload is None:
        return None
    try:
        return BackendWorkerConfigPayload.model_validate(payload).to_pulled_config()
    except ValidationError:
        print("worker config pull skipped: malformed payload", file=sys.stderr)
        return None


def load_worker_config_from_relay(
    relay_url: str,
    relay_token: str | None,
    *,
    timeout_sec: float = 5.0,
    store: WorkerConfigLkgStore | None = None,
    urlopen: UrlOpen | None = None,
    models: WorkerModelsConfig | None = None,
    clip: ClipRecordingConfig | None = None,
    dev_mjpeg: DevMjpegConfig | None = None,
) -> ConfigSnapshot | None:
    lkg_store = store or WorkerConfigLkgStore()
    payload = _pull_payload(
        relay_url,
        relay_token,
        timeout_sec=timeout_sec,
        urlopen=urlopen,
    )
    if payload is not None:
        try:
            fresh = _snapshot_from_payload(
                payload,
                relay_url,
                relay_token,
                source=ConfigSource.PULLED,
                stale=False,
                models=models,
                clip=clip,
                dev_mjpeg=dev_mjpeg,
            )
        except (ValidationError, WorkerConfigError):
            print("worker config pull skipped: malformed payload", file=sys.stderr)
        else:
            saved = lkg_store.save(payload, fresh.directive)
            if saved:
                return fresh
            stored = lkg_store.load()
            if stored is None or _stored_key(stored) <= _snapshot_key(fresh):
                return fresh
            try:
                return _snapshot_from_stored(
                    stored, relay_url, relay_token, models=models, clip=clip, dev_mjpeg=dev_mjpeg
                )
            except (ValidationError, WorkerConfigError) as exc:
                print(
                    f"WARNING: worker config LKG at {lkg_store.database_path} "
                    f"failed re-validation ({exc}); deleting corrupt LKG and "
                    "using fresh (race-losing) snapshot instead",
                    file=sys.stderr,
                )
                lkg_store.clear()
                return fresh
    stored = lkg_store.load()
    if stored is None:
        return None
    try:
        return _snapshot_from_stored(
            stored, relay_url, relay_token, models=models, clip=clip, dev_mjpeg=dev_mjpeg
        )
    except (ValidationError, WorkerConfigError):
        print("worker config LKG skipped: malformed payload", file=sys.stderr)
        return None


def resolve_startup_config(
    yaml_config: WorkerConfig,
    relay_url: str,
    relay_token: str | None,
    *,
    timeout_sec: float = 5.0,
    store: WorkerConfigLkgStore | None = None,
    urlopen: UrlOpen | None = None,
    environ: Mapping[str, str] | None = None,
) -> ConfigSnapshot:
    effective_token = relay_token or yaml_config.relay.token.get_secret_value()
    models, clip, dev_mjpeg = resolve_local_overrides(yaml_config, environ)
    pulled = load_worker_config_from_relay(
        relay_url,
        effective_token,
        timeout_sec=timeout_sec,
        store=store,
        urlopen=urlopen,
        models=models,
        clip=clip,
        dev_mjpeg=dev_mjpeg,
    )
    if pulled is not None:
        return pulled
    return ConfigSnapshot(
        config=yaml_config,
        registry_version=0,
        directive=RestartDirective(generation=0, version=0),
        source=ConfigSource.YAML,
        stale=True,
    )


def _pull_payload(
    relay_url: str,
    relay_token: str | None,
    *,
    timeout_sec: float,
    urlopen: UrlOpen | None,
) -> JsonObject | None:
    headers = {"Accept": "application/json"}
    if relay_token:
        headers["X-Edge-Relay-Token"] = relay_token
    request = urllib.request.Request(
        f"{relay_url.rstrip('/')}{WORKER_CONFIG_PATH}",
        headers=headers,
        method="GET",
    )
    opener = stdlib_urlopen if urlopen is None else urlopen
    try:
        with opener(request, timeout_sec) as response:
            if response.status != 200:
                print(
                    f"worker config pull skipped: HTTP {response.status}",
                    file=sys.stderr,
                )
                return None
            return _JSON_OBJECT_ADAPTER.validate_json(response.read())
    except (OSError, UnicodeDecodeError, ValidationError, WorkerConfigError):
        print("worker config pull skipped: request failed", file=sys.stderr)
        return None
def _snapshot_from_payload(
    payload: JsonObject,
    relay_url: str,
    relay_token: str | None,
    *,
    source: ConfigSource,
    stale: bool,
    models: WorkerModelsConfig | None = None,
    clip: ClipRecordingConfig | None = None,
    dev_mjpeg: DevMjpegConfig | None = None,
) -> ConfigSnapshot:
    parsed = BackendWorkerConfigPayload.model_validate(payload)
    return ConfigSnapshot(
        config=parsed.to_worker_config(
            relay_url, relay_token, models=models, clip=clip, dev_mjpeg=dev_mjpeg
        ),
        registry_version=parsed.resolved_registry_version,
        directive=parsed.directive,
        source=source,
        stale=stale,
    )


def _snapshot_from_stored(
    stored: StoredConfigPayload,
    relay_url: str,
    relay_token: str | None,
    *,
    models: WorkerModelsConfig | None = None,
    clip: ClipRecordingConfig | None = None,
    dev_mjpeg: DevMjpegConfig | None = None,
) -> ConfigSnapshot:
    snapshot = _snapshot_from_payload(
        stored.payload,
        relay_url,
        relay_token,
        source=ConfigSource.LKG,
        stale=True,
        models=models,
        clip=clip,
        dev_mjpeg=dev_mjpeg,
    )
    if _snapshot_key(snapshot) != _stored_key(stored):
        raise WorkerConfigError("worker config LKG revision mismatch")
    return snapshot


def _snapshot_key(snapshot: ConfigSnapshot) -> tuple[RestartDirective, int]:
    return snapshot.directive, snapshot.registry_version


def _stored_key(stored: StoredConfigPayload) -> tuple[RestartDirective, int]:
    return stored.directive, stored.registry_version


__all__ = [
    "API_FACILITY_ID_ENV",
    "RELAY_TOKEN_ENV",
    "RELAY_URL_ENV",
    "ConfigSnapshot",
    "ConfigSource",
    "UrlOpen",
    "load_worker_config_from_relay",
    "pull_worker_config",
    "resolve_startup_config",
]
