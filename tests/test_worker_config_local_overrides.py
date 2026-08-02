"""Issues #66/#68: locally-sourced ``models.fall``/``clip.enabled`` must
survive a relay pull.

Before this change, ``BackendWorkerConfigPayload.to_worker_config()``
(``worker/runtime/config/pull_models.py``) built the effective ``WorkerConfig``
from only ``relay``/``domains``/``cameras``, so ``models``/``clip`` always
resolved to their pydantic defaults (``fall=None``, ``enabled=False``) on
every relay pull -- including the shipped pull-first production topology
(``compose.edge.yaml``'s default empty ``EDGE_CAMERA_CONFIG``), where a fall
model could never be configured (the #43 boot gate then refuses to start) and
clip recording could never be turned on.

``worker/runtime/config/local_env.py`` gives both fields a real,
production-reachable env-var surface, and ``resolve_local_overrides`` settles
precedence between an explicit local YAML value and the environment (mirroring
``WorkerRuntime._resolve_mjpeg_config``'s ``dev_mjpeg`` precedent): an explicit
YAML value wins outright; YAML silence defers to env.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from types import TracebackType
from typing import Self, final

import pytest

from worker.runtime.config import (
    ConfigSource,
    JsonObject,
    WorkerConfig,
    WorkerConfigError,
    WorkerConfigLkgStore,
    load_worker_config_from_relay,
    resolve_local_overrides,
)
from worker.runtime.config.local_env import (
    ML_WORKER_CLIP_RECORDING_ENABLED_ENV,
    ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV,
    ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV,
    ML_WORKER_FALL_MODEL_STRIDE_ENV,
    ML_WORKER_FALL_MODEL_WINDOW_ENV,
    fall_model_config_from_environment,
)


@final
class FakeResponse:
    def __init__(self, payload: JsonObject, status: int = 200) -> None:
        self._payload: JsonObject = payload
        self.status: int = status

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _write_fall_artifact(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(b"placeholder")
    (path / "arch.json").write_text('{"hidden":4,"layers":1,"dropout":0.0}', encoding="utf-8")
    (path / "metadata.yaml").write_text("type: lstm\n", encoding="utf-8")
    return path


def _fall_env(artifact_dir: Path) -> dict[str, str]:
    return {
        ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV: str(artifact_dir),
        ML_WORKER_FALL_MODEL_WINDOW_ENV: "3",
        ML_WORKER_FALL_MODEL_STRIDE_ENV: "1",
        ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV: "0.5",
    }


def _payload(*, registry_version: int, config_version: int, restart_epoch: int) -> JsonObject:
    return {
        "registry_version": registry_version,
        "config_version": config_version,
        "restart_epoch": restart_epoch,
        "cameras": [
            {
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "rtsp_url": "rtsp://user:camera-pass@camera/live",
                "fps": 7.5,
                "domains": ["fall"],
            }
        ],
    }


def test_pull_with_fall_env_vars_set_configures_models_fall_and_clip_enabled(
    tmp_path: Path,
) -> None:
    """Requirement (a): a relay pull with the fall env vars set must carry the
    configured ``models.fall``/``clip.enabled`` into the resulting
    ``WorkerConfig``, not silently drop them."""
    artifact_dir = _write_fall_artifact(tmp_path / "models" / "fall" / "lstm")
    environ = {**_fall_env(artifact_dir), ML_WORKER_CLIP_RECORDING_ENABLED_ENV: "true"}
    models, clip = resolve_local_overrides(None, environ)

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=WorkerConfigLkgStore(tmp_path / "worker-state.sqlite3"),
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=1, config_version=1, restart_epoch=0)
        ),
        models=models,
        clip=clip,
    )

    assert snapshot is not None
    assert snapshot.config.models.fall is not None
    assert snapshot.config.models.fall.artifact_dir == artifact_dir.resolve()
    assert snapshot.config.clip.enabled is True


def test_pull_with_no_fall_config_leaves_models_fall_none_for_the_boot_gate(
    tmp_path: Path,
) -> None:
    """Requirement (b): with no local fall configuration at all (env unset,
    no YAML), the pulled config's ``models.fall`` must remain ``None`` so the
    #43 boot gate (``WorkerRuntime._create_fall_model``) still fires -- this
    fix must not weaken that gate by inventing a default fall model."""
    models, clip = resolve_local_overrides(None, {})
    assert models.fall is None
    assert clip.enabled is False

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=WorkerConfigLkgStore(tmp_path / "worker-state.sqlite3"),
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=1, config_version=1, restart_epoch=0)
        ),
        models=models,
        clip=clip,
    )

    assert snapshot is not None
    assert snapshot.config.models.fall is None
    assert snapshot.config.clip.enabled is False


def test_local_yaml_fall_config_wins_over_env_when_both_are_set(tmp_path: Path) -> None:
    """Requirement (c): precedence mirrors
    ``WorkerRuntime._resolve_mjpeg_config``'s ``dev_mjpeg`` precedent -- an
    explicit local YAML value wins outright over env, env only decides when
    the YAML is silent."""
    yaml_artifact_dir = _write_fall_artifact(tmp_path / "yaml-fall")
    env_artifact_dir = _write_fall_artifact(tmp_path / "env-fall")
    yaml_config = WorkerConfig.model_validate(
        {
            "relay": {"url": "http://ml-api:8000", "token": "relay-secret"},
            "cameras": [
                {
                    "camera_id": "yaml-camera",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://yaml/camera",
                }
            ],
            "models": {
                "fall": {
                    "type": "lstm",
                    "framework": "pytorch",
                    "mode": "sequence",
                    "artifact_dir": str(yaml_artifact_dir),
                    "window": 3,
                    "stride": 1,
                    "input_shape": [3, 51],
                    "operating_threshold": 0.5,
                }
            },
            "clip": {"enabled": True},
        }
    )
    environ = {**_fall_env(env_artifact_dir), ML_WORKER_CLIP_RECORDING_ENABLED_ENV: "false"}

    models, clip = resolve_local_overrides(yaml_config, environ)

    assert models.fall is not None
    assert models.fall.artifact_dir == yaml_artifact_dir.resolve()
    assert clip.enabled is True


def test_malformed_fall_env_value_raises_loudly_instead_of_silently_defaulting() -> None:
    """Requirement (d): fail-closed per ADR-0002 -- a malformed env value
    (here, a non-integer window) must raise ``WorkerConfigError`` rather than
    silently falling back to an unconfigured fall model."""
    environ = {
        ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV: "/nonexistent/artifact/dir",
        ML_WORKER_FALL_MODEL_WINDOW_ENV: "not-a-number",
        ML_WORKER_FALL_MODEL_STRIDE_ENV: "1",
        ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV: "0.5",
    }

    with pytest.raises(WorkerConfigError, match=ML_WORKER_FALL_MODEL_WINDOW_ENV):
        fall_model_config_from_environment(environ)


def test_lkg_restore_path_preserves_locally_sourced_models_and_clip(tmp_path: Path) -> None:
    """Requirement (e): the LKG-restore path (a fresh pull failing, falling
    back to the last-known-good payload) also goes through
    payload -> WorkerConfig conversion (``_snapshot_from_stored`` ->
    ``_snapshot_from_payload``), so locally-sourced ``models``/``clip`` must be
    re-attached there too, not only on the live-pull path."""
    artifact_dir = _write_fall_artifact(tmp_path / "models" / "fall" / "lstm")
    environ = {**_fall_env(artifact_dir), ML_WORKER_CLIP_RECORDING_ENABLED_ENV: "true"}
    models, clip = resolve_local_overrides(None, environ)
    store = WorkerConfigLkgStore(tmp_path / "worker-state.sqlite3")

    fresh = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=store,
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=3, config_version=3, restart_epoch=1)
        ),
        models=models,
        clip=clip,
    )
    assert fresh is not None
    assert fresh.source is ConfigSource.PULLED

    def offline(_request: urllib.request.Request, _timeout: float) -> FakeResponse:
        raise urllib.error.URLError("offline")

    stale = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=store,
        urlopen=offline,
        models=models,
        clip=clip,
    )

    assert stale is not None
    assert stale.source is ConfigSource.LKG
    assert stale.stale is True
    assert stale.config.models.fall is not None
    assert stale.config.models.fall.artifact_dir == artifact_dir.resolve()
    assert stale.config.clip.enabled is True
