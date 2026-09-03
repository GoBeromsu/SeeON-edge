"""Issues #66/#68: locally-sourced ``models.fall``/``clip.enabled`` must
survive a relay pull.

Before this change, ``BackendWorkerConfigPayload.to_worker_config()``
(``worker/runtime/config/pull_models.py``) built the effective ``WorkerConfig``
from only ``relay``/``domains``/``cameras``, so ``models``/``clip`` always
resolved to their pydantic defaults (``fall=None``, ``enabled=False``) on
every relay pull -- including the shipped pull-first production topology
(``compose.edge.yaml`` passes no ``--config``, so the worker always pulls
live), where a fall model could never be configured (the #43 boot gate then
refuses to start) and clip recording could never be turned on.

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

from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.runtime.config import (
    ClipRecordingConfig,
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
    return write_pose_bbox56_bundle(path)


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
    artifact_dir = _write_fall_artifact(tmp_path / "models" / "fall" / "pose-bbox56-gru")
    environ = {**_fall_env(artifact_dir), ML_WORKER_CLIP_RECORDING_ENABLED_ENV: "true"}
    models, clip, dev_mjpeg = resolve_local_overrides(None, environ)

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=WorkerConfigLkgStore(tmp_path / "worker-state.sqlite3"),
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=1, config_version=1, restart_epoch=0)
        ),
        models=models,
        clip=clip,
        dev_mjpeg=dev_mjpeg,
    )

    assert snapshot is not None
    assert snapshot.config.models.fall is not None
    assert snapshot.config.models.fall.artifact_dir == artifact_dir.resolve()
    assert snapshot.config.clip.enabled is True


def test_pull_with_no_fall_config_resolves_the_packaged_default_bundle(
    tmp_path: Path, packaged_fall_bundle: Path
) -> None:
    """Issue #133: with no local fall configuration at all (env unset, no
    YAML), ``models.fall`` must resolve to the packaged default fall bundle
    instead of staying ``None`` -- the worker boots with zero env vars using
    a default fall model rather than refusing to start. This supersedes the
    previous #43-boot-gate contract asserted here (fall must stay
    unconfigured absent explicit env/YAML); the #43 gate itself
    (``WorkerRuntime._create_fall_model``) is unchanged and still refuses to
    boot when ``models.fall`` truly cannot be resolved (e.g. the packaged
    default's weights are missing -- see the skip reason above and
    tests/test_local_env_defaults.py)."""
    models, clip, dev_mjpeg = resolve_local_overrides(None, {})
    assert models.fall is not None
    assert models.fall.type == "pose-bbox56-proxy-v0"
    assert models.fall.artifact_dir == packaged_fall_bundle.resolve()
    # Deliberately not a literal: this test's concern is fall-model
    # resolution, not clip. Deferring to ClipRecordingConfig's own default
    # documents "env/YAML silence defers to the model default" (this
    # branch's intent) and stays correct regardless of merge order with
    # #137 (which flips that default to always-on).
    assert clip.enabled is ClipRecordingConfig().enabled
    assert dev_mjpeg is None

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=WorkerConfigLkgStore(tmp_path / "worker-state.sqlite3"),
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=1, config_version=1, restart_epoch=0)
        ),
        models=models,
        clip=clip,
        dev_mjpeg=dev_mjpeg,
    )

    assert snapshot is not None
    assert snapshot.config.models.fall is not None
    assert snapshot.config.models.fall.type == "pose-bbox56-proxy-v0"
    # Same rationale as the `clip.enabled` assertion above: defers to the
    # model default rather than asserting a stale literal.
    assert snapshot.config.clip.enabled is ClipRecordingConfig().enabled
    assert snapshot.config.dev_mjpeg.enabled is True
    assert snapshot.config.dev_mjpeg.host == "0.0.0.0"
    assert snapshot.config.dev_mjpeg.port == 8090


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
                    "type": "pose-bbox56-proxy-v0",
                    "framework": "pytorch",
                    "mode": "sequence",
                    "artifact_dir": str(yaml_artifact_dir),
                    "window": 30,
                    "stride": 5,
                    "input_shape": [30, 56],
                    "operating_threshold": 0.5,
                }
            },
            "clip": {"enabled": True},
        }
    )
    environ = {**_fall_env(env_artifact_dir), ML_WORKER_CLIP_RECORDING_ENABLED_ENV: "false"}

    models, clip, dev_mjpeg = resolve_local_overrides(yaml_config, environ)

    assert models.fall is not None
    assert models.fall.artifact_dir == yaml_artifact_dir.resolve()
    assert clip.enabled is True
    assert dev_mjpeg is None


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
    artifact_dir = _write_fall_artifact(tmp_path / "models" / "fall" / "pose-bbox56-gru")
    environ = {**_fall_env(artifact_dir), ML_WORKER_CLIP_RECORDING_ENABLED_ENV: "true"}
    models, clip, dev_mjpeg = resolve_local_overrides(None, environ)
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
        dev_mjpeg=dev_mjpeg,
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
        dev_mjpeg=dev_mjpeg,
    )

    assert stale is not None
    assert stale.source is ConfigSource.LKG
    assert stale.stale is True
    assert stale.config.models.fall is not None
    assert stale.config.models.fall.artifact_dir == artifact_dir.resolve()
    assert stale.config.clip.enabled is True


def test_pull_with_yaml_dev_mjpeg_enabled_survives_the_pull(
    tmp_path: Path, packaged_fall_bundle: Path
) -> None:
    """Issue #113: ``BackendWorkerConfigPayload.to_worker_config()`` never
    threaded ``dev_mjpeg`` through a relay pull, so an explicit local
    ``dev_mjpeg.enabled: true`` was silently reset to the pydantic default
    (disabled) on every successful pull -- with no failure and no log line,
    the live-view MJPEG port would simply never bind. This must not regress:
    a YAML with ``dev_mjpeg.enabled: true`` must still be enabled in the
    post-pull ``WorkerConfig``."""
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
            "dev_mjpeg": {"enabled": True, "host": "127.0.0.1", "port": 8090},
        }
    )
    _models, _clip, dev_mjpeg = resolve_local_overrides(yaml_config, {})
    assert dev_mjpeg is not None
    assert dev_mjpeg.enabled is True

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=WorkerConfigLkgStore(tmp_path / "worker-state.sqlite3"),
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=1, config_version=1, restart_epoch=0)
        ),
        models=_models,
        clip=_clip,
        dev_mjpeg=dev_mjpeg,
    )

    assert snapshot is not None
    assert snapshot.config.dev_mjpeg.enabled is True
    assert snapshot.config.dev_mjpeg.port == 8090


def test_pull_with_no_yaml_dev_mjpeg_leaves_it_disabled_for_env_fallback(
    packaged_fall_bundle: Path,
) -> None:
    """The default (no local YAML, or YAML silent on ``dev_mjpeg``) must keep
    resolving to ``None`` here so ``WorkerRuntime._resolve_mjpeg_config``'s
    existing ``ML_WORKER_DEV_MJPEG*`` env fallback is left untouched -- this
    fix only needed to add the missing YAML-survives-a-pull half."""
    _models, _clip, dev_mjpeg = resolve_local_overrides(None, {})
    assert dev_mjpeg is None
