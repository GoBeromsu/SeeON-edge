"""Tests for `worker/__main__.py`'s startup config resolution (issue #27).

Before this wiring, the live `WorkerConfig` handed to `WorkerRuntime` came
exclusively from local YAML (`load_worker_config`); the relay-pull path
(`resolve_startup_config` / `load_worker_config_from_relay`,
`worker/runtime/config/config_pull.py`) had zero non-test callers, so the
shipped `compose.edge.yaml` (leaving `EDGE_CAMERA_CONFIG` unset) crash-looped
the production worker and dashboard-configured cameras never reached it.

`worker/__main__.py:main` now has two startup branches:

- No `--config`/`EDGE_CAMERA_CONFIG`: pull directly via
  `load_worker_config_from_relay`, which returns `None` only when there is
  neither a fresh pull nor a last-known-good (LKG) cache.
- `--config`/`EDGE_CAMERA_CONFIG` set: load the YAML, then let
  `resolve_startup_config` attempt a pull that takes precedence, falling
  back to the YAML on any failure.

These tests monkeypatch `worker_main.load_worker_config_from_relay` /
`worker_main.resolve_startup_config` directly (the same seam
`tests/test_worker_entrypoint.py` uses for `make_restart_check`) rather than
injecting fake `urlopen`/`store` collaborators through `worker/__main__.py`
itself, since `main()` does not expose those seams to its caller -- the
`urlopen`/`store` injection points are exercised directly against
`config_pull.py` in `tests/test_worker_config_pull_lkg.py`. No network call
is ever made from this file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from edge_worker_fixtures import edge_config_payload

import worker.__main__ as worker_main
from worker.runtime.config import (
    CameraRuntimeConfig,
    ConfigSnapshot,
    ConfigSource,
    RelayConfig,
    RestartDirective,
    WorkerConfig,
)
from worker.runtime.worker import WorkerRuntime


def _fake_loop_factory(camera: object, bus: object, reporter: object) -> None:
    raise AssertionError("loop factory must not be invoked by CLI tests")


@pytest.fixture(autouse=True)
def _isolate_from_default_ingest_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors `tests/test_worker_entrypoint.py`'s fixture of the same name:
    never let CLI tests construct the real per-camera ingest loop.
    """
    real_init = WorkerRuntime.__init__

    def _init_with_fake_loop_factory(
        self: WorkerRuntime, *args: object, **kwargs: object
    ) -> None:
        kwargs.setdefault("loop_factory", _fake_loop_factory)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(WorkerRuntime, "__init__", _init_with_fake_loop_factory)


@pytest.fixture(autouse=True)
def _no_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDGE_CAMERA_CONFIG", raising=False)
    monkeypatch.delenv("RELAY_URL", raising=False)
    monkeypatch.delenv("RELAY_TOKEN", raising=False)


def _write_yaml_config(tmp_path: Path, *, camera_count: int = 1) -> Path:
    payload: dict[str, Any] = dict(
        edge_config_payload(
            camera_count=camera_count,
            include_optional_fields=False,
            resident_ids=False,
        )
    )
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def _worker_config(camera_id: str) -> WorkerConfig:
    return WorkerConfig(
        relay=RelayConfig(url="http://ml-api:8000", token="relay-token"),
        cameras=(
            CameraRuntimeConfig(
                camera_id=camera_id,
                facility_id="facility-1",
                rtsp_url=f"rtsp://{camera_id}/sub",
            ),
        ),
    )


def _spy_workerruntime_config(monkeypatch: pytest.MonkeyPatch) -> list[WorkerRuntime]:
    constructed: list[WorkerRuntime] = []
    real_init = WorkerRuntime.__init__

    def _spy_init(self: WorkerRuntime, *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)
        constructed.append(self)

    monkeypatch.setattr(WorkerRuntime, "__init__", _spy_init)
    monkeypatch.setattr(WorkerRuntime, "run", lambda self: None)
    return constructed


# --- no YAML: pull-only startup path ---------------------------------------


def test_no_yaml_successful_pull_becomes_live_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    pulled_config = _worker_config("pulled-camera")
    snapshot = ConfigSnapshot(
        config=pulled_config,
        registry_version=4,
        directive=RestartDirective(generation=1, version=4),
        source=ConfigSource.PULLED,
        stale=False,
    )

    def _fake_load_from_relay(relay_url: str, relay_token: str | None) -> ConfigSnapshot:
        assert relay_url == "http://ml-api:8000"
        assert relay_token == "relay-token"
        return snapshot

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fake_load_from_relay)
    constructed = _spy_workerruntime_config(monkeypatch)

    exit_code = worker_main.main([])

    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0].config is pulled_config


def test_no_yaml_failed_pull_with_lkg_uses_lkg_config_and_logs_stale(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    lkg_config = _worker_config("lkg-camera")
    snapshot = ConfigSnapshot(
        config=lkg_config,
        registry_version=2,
        directive=RestartDirective(generation=1, version=2),
        source=ConfigSource.LKG,
        stale=True,
    )
    monkeypatch.setattr(
        worker_main, "load_worker_config_from_relay", lambda *_a, **_k: snapshot
    )
    constructed = _spy_workerruntime_config(monkeypatch)

    with caplog.at_level(logging.INFO):
        exit_code = worker_main.main([])

    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0].config is lkg_config
    assert any(
        "lkg" in record.getMessage().lower() and "stale=True" in record.getMessage()
        for record in caplog.records
    )


def test_no_yaml_failed_pull_no_lkg_exits_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", lambda *_a, **_k: None)

    with caplog.at_level(logging.ERROR):
        exit_code = worker_main.main([])

    assert exit_code == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "http://ml-api:8000" in message
        and "RELAY_URL" in message
        and "RELAY_TOKEN" in message
        for message in messages
    )


def test_no_yaml_and_no_relay_url_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not attempt a pull without RELAY_URL")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    assert worker_main.main([]) == 2


# --- YAML set: resolve_startup_config governs precedence --------------------


def test_yaml_set_successful_pull_wins_over_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_yaml_config(tmp_path)
    pulled_config = _worker_config("pulled-camera")
    snapshot = ConfigSnapshot(
        config=pulled_config,
        registry_version=9,
        directive=RestartDirective(generation=2, version=9),
        source=ConfigSource.PULLED,
        stale=False,
    )
    monkeypatch.setattr(worker_main, "resolve_startup_config", lambda *_a, **_k: snapshot)
    constructed = _spy_workerruntime_config(monkeypatch)

    exit_code = worker_main.main(["--config", str(config_path)])

    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0].config is pulled_config


def test_yaml_set_failed_pull_no_lkg_falls_back_to_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_yaml_config(tmp_path)
    captured_yaml_config: list[WorkerConfig] = []

    def _fake_resolve_startup_config(
        yaml_config: WorkerConfig, relay_url: str, relay_token: str | None
    ) -> ConfigSnapshot:
        captured_yaml_config.append(yaml_config)
        return ConfigSnapshot(
            config=yaml_config,
            registry_version=0,
            directive=RestartDirective(generation=0, version=0),
            source=ConfigSource.YAML,
            stale=True,
        )

    monkeypatch.setattr(worker_main, "resolve_startup_config", _fake_resolve_startup_config)
    constructed = _spy_workerruntime_config(monkeypatch)

    exit_code = worker_main.main(["--config", str(config_path)])

    assert exit_code == 0
    assert len(constructed) == 1
    assert len(captured_yaml_config) == 1
    assert constructed[0].config is captured_yaml_config[0]


# --- --check-config keeps working in both branches, without a relay pull ---


def test_check_config_no_yaml_pulls_but_has_no_further_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    snapshot = ConfigSnapshot(
        config=_worker_config("pulled-camera"),
        registry_version=1,
        directive=RestartDirective(generation=0, version=1),
        source=ConfigSource.PULLED,
        stale=False,
    )
    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", lambda *_a, **_k: snapshot)

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("--check-config must not construct WorkerRuntime")

    monkeypatch.setattr(worker_main, "make_restart_check", _fail)
    monkeypatch.setattr(WorkerRuntime, "__init__", _fail)

    assert worker_main.main(["--check-config"]) == 0


def test_check_config_with_yaml_never_calls_resolve_startup_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_yaml_config(tmp_path)

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "--check-config with an explicit YAML must validate the YAML alone, "
            "matching the pre-#27 no-relay-side-effect contract"
        )

    monkeypatch.setattr(worker_main, "resolve_startup_config", _fail)
    monkeypatch.setattr(worker_main, "make_restart_check", _fail)
    monkeypatch.setattr(WorkerRuntime, "__init__", _fail)

    assert worker_main.main(["--config", str(config_path), "--check-config"]) == 0
