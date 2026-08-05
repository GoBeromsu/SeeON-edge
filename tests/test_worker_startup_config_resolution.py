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
from pydantic import ValidationError

import worker.__main__ as worker_main
from worker.runtime.config import (
    CameraRuntimeConfig,
    ConfigSnapshot,
    ConfigSource,
    RelayConfig,
    RestartDirective,
    WorkerConfig,
    WorkerConfigError,
    WorkerConfigLkgStore,
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

    def _fake_load_from_relay(
        relay_url: str, relay_token: str | None, **_kwargs: object
    ) -> ConfigSnapshot:
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


def test_check_config_no_yaml_is_strictly_static_no_pull_no_lkg_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--check-config` with no YAML must not touch the network or the disk.

    Regression guard for a review finding on the first cut of this PR: the
    no-YAML branch called `load_worker_config_from_relay` *before* the
    `--check-config` early return, and a successful pull inside that function
    performs an fsync'd write to the last-known-good (LKG) store on disk --
    i.e. a flag documented as side-effect-free
    (worker/runtime/AGENTS.md:65, "`--check-config` performs no model,
    camera, or relay side effect") was mutating persistent device state.
    `--check-config` must only validate that RELAY_URL/RELAY_TOKEN are set
    and well-formed; the live pull (and any LKG write) stays deferred to
    boot.
    """
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("--check-config must never attempt a relay pull")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    def _fail_save(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("--check-config must never write the LKG store")

    monkeypatch.setattr(WorkerConfigLkgStore, "save", _fail_save)

    def _fail_construct(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("--check-config must not construct WorkerRuntime")

    monkeypatch.setattr(worker_main, "make_restart_check", _fail_construct)
    monkeypatch.setattr(WorkerRuntime, "__init__", _fail_construct)

    assert worker_main.main(["--check-config"]) == 0


def test_check_config_no_yaml_reports_lkg_presence_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`--check-config` may read the LKG store to report whether a cache
    exists (a pure read via `WorkerConfigLkgStore.load`), but must not write
    it.
    """
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setattr(worker_main, "resolve_state_dir", lambda: tmp_path)

    with caplog.at_level(logging.INFO):
        exit_code = worker_main.main(["--check-config"])

    assert exit_code == 0
    assert any(
        "no last-known-good cache yet" in record.getMessage() for record in caplog.records
    )


def test_no_yaml_missing_relay_token_exits_before_any_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.delenv("RELAY_TOKEN", raising=False)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not attempt a pull without RELAY_TOKEN")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    assert worker_main.main([]) == 2


def test_no_yaml_malformed_relay_url_exits_before_any_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELAY_URL", "not-a-url")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not attempt a pull with a malformed RELAY_URL")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    assert worker_main.main([]) == 2


# --- exit codes stay owned by the entrypoint on an unexpected raise --------


def test_no_yaml_pull_raising_workerconfigerror_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards `config_pull.load_worker_config_from_relay`'s one unguarded
    branch: a fresh pull can validate, lose the LKG race (a strictly newer
    directive/registry_version already on disk), and then re-derive the
    snapshot from the stored payload via `_snapshot_from_stored` with no
    try/except -- so a `WorkerConfigError` there (e.g. an LKG revision
    mismatch) would otherwise escape `main()` as a raw traceback instead of
    the documented exit code.
    """
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _raise(*_args: object, **_kwargs: object) -> ConfigSnapshot:
        raise WorkerConfigError("worker config LKG revision mismatch")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _raise)

    assert worker_main.main([]) == 2


def test_yaml_set_pull_raising_workerconfigerror_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_yaml_config(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> ConfigSnapshot:
        raise WorkerConfigError("worker config LKG revision mismatch")

    monkeypatch.setattr(worker_main, "resolve_startup_config", _raise)

    assert worker_main.main(["--config", str(config_path)]) == 2


def test_no_yaml_pull_raising_validationerror_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors the `WorkerConfigError` guard above for its sibling exception:
    `load_worker_config_from_relay`'s unguarded "lost the LKG race" branch
    calls `_snapshot_from_stored` -> `_snapshot_from_payload` ->
    `BackendWorkerConfigPayload.model_validate(stored.payload)`, which raises
    `pydantic.ValidationError` (not `WorkerConfigError`) when the stored LKG
    payload itself fails schema validation. Both exception types must be
    caught, not just one.
    """
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _raise(*_args: object, **_kwargs: object) -> ConfigSnapshot:
        # Real pydantic ValidationError, produced the same way the guarded
        # branch would encounter one -- not a hand-constructed stand-in.
        RelayConfig.model_validate({"url": "not-a-url", "token": "relay-token"})
        raise AssertionError("RelayConfig.model_validate should have raised")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _raise)

    assert worker_main.main([]) == 2


def test_yaml_set_pull_raising_validationerror_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_yaml_config(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> ConfigSnapshot:
        RelayConfig.model_validate({"url": "not-a-url", "token": "relay-token"})
        raise AssertionError("RelayConfig.model_validate should have raised")

    monkeypatch.setattr(worker_main, "resolve_startup_config", _raise)

    assert worker_main.main(["--config", str(config_path)]) == 2


# --- the fatal no-config message must not assert an unestablished cause ----


def test_no_yaml_no_config_message_names_both_plausible_causes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: a relay that is up and returns a valid payload where every
    camera lacks an `rtsp_url` (realistic mid-onboarding, cameras added in
    the dashboard before stream URLs are entered) makes
    `BackendWorkerConfigPayload.to_worker_config` raise "must include at
    least one camera"; `config_pull.py` swallows that and the pull is
    reported as failed identically to an unreachable relay. The fatal
    message must not claim the relay was unreachable -- it wasn't
    established -- so it must name both possible causes rather than assert
    one.
    """
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", lambda *_a, **_k: None)

    with caplog.at_level(logging.ERROR):
        exit_code = worker_main.main([])

    assert exit_code == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "unreachable" in message and "no usable camera" in message for message in messages
    )


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


# --- issue #150: "카메라 없음" boots, "설정 없음"/malformed still fails fast ---
#
# The two paths this issue's fix must keep distinct:
#
# 1. No config at all, or a config that fails to parse/validate (unreadable
#    YAML, a relay pull with neither a fresh payload nor a last-known-good
#    cache, a config missing required fields like `relay`) -- issue #43's
#    fail-fast boot is untouched here and still refuses to start.
# 2. A config that resolves cleanly but with zero cameras (a fresh install,
#    or every camera still missing its RTSP URL) -- this used to collapse
#    into the same failure as (1) because `WorkerConfig.cameras` required
#    `min_length=1` and `BackendWorkerConfigPayload.to_worker_config` raised
#    on an empty resolved roster. Both gates are relaxed; this must now boot.


def test_worker_config_accepts_zero_cameras() -> None:
    """"카메라 없음" is a valid, bootable state -- `WorkerConfig.cameras` no
    longer requires at least one entry."""
    config = WorkerConfig(relay=RelayConfig(url="http://ml-api:8000", token="relay-token"))

    assert config.cameras == ()


def test_worker_config_still_rejects_a_config_missing_the_relay_section() -> None:
    """"설정 없음"/malformed config keeps failing fast -- only the camera-count
    gate was relaxed by issue #150, nothing else about `WorkerConfig`
    validation changed."""
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate({"cameras": []})


def test_no_yaml_pull_resolving_to_zero_cameras_still_boots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #150: a relay pull that resolves to a zero-camera `WorkerConfig`
    (a fresh install, or every camera still missing its RTSP URL) must reach
    `WorkerRuntime` construction, not the "worker has no usable
    configuration" exit -- unlike before this fix, where
    `BackendWorkerConfigPayload.to_worker_config` raised on an empty roster
    and `config_pull.py` swallowed that as a malformed pull, making
    `load_worker_config_from_relay` return `None` for this exact case."""
    monkeypatch.setenv("RELAY_URL", "http://ml-api:8000")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    zero_camera_config = WorkerConfig(
        relay=RelayConfig(url="http://ml-api:8000", token="relay-token"),
    )
    snapshot = ConfigSnapshot(
        config=zero_camera_config,
        registry_version=1,
        directive=RestartDirective(generation=0, version=1),
        source=ConfigSource.PULLED,
        stale=False,
    )
    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", lambda *_a, **_k: snapshot)
    constructed = _spy_workerruntime_config(monkeypatch)

    exit_code = worker_main.main([])

    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0].config.cameras == ()


def test_no_yaml_and_no_relay_url_still_exits_fast_with_zero_cameras_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Companion to the boot-succeeds test above: "설정 없음" (no `--config`/
    `EDGE_CAMERA_CONFIG`, and no `RELAY_URL` either -- there is no config to
    resolve at all, not even one with zero cameras) still refuses to start,
    exactly as `test_no_yaml_and_no_relay_url_exits_with_config_error_code`
    already pins -- restated here to keep both halves of issue #150's
    contract next to each other in one file."""

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not attempt a pull without RELAY_URL")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    assert worker_main.main([]) == 2
