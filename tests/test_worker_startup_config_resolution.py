"""Tests for ``worker.__main__`` startup config resolution.

Normal Edge startup uses the baked private relay endpoint
``http://ml-api:8000`` plus the projected ``RELAY_TOKEN`` secret. ``RELAY_URL``
is retired for normal runtime and must be rejected rather than becoming a
second topology authority. With no ``--config``, startup pulls the versioned
worker config and may fall back to its last-known-good (LKG) cache.

An explicit YAML file is a developer/e2e bootstrap escape hatch only. It may
supply relay credentials, but static camera roster, model, domain, and clip
policy are rejected. A successful versioned pull takes precedence over this
zero-camera bootstrap config; a failed pull may fall back to it.

These tests monkeypatch ``load_worker_config_from_relay`` and
``resolve_startup_config`` at the entrypoint seam. Network and LKG persistence
behavior is covered directly by the config-pull tests; no network call is made
from this file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
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


def _write_yaml_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "relay": {"url": "http://ml-api:8000", "token": "relay-token"},
            }
        ),
        encoding="utf-8",
    )
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
    packaged_fall_bundle: Path,
) -> None:
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
    packaged_fall_bundle: Path,
) -> None:
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
    packaged_fall_bundle: Path,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", lambda *_a, **_k: None)

    with caplog.at_level(logging.ERROR):
        exit_code = worker_main.main([])

    assert exit_code == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "http://ml-api:8000" in message and "RELAY_TOKEN" in message and "dashboard" in message
        for message in messages
    )


def test_check_config_no_yaml_missing_relay_token_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("check-config must not pull without RELAY_TOKEN")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    assert worker_main.main(["--check-config"]) == 2


# --- supported relay-only YAML: resolve_startup_config governs precedence ---


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
    `--check-config` must only validate the projected RELAY_TOKEN against the
    baked relay endpoint; the live pull (and any LKG write) stays deferred to
    boot.
    """
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
    monkeypatch.delenv("RELAY_TOKEN", raising=False)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not attempt a pull without RELAY_TOKEN")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    assert worker_main.main([]) == 2


def test_normal_runtime_rejects_retired_relay_url_before_any_pull(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("RELAY_URL", "http://other-relay.test")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not pull when retired RELAY_URL is present")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    with caplog.at_level(logging.ERROR):
        assert worker_main.main([]) == 2

    assert any(
        "RELAY_URL" in record.getMessage()
        and "versioned worker config authority" in record.getMessage()
        for record in caplog.records
    )


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


# --- packaged-model resolution runs before the guarded pull -----------------
#
# On the env-only production topology (no `--config`), `resolve_local_overrides`
# resolves the packaged default fall model BEFORE the relay pull. The CI
# `Dockerfile.edge` image bakes empty model directories (weights stay
# gitignored, mounted at runtime), so on that image this call raises
# `WorkerConfigError` ("packaged default LSTM fall model is not fully
# provisioned"). That call sits before the guarded pull, so the failure must
# still surface as the documented CONFIG_ERROR_EXIT_CODE (2) with a logged
# message -- not escape `main()` as a raw traceback that reports the generic
# runtime exit code (1) and misrepresents a config/packaging fault. The guard
# must not weaken the validation itself: a missing packaged model still refuses
# to boot.


def test_no_yaml_missing_packaged_model_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise WorkerConfigError(
            "packaged default LSTM fall model is not fully provisioned at "
            "'models/fall/lstm'; run scripts/fetch-models.sh"
        )

    monkeypatch.setattr(worker_main, "resolve_local_overrides", _raise)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "a relay pull must not be attempted once local model resolution has failed"
        )

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    with caplog.at_level(logging.ERROR):
        exit_code = worker_main.main([])

    assert exit_code == 2
    assert any("model" in record.getMessage().lower() for record in caplog.records)


def test_no_yaml_local_override_validationerror_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling of the WorkerConfigError guard above: a malformed packaged
    manifest can surface a raw `pydantic.ValidationError` out of
    `resolve_local_overrides`. Both exception types must map to the documented
    config-error exit code, mirroring the guarded relay-pull branch."""
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _raise(*_args: object, **_kwargs: object) -> object:
        RelayConfig.model_validate({"url": "not-a-url", "token": "relay-token"})
        raise AssertionError("RelayConfig.model_validate should have raised")

    monkeypatch.setattr(worker_main, "resolve_local_overrides", _raise)
    monkeypatch.setattr(
        worker_main,
        "load_worker_config_from_relay",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("must not pull once local model resolution has failed")
        ),
    )

    assert worker_main.main([]) == 2


def test_check_config_no_yaml_never_resolves_local_model_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The boot-smoke contract: `--check-config` is a static import/relay-token
    check and must exit 0 even when the packaged model is absent (the CI image
    bakes empty model dirs). It must therefore never reach
    `resolve_local_overrides`, which is the runtime model-provisioning gate.
    """
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "--check-config must not resolve local model overrides -- it is a "
            "static, no-side-effect check that must pass without a provisioned model"
        )

    monkeypatch.setattr(worker_main, "resolve_local_overrides", _fail)

    assert worker_main.main(["--check-config"]) == 0


# --- the fatal no-config message must not assert an unestablished cause ----


def test_no_yaml_no_config_message_names_both_plausible_causes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    packaged_fall_bundle: Path,
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
    packaged_fall_bundle: Path,
) -> None:
    """Issue #150: a relay pull that resolves to a zero-camera `WorkerConfig`
    (a fresh install, or every camera still missing its RTSP URL) must reach
    `WorkerRuntime` construction, not the "worker has no usable
    configuration" exit -- unlike before this fix, where
    `BackendWorkerConfigPayload.to_worker_config` raised on an empty roster
    and `config_pull.py` swallowed that as a malformed pull, making
    `load_worker_config_from_relay` return `None` for this exact case."""
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


def test_no_yaml_and_no_relay_token_still_exits_fast_with_zero_cameras_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Companion to the boot-succeeds test above: without the projected relay
    token there is no authenticated config to resolve, not even one with zero
    cameras, so startup still refuses before attempting a pull."""

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not attempt a pull without RELAY_TOKEN")

    monkeypatch.setattr(worker_main, "load_worker_config_from_relay", _fail)

    assert worker_main.main([]) == 2
