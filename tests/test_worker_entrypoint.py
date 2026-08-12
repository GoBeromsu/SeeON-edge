"""Tests for the canonical `python -m worker` CLI (`worker/__main__.py`).

Covers the argparse surface, the documented exit-code table
(docs/architecture.md "Entrypoint"), `--check-config`'s no-side-effect
contract, `--heartbeat-on-start` passthrough, `restart_check` wiring via
`make_restart_check`, and SIGINT/SIGTERM clean shutdown. `WorkerRuntime.run`
is monkeypatched throughout (it would otherwise require real cameras/models);
one test constructs the real `WorkerRuntime` end to end with fake
collaborators to prove the historical `WorkerRuntime(config)` positional-call
TypeError cannot recur.
"""

from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any

import pytest
import yaml
from edge_worker_fixtures import edge_config_payload

import worker.__main__ as worker_main
from worker.runtime.config import RestartDirective
from worker.runtime.worker import WorkerRuntime
from worker.system_test import SystemTestOutcome


def _write_config(tmp_path: Path, *, camera_count: int = 1, version: int = 1) -> Path:
    payload: dict[str, Any] = dict(
        edge_config_payload(
            camera_count=camera_count,
            include_optional_fields=False,
            resident_ids=False,
        )
    )
    payload["version"] = version
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def _fake_loop_factory(camera: object, bus: object, reporter: object) -> None:
    raise AssertionError("loop factory must not be invoked by CLI tests")


class _FakeServingClient:
    def create(self, task: str, **_options: object) -> None:
        raise AssertionError("serving client must not be used by CLI tests")


@pytest.fixture(autouse=True)
def _isolate_from_default_ingest_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this file's tests independent of `WorkerRuntime`'s default
    `loop_factory` composition.

    `worker/__main__.py` intentionally omits `loop_factory` when constructing
    `WorkerRuntime`: composing the real per-camera ingest loop (decode
    adapter selection, source wiring) is composition-root territory
    (`worker/runtime/worker.py`), not the CLI's. That default composition has
    its own dedicated coverage in `tests/test_worker_ingest_composition.py`,
    including a bare-construction, no-`loop_factory` test proving the CLI's
    omitted-kwarg call is safe end to end. This file's job is the CLI
    contract (argparse, exit codes, config resolution, restart_check wiring,
    signal handling), not ingest composition, so every test here injects a
    fake `loop_factory` instead of exercising the real default -- CLI tests
    must never construct real decode adapters. Applied globally here rather
    than per test to keep that isolation guaranteed rather than opt-in.
    """
    real_init = WorkerRuntime.__init__

    def _init_with_fake_loop_factory(
        self: WorkerRuntime, *args: object, **kwargs: object
    ) -> None:
        kwargs.setdefault("loop_factory", _fake_loop_factory)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(WorkerRuntime, "__init__", _init_with_fake_loop_factory)


# --- argparse surface -------------------------------------------------


def test_help_flag_exits_zero_and_documents_all_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        worker_main.main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--config" in output
    assert "--check-config" in output
    assert "--heartbeat-on-start" in output
    assert "--max-frames-per-camera" in output
    assert "--system-test" in output
    assert "--system-test-validation-run-id" in output
    assert "--system-test-edge-event-id" in output
    assert "--confirm-system-test" in output


def test_system_test_is_disabled_by_default_before_config_or_camera_boot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: no typed SYSTEM_TEST gate and traps on normal worker boot seams.
    monkeypatch.delenv("ML_WORKER_SYSTEM_TEST_GATE", raising=False)

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("disabled SYSTEM_TEST must not load config or start inference")

    monkeypatch.setattr(worker_main, "load_worker_config", _fail)
    monkeypatch.setattr(WorkerRuntime, "__init__", _fail)

    # When: an operator-shaped invocation is attempted without the gate.
    exit_code = worker_main.main(
        [
            "--system-test",
            "emit",
            "--system-test-validation-run-id",
            "0197f671-3a31-7a6c-a6e4-83ed412de80f",
            "--confirm-system-test",
            "SYSTEM_TEST",
        ]
    )

    # Then: it fails closed before any normal inference/config path.
    assert exit_code == 2


def test_system_test_cli_emits_safe_machine_outcome_and_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: the exact gate and an injected successful operator service outcome.
    monkeypatch.setenv("ML_WORKER_SYSTEM_TEST_GATE", "SYSTEM_TEST_OPERATOR_ENABLED")
    monkeypatch.setenv("RELAY_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("RELAY_TOKEN", "local-relay-token")
    monkeypatch.setattr(worker_main, "resolve_state_dir", lambda: tmp_path)
    calls: list[object] = []

    def _fail_runtime(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("SYSTEM_TEST must not construct the inference runtime")

    monkeypatch.setattr(WorkerRuntime, "__init__", _fail_runtime)

    def _execute(request: object, environment: object, state_dir: Path) -> object:
        calls.append((request, environment, state_dir))
        return SystemTestOutcome(
            status=worker_main.SystemTestStatus.ACKED,
            edge_event_id="00000000-0000-4000-8000-000000000099",
            correlation_id="00000000-0000-4000-8000-000000000099",
            backend_event_id="backend-system-test",
        )

    monkeypatch.setattr(worker_main, "execute_system_test", _execute)

    # When: the explicit one-shot action is invoked.
    exit_code = worker_main.main(
        [
            "--system-test",
            "emit",
            "--system-test-validation-run-id",
            "0197f671-3a31-7a6c-a6e4-83ed412de80f",
            "--confirm-system-test",
            "SYSTEM_TEST",
        ]
    )

    # Then: no worker runtime is constructed and stdout contains only safe IDs/status.
    assert exit_code == 0
    assert len(calls) == 1
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "status": "ACKED",
        "edge_event_id": "00000000-0000-4000-8000-000000000099",
        "correlation_id": "00000000-0000-4000-8000-000000000099",
        "backend_event_id": "backend-system-test",
        "error_code": None,
    }


def test_system_test_cli_treats_previously_acked_recovery_as_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ML_WORKER_SYSTEM_TEST_GATE", "SYSTEM_TEST_OPERATOR_ENABLED")
    monkeypatch.setattr(worker_main, "resolve_state_dir", lambda: tmp_path)
    monkeypatch.setattr(
        worker_main,
        "execute_system_test",
        lambda *_args: SystemTestOutcome(
            status=worker_main.SystemTestStatus.PREVIOUSLY_ACKED,
            edge_event_id="00000000-0000-4000-8000-000000000099",
            correlation_id="00000000-0000-4000-8000-000000000099",
            backend_event_id="backend-system-test",
        ),
    )

    exit_code = worker_main.main(
        [
            "--system-test",
            "emit",
            "--system-test-validation-run-id",
            "0197f671-3a31-7a6c-a6e4-83ed412de80f",
            "--confirm-system-test",
            "SYSTEM_TEST",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PREVIOUSLY_ACKED"


# --- config resolution / exit code 2 -----------------------------------


def test_missing_config_path_exits_with_config_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDGE_CAMERA_CONFIG", raising=False)

    assert worker_main.main([]) == 2


def test_invalid_yaml_exits_with_config_error_code(tmp_path: Path) -> None:
    bad_path = tmp_path / "broken.yaml"
    bad_path.write_text("relay: [unterminated", encoding="utf-8")

    assert worker_main.main(["--config", str(bad_path)]) == 2


def test_config_missing_required_field_exits_with_config_error_code(
    tmp_path: Path,
) -> None:
    """필수 섹션이 빠진 설정은 계속 config-error(2)로 죽는다.

    이 픽스처는 원래 `relay`만 있고 `cameras`가 없는 YAML이었다. 이슈 #150
    이후로는 그게 "필수 필드 누락"이 아니라 **유효한 0대 로스터**다 -- 새로
    설치한 노드가 첫 카메라를 등록하기 전 상태이므로 부팅해야 한다. 그래서
    실제로 여전히 필수인 `relay`를 빼도록 바꿨다. "설정 없음"은 예전처럼
    죽고 "카메라 없음"은 뜬다는 구분(이슈 #43 대 #150)이 이 테스트가 지키는
    선이다.
    """
    incomplete_path = tmp_path / "incomplete.yaml"
    incomplete_path.write_text(
        yaml.safe_dump({"version": 1}),
        encoding="utf-8",
    )

    assert worker_main.main(["--config", str(incomplete_path)]) == 2


def test_edge_camera_config_env_has_no_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression guard: the camera roster/config path must never be
    provisionable through the environment (an env var reaches the runtime via
    compose, and compose is tracked in Git) -- see
    ``scripts/verify_scope_fidelity.py``'s ``ROSTER_PATTERN``. A static-YAML
    boot is CLI-flag-only (``--config``); setting ``EDGE_CAMERA_CONFIG`` must
    be inert, so with no ``--config`` and no ``RELAY_URL`` this still exits
    config-error(2) exactly as if the env var were never set."""
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("EDGE_CAMERA_CONFIG", str(config_path))
    monkeypatch.delenv("RELAY_URL", raising=False)

    assert worker_main.main(["--check-config"]) == 2


# --- --check-config has zero model/camera/relay side effects -----------


def test_check_config_validates_without_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("check-config must not touch relay or construct WorkerRuntime")

    monkeypatch.setattr(worker_main, "bounded_request", _fail)
    monkeypatch.setattr(worker_main, "make_restart_check", _fail)
    monkeypatch.setattr(WorkerRuntime, "__init__", _fail)

    assert worker_main.main(["--config", str(config_path), "--check-config"]) == 0


def test_check_config_ignores_heartbeat_on_start_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("check-config must have zero relay side effects")

    monkeypatch.setattr(worker_main, "bounded_request", _fail)
    monkeypatch.setattr(WorkerRuntime, "__init__", _fail)

    exit_code = worker_main.main(
        ["--config", str(config_path), "--check-config", "--heartbeat-on-start"]
    )

    assert exit_code == 0


# --- real WorkerRuntime construction: proves no TypeError ---------------


def test_real_workerruntime_constructs_with_fake_collaborators_without_typeerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guards the historical bug: `WorkerRuntime(config)` positionally, with
    no `loop_factory`/`serving_client`, raised TypeError at runtime.

    `worker/__main__.py` no longer passes `loop_factory` itself (composing
    the real per-camera ingest loop is `WorkerRuntime`'s own responsibility);
    this test supplies one explicitly as a fake seam, alongside the other
    collaborators, so it keeps proving "real WorkerRuntime + fake
    collaborators construct without TypeError" independent of whether
    `loop_factory` is mandatory or has a real default at construction time.
    """
    config_path = _write_config(tmp_path)
    constructed: list[WorkerRuntime] = []
    real_init = WorkerRuntime.__init__
    fake_serving = _FakeServingClient()

    def _spy_init(self: WorkerRuntime, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("loop_factory", _fake_loop_factory)
        real_init(self, *args, **kwargs)
        constructed.append(self)

    monkeypatch.setattr(WorkerRuntime, "__init__", _spy_init)
    monkeypatch.setattr(WorkerRuntime, "run", lambda self: None)
    monkeypatch.setattr(worker_main, "InProcessServingClient", lambda: fake_serving)

    exit_code = worker_main.main(["--config", str(config_path)])

    assert exit_code == 0
    assert len(constructed) == 1
    runtime = constructed[0]
    assert runtime._loop_factory is _fake_loop_factory  # noqa: SLF001
    assert runtime._serving is fake_serving  # noqa: SLF001


# --- restart_check wiring ------------------------------------------------


def test_restart_check_wired_from_config_relay_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.delenv("RELAY_URL", raising=False)
    monkeypatch.delenv("RELAY_TOKEN", raising=False)
    calls: list[tuple[str, str, RestartDirective, object]] = []
    sentinel_check = lambda: False  # noqa: E731

    def _fake_make_restart_check(
        relay_url: str,
        relay_token: str,
        boot_directive: RestartDirective,
        *,
        pull_config: object,
        **_kwargs: object,
    ) -> object:
        calls.append((relay_url, relay_token, boot_directive, pull_config))
        return sentinel_check

    constructed: list[WorkerRuntime] = []
    real_init = WorkerRuntime.__init__

    def _spy_init(self: WorkerRuntime, *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)
        constructed.append(self)

    monkeypatch.setattr(worker_main, "make_restart_check", _fake_make_restart_check)
    monkeypatch.setattr(WorkerRuntime, "__init__", _spy_init)
    monkeypatch.setattr(WorkerRuntime, "run", lambda self: None)

    exit_code = worker_main.main(["--config", str(config_path)])

    assert exit_code == 0
    assert len(calls) == 1
    relay_url, relay_token, boot_directive, pull_config = calls[0]
    assert relay_url == "http://127.0.0.1:8000"
    assert relay_token == "relay-token-1"
    assert boot_directive == RestartDirective(generation=0, version=0)
    assert pull_config is worker_main.pull_worker_config
    assert constructed[0]._restart_check is sentinel_check  # noqa: SLF001


def test_restart_check_relay_env_vars_override_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("RELAY_URL", "http://relay.test")
    monkeypatch.setenv("RELAY_TOKEN", "relay-token")
    calls: list[tuple[str, str]] = []

    def _fake_make_restart_check(
        relay_url: str, relay_token: str, _boot_directive: object, *, pull_config: object
    ) -> object:
        calls.append((relay_url, relay_token))
        return lambda: False

    monkeypatch.setattr(worker_main, "make_restart_check", _fake_make_restart_check)
    monkeypatch.setattr(WorkerRuntime, "run", lambda self: None)

    assert worker_main.main(["--config", str(config_path)]) == 0
    assert calls == [("http://relay.test", "relay-token")]


# --- --max-frames-per-camera passthrough ----------------------------------


def test_max_frames_per_camera_wired_to_workerruntime_constructor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)
    constructed: list[WorkerRuntime] = []
    real_init = WorkerRuntime.__init__

    def _spy_init(self: WorkerRuntime, *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)
        constructed.append(self)

    monkeypatch.setattr(WorkerRuntime, "__init__", _spy_init)
    monkeypatch.setattr(WorkerRuntime, "run", lambda self: None)

    exit_code = worker_main.main(
        ["--config", str(config_path), "--max-frames-per-camera", "3200"]
    )

    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0]._max_frames_per_camera == 3200  # noqa: SLF001


def test_max_frames_per_camera_defaults_to_none_when_flag_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)
    constructed: list[WorkerRuntime] = []
    real_init = WorkerRuntime.__init__

    def _spy_init(self: WorkerRuntime, *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)
        constructed.append(self)

    monkeypatch.setattr(WorkerRuntime, "__init__", _spy_init)
    monkeypatch.setattr(WorkerRuntime, "run", lambda self: None)

    exit_code = worker_main.main(["--config", str(config_path)])

    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0]._max_frames_per_camera is None  # noqa: SLF001


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "3.5"])
def test_max_frames_per_camera_rejects_non_positive_or_non_integer_values(raw: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        worker_main.main(["--max-frames-per-camera", raw])

    assert exc_info.value.code == 2


# --- heartbeat-on-start passthrough ---------------------------------------


def test_heartbeat_on_start_sends_canonical_heartbeat_per_camera(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, camera_count=2, version=3)
    requests: list[tuple[str, str, dict[str, str], bytes | None, float]] = []

    def _fake_bounded_request(
        url: str,
        method: str,
        headers: dict[str, str],
        data: bytes | None,
        timeout_sec: float,
        on_response: object = None,
    ) -> tuple[int, dict[str, str], bytes]:
        del on_response
        requests.append((url, method, headers, data, timeout_sec))
        return 204, {}, b""

    monkeypatch.setattr(worker_main, "bounded_request", _fake_bounded_request)
    monkeypatch.setattr(WorkerRuntime, "run", lambda self: None)

    exit_code = worker_main.main(["--config", str(config_path), "--heartbeat-on-start"])

    assert exit_code == 0
    assert len(requests) == 2
    for (url, method, headers, data, _timeout), index in zip(requests, (1, 2), strict=True):
        assert url == "http://127.0.0.1:8000/api/v1/relay/heartbeat"
        assert method == "POST"
        assert headers["X-Edge-Relay-Token"] == "relay-token-1"
        assert data is not None
        assert json.loads(data) == {
            "camera_id": f"camera-{index}",
            "facility_id": "facility-1",
            "config_version": 3,
        }


def test_heartbeat_on_start_flag_off_sends_no_heartbeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("heartbeat must not be sent without --heartbeat-on-start")

    monkeypatch.setattr(worker_main, "bounded_request", _fail)
    monkeypatch.setattr(WorkerRuntime, "run", lambda self: None)

    assert worker_main.main(["--config", str(config_path)]) == 0


def test_heartbeat_on_start_failure_is_nonfatal_and_run_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)

    def _failing_bounded_request(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("relay unreachable")

    ran: list[bool] = []
    monkeypatch.setattr(worker_main, "bounded_request", _failing_bounded_request)
    monkeypatch.setattr(WorkerRuntime, "run", lambda self: ran.append(True))

    exit_code = worker_main.main(["--config", str(config_path), "--heartbeat-on-start"])

    assert exit_code == 0
    assert ran == [True]


# --- exit codes 0 / 1 / 3 from runtime.run() ------------------------------


def test_clean_run_returns_zero_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(WorkerRuntime, "run", lambda self: None)

    assert worker_main.main(["--config", str(config_path)]) == 0


def test_bootstrap_systemexit_translates_to_its_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`bootstrap.bootstrap_or_exit` calls `sys.exit(stage.exit_code)` directly
    on a global stage failure; `WorkerRuntime.run` never intercepts it, so it
    reaches `main()` as `SystemExit`, not `BootstrapStageError`."""
    config_path = _write_config(tmp_path)

    def _fake_run(self: WorkerRuntime) -> None:
        raise SystemExit(3)

    monkeypatch.setattr(WorkerRuntime, "run", _fake_run)

    assert worker_main.main(["--config", str(config_path)]) == 3


def test_systemexit_with_non_int_code_falls_back_to_generic_error_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)

    def _fake_run(self: WorkerRuntime) -> None:
        raise SystemExit("boom")

    monkeypatch.setattr(WorkerRuntime, "run", _fake_run)

    assert worker_main.main(["--config", str(config_path)]) == 1


def test_generic_runtime_error_returns_generic_error_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)

    def _fake_run(self: WorkerRuntime) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(WorkerRuntime, "run", _fake_run)

    assert worker_main.main(["--config", str(config_path)]) == 1


# --- signal shutdown -------------------------------------------------------


def test_sigint_triggers_clean_shutdown_and_restores_previous_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)
    stop_calls: list[WorkerRuntime] = []

    def _fake_run(self: WorkerRuntime) -> None:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)

    monkeypatch.setattr(WorkerRuntime, "run", _fake_run)
    monkeypatch.setattr(WorkerRuntime, "stop", lambda self: stop_calls.append(self))
    previous_handler = signal.getsignal(signal.SIGINT)

    exit_code = worker_main.main(["--config", str(config_path)])

    assert exit_code == 0
    assert len(stop_calls) == 1
    assert signal.getsignal(signal.SIGINT) is previous_handler


def test_sigterm_triggers_clean_shutdown_and_restores_previous_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)
    stop_calls: list[WorkerRuntime] = []

    def _fake_run(self: WorkerRuntime) -> None:
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)

    monkeypatch.setattr(WorkerRuntime, "run", _fake_run)
    monkeypatch.setattr(WorkerRuntime, "stop", lambda self: stop_calls.append(self))
    previous_handler = signal.getsignal(signal.SIGTERM)

    exit_code = worker_main.main(["--config", str(config_path)])

    assert exit_code == 0
    assert len(stop_calls) == 1
    assert signal.getsignal(signal.SIGTERM) is previous_handler
