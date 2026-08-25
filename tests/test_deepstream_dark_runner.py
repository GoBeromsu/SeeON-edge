"""Dedicated dark PID-1 runner fatal containment regressions."""

from __future__ import annotations

import os
import signal
import stat
import sys
import textwrap
import uuid
from pathlib import Path

import pytest
from pydantic import TypeAdapter

import worker.runtime.deepstream.__main__ as dark_main
from worker.runtime.deepstream import ChildConfig, DarkRunRequest, run_dark_child
from worker.runtime.deepstream.fault import DarkFirstFault


def _fake_child(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-native-child.py"
    _ = executable.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
            import os, socket, struct, sys
            sys.path.insert(0, os.getcwd())
            from worker.native.deepstream.ipc import (
                MessageKind, decode_control_message, encode_message,
            )

            args = dict(zip(sys.argv[1::2], sys.argv[2::2], strict=True))
            control = socket.socket(fileno=int(args["--control-fd"]))
            identity_fd = int(args["--identity-fd"])
            identity = os.read(identity_fd, 36)
            os.close(identity_fd)
            mode = os.environ.get("FAKE_DEEPSTREAM_CHILD_MODE", "normal")
            if mode == "no_ready":
                signal_fd = os.eventfd(0)
                os.read(signal_fd, 8)
            os.write(int(args["--ready-fd"]), b"R")
            os.close(int(args["--ready-fd"]))
            if mode == "hung":
                signal_fd = os.eventfd(0)
                os.read(signal_fd, 8)
            while True:
                raw = control.recv(65535)
                if not raw:
                    raise SystemExit(4)
                request = decode_control_message(raw)
                if request.kind is MessageKind.STATUS:
                    reply = request.__class__(
                        MessageKind.STATUS_REPLY, request.worker_boot_id, request.child_instance_id,
                        request.camera_id, 0, 0, 0, 0, 0, request.request_id,
                        request.transform_id, struct.pack("<QQQQQIB", 0, 0, 0, 0, 0, 0, 1),
                    )
                    control.sendall(encode_message(reply))
                    if mode == "unexpected":
                        raise SystemExit(9)
                elif request.kind is MessageKind.FATAL:
                    reply = request.__class__(
                        MessageKind.ACK, request.worker_boot_id, request.child_instance_id,
                        request.camera_id, 0, 0, 0, 0, 0, request.request_id,
                        request.transform_id, b"",
                    )
                    control.sendall(encode_message(reply))
                    raise SystemExit(4)
                elif request.kind is MessageKind.SHUTDOWN:
                    reply = request.__class__(
                        MessageKind.ACK, request.worker_boot_id, request.child_instance_id,
                        request.camera_id, 0, 0, 0, 0, 0, request.request_id,
                        request.transform_id, b"",
                    )
                    control.sendall(encode_message(reply))
                    raise SystemExit(0)
                else:
                    raise SystemExit(4)
            """
        ),
        encoding="utf-8",
    )
    _ = executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return executable


def _request(tmp_path: Path, executable: Path, *, fatal: str | None = None) -> DarkRunRequest:
    state = tmp_path / "state"
    return DarkRunRequest(
        child=ChildConfig(
            executable=executable,
            worker_boot_id=uuid.uuid4(),
            socket_dir=state / "ipc",
            first_fault_path=state / "deepstream-first-fault.json",
            lease_state_dir=state,
            startup_timeout_sec=0.2,
            stop_timeout_sec=0.2,
        ),
        inject_fatal=fatal,
    )


def test_dark_cli_keeps_installed_child_default_with_typed_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    captured: list[DarkRunRequest] = []

    def capture(request: DarkRunRequest) -> int:
        captured.append(request)
        return 0

    monkeypatch.setattr(dark_main, "run_dark_child", capture)

    # When
    exit_code = dark_main.main(["--state-dir", str(tmp_path)])

    # Then
    assert exit_code == 0
    assert captured[0].child.executable == Path("/usr/local/bin/seeon-deepstream-child")


def test_graceful_sigterm_exits_zero_without_fault(tmp_path: Path) -> None:
    # Given
    class Sources:
        def add(self, camera_id: str, uri: str) -> object:
            del camera_id, uri
            return object()

    class GracefulSupervisor:
        def __init__(self, config: ChildConfig) -> None:
            del config
            self.stopped: bool = False

        @property
        def sources(self) -> Sources:
            return Sources()

        def start(self) -> None:
            os.kill(os.getpid(), signal.SIGTERM)

        def stop(self) -> None:
            self.stopped = True

        def fatal(self, category: str) -> None:
            del category

        def wait(self) -> int:
            return 0

    request = _request(tmp_path, tmp_path / "unused")

    # When
    exit_code = run_dark_child(request, supervisor_factory=GracefulSupervisor)

    # Then
    assert exit_code == 0
    assert not request.child.first_fault_path.exists()


def test_injected_fatal_makes_runner_exit_four_with_typed_durable_fault(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    monkeypatch.setenv("FAKE_DEEPSTREAM_CHILD_MODE", "normal")
    request = _request(tmp_path, _fake_child(tmp_path), fatal="cuda")

    # When
    exit_code = run_dark_child(request)

    # Then
    fault = TypeAdapter(DarkFirstFault).validate_json(request.child.first_fault_path.read_bytes())
    assert exit_code == 4
    assert fault.category == "cuda"
    assert fault.exit_code == 4


def test_unexpected_child_exit_is_fatal_without_respawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    monkeypatch.setenv("FAKE_DEEPSTREAM_CHILD_MODE", "unexpected")
    request = _request(tmp_path, _fake_child(tmp_path))

    # When
    exit_code = run_dark_child(request)

    # Then
    fault = TypeAdapter(DarkFirstFault).validate_json(request.child.first_fault_path.read_bytes())
    assert exit_code == 4
    assert fault.category == "child_exit"
    assert fault.stage == "deepstream_child"


def test_missing_ready_signal_is_bounded_fatal_and_reaped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    monkeypatch.setenv("FAKE_DEEPSTREAM_CHILD_MODE", "no_ready")
    request = _request(tmp_path, _fake_child(tmp_path))

    # When
    exit_code = run_dark_child(request)

    # Then
    assert exit_code == 4
    assert not request.child.socket_dir.exists() or not any(request.child.socket_dir.iterdir())


def test_hung_child_handshake_is_fatal_and_reaped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    monkeypatch.setenv("FAKE_DEEPSTREAM_CHILD_MODE", "hung")
    request = _request(tmp_path, _fake_child(tmp_path))

    # When
    exit_code = run_dark_child(request)

    # Then
    assert exit_code == 4
    assert request.child.first_fault_path.exists()
    assert not tuple(request.child.socket_dir.glob("*.sock"))
