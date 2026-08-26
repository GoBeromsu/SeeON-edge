"""Round-2 trust-boundary and dead-surface regressions for PR #401."""

from __future__ import annotations

import socket
import threading
import uuid
from pathlib import Path
from typing import cast

import pytest

import worker.runtime.deepstream as dark_runtime
import worker.runtime.deepstream.fault as fault_module
from worker.native.deepstream.control import (
    ChildControlError,
    ControlIdentity,
    DeepStreamControlClient,
    parse_source_uri,
)
from worker.runtime.deepstream import ChildConfig
from worker.runtime.deepstream.failure_receiver import NativeFailureReceiver
from worker.runtime.deepstream.fault import persist_first_fault

_BOOT = uuid.UUID("12345678-1234-5678-1234-567812345678")
_CHILD = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_TRANSFORM = "seeon-perception-v1"


def _socket_type_only(value: object) -> socket.socket:
    return cast(socket.socket, value)


def test_first_fault_returns_record_actually_persisted(tmp_path: Path) -> None:
    path = tmp_path / "private" / "fault.json"
    first = persist_first_fault(
        path, category="cuda", exit_code=4, worker_boot_id=_BOOT, child_instance_id=_CHILD
    )
    second = persist_first_fault(
        path, category="xid", exit_code=4, worker_boot_id=_BOOT, child_instance_id=_CHILD
    )
    assert first.category == second.category == "cuda"


def test_first_fault_publication_is_atomic_for_concurrent_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "private" / "fault.json"
    first_writer_entered = threading.Event()
    release_first_writer = threading.Event()
    real_fdopen = fault_module.os.fdopen
    delayed = False

    def controlled_fdopen(descriptor: int, mode: str, *args, **kwargs):
        nonlocal delayed
        if mode == "wb" and not delayed:
            delayed = True
            first_writer_entered.set()
            assert release_first_writer.wait(timeout=2.0)
        return real_fdopen(descriptor, mode, *args, **kwargs)

    monkeypatch.setattr(fault_module.os, "fdopen", controlled_fdopen)
    results = []
    errors: list[Exception] = []

    def write_first() -> None:
        try:
            results.append(
                persist_first_fault(
                    path,
                    category="cuda",
                    exit_code=4,
                    worker_boot_id=_BOOT,
                    child_instance_id=_CHILD,
                )
            )
        except Exception as error:  # noqa: BLE001 - test-thread boundary
            errors.append(error)

    thread = threading.Thread(target=write_first)
    thread.start()
    assert first_writer_entered.wait(timeout=2.0)
    try:
        results.append(
            persist_first_fault(
                path,
                category="xid",
                exit_code=4,
                worker_boot_id=_BOOT,
                child_instance_id=_CHILD,
            )
        )
    finally:
        release_first_writer.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1]


def test_first_fault_refuses_symlink_target(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    target = tmp_path / "target"
    _ = target.write_text("trusted", encoding="utf-8")
    path = private / "fault.json"
    path.symlink_to(target)
    with pytest.raises(OSError):
        _ = persist_first_fault(
            path, category="cuda", exit_code=4, worker_boot_id=_BOOT, child_instance_id=_CHILD
        )
    assert target.read_text(encoding="utf-8") == "trusted"


def test_unexpected_failure_channel_eof_is_typed_child_exit() -> None:
    sender, receiver_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    fatal = threading.Event()
    categories: list[str] = []

    def on_fatal(category: str) -> None:
        categories.append(category)
        fatal.set()

    receiver = NativeFailureReceiver(
        receiver_socket,
        _BOOT,
        _CHILD,
        lambda camera, category: None,
        on_fatal,
    )
    receiver.start()
    sender.close()
    assert fatal.wait(timeout=1.0)
    receiver.close()
    assert categories == ["child_exit"]


def test_uri_validator_caps_length_and_accepts_quote_data() -> None:
    credentialed_uri = "rtsp://user:p'ass" + chr(64) + "camera.example/live"
    assert parse_source_uri(credentialed_uri) == credentialed_uri
    with pytest.raises(ChildControlError):
        _ = parse_source_uri("rtsp://camera.example/" + "x" * 8_192)
    with pytest.raises(ChildControlError):
        _ = parse_source_uri("rtsp:///missing-host")


def test_transport_child_command_has_no_gpu_id_argument(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import worker.runtime.deepstream.transport as transport

    captured: list[transport.SpawnRequest] = []

    def fail(request: transport.SpawnRequest) -> None:
        captured.append(request)
        raise OSError("captured")

    monkeypatch.setattr(transport, "spawn_process", fail)
    with pytest.raises(OSError):
        _ = transport.spawn_child(
            ChildConfig(
                executable=tmp_path / "child",
                worker_boot_id=_BOOT,
                socket_dir=tmp_path / "ipc",
                first_fault_path=tmp_path / "fault",
            )
        )
    assert "--gpu-id" not in captured[0].command
    assert captured[0].environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_transport_closes_socketpair_after_partial_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import worker.runtime.deepstream.transport as transport

    real_socketpair = socket.socketpair
    created: list[socket.socket] = []
    calls = 0

    def fail_second(
        family: socket.AddressFamily = socket.AF_UNIX,
        type_: socket.SocketKind = socket.SOCK_STREAM,
        protocol: int = 0,
    ) -> tuple[socket.socket, socket.socket]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("partial initialization")
        endpoints = real_socketpair(family, type_, protocol)
        created.extend(endpoints)
        return endpoints

    monkeypatch.setattr(socket, "socketpair", fail_second)
    with pytest.raises(OSError):
        _ = transport.spawn_child(
            ChildConfig(
                executable=tmp_path / "child",
                worker_boot_id=_BOOT,
                socket_dir=tmp_path / "ipc",
                first_fault_path=tmp_path / "fault",
            )
        )
    assert created and all(endpoint.fileno() == -1 for endpoint in created)


def test_dark_runtime_exports_continuously_monitored_pid1_runner() -> None:
    assert callable(dark_runtime.run_dark_child)
    assert dark_runtime.FATAL_CHILD_EXIT_CODE == 4


def test_named_endpoint_dead_surfaces_are_absent() -> None:
    identity = ControlIdentity(_BOOT, _CHILD, _TRANSFORM)
    with pytest.raises(TypeError):
        _ = DeepStreamControlClient(_socket_type_only(Path("/tmp/dead")), identity)
