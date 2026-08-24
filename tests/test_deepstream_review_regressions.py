"""Regression reproductions from the failed PR #401 five-lane review."""

from __future__ import annotations

import socket
import threading
import uuid
from pathlib import Path
from typing import override

import pytest

import worker.native.deepstream.control as control_module
import worker.runtime.deepstream as dark_runtime
import worker.runtime.deepstream.supervisor as supervisor_module
import worker.runtime.deepstream.transport as transport_module
from worker.native.deepstream.ipc import (
    ControlMessage,
    MessageKind,
    MetadataFrame,
    encode_message,
)
from worker.native.deepstream.metadata import (
    LatestMetadataSlot,
    MetadataPullFailure,
    MetadataReceiver,
    SourceBinding,
)
from worker.runtime.deepstream.config import ChildConfig
from worker.runtime.deepstream.failure_receiver import NativeFailureReceiver
from worker.runtime.deepstream.source_control import DarkSourceController, SourceState

_BOOT = uuid.UUID("12345678-1234-5678-1234-567812345678")
_CHILD = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_TRANSFORM = "seeon-perception-v1"


def _binding(camera_id: str, *, generation: int = 1, epoch: int = 1) -> SourceBinding:
    return SourceBinding(str(_BOOT), str(_CHILD), camera_id, generation, epoch, _TRANSFORM)


def _frame(
    camera_id: str,
    *,
    generation: int = 1,
    epoch: int = 1,
    sequence: int = 1,
) -> MetadataFrame:
    return MetadataFrame.empty(
        ControlMessage(
            kind=MessageKind.METADATA,
            worker_boot_id=_BOOT,
            child_instance_id=_CHILD,
            camera_id=camera_id,
            source_generation=generation,
            stream_epoch=epoch,
            source_pts=sequence * 1_000,
            source_sequence=sequence,
            native_publish_sequence=sequence,
            request_id=0,
            transform_id=_TRANSFORM,
        )
    )


class _NoopReceiver:
    def pull_now(self, camera_id: str) -> MetadataFrame | None:
        del camera_id
        return None


class _RecoveringPuller:
    def __init__(self) -> None:
        self.calls: int = 0
        self.first_pull: threading.Event = threading.Event()

    def pull_latest(self, camera_id: str) -> MetadataFrame | None:
        self.calls += 1
        if self.calls == 1:
            self.first_pull.set()
            raise MetadataPullFailure("temporary_pull")
        return _frame(camera_id, sequence=2)


def test_metadata_receiver_recovers_after_one_pull_failure() -> None:
    # Given
    slot = LatestMetadataSlot()
    _ = slot.register_source(_binding("camera-a"))
    sender, wake_receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    puller = _RecoveringPuller()
    token = slot.subscribe(_binding("camera-a"))

    # When
    with MetadataReceiver(wake_receiver, slot, puller):
        try:
            _ = sender.send(b"camera-a")
            assert puller.first_pull.wait(timeout=1.0)
            _ = sender.send(b"camera-a")
            _ = slot.wait_accepted(token, timeout_sec=1.0)
        finally:
            sender.close()

    # Then
    assert slot.peek("camera-a") is not None
    assert slot.counters().pull_failures == 1


def test_empty_metadata_datagram_is_rejected_without_stopping_receiver() -> None:
    # Given
    slot = LatestMetadataSlot()
    _ = slot.register_source(_binding("camera-a"))
    sender, wake_receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)

    class Puller:
        def pull_latest(self, camera_id: str) -> MetadataFrame | None:
            return _frame(camera_id)

    # When
    token = slot.subscribe(_binding("camera-a"))
    with MetadataReceiver(wake_receiver, slot, Puller()):
        try:
            _ = sender.send(b"")
            _ = sender.send(b"camera-a")
            _ = slot.wait_accepted(token, timeout_sec=1.0)
        finally:
            sender.close()

    # Then
    assert slot.peek("camera-a") is not None
    assert slot.counters().malformed == 1


def test_synthetic_metadata_cannot_establish_source_ready() -> None:
    # Given
    slot = LatestMetadataSlot()

    class Control:
        def add_source(self, camera_id: str, uri: str) -> SourceBinding:
            del uri
            return _binding(camera_id)

        def remove_source(self, camera_id: str) -> None:
            del camera_id

        def source_failure(self, camera_id: str, category: str) -> SourceBinding:
            del category
            return _binding(camera_id, epoch=2)

    controller = DarkSourceController(Control(), slot, _NoopReceiver())

    # When / Then
    with pytest.raises(TimeoutError):
        _ = controller.add("camera-a", "loopback://camera-a")
    assert controller.snapshot("camera-a").state is SourceState.TOMBSTONED


def test_readiness_wait_is_scoped_to_camera_and_binding() -> None:
    # Given
    slot = LatestMetadataSlot()
    _ = slot.register_source(_binding("camera-a"))
    _ = slot.register_source(_binding("camera-b"))
    token = slot.subscribe(_binding("camera-a"))

    # When
    assert slot.publish(_frame("camera-b", sequence=4))

    # Then
    with pytest.raises(TimeoutError):
        _ = slot.wait_accepted(token, timeout_sec=0.01)


def test_latest_selection_uses_native_publish_high_water_after_pause() -> None:
    # Given
    slot = LatestMetadataSlot()
    _ = slot.register_source(_binding("camera-a"))
    for sequence in range(1, 65):
        assert slot.publish(_frame("camera-a", sequence=sequence))

    # When
    latest = slot.peek("camera-a")

    # Then
    assert latest is not None
    assert latest.native_publish_sequence == 64
    assert slot.counters().overwritten == 63


def test_failed_add_rolls_back_to_recoverable_tombstone() -> None:
    # Given
    class FailedControl:
        def add_source(self, camera_id: str, uri: str) -> SourceBinding:
            del camera_id, uri
            raise control_module.ChildControlError("source_add_failed", "typed")

        def remove_source(self, camera_id: str) -> None:
            del camera_id

        def source_failure(self, camera_id: str, category: str) -> SourceBinding:
            del category
            return _binding(camera_id)

    controller = DarkSourceController(
        FailedControl(),
        LatestMetadataSlot(),
        _NoopReceiver(),
    )

    # When
    with pytest.raises(control_module.ChildControlError):
        _ = controller.add("camera-a", "loopback://camera-a")

    # Then
    assert controller.snapshot("camera-a").state is SourceState.TOMBSTONED


def test_failed_rebuild_leaves_recoverable_degraded_binding() -> None:
    # Given
    slot = LatestMetadataSlot()

    class FailedControl:
        def add_source(self, camera_id: str, uri: str) -> SourceBinding:
            del uri
            return _binding(camera_id, generation=3, epoch=7)

        def remove_source(self, camera_id: str) -> None:
            del camera_id

        def source_failure(self, camera_id: str, category: str) -> SourceBinding:
            del camera_id, category
            raise control_module.ChildControlError("source_rebuild_failed", "typed")

    class PublishingReceiver(_NoopReceiver):
        @override
        def pull_now(self, camera_id: str) -> MetadataFrame | None:
            frame = _frame(camera_id, generation=3, epoch=7)
            assert slot.publish(frame)
            return frame

    controller = DarkSourceController(FailedControl(), slot, PublishingReceiver())
    assert controller.add("camera-a", "loopback://camera-a").state is SourceState.SOURCE_READY

    # When
    with pytest.raises(control_module.ChildControlError):
        _ = controller.rebuild("camera-a", "rtsp_auth")

    # Then
    assert controller.snapshot("camera-a") == dark_runtime.SourceSnapshot(
        "camera-a", SourceState.DEGRADED, 3, 7
    )


def test_control_supplies_inherited_ipc_and_keeps_secrets_off_argv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    captured: list[tuple[str, ...]] = []

    def fail_spawn(request: transport_module.SpawnRequest) -> None:
        captured.append(request.command)
        raise OSError("stop after capture")

    monkeypatch.setattr(transport_module, "spawn_process", fail_spawn)
    child = supervisor_module.DeepStreamChildSupervisor(
        ChildConfig(
            executable=tmp_path / "child",
            worker_boot_id=_BOOT,
            socket_dir=tmp_path / "ipc",
            first_fault_path=tmp_path / "fault",
            lease_state_dir=tmp_path / "lease",
        )
    )

    # When
    with pytest.raises(supervisor_module.ChildStartupError):
        child.start()

    # Then
    assert "--control-fd" in captured[0]
    assert "--wake-fd" in captured[0]
    assert "--boot-id" not in captured[0]
    assert "--child-id" not in captured[0]


def test_reliable_failure_channel_delivers_source_and_fatal_events() -> None:
    # Given
    sender, receiver_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    source_received = threading.Event()
    fatal_received = threading.Event()
    received: list[tuple[str, str]] = []

    def on_source(camera: str, category: str) -> None:
        received.append((camera, category))
        source_received.set()

    def on_fatal(category: str) -> None:
        received.append(("_worker", category))
        fatal_received.set()

    receiver = NativeFailureReceiver(
        receiver_socket,
        _BOOT,
        _CHILD,
        on_source,
        on_fatal,
    )
    receiver.start()

    def event(kind: MessageKind, payload: bytes) -> bytes:
        return encode_message(
            ControlMessage(
                kind, _BOOT, _CHILD, "camera-a", 1, 1, 0, 0, 0, 0,
                "seeon-perception-v1", payload,
            )
        )

    # When
    _ = sender.send(event(MessageKind.SOURCE_FAILURE, b"eos"))
    assert source_received.wait(timeout=1.0)
    _ = sender.send(event(MessageKind.FATAL, b"cuda"))
    assert fatal_received.wait(timeout=1.0)
    receiver.close()
    sender.close()

    # Then
    assert received == [("camera-a", "eos"), ("_worker", "cuda")]


def test_uri_boundary_rejects_file_scheme_and_preserves_rtsp_quotes() -> None:
    # Given
    parser = control_module.parse_source_uri

    # When / Then
    with pytest.raises(control_module.ChildControlError):
        _ = parser("file:///tmp/injected ! filesink location=/tmp/pwned")
    credentialed_uri = "rtsp://user:p'ass" + chr(64) + "camera.example/live"
    parsed = parser(credentialed_uri)
    assert parsed.encode() == b"rtsp://user:p'ass" + bytes((64,)) + b"camera.example/live"
