"""Round-3 regressions for PR #401 concurrency and exit-status findings."""

from __future__ import annotations

import socket
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

import worker.runtime.deepstream.runner as runner_module
from worker.native.deepstream.control import ChildControlError
from worker.native.deepstream.ipc import ControlMessage, MessageKind, MetadataFrame
from worker.native.deepstream.metadata import AcceptanceToken, LatestMetadataSlot, SourceBinding
from worker.native.deepstream.metadata_receiver import MetadataReceiver
from worker.runtime.deepstream import ChildConfig, DarkRunRequest, DarkSource
from worker.runtime.deepstream.source_control import DarkSourceController, SourceState

_BOOT = uuid.UUID("12345678-1234-5678-1234-567812345678")
_CHILD = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_TRANSFORM = "seeon-perception-v1"


def _binding(*, epoch: int) -> SourceBinding:
    return SourceBinding(str(_BOOT), str(_CHILD), "camera-a", 1, epoch, _TRANSFORM)


def _frame(*, epoch: int, pts: int, sequence: int, publish_sequence: int) -> MetadataFrame:
    return MetadataFrame.empty(
        ControlMessage(
            MessageKind.METADATA,
            _BOOT,
            _CHILD,
            "camera-a",
            1,
            epoch,
            pts,
            sequence,
            publish_sequence,
            0,
            _TRANSFORM,
        )
    )


class _LifecycleControl:
    def add_source(self, camera_id: str, uri: str) -> SourceBinding:
        del camera_id, uri
        return _binding(epoch=1)

    def remove_source(self, camera_id: str) -> None:
        del camera_id

    def source_failure(self, camera_id: str, category: str) -> SourceBinding:
        del camera_id, category
        return _binding(epoch=2)


class _BindingReceiver:
    def __init__(self, slot: LatestMetadataSlot) -> None:
        self._slot: LatestMetadataSlot = slot
        self._handler: Callable[[SourceBinding, MetadataFrame], None] | None = None

    def set_binding_handler(
        self,
        handler: Callable[[SourceBinding, MetadataFrame], None],
    ) -> None:
        self._handler = handler

    def pull_now(self, camera_id: str) -> MetadataFrame | None:
        del camera_id
        return None

    def deliver(self, binding: SourceBinding, frame: MetadataFrame) -> None:
        if self._handler is not None:
            self._handler(binding, frame)
        assert self._slot.publish(frame)


class _ObservedSlot:
    def __init__(self) -> None:
        self.slot: LatestMetadataSlot = LatestMetadataSlot()
        self.wait_count: int = 0
        self.rebuild_waiting: threading.Event = threading.Event()

    def register_source(self, binding: SourceBinding) -> AcceptanceToken:
        return self.slot.register_source(binding)

    def remove_source(self, camera_id: str) -> None:
        self.slot.remove_source(camera_id)

    def wait_accepted(self, token: AcceptanceToken, *, timeout_sec: float) -> MetadataFrame:
        del timeout_sec
        self.wait_count += 1
        if self.wait_count == 1:
            assert self.slot.publish(_frame(epoch=1, pts=1, sequence=1, publish_sequence=1))
        else:
            self.rebuild_waiting.set()
        return self.slot.wait_accepted(token, timeout_sec=0.25)


def test_binding_delivery_cannot_deadlock_rebuild_readiness() -> None:
    admission = _ObservedSlot()
    receiver = _BindingReceiver(admission.slot)
    controller = DarkSourceController(_LifecycleControl(), admission, receiver)
    assert controller.add("camera-a", "loopback://camera-a").state is SourceState.SOURCE_READY

    delivered = threading.Event()

    def deliver_rebuild_frame() -> None:
        assert admission.rebuild_waiting.wait(timeout=1.0)
        receiver.deliver(
            _binding(epoch=2),
            _frame(epoch=2, pts=1, sequence=1, publish_sequence=2),
        )
        delivered.set()

    delivery = threading.Thread(target=deliver_rebuild_frame)
    delivery.start()
    rebuilt = controller.rebuild("camera-a", "eos")
    delivery.join(timeout=1.0)

    assert delivered.is_set()
    assert rebuilt.state is SourceState.SOURCE_READY
    assert rebuilt.stream_epoch == 2


def test_metadata_receiver_second_close_is_harmless() -> None:
    class Puller:
        def pull_latest(self, camera_id: str) -> MetadataFrame | None:
            del camera_id
            return None

    sender, receiver_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    receiver = MetadataReceiver(receiver_socket, LatestMetadataSlot(), Puller())
    try:
        _ = receiver.__enter__()
        receiver.close()
        receiver.close()
    finally:
        sender.close()


def test_runner_contains_child_control_error_as_typed_exit_four(tmp_path: Path) -> None:
    class Sources:
        def add(self, camera_id: str, uri: str) -> None:
            del camera_id, uri
            raise ChildControlError("control_protocol", "bad reply")

    class Supervisor:
        def __init__(self, config: ChildConfig) -> None:
            del config

        @property
        def sources(self) -> Sources:
            return Sources()

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def fatal(self, category: str) -> None:
            del category

        def wait(self) -> int:
            return 0

    request = DarkRunRequest(
        ChildConfig(
            executable=tmp_path / "child",
            worker_boot_id=_BOOT,
            socket_dir=tmp_path / "ipc",
            first_fault_path=tmp_path / "fault.json",
            lease_state_dir=tmp_path,
        ),
        (DarkSource("camera-a", "loopback://camera-a"),),
    )

    assert runner_module.run_dark_child(request, supervisor_factory=Supervisor) == 4
    assert b'"category":"control_protocol"' in request.child.first_fault_path.read_bytes()
