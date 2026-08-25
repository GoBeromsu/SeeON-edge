"""Round-2 regressions for PR #401 lifecycle and trust-boundary findings."""

from __future__ import annotations

import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import override

import pytest

import worker.runtime.deepstream.runner as runner_module
from worker.native.deepstream.ipc import ControlMessage, MessageKind, MetadataFrame
from worker.native.deepstream.metadata import AcceptanceToken, LatestMetadataSlot, SourceBinding
from worker.runtime.deepstream import ChildConfig, DarkRunRequest, DarkSource
from worker.runtime.deepstream.source_control import DarkSourceController, SourceState
from worker.types import AssociationResult

_BOOT = uuid.UUID("12345678-1234-5678-1234-567812345678")
_CHILD = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_TRANSFORM = "seeon-perception-v1"


def _binding(*, generation: int = 1, epoch: int = 1) -> SourceBinding:
    return SourceBinding(str(_BOOT), str(_CHILD), "camera-a", generation, epoch, _TRANSFORM)


def _frame(
    *,
    epoch: int,
    pts: int,
    sequence: int,
    publish_sequence: int,
    generation: int = 1,
) -> MetadataFrame:
    empty = MetadataFrame.empty(
        ControlMessage(
            MessageKind.METADATA,
            _BOOT,
            _CHILD,
            "camera-a",
            generation,
            epoch,
            pts,
            sequence,
            publish_sequence,
            0,
            _TRANSFORM,
        )
    )
    identity = empty.frame.identity
    return replace(
        empty,
        frame=replace(
            empty.frame,
            association=AssociationResult("legacy-greedy-bbox-iou.v1", (), (), identity),
        ),
        source_width=640,
        source_height=360,
        source_time_ns=max(1, pts),
    )


def test_rebind_registration_returns_zero_baseline_token_atomically() -> None:
    slot = LatestMetadataSlot()
    _ = slot.register_source(_binding(epoch=1))
    assert slot.publish(_frame(epoch=1, pts=100, sequence=100, publish_sequence=100))

    token = slot.register_source(_binding(epoch=2))

    assert token == AcceptanceToken(_binding(epoch=2), 0)
    assert slot.peek("camera-a") is None


def test_generation_change_accepts_reset_pts_and_publish_sequence() -> None:
    slot = LatestMetadataSlot()
    _ = slot.register_source(_binding(generation=1, epoch=4))
    assert slot.publish(_frame(epoch=4, pts=90_000, sequence=90, publish_sequence=90))

    _ = slot.register_source(_binding(generation=2, epoch=1))

    assert slot.publish(
        _frame(epoch=1, pts=1_000, sequence=1, publish_sequence=1, generation=2)
    )


def test_new_epoch_accepts_reset_pts_with_higher_native_publish_sequence() -> None:
    slot = LatestMetadataSlot()
    _ = slot.register_source(_binding(epoch=1))
    assert slot.publish(_frame(epoch=1, pts=90_000, sequence=90, publish_sequence=90))

    _ = slot.register_source(_binding(epoch=2))

    assert slot.publish(_frame(epoch=2, pts=1_000, sequence=1, publish_sequence=91))
    assert slot.peek("camera-a") == _frame(epoch=2, pts=1_000, sequence=1, publish_sequence=91)


class _TimeoutThenReadySlot:
    def __init__(self) -> None:
        self.waits: int = 0
        self.registered: SourceBinding | None = None
        self.removed: list[str] = []

    def register_source(self, binding: SourceBinding) -> AcceptanceToken:
        self.registered = binding
        return AcceptanceToken(binding, 0)

    def remove_source(self, camera_id: str) -> None:
        self.registered = None
        self.removed.append(camera_id)

    def wait_accepted(self, token: AcceptanceToken, *, timeout_sec: float) -> MetadataFrame:
        del token, timeout_sec
        self.waits += 1
        if self.waits == 1:
            raise TimeoutError("metadata binding deadline elapsed")
        assert self.registered is not None
        return _frame(
            epoch=self.registered.stream_epoch,
            pts=1_000,
            sequence=1,
            publish_sequence=self.waits,
        )


class _NoopReceiver:
    def pull_now(self, camera_id: str) -> MetadataFrame | None:
        del camera_id
        return None


class _ReAddControl:
    def __init__(self) -> None:
        self.generation: int = 0
        self.removed: list[str] = []

    def add_source(self, camera_id: str, uri: str) -> SourceBinding:
        del uri
        self.generation += 1
        return SourceBinding(str(_BOOT), str(_CHILD), camera_id, self.generation, 1, _TRANSFORM)

    def remove_source(self, camera_id: str) -> None:
        self.removed.append(camera_id)

    def source_failure(self, camera_id: str, category: str) -> SourceBinding:
        del category
        return SourceBinding(str(_BOOT), str(_CHILD), camera_id, self.generation, 2, _TRANSFORM)


def test_add_timeout_rolls_back_native_and_allows_readd() -> None:
    control = _ReAddControl()
    slot = _TimeoutThenReadySlot()
    controller = DarkSourceController(control, slot, _NoopReceiver())

    with pytest.raises(TimeoutError):
        _ = controller.add("camera-a", "loopback://camera-a")

    assert control.removed == ["camera-a"]
    assert slot.removed == ["camera-a"]
    assert controller.snapshot("camera-a").state is SourceState.TOMBSTONED
    assert controller.add("camera-a", "loopback://camera-a").state is SourceState.SOURCE_READY


def test_rebuild_timeout_removes_native_and_allows_readd() -> None:
    class ReadyTimeoutReadySlot(_TimeoutThenReadySlot):
        waits: int
        @override
        def wait_accepted(
            self,
            token: AcceptanceToken,
            *,
            timeout_sec: float,
        ) -> MetadataFrame:
            del token, timeout_sec
            self.waits += 1
            if self.waits == 2:
                raise TimeoutError("metadata binding deadline elapsed")
            assert self.registered is not None
            return _frame(
                epoch=self.registered.stream_epoch,
                pts=1_000,
                sequence=1,
                publish_sequence=self.waits,
            )

    control = _ReAddControl()
    slot = ReadyTimeoutReadySlot()
    controller = DarkSourceController(control, slot, _NoopReceiver())
    assert controller.add("camera-a", "loopback://camera-a").state is SourceState.SOURCE_READY

    with pytest.raises(TimeoutError):
        _ = controller.rebuild("camera-a", "eos")

    assert control.removed == ["camera-a"]
    assert controller.snapshot("camera-a").state is SourceState.DEGRADED
    assert controller.add("camera-a", "loopback://camera-a").state is SourceState.SOURCE_READY


def test_add_and_rebuild_are_serialized_on_current_binding() -> None:
    add_waiting = threading.Event()
    release_add = threading.Event()
    rebuild_called = threading.Event()
    rebuild_thread_started = threading.Event()
    operations_done = threading.Event()
    release_observed = threading.Event()

    class Slot(_TimeoutThenReadySlot):
        waits: int
        @override
        def wait_accepted(
            self,
            token: AcceptanceToken,
            *,
            timeout_sec: float,
        ) -> MetadataFrame:
            del token, timeout_sec
            self.waits += 1
            assert self.registered is not None
            if self.waits == 1:
                add_waiting.set()
                assert release_add.wait(timeout=1.0)
            return _frame(
                epoch=self.registered.stream_epoch,
                pts=1_000,
                sequence=1,
                publish_sequence=self.waits,
            )

    class Control(_ReAddControl):
        @override
        def source_failure(self, camera_id: str, category: str) -> SourceBinding:
            assert release_observed.is_set()
            rebuild_called.set()
            return super().source_failure(camera_id, category)

    control, slot = Control(), Slot()
    controller = DarkSourceController(control, slot, _NoopReceiver())
    add_thread = threading.Thread(
        target=controller.add,
        args=("camera-a", "loopback://camera-a"),
    )

    def rebuild() -> None:
        rebuild_thread_started.set()
        _ = controller.rebuild("camera-a", "eos")
        operations_done.set()

    add_thread.start()
    assert add_waiting.wait(timeout=1.0)
    rebuild_thread = threading.Thread(target=rebuild)
    rebuild_thread.start()
    assert rebuild_thread_started.wait(timeout=1.0)
    assert not rebuild_called.is_set()
    release_observed.set()
    release_add.set()
    assert operations_done.wait(timeout=1.0)
    add_thread.join(timeout=1.0)
    rebuild_thread.join(timeout=1.0)

    assert controller.snapshot("camera-a").state is SourceState.SOURCE_READY
    assert controller.snapshot("camera-a").stream_epoch == 2


def test_runner_maps_source_readiness_timeout_to_typed_exit_four(tmp_path: Path) -> None:
    class Sources:
        def add(self, camera_id: str, uri: str) -> None:
            del camera_id, uri
            raise TimeoutError("exact frame absent")

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
    assert b'"category":"source_ready_timeout"' in request.child.first_fault_path.read_bytes()
