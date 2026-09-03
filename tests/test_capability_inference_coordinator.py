from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import final

import numpy as np
import pytest

from contracts.frame import Frame
from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from contracts.runner import RunnerProtocol, RunnerResult, pose_result
from worker.adapters.model.errors import FatalAcceleratorError
from worker.interfaces.output import EventSink
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.analytics.composite import CompositeResult
from worker.pipeline.bus import Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump, EvidenceAttacher
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.inference_coordinator import (
    CapabilityInferenceCoordinator,
    CoordinatedInference,
    InferenceResultSlot,
)
from worker.pipeline.inference_telemetry import InferenceGeometryTelemetry
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.types import BusinessEvent, DecisionInput, FramePacket, ModuleResult


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass(frozen=True)
class _Metrics:
    published: int
    taken: int
    dropped: int
    queue_age_sec: float


class _LatestSlot:
    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self._value: tuple[FramePacket, float] | None = None
        self.published = self.taken = self.dropped = 0

    def publish(self, packet: FramePacket) -> None:
        self.published += 1
        if self._value is not None:
            self.dropped += 1
            self._value[0].release()
        self._value = (packet, self._clock())

    def take(self, *, timeout_sec: float | None = None) -> FramePacket | None:
        del timeout_sec
        if self._value is None:
            return None
        packet, _at = self._value
        self._value = None
        self.taken += 1
        return packet

    def metrics(self) -> _Metrics:
        age = 0.0 if self._value is None else self._clock() - self._value[1]
        return _Metrics(self.published, self.taken, self.dropped, age)


class _Watchdog:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []
        self.guard_calls = 0

    @contextmanager
    def guard(self, *, camera_id: str, task: str, **_kwargs: object):
        self.guard_calls += 1
        self.entries.append((camera_id, task))
        try:
            yield self.guard_calls
        finally:
            self.entries.pop()

    def in_flight(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.entries)


@final
class _BatchClient:
    def __init__(
        self,
        clock: _Clock,
        watchdog: _Watchdog,
        *,
        raise_for: Callable[[Sequence[FramePacket]], BaseException | None] | None = None,
        result_count_for: Callable[[Sequence[FramePacket]], int | None] | None = None,
    ) -> None:
        self.clock, self.watchdog = clock, watchdog
        self._raise_for = raise_for
        self._result_count_for = result_count_for
        self.batches: list[tuple[tuple[str, int], ...]] = []
        self.batch_geometries: list[tuple[tuple[int, int], ...]] = []
        self.options: list[dict[str, object]] = []

    def create(self, task: str, **kwargs: object) -> RunnerProtocol:
        del task, kwargs
        raise NotImplementedError

    def infer_batch(
        self, task: str, frames: Sequence[FramePacket], **kwargs: object
    ) -> tuple[RunnerResult, ...]:
        self.options.append(dict(kwargs))
        assert task == "pose"
        assert len(self.watchdog.in_flight()) == 1
        self.batches.append(tuple((frame.camera_id, frame.seq) for frame in frames))
        self.batch_geometries.append(tuple((frame.height, frame.width) for frame in frames))
        if self._raise_for is not None:
            error = self._raise_for(frames)
            if error is not None:
                raise error
        self.clock.now += 0.020
        if self._result_count_for is not None:
            count = self._result_count_for(frames)
            if count is not None:
                return tuple(
                    pose_result((), ((float(index), 0.0, 1.0, 1.0, 0.9),)) for index in range(count)
                )
        return tuple(pose_result((), ((float(frame.seq), 0.0, 1.0, 1.0, 0.9),)) for frame in frames)


def _packet(camera_id: str, seq: int, *, height: int = 2, width: int = 2) -> FramePacket:
    return FramePacket(
        camera_id,
        Frame(
            index=seq,
            time_sec=float(seq),
            image=np.zeros((height, width, 3), dtype=np.uint8),
        ),
        float(seq),
        seq,
        width,
        height,
        0.0,
    )


def _coordinator(
    camera_ids: Sequence[str],
    *,
    raise_for: Callable[[Sequence[FramePacket]], BaseException | None] | None = None,
    result_count_for: Callable[[Sequence[FramePacket]], int | None] | None = None,
):
    clock, watchdog = _Clock(), _Watchdog()
    client = _BatchClient(clock, watchdog, raise_for=raise_for, result_count_for=result_count_for)
    coordinator = CapabilityInferenceCoordinator(client, watchdog, clock=clock)
    lanes: dict[str, tuple[_LatestSlot, InferenceResultSlot]] = {}
    for camera_id in camera_ids:
        source, results = _LatestSlot(clock), InferenceResultSlot()
        coordinator.register(camera_id, source, results)
        lanes[camera_id] = source, results
    return coordinator, client, watchdog, clock, lanes


def _release_results(lanes: dict[str, tuple[_LatestSlot, InferenceResultSlot]]) -> None:
    for _source, results in lanes.values():
        value = results.take(timeout_sec=0)
        if value is not None:
            value.packet.release()


def test_all_thirteen_ready_cameras_are_admitted_every_cycle_without_starvation() -> None:
    camera_ids = tuple(f"camera-{index}" for index in range(13))
    coordinator, client, _watchdog, _clock, lanes = _coordinator(camera_ids)
    try:
        for cycle in range(2):
            for camera_id, (source, _results) in lanes.items():
                source.publish(_packet(camera_id, cycle))
            assert coordinator.run_cycle() == 13
            _release_results(lanes)
        assert [len(batch) for batch in client.batches] == [13, 13]
        assert all(value.admitted == 2 for value in coordinator.snapshot().cameras.values())
    finally:
        coordinator.stop()


def test_forward_threads_the_provisioned_pose_device_to_batch_serving() -> None:
    clock, watchdog = _Clock(), _Watchdog()
    client = _BatchClient(clock, watchdog)
    coordinator = CapabilityInferenceCoordinator(
        client,
        watchdog,
        clock=clock,
        pose_device="cuda",
    )
    source, results = _LatestSlot(clock), InferenceResultSlot()
    coordinator.register("camera-a", source, results)
    try:
        source.publish(_packet("camera-a", 1))
        assert coordinator.run_cycle() == 1
        assert client.options == [{"device": "cuda"}]
        delivered = results.take(timeout_sec=0)
        assert delivered is not None
        delivered.packet.release()
    finally:
        coordinator.stop()


def test_partial_drain_preserves_camera_to_result_mapping() -> None:
    coordinator, _client, _watchdog, _clock, lanes = _coordinator(
        ("camera-a", "camera-b", "camera-c")
    )
    try:
        lanes["camera-a"][0].publish(_packet("camera-a", 11))
        lanes["camera-c"][0].publish(_packet("camera-c", 33))
        assert coordinator.run_cycle() == 2
        for camera_id, expected in (("camera-a", 11), ("camera-c", 33)):
            delivered = lanes[camera_id][1].take(timeout_sec=0)
            assert delivered is not None
            assert delivered.packet.camera_id == camera_id
            assert delivered.pose.result.boxes[0][0] == expected  # type: ignore[union-attr]
            delivered.packet.release()
        assert lanes["camera-b"][1].take(timeout_sec=0) is None
    finally:
        coordinator.stop()


@pytest.mark.parametrize(
    ("width", "height"),
    ((640, 360), (640, 480)),
    ids=("640x360", "640x480"),
)
def test_homogeneous_geometry_cycle_keeps_one_model_batch(width: int, height: int) -> None:
    coordinator, client, _watchdog, _clock, lanes = _coordinator(
        ("camera-a", "camera-b", "camera-c")
    )
    try:
        held = {camera_id: _packet(camera_id, 7, height=height, width=width) for camera_id in lanes}
        for camera_id, (source, _results) in lanes.items():
            source.publish(held[camera_id])
        assert coordinator.run_cycle() == 3
        assert client.batches == [
            (("camera-a", 7), ("camera-b", 7), ("camera-c", 7)),
        ]
        assert client.batch_geometries == [((height, width), (height, width), (height, width))]
        for camera_id in ("camera-a", "camera-b", "camera-c"):
            delivered = lanes[camera_id][1].take(timeout_sec=0)
            assert delivered is not None
            assert delivered.packet is held[camera_id]
            assert delivered.packet.camera_id == camera_id
            assert delivered.packet.height == height
            assert delivered.packet.width == width
            assert delivered.pose.result.boxes[0][0] == 7  # type: ignore[union-attr]
            assert delivered.packet.released is False
            delivered.packet.release()
    finally:
        coordinator.stop()


def test_mixed_geometry_cycle_dispatches_one_batch_per_first_seen_key() -> None:
    coordinator, client, _watchdog, _clock, lanes = _coordinator(
        ("camera-a", "camera-b", "camera-c", "camera-d")
    )
    try:
        lanes["camera-a"][0].publish(_packet("camera-a", 11, height=360, width=640))
        lanes["camera-b"][0].publish(_packet("camera-b", 22, height=480, width=640))
        lanes["camera-c"][0].publish(_packet("camera-c", 33, height=360, width=640))
        lanes["camera-d"][0].publish(_packet("camera-d", 44, height=1080, width=1920))
        assert coordinator.run_cycle() == 4
        assert client.batches == [
            (("camera-a", 11), ("camera-c", 33)),
            (("camera-b", 22),),
            (("camera-d", 44),),
        ]
        assert client.batch_geometries == [
            ((360, 640), (360, 640)),
            ((480, 640),),
            ((1080, 1920),),
        ]
        for camera_id, expected in (
            ("camera-a", 11),
            ("camera-b", 22),
            ("camera-c", 33),
            ("camera-d", 44),
        ):
            delivered = lanes[camera_id][1].take(timeout_sec=0)
            assert delivered is not None
            assert delivered.packet.camera_id == camera_id
            assert delivered.pose.result.boxes[0][0] == expected  # type: ignore[union-attr]
            delivered.packet.release()
    finally:
        coordinator.stop()


def test_geometry_observation_failure_releases_every_selected_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, client, _watchdog, _clock, lanes = _coordinator(("camera-a", "camera-b"))
    held = {
        "camera-a": _packet("camera-a", 11, height=360, width=640),
        "camera-b": _packet("camera-b", 22, height=480, width=640),
    }
    observations = 0

    def fail_observation(
        _telemetry: InferenceGeometryTelemetry,
        camera_id: str,
        geometry: tuple[int, int],
    ) -> None:
        nonlocal observations
        del camera_id, geometry
        observations += 1
        if observations == 2:
            raise OSError("telemetry handler failed")

    monkeypatch.setattr(InferenceGeometryTelemetry, "observe_geometry", fail_observation)
    try:
        for camera_id, packet in held.items():
            lanes[camera_id][0].publish(packet)

        with pytest.raises(OSError, match="telemetry handler failed"):
            coordinator.run_cycle()

        assert observations == 2
        assert client.batches == []
        assert all(packet.released for packet in held.values())
        assert all(results.take(timeout_sec=0) is None for _source, results in lanes.values())
    finally:
        coordinator.stop()


def test_base_exception_during_forward_releases_every_selected_packet() -> None:
    coordinator, client, _watchdog, _clock, lanes = _coordinator(("camera-a", "camera-b"))
    held = {
        "camera-a": _packet("camera-a", 11, height=360, width=640),
        "camera-b": _packet("camera-b", 22, height=360, width=640),
    }

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    client.infer_batch = interrupt  # type: ignore[method-assign]
    try:
        for camera_id, packet in held.items():
            lanes[camera_id][0].publish(packet)

        with pytest.raises(KeyboardInterrupt):
            coordinator.run_cycle()

        assert all(packet.released for packet in held.values())
        assert all(results.take(timeout_sec=0) is None for _source, results in lanes.values())
    finally:
        coordinator.stop()


def test_nonfatal_geometry_bucket_failure_releases_only_that_bucket(
    caplog: pytest.LogCaptureFixture,
) -> None:
    held: dict[str, FramePacket] = {}

    def raise_for(frames: Sequence[FramePacket]) -> BaseException | None:
        if frames[0].height == 480:
            return RuntimeError("middle geometry failed")
        return None

    coordinator, client, watchdog, _clock, lanes = _coordinator(
        ("camera-a", "camera-b", "camera-c"),
        raise_for=raise_for,
    )
    try:
        held["camera-a"] = _packet("camera-a", 11, height=360, width=640)
        held["camera-b"] = _packet("camera-b", 22, height=480, width=640)
        held["camera-c"] = _packet("camera-c", 33, height=1080, width=1920)
        lanes["camera-a"][0].publish(held["camera-a"])
        lanes["camera-b"][0].publish(held["camera-b"])
        lanes["camera-c"][0].publish(held["camera-c"])
        with caplog.at_level(logging.ERROR):
            assert coordinator.run_cycle() == 2
        assert client.batches == [
            (("camera-a", 11),),
            (("camera-b", 22),),
            (("camera-c", 33),),
        ]
        assert watchdog.guard_calls == 3
        first = lanes["camera-a"][1].take(timeout_sec=0)
        third = lanes["camera-c"][1].take(timeout_sec=0)
        assert first is not None
        assert third is not None
        assert first.packet.camera_id == "camera-a"
        assert third.packet.camera_id == "camera-c"
        assert first.pose.result.boxes[0][0] == 11  # type: ignore[union-attr]
        assert third.pose.result.boxes[0][0] == 33  # type: ignore[union-attr]
        assert lanes["camera-b"][1].take(timeout_sec=0) is None
        assert held["camera-a"].released is False
        assert held["camera-b"].released is True
        assert held["camera-c"].released is False
        snapshot = coordinator.snapshot()
        assert snapshot.cameras["camera-a"].inferred == 1
        assert snapshot.cameras["camera-b"].inferred == 0
        assert snapshot.cameras["camera-c"].inferred == 1
        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "RuntimeError" in message and "480x640" in message and "camera-b" in message
            for message in messages
        )
        first.packet.release()
        third.packet.release()
    finally:
        coordinator.stop()


def test_bucket_results_return_to_original_camera_lanes() -> None:
    def result_count_for(frames: Sequence[FramePacket]) -> int | None:
        if frames[0].height == 480:
            return 0
        return None

    coordinator, client, _watchdog, _clock, lanes = _coordinator(
        ("camera-a", "camera-b", "camera-c", "camera-d"),
        result_count_for=result_count_for,
    )
    try:
        lanes["camera-a"][0].publish(_packet("camera-a", 11, height=360, width=640))
        held_b = _packet("camera-b", 22, height=480, width=640)
        lanes["camera-b"][0].publish(held_b)
        lanes["camera-c"][0].publish(_packet("camera-c", 33, height=360, width=640))
        lanes["camera-d"][0].publish(_packet("camera-d", 44, height=1080, width=1920))
        assert coordinator.run_cycle() == 3
        assert client.batches == [
            (("camera-a", 11), ("camera-c", 33)),
            (("camera-b", 22),),
            (("camera-d", 44),),
        ]
        for camera_id, expected in (("camera-a", 11), ("camera-c", 33), ("camera-d", 44)):
            delivered = lanes[camera_id][1].take(timeout_sec=0)
            assert delivered is not None
            assert delivered.packet.camera_id == camera_id
            assert delivered.pose.result.boxes[0][0] == expected  # type: ignore[union-attr]
            delivered.packet.release()
        assert lanes["camera-b"][1].take(timeout_sec=0) is None
        assert held_b.released is True
        snapshot = coordinator.snapshot()
        assert snapshot.cameras["camera-a"].inferred == 1
        assert snapshot.cameras["camera-b"].inferred == 0
        assert snapshot.cameras["camera-c"].inferred == 1
        assert snapshot.cameras["camera-d"].inferred == 1
    finally:
        coordinator.stop()


def test_fatal_accelerator_error_still_escapes_the_cycle() -> None:
    held: dict[str, FramePacket] = {}

    def raise_for(frames: Sequence[FramePacket]) -> BaseException | None:
        if frames[0].height == 480:
            return FatalAcceleratorError("cuda context lost", camera_id="camera-b", task="pose")
        return None

    coordinator, client, _watchdog, _clock, lanes = _coordinator(
        ("camera-a", "camera-b", "camera-c"),
        raise_for=raise_for,
    )
    try:
        held["camera-a"] = _packet("camera-a", 11, height=360, width=640)
        held["camera-b"] = _packet("camera-b", 22, height=480, width=640)
        held["camera-c"] = _packet("camera-c", 33, height=1080, width=1920)
        lanes["camera-a"][0].publish(held["camera-a"])
        lanes["camera-b"][0].publish(held["camera-b"])
        lanes["camera-c"][0].publish(held["camera-c"])
        with pytest.raises(FatalAcceleratorError, match="cuda context lost"):
            coordinator.run_cycle()
        assert client.batches == [
            (("camera-a", 11),),
            (("camera-b", 22),),
        ]
        first = lanes["camera-a"][1].take(timeout_sec=0)
        assert first is not None
        assert first.packet.camera_id == "camera-a"
        assert first.pose.result.boxes[0][0] == 11  # type: ignore[union-attr]
        assert lanes["camera-b"][1].take(timeout_sec=0) is None
        assert lanes["camera-c"][1].take(timeout_sec=0) is None
        assert held["camera-a"].released is False
        assert held["camera-b"].released is True
        assert held["camera-c"].released is True
        snapshot = coordinator.snapshot()
        assert snapshot.cameras["camera-a"].inferred == 1
        assert snapshot.cameras["camera-b"].inferred == 0
        assert snapshot.cameras["camera-c"].inferred == 0
        first.packet.release()
    finally:
        coordinator.stop()


def _stale_occupant(slot: InferenceResultSlot, camera_id: str) -> FramePacket:
    occupant = _packet(camera_id, 0)
    slot.publish(
        CoordinatedInference(
            occupant,
            ModuleResult("pose", pose_result((), ((0.0, 0.0, 1.0, 1.0, 0.9),)), 0.0, "pose"),
        )
    )
    occupant.release()
    return occupant


def test_nonfatal_result_slot_publish_failure_releases_unpublished_rows() -> None:
    coordinator, client, _watchdog, _clock, lanes = _coordinator(
        ("camera-a", "camera-b", "camera-c")
    )
    try:
        held = {
            "camera-a": _packet("camera-a", 11, height=360, width=640),
            "camera-b": _packet("camera-b", 22, height=480, width=640),
            "camera-c": _packet("camera-c", 33, height=1080, width=1920),
        }
        _stale_occupant(lanes["camera-b"][1], "camera-b")
        for camera_id, packet in held.items():
            lanes[camera_id][0].publish(packet)
        assert coordinator.run_cycle() == 2
        assert client.batches == [
            (("camera-a", 11),),
            (("camera-b", 22),),
            (("camera-c", 33),),
        ]
        first = lanes["camera-a"][1].take(timeout_sec=0)
        third = lanes["camera-c"][1].take(timeout_sec=0)
        assert first is not None and first.packet is held["camera-a"]
        assert third is not None and third.packet is held["camera-c"]
        assert lanes["camera-b"][1].take(timeout_sec=0) is None
        assert held["camera-a"].released is False
        assert held["camera-b"].released is True
        assert held["camera-c"].released is False
        snapshot = coordinator.snapshot()
        assert snapshot.cameras["camera-a"].inferred == 1
        assert snapshot.cameras["camera-b"].inferred == 0
        assert snapshot.cameras["camera-c"].inferred == 1
        first.packet.release()
        third.packet.release()
    finally:
        coordinator.stop()


def test_same_geometry_publish_failure_releases_unvisited_siblings() -> None:
    coordinator, client, _watchdog, _clock, lanes = _coordinator(
        ("camera-a", "camera-c", "camera-e", "camera-f")
    )
    try:
        held = {
            "camera-a": _packet("camera-a", 11, height=360, width=640),
            "camera-c": _packet("camera-c", 33, height=360, width=640),
            "camera-e": _packet("camera-e", 55, height=360, width=640),
            "camera-f": _packet("camera-f", 66, height=1080, width=1920),
        }
        _stale_occupant(lanes["camera-c"][1], "camera-c")
        for camera_id, packet in held.items():
            lanes[camera_id][0].publish(packet)
        assert coordinator.run_cycle() == 2
        assert client.batches == [
            (("camera-a", 11), ("camera-c", 33), ("camera-e", 55)),
            (("camera-f", 66),),
        ]
        first = lanes["camera-a"][1].take(timeout_sec=0)
        later = lanes["camera-f"][1].take(timeout_sec=0)
        assert first is not None and first.packet is held["camera-a"]
        assert later is not None and later.packet is held["camera-f"]
        assert lanes["camera-c"][1].take(timeout_sec=0) is None
        assert lanes["camera-e"][1].take(timeout_sec=0) is None
        assert held["camera-a"].released is False
        assert held["camera-c"].released is True
        assert held["camera-e"].released is True
        assert held["camera-f"].released is False
        snapshot = coordinator.snapshot()
        assert snapshot.cameras["camera-a"].inferred == 1
        assert snapshot.cameras["camera-c"].inferred == 0
        assert snapshot.cameras["camera-e"].inferred == 0
        assert snapshot.cameras["camera-f"].inferred == 1
        first.packet.release()
        later.packet.release()
    finally:
        coordinator.stop()


def test_watchdog_has_exactly_one_entry_during_thirteen_camera_forward() -> None:
    coordinator, _client, watchdog, _clock, lanes = _coordinator(
        tuple(f"camera-{index}" for index in range(13))
    )
    try:
        for camera_id, (source, _results) in lanes.items():
            source.publish(_packet(camera_id, 1))
        coordinator.run_cycle()
        assert watchdog.guard_calls == 1
        assert watchdog.in_flight() == ()
        _release_results(lanes)
    finally:
        coordinator.stop()


class _Analytics:
    def __init__(self, camera_id: str, completed: threading.Event) -> None:
        self.state: list[int] = []
        self.camera_id, self.completed = camera_id, completed
        self.owner_thread: int | None = None

    def process(self, packet: FramePacket, *, prefetched_results: object) -> CompositeResult:
        del prefetched_results
        current = threading.get_ident()
        self.owner_thread = current if self.owner_thread is None else self.owner_thread
        assert current == self.owner_thread
        assert threading.current_thread().name == f"pump-{self.camera_id}"
        self.state.append(packet.seq)
        self.completed.set()
        observation = FrameObservation()
        return CompositeResult((), observation, object())  # type: ignore[arg-type]


class _Decision:
    def update(self, _value: object) -> tuple[()]:
        return ()


class _Sink:
    def emit(self, _event: object) -> None:
        raise AssertionError("no events expected")


def test_per_camera_state_is_distinct_and_mutated_only_on_its_pump_thread() -> None:
    coordinator, _client, _watchdog, _clock, lanes = _coordinator(("camera-a", "camera-b"))
    completed = {camera_id: threading.Event() for camera_id in lanes}
    analytics = {camera_id: _Analytics(camera_id, completed[camera_id]) for camera_id in lanes}
    pumps = {
        camera_id: CameraPipelinePump(
            camera_id, results, analytics[camera_id], _Decision(), _Sink(), poll_timeout_sec=0.01
        )
        for camera_id, (_source, results) in lanes.items()
    }
    threads = [
        threading.Thread(target=pump.run, name=f"pump-{camera_id}")
        for camera_id, pump in pumps.items()
    ]
    for thread in threads:
        thread.start()
    try:
        for camera_id, (source, _results) in lanes.items():
            source.publish(_packet(camera_id, 7))
        assert coordinator.run_cycle() == 2
        assert all(event.wait(1.0) for event in completed.values())
        assert analytics["camera-a"].state is not analytics["camera-b"].state
        assert analytics["camera-a"].owner_thread != analytics["camera-b"].owner_thread
    finally:
        for pump in pumps.values():
            pump.stop()
        coordinator.stop()
        for thread in threads:
            thread.join(1.0)
            assert not thread.is_alive()


def test_missing_camera_shrinks_batches_without_affecting_siblings_and_is_telemetried() -> None:
    coordinator, _client, _watchdog, clock, lanes = _coordinator(
        ("camera-a", "camera-gap", "camera-c")
    )
    try:
        lanes["camera-a"][0].publish(_packet("camera-a", 0))
        lanes["camera-a"][0].publish(_packet("camera-a", 1))
        lanes["camera-c"][0].publish(_packet("camera-c", 1))
        clock.now = 0.125
        assert coordinator.run_cycle() == 2
        snapshot = coordinator.snapshot()
        assert snapshot.batch_sizes == {2: 1}
        assert snapshot.cameras["camera-gap"].admitted == 0
        assert snapshot.cameras["camera-a"].inferred == 1
        assert snapshot.cameras["camera-a"].overwritten == 1
        assert snapshot.cameras["camera-c"].inferred == 1
        assert snapshot.cameras["camera-a"].queue_age_sec == 0.125
        assert snapshot.forward_p50_sec == pytest.approx(0.020)
        assert snapshot.forward_p95_sec == pytest.approx(0.020)
        _release_results(lanes)
    finally:
        coordinator.stop()


def test_composite_observes_only_pose_results_and_coasts_a_missing_result() -> None:
    tracker = GreedyIouTracker(max_misses=0)
    scene = SceneState("camera-a")
    analytics = CompositeExtractor(
        extractors=(),
        scheduler=Scheduler({"pose": 1}),
        tracker=tracker,
        scene_state=scene,
    )
    observed = analytics.process(
        _packet("camera-a", 1),
        prefetched_results=(
            ModuleResult(
                "pose",
                pose_result((), ((0.0, 0.0, 1.0, 1.0, 0.9),)),
                1.0,
                "pose",
            ),
        ),
    )
    latest = scene.latest_observation
    skipped = analytics.process(_packet("camera-a", 2))

    assert observed.observation.track_ids == (0,)
    assert skipped.observation is latest
    assert scene.latest_observation is latest
    assert scene.track_ids == (0,)
    assert tracker.live_ids == frozenset({0})


def test_runtime_snapshot_and_rendered_log_include_local_inference_telemetry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator, _client, _watchdog, _clock, lanes = _coordinator(("camera-a",))
    diagnostics = WorkerDiagnostics()
    diagnostics.register_inference(coordinator)
    try:
        lanes["camera-a"][0].publish(_packet("camera-a", 1))
        coordinator.run_cycle()
        snapshot = diagnostics.snapshot().cameras[0]
        assert snapshot.inference is not None
        assert snapshot.inference.admitted == snapshot.inference.inferred == 1
        assert snapshot.batch_sizes == ((1, 1),)
        with caplog.at_level(logging.INFO):
            diagnostics.log_snapshot()
        message = caplog.records[-1].getMessage()
        assert "inference={'admitted': 1, 'overwritten': 0, 'inferred': 1" in message
        assert "'batch_sizes': {1: 1}" in message
        assert "'forward_p50_sec': 0.02" in message
        assert "inference" not in repr(diagnostics.to_payload("facility-a", None, 1))
        _release_results(lanes)
    finally:
        coordinator.stop()


def test_zero_ready_cycle_sleeps_once_instead_of_busy_spinning() -> None:
    sleeps: list[float] = []
    coordinator, _client, _watchdog, _clock, _lanes = _coordinator(("camera-a",))

    def idle_wait(delay: float) -> None:
        sleeps.append(delay)
        coordinator.stop()

    coordinator._idle_wait = idle_wait  # noqa: SLF001 - deterministic idle-loop proof
    coordinator.run()

    assert sleeps == [0.005]


def _blank_decision_input() -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(),
        frame_width=2,
        frame_height=2,
        live_track_ids=(),
        time_sec=1.0,
        frame_index=1,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.EMPTY),
    )


class _ZeroEventAnalytics:
    def process(self, packet: FramePacket, *, prefetched_results: object) -> CompositeResult:
        del packet, prefetched_results
        return CompositeResult((), FrameObservation(), _blank_decision_input())


def _blank_analytics(camera_id: str) -> CompositeExtractor:
    return CompositeExtractor(
        extractors=(),
        scheduler=Scheduler({}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState(camera_id),
    )


@final
class _CountingZeroEventDecider:
    def __init__(self) -> None:
        self.calls: int = 0

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        del input_value
        self.calls += 1
        return ()


@final
class _RaisingAnalytics(CompositeExtractor):
    def process(
        self, packet: FramePacket, *, prefetched_results: Sequence[ModuleResult] = ()
    ) -> CompositeResult:
        del packet, prefetched_results
        raise RuntimeError("analytics exploded")


def _raising_analytics(camera_id: str) -> CompositeExtractor:
    return _RaisingAnalytics(
        extractors=(),
        scheduler=Scheduler({}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState(camera_id),
    )


@final
class _RaisingDecider:
    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        del input_value
        raise RuntimeError("decision exploded")


@final
class _OneEventDecider:
    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        del input_value
        return (
            BusinessEvent(
                domain="probe",
                event_type="tick",
                identity="frame-1",
                camera_id="camera-a",
                facility_id="facility-a",
                time_sec=1.0,
                probability=1.0,
            ),
        )


@final
class _RaisingAttacher:
    def attach(
        self,
        event: BusinessEvent,
        packet: FramePacket,
        observation: FrameObservation,
    ) -> BusinessEvent:
        del event, packet, observation
        raise RuntimeError("evidence attach exploded")


@final
class _RaisingSink:
    def emit(self, event: BusinessEvent) -> None:
        del event
        raise RuntimeError("sink exploded")


@final
class _NoEventSink:
    def emit(self, event: BusinessEvent) -> None:
        del event
        raise AssertionError("no events expected")


def _zero_event_aggregator(decider: _CountingZeroEventDecider | None = None) -> EventAggregator:
    return EventAggregator(
        deciders=() if decider is None else (decider,),
        incidents=IncidentManager(),
    )


def _one_event_aggregator() -> EventAggregator:
    return EventAggregator(deciders=(_OneEventDecider(),), incidents=IncidentManager())


def _raising_decision() -> EventAggregator:
    return EventAggregator(deciders=(_RaisingDecider(),), incidents=IncidentManager())


def _pump_one_frame(
    *,
    camera_id: str = "camera-a",
    analytics: CompositeExtractor | None = None,
    decision: EventAggregator | None = None,
    sink: EventSink | None = None,
    diagnostics: WorkerDiagnostics | None = None,
    evidence_attacher: EvidenceAttacher | None = None,
) -> CameraPipelinePump:
    pump = CameraPipelinePump(
        camera_id,
        InferenceResultSlot(),
        _blank_analytics(camera_id) if analytics is None else analytics,
        _zero_event_aggregator() if decision is None else decision,
        _NoEventSink() if sink is None else sink,
        diagnostics=diagnostics,
        evidence_attacher=evidence_attacher,
    )
    pump._pump_one(
        _packet(camera_id, 1),
        ModuleResult("pose", pose_result((), ()), 0.0, "pose"),
    )
    return pump


def test_zero_event_decision_path_emits_nothing() -> None:
    """Baseline: a completed no-event decision still emits nothing.

    Recorded green before production edits. After the additive counter lands,
    the unchanged contract is still: decision.update() ran once and the sink
    was never invoked.
    """
    decider = _CountingZeroEventDecider()
    _pump_one_frame(decision=_zero_event_aggregator(decider))
    assert decider.calls == 1


def test_zero_event_decision_records_one_detection_completion() -> None:
    diagnostics = WorkerDiagnostics()
    _pump_one_frame(diagnostics=diagnostics)
    snapshot = diagnostics.snapshot().cameras
    assert len(snapshot) == 1
    assert snapshot[0].camera_id == "camera-a"
    assert snapshot[0].decision_completed == 1


def test_analytics_or_decision_error_records_zero_detection_completions() -> None:
    analytics_diagnostics = WorkerDiagnostics()
    with pytest.raises(RuntimeError, match="analytics exploded"):
        _pump_one_frame(
            analytics=_raising_analytics("camera-a"),
            diagnostics=analytics_diagnostics,
        )
    assert analytics_diagnostics.snapshot().cameras == ()

    decision_diagnostics = WorkerDiagnostics()
    with pytest.raises(RuntimeError, match="decision exploded"):
        _pump_one_frame(
            decision=_raising_decision(),
            diagnostics=decision_diagnostics,
        )
    assert decision_diagnostics.snapshot().cameras == ()


def test_post_decision_evidence_or_sink_error_keeps_detection_completion() -> None:
    attach_diagnostics = WorkerDiagnostics()
    with pytest.raises(RuntimeError, match="evidence attach exploded"):
        _pump_one_frame(
            decision=_one_event_aggregator(),
            diagnostics=attach_diagnostics,
            evidence_attacher=_RaisingAttacher(),
            sink=_NoEventSink(),
        )
    assert attach_diagnostics.snapshot().cameras[0].decision_completed == 1

    sink_diagnostics = WorkerDiagnostics()
    with pytest.raises(RuntimeError, match="sink exploded"):
        _pump_one_frame(
            decision=_one_event_aggregator(),
            diagnostics=sink_diagnostics,
            sink=_RaisingSink(),
        )
    assert sink_diagnostics.snapshot().cameras[0].decision_completed == 1


def test_round_robin_admits_every_continuously_ready_lane_within_bounded_cycles() -> None:
    camera_ids = tuple(f"camera-{index}" for index in range(50))
    coordinator, client, _watchdog, _clock, lanes = _coordinator(camera_ids)
    admitted: set[str] = set()
    first_cycle: dict[str, int] = {}
    try:
        for cycle in range(4):
            for camera_id, (source, _results) in lanes.items():
                source.publish(_packet(camera_id, cycle))
            before = len(client.batches)
            selected = coordinator.run_cycle()
            assert 1 <= selected <= 16
            cycle_ids = [
                camera_id for batch in client.batches[before:] for camera_id, _seq in batch
            ]
            assert len(cycle_ids) == selected
            for camera_id in cycle_ids:
                first_cycle.setdefault(camera_id, cycle)
                admitted.add(camera_id)
            _release_results(lanes)
        assert admitted == set(camera_ids)
        assert max(first_cycle.values()) <= 3
        assert all(count <= 16 for count in coordinator.snapshot().batch_sizes)
    finally:
        coordinator.stop()


def test_round_robin_admits_every_lane_when_fewer_than_max_batch() -> None:
    camera_ids = tuple(f"camera-{index}" for index in range(5))
    coordinator, client, _watchdog, _clock, lanes = _coordinator(camera_ids)
    try:
        for cycle in range(2):
            for camera_id, (source, _results) in lanes.items():
                source.publish(_packet(camera_id, cycle))
            assert coordinator.run_cycle() == 5
            cycle_ids = [camera_id for camera_id, _seq in client.batches[cycle]]
            assert cycle_ids == list(camera_ids)
            _release_results(lanes)
    finally:
        coordinator.stop()


def test_round_robin_skips_empty_lanes_interspersed_with_ready_lanes() -> None:
    camera_ids = tuple(f"camera-{index}" for index in range(20))
    coordinator, client, _watchdog, _clock, lanes = _coordinator(camera_ids)
    ready = tuple(camera_id for index, camera_id in enumerate(camera_ids) if index % 2 == 0)
    try:
        for camera_id in ready:
            lanes[camera_id][0].publish(_packet(camera_id, 1))
        assert coordinator.run_cycle() == 10
        assert [camera_id for camera_id, _seq in client.batches[0]] == list(ready)
        for camera_id in camera_ids:
            delivered = lanes[camera_id][1].take(timeout_sec=0)
            if camera_id in ready:
                assert delivered is not None
                delivered.packet.release()
            else:
                assert delivered is None
    finally:
        coordinator.stop()


def test_zero_ready_cycle_preserves_fairness_cursor_and_waits_once() -> None:
    camera_ids = tuple(f"camera-{index}" for index in range(20))
    sleeps: list[float] = []
    coordinator, client, _watchdog, _clock, lanes = _coordinator(camera_ids)
    try:
        for camera_id, (source, _results) in lanes.items():
            source.publish(_packet(camera_id, 0))
        assert coordinator.run_cycle() == 16
        first = [camera_id for camera_id, _seq in client.batches[0]]
        assert first == [f"camera-{index}" for index in range(16)]
        _release_results(lanes)
        for source, _results in lanes.values():
            leftover = source.take(timeout_sec=0)
            if leftover is not None:
                leftover.release()

        def idle_wait(delay: float) -> None:
            sleeps.append(delay)
            coordinator._stop_event.set()  # noqa: SLF001 - keep slots open for resume

        coordinator._idle_wait = idle_wait  # noqa: SLF001 - deterministic idle-loop proof
        coordinator.run()
        assert sleeps == [0.005]
        assert len(client.batches) == 1

        for camera_id, (source, _results) in lanes.items():
            source.publish(_packet(camera_id, 1))
        assert coordinator.run_cycle() == 16
        resumed = [camera_id for camera_id, _seq in client.batches[1]]
        assert resumed[:4] == ["camera-16", "camera-17", "camera-18", "camera-19"]
        assert "camera-12" not in resumed
        _release_results(lanes)
    finally:
        coordinator.stop()


def test_mixed_geometry_keys_do_not_multiply_the_global_batch_cap() -> None:
    camera_ids = tuple(f"camera-{index}" for index in range(32))
    coordinator, client, _watchdog, _clock, lanes = _coordinator(camera_ids)
    try:
        for index, camera_id in enumerate(camera_ids):
            height = 360 if index % 2 == 0 else 480
            lanes[camera_id][0].publish(_packet(camera_id, 1, height=height, width=640))
        assert coordinator.run_cycle() == 16
        selected = [camera_id for batch in client.batches for camera_id, _seq in batch]
        assert len(selected) == 16
        assert all(len(set(geometry)) == 1 for geometry in client.batch_geometries)
        assert sum(len(batch) for batch in client.batches) == 16
        _release_results(lanes)
    finally:
        coordinator.stop()


def _publish_geometries(
    lanes: dict[str, tuple[_LatestSlot, InferenceResultSlot]],
    seq: int,
    geometries: dict[str, tuple[int, int]],
) -> None:
    for camera_id, (width, height) in geometries.items():
        lanes[camera_id][0].publish(_packet(camera_id, seq, height=height, width=width))


def _geometry_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "x" in record.getMessage()
        and any(camera in record.getMessage() for camera in ("camera-a", "camera-b", "camera-c"))
    ]


def test_camera_telemetry_observed_geometry_is_none_before_admission() -> None:
    # Given
    coordinator, _client, _watchdog, _clock, lanes = _coordinator(("camera-a", "camera-b"))
    try:
        # When
        snapshot = coordinator.snapshot()
        # Then
        assert snapshot.cameras["camera-a"].observed_geometry is None
        assert snapshot.cameras["camera-b"].observed_geometry is None
        assert snapshot.geometry_batch_sizes == {}
    finally:
        coordinator.stop()
        del lanes


def test_camera_telemetry_exposes_last_observed_width_height() -> None:
    # Given
    coordinator, _client, _watchdog, _clock, lanes = _coordinator(("camera-a", "camera-b"))
    try:
        _publish_geometries(lanes, 1, {"camera-a": (640, 360)})
        # When
        assert coordinator.run_cycle() == 1
        snapshot = coordinator.snapshot()
        # Then
        assert snapshot.cameras["camera-a"].observed_geometry == (640, 360)
        assert snapshot.cameras["camera-b"].observed_geometry is None
        _release_results(lanes)
    finally:
        coordinator.stop()


def test_inference_snapshot_partitions_physical_batch_histograms_by_geometry() -> None:
    # Given
    coordinator, client, _watchdog, _clock, lanes = _coordinator(
        ("camera-a", "camera-b", "camera-c")
    )
    try:
        _publish_geometries(
            lanes,
            1,
            {"camera-a": (640, 360), "camera-b": (640, 480), "camera-c": (640, 360)},
        )
        # When
        assert coordinator.run_cycle() == 3
        snapshot = coordinator.snapshot()
        # Then
        assert client.batches == [
            (("camera-a", 1), ("camera-c", 1)),
            (("camera-b", 1),),
        ]
        assert snapshot.batch_sizes == {2: 1, 1: 1}
        assert snapshot.geometry_batch_sizes == {
            (640, 360): {2: 1},
            (640, 480): {1: 1},
        }
        _release_results(lanes)
    finally:
        coordinator.stop()


def test_stable_distinct_geometries_emit_no_geometry_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given
    coordinator, _client, _watchdog, _clock, lanes = _coordinator(
        ("camera-a", "camera-b", "camera-c")
    )
    geometries = {"camera-a": (640, 360), "camera-b": (640, 480), "camera-c": (1920, 1080)}
    try:
        # When
        with caplog.at_level(logging.WARNING):
            for seq in (1, 2):
                _publish_geometries(lanes, seq, geometries)
                assert coordinator.run_cycle() == 3
                _release_results(lanes)
        # Then
        assert _geometry_warnings(caplog) == []
        snapshot = coordinator.snapshot()
        assert snapshot.cameras["camera-a"].observed_geometry == (640, 360)
        assert snapshot.cameras["camera-b"].observed_geometry == (640, 480)
        assert snapshot.cameras["camera-c"].observed_geometry == (1920, 1080)
    finally:
        coordinator.stop()


def test_one_camera_geometry_transition_emits_one_rendered_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given
    coordinator, _client, _watchdog, _clock, lanes = _coordinator(("camera-a", "camera-b"))
    try:
        _publish_geometries(lanes, 1, {"camera-a": (640, 360), "camera-b": (640, 480)})
        assert coordinator.run_cycle() == 2
        _release_results(lanes)
        _publish_geometries(lanes, 2, {"camera-a": (1920, 1080), "camera-b": (640, 480)})
        # When
        with caplog.at_level(logging.WARNING):
            assert coordinator.run_cycle() == 2
        # Then
        warnings = _geometry_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "camera-a" in message
        assert "640x360" in message
        assert "1920x1080" in message
        assert message.index("640x360") < message.index("1920x1080")
        snapshot = coordinator.snapshot()
        assert snapshot.cameras["camera-a"].observed_geometry == (1920, 1080)
        assert snapshot.cameras["camera-b"].observed_geometry == (640, 480)
        assert snapshot.geometry_batch_sizes[(1920, 1080)] == {1: 1}
        _release_results(lanes)
    finally:
        coordinator.stop()


def test_geometry_transition_warning_does_not_repeat_while_stable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given
    coordinator, _client, _watchdog, _clock, lanes = _coordinator(("camera-a", "camera-b"))
    try:
        _publish_geometries(lanes, 1, {"camera-a": (640, 360), "camera-b": (640, 480)})
        assert coordinator.run_cycle() == 2
        _release_results(lanes)
        _publish_geometries(lanes, 2, {"camera-a": (1920, 1080), "camera-b": (640, 480)})
        assert coordinator.run_cycle() == 2
        _release_results(lanes)
        _publish_geometries(lanes, 3, {"camera-a": (1920, 1080), "camera-b": (640, 480)})
        # When
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert coordinator.run_cycle() == 2
        # Then
        assert _geometry_warnings(caplog) == []
        snapshot = coordinator.snapshot()
        assert snapshot.cameras["camera-a"].observed_geometry == (1920, 1080)
        assert snapshot.geometry_batch_sizes[(1920, 1080)] == {1: 2}
        _release_results(lanes)
    finally:
        coordinator.stop()


def test_inference_telemetry_excludes_rtsp_credentials_and_frame_payload() -> None:
    # Given
    coordinator, _client, _watchdog, _clock, lanes = _coordinator(("camera-a",))
    try:
        lanes["camera-a"][0].publish(_packet("camera-a", 1, height=360, width=640))
        # When
        assert coordinator.run_cycle() == 1
        snapshot = coordinator.snapshot()
        # Then
        telemetry = snapshot.cameras["camera-a"]
        assert telemetry.observed_geometry == (640, 360)
        dumped = repr(snapshot)
        assert "rtsp://" not in dumped
        assert "password" not in dumped
        assert "array(" not in dumped
        assert "uint8" not in dumped
        _release_results(lanes)
    finally:
        coordinator.stop()
