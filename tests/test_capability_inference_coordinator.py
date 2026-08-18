from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import final

import numpy as np
import pytest

from contracts.frame import Frame
from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from contracts.runner import RunnerProtocol, RunnerResult, pose_result
from worker.interfaces.output import EventSink
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.analytics.composite import CompositeResult
from worker.pipeline.bus import Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump, EvidenceAttacher
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.inference_coordinator import (
    CapabilityInferenceCoordinator,
    InferenceResultSlot,
)
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
    def __init__(self, clock: _Clock, watchdog: _Watchdog) -> None:
        self.clock, self.watchdog = clock, watchdog
        self.batches: list[tuple[tuple[str, int], ...]] = []
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
        self.clock.now += 0.020
        return tuple(
            pose_result((), ((float(frame.seq), 0.0, 1.0, 1.0, 0.9),))
            for frame in frames
        )


def _packet(camera_id: str, seq: int) -> FramePacket:
    return FramePacket(
        camera_id,
        Frame(index=seq, time_sec=float(seq), image=np.zeros((2, 2, 3), dtype=np.uint8)),
        float(seq),
        seq,
        2,
        2,
        0.0,
    )


def _coordinator(camera_ids: Sequence[str]):
    clock, watchdog = _Clock(), _Watchdog()
    client = _BatchClient(clock, watchdog)
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
