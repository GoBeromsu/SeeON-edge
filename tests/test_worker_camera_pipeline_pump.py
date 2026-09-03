"""Per-camera pipeline pump: ``bus.inference`` -> analytics -> decision -> sink.

Covers the properties the composition root relies on but does not itself
prove: admitted events actually reach the sink through a real ``emit()``
call, the scheduler still gates which extractor runs, one camera's pump
failure never stops another camera's pump, and ``IngestSupervisor.stop()``
actually reaps real pump threads.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import final

import numpy as np
import pytest

from contracts.frame import Frame
from contracts.runner import pose_result
from worker.adapters.model.errors import FatalAcceleratorError
from worker.domains.fall import FallPolicyDeciderV2, FallPolicyV2, FallV2Probabilities
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.bus import Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.inference_coordinator import CoordinatedInference, InferenceResultSlot
from worker.pipeline.ingest.lifecycle import IngestSupervisor
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.pipeline.trace.models import TracePersistenceError
from worker.types import BusinessEvent, DecisionInput, FramePacket, ModuleResult


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.full((2, 3, 3), seq, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=seq / 5.0, image=image)
    return FramePacket(camera_id, frame, seq / 5.0, seq, 3, 2, 0.25)


def _deliver(results: InferenceResultSlot, packet: FramePacket) -> None:
    results.publish(
        CoordinatedInference(
            packet,
            ModuleResult("pose", pose_result((), ()), 0.0, "pose"),
        )
    )


def _blank_analytics(
    camera_id: str,
    *,
    extractors: tuple[object, ...] = (),
    task_intervals: dict[str, int] | None = None,
) -> CompositeExtractor:
    return CompositeExtractor(
        extractors=extractors,  # type: ignore[arg-type]
        scheduler=Scheduler(task_intervals=task_intervals or {}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState(camera_id=camera_id),
    )


def _wait_for(predicate: Callable[[], bool], *, timeout_sec: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@final
class _RecordingExtractor:
    """Fake ``NamedExtractor``: records exactly which frames it was asked to extract."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.calls: list[int] = []

    def extract(self, packet: FramePacket) -> ModuleResult:
        self.calls.append(packet.frame.index)
        return ModuleResult(module_name=self.module_name, result=(), elapsed_ms=0.0)


@final
class _FrameEchoDecider:
    """Emits one uniquely-identified event per decision input it sees."""

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        return (
            BusinessEvent(
                domain="probe",
                event_type="tick",
                identity=f"frame-{input_value.frame_index}",
                camera_id=self._camera_id,
                facility_id=f"facility-{self._camera_id}",
                time_sec=input_value.time_sec or 0.0,
                probability=1.0,
            ),
        )


@final
class _RaisingDecider:
    """Always fails: exercises the pump's per-frame failure isolation branch."""

    def __init__(self) -> None:
        self.call_count = 0

    def update(self, _input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        self.call_count += 1
        raise RuntimeError("decision stage exploded")


@final
class _FatalDecider:
    def update(self, _input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        raise FatalAcceleratorError(
            "injected accelerator failure", camera_id="camera-a", task="pose"
        )


@final
class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[BusinessEvent] = []
        self.on_emit: Callable[[BusinessEvent], None] | None = None

    def emit(self, event: BusinessEvent) -> None:
        self.events.append(event)
        if self.on_emit is not None:
            self.on_emit(event)


def test_pump_forwards_a_coordinated_result_to_the_sink_via_real_emit() -> None:
    results = InferenceResultSlot()
    analytics = _blank_analytics("camera-a", task_intervals={"pose": 1})
    decision = EventAggregator(
        deciders=(_FrameEchoDecider("camera-a"),), incidents=IncidentManager()
    )
    sink = _RecordingSink()
    pump = CameraPipelinePump("camera-a", results, analytics, decision, sink, poll_timeout_sec=0.02)
    sink.on_emit = lambda _event: pump.stop()

    _deliver(results, _packet("camera-a", 1))
    pump.run()

    assert len(sink.events) == 1
    admitted = sink.events[0]
    assert admitted.domain == "probe"
    assert admitted.event_type == "tick"
    assert admitted.camera_id == "camera-a"
    # IncidentManager.admit() resolves the decider's identity into a durable
    # edge_event_id, so only its presence (not its exact value) is asserted here.
    assert isinstance(admitted.identity, str) and admitted.identity


def test_scheduler_gates_extraction_so_skipped_frames_never_reach_the_extractor() -> None:
    """analytics.process only calls extractors for modules the scheduler marks due."""
    results = InferenceResultSlot()
    extractor = _RecordingExtractor("probe")
    # interval 2: frame 1 is skipped (1 % 2 != 0), frame 2 is due (2 % 2 == 0).
    analytics = _blank_analytics("camera-a", extractors=(extractor,), task_intervals={"probe": 2})
    decision = EventAggregator(
        deciders=(_FrameEchoDecider("camera-a"),), incidents=IncidentManager()
    )
    sink = _RecordingSink()
    pump = CameraPipelinePump("camera-a", results, analytics, decision, sink, poll_timeout_sec=0.02)

    def _on_emit(_event: BusinessEvent) -> None:
        if len(sink.events) == 1:
            _deliver(results, _packet("camera-a", 2))
        else:
            pump.stop()

    sink.on_emit = _on_emit
    _deliver(results, _packet("camera-a", 1))
    pump.run()

    # Decision still runs on every frame (identity/track state must advance regardless),
    # but the scheduled-out module is never asked to extract the skipped frame.
    assert len(sink.events) == 2
    assert extractor.calls == [2]


def test_one_camera_pump_failure_does_not_stop_the_other_camera_pump(
    caplog: pytest.LogCaptureFixture,
) -> None:
    results_a = InferenceResultSlot()
    results_b = InferenceResultSlot()
    raising = _RaisingDecider()
    sink_a = _RecordingSink()
    decision_a = EventAggregator(deciders=(raising,), incidents=IncidentManager())
    pump_a = CameraPipelinePump(
        "camera-a",
        results_a,
        _blank_analytics("camera-a"),
        decision_a,
        sink_a,
        poll_timeout_sec=0.02,
    )

    sink_b = _RecordingSink()
    decision_b = EventAggregator(
        deciders=(_FrameEchoDecider("camera-b"),), incidents=IncidentManager()
    )
    pump_b = CameraPipelinePump(
        "camera-b",
        results_b,
        _blank_analytics("camera-b"),
        decision_b,
        sink_b,
        poll_timeout_sec=0.02,
    )

    supervisor = IngestSupervisor([pump_a, pump_b])
    supervisor.start()
    try:
        failed_packet = _packet("camera-a", 1)
        _deliver(results_a, failed_packet)
        _deliver(results_b, _packet("camera-b", 1))

        assert _wait_for(lambda: len(sink_b.events) >= 1)
        assert _wait_for(lambda: pump_a.failure_count >= 1)

        # Publish a second frame to camera-a: if the exception had killed the
        # thread instead of being isolated, this would never be processed.
        _deliver(results_a, _packet("camera-a", 2))
        assert _wait_for(lambda: pump_a.failure_count >= 2)
    finally:
        supervisor.stop(join_timeout_sec=2.0)

    assert pump_a.failure_count == 2
    assert failed_packet.released
    failure_record = next(
        record for record in caplog.records if record.name == "worker.pipeline.camera_pipeline"
    )
    assert "camera_id=camera-a" in failure_record.getMessage()
    assert "error=RuntimeError" in failure_record.getMessage()
    assert "decision stage exploded" not in failure_record.getMessage()
    assert raising.call_count == 2
    assert len(sink_b.events) == 1
    assert sink_a.events == []


def test_fatal_accelerator_error_propagates_out_of_run_instead_of_being_swallowed() -> None:
    results = InferenceResultSlot()
    decision = EventAggregator(deciders=(_FatalDecider(),), incidents=IncidentManager())
    sink = _RecordingSink()
    pump = CameraPipelinePump(
        "camera-a",
        results,
        _blank_analytics("camera-a"),
        decision,
        sink,
        poll_timeout_sec=0.02,
    )

    _deliver(results, _packet("camera-a", 1))

    with pytest.raises(FatalAcceleratorError):
        pump.run()


def test_pump_self_terminates_once_max_frames_processed() -> None:
    results = InferenceResultSlot()
    decision = EventAggregator(deciders=(), incidents=IncidentManager())
    sink = _RecordingSink()
    pump = CameraPipelinePump(
        "camera-a",
        results,
        _blank_analytics("camera-a"),
        decision,
        sink,
        poll_timeout_sec=0.02,
        max_frames=2,
    )

    thread = threading.Thread(target=pump.run, daemon=True)
    thread.start()
    _deliver(results, _packet("camera-a", 1))
    assert _wait_for(lambda: pump.processed_count >= 1)
    _deliver(results, _packet("camera-a", 2))
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert pump.processed_count == 2


def test_pump_without_max_frames_keeps_polling_past_what_a_cap_would_have_stopped() -> None:
    results = InferenceResultSlot()
    decision = EventAggregator(deciders=(), incidents=IncidentManager())
    sink = _RecordingSink()
    pump = CameraPipelinePump(
        "camera-a",
        results,
        _blank_analytics("camera-a"),
        decision,
        sink,
        poll_timeout_sec=0.02,
    )
    thread = threading.Thread(target=pump.run, daemon=True)
    thread.start()
    for frame_index in (1, 2, 3):
        _deliver(results, _packet("camera-a", frame_index))
        assert _wait_for(lambda frame_index=frame_index: pump.processed_count >= frame_index)
    pump.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert pump.processed_count >= 3


def test_supervisor_stop_joins_real_pump_threads() -> None:
    results = InferenceResultSlot()
    decision = EventAggregator(deciders=(), incidents=IncidentManager())
    sink = _RecordingSink()
    pump = CameraPipelinePump(
        "camera-a",
        results,
        _blank_analytics("camera-a"),
        decision,
        sink,
        poll_timeout_sec=0.02,
    )
    supervisor = IngestSupervisor([pump])

    supervisor.start()
    thread_name = "worker-ingest-camera-a"
    started_thread = next(thread for thread in threading.enumerate() if thread.name == thread_name)
    assert started_thread.is_alive()

    supervisor.stop(join_timeout_sec=2.0)

    assert not started_thread.is_alive()


class _FailingTraceCapture:
    """Every trace attempt fails, the way a stopped or full writer does."""

    def __init__(self) -> None:
        self.calls = 0

    def capture(self, *_args: object, **_kwargs: object) -> object:
        self.calls += 1
        raise TracePersistenceError("admitted event trace could not be persisted")


def test_a_failing_trace_capture_does_not_stop_the_event_reaching_the_sink() -> None:
    """Detection must survive any analysis-tracing failure.

    Emission used to be conditional on QA trace persistence: `capture` raised
    `TracePersistenceError` and that propagated out of the pump, so a writer
    that was not yet started, a full handoff queue, or a failing trace store
    silently stopped a resident's fall event from reaching anyone.

    The decision basis an admitted event needs travels in its delivery-queue
    envelope, not in this cache, so a trace failure may cost the trace pointer
    and nothing else.
    """
    results = InferenceResultSlot()
    analytics = _blank_analytics("camera-a", task_intervals={"pose": 1})
    decision = EventAggregator(
        deciders=(_FrameEchoDecider("camera-a"),), incidents=IncidentManager()
    )
    sink = _RecordingSink()
    capture = _FailingTraceCapture()
    pump = CameraPipelinePump(
        "camera-a",
        results,
        analytics,
        decision,
        sink,
        poll_timeout_sec=0.02,
        trace_capture=capture,
        trace_writer=object(),
    )
    sink.on_emit = lambda _event: pump.stop()

    _deliver(results, _packet("camera-a", 1))
    pump.run()

    assert capture.calls >= 1, "the failing capture was never exercised"
    assert len(sink.events) == 1, (
        "the event never reached the sink; an analysis-tracing failure suppressed a resident alert"
    )


class _ExplodingDiagnostics:
    """Telemetry that fails the way a broken counter or full buffer does."""

    def __init__(self) -> None:
        self.calls = 0

    def record_measured_fps(self, *_args: object, **_kwargs: object) -> None:
        return None

    def record_detection_completed(self, *_args: object, **_kwargs: object) -> None:
        self.calls += 1
        raise RuntimeError("diagnostics sink unavailable")


def test_failing_diagnostics_do_not_stop_the_event_reaching_the_sink() -> None:
    """Telemetry is reporting; it must never suppress a resident alert.

    `record_detection_completed` sat unguarded on the frame path ahead of the
    emission loop, so a raising telemetry sink meant the events computed from
    that frame were never emitted. This is the fourth auxiliary capability in
    this runtime found holding that power, after trace capture failing camera
    startup, trace publication killing the writer thread, and the trace
    persistence coupling itself.
    """
    results = InferenceResultSlot()
    analytics = _blank_analytics("camera-a", task_intervals={"pose": 1})
    decision = EventAggregator(
        deciders=(_FrameEchoDecider("camera-a"),), incidents=IncidentManager()
    )
    sink = _RecordingSink()
    diagnostics = _ExplodingDiagnostics()
    pump = CameraPipelinePump(
        "camera-a",
        results,
        analytics,
        decision,
        sink,
        poll_timeout_sec=0.02,
        diagnostics=diagnostics,
    )
    sink.on_emit = lambda _event: pump.stop()

    _deliver(results, _packet("camera-a", 1))
    pump.run()

    assert diagnostics.calls >= 1, "the failing telemetry was never exercised"
    assert len(sink.events) == 1, (
        "the event never reached the sink; a telemetry failure suppressed a resident alert"
    )


@final
class _FallDecider:
    """Emits a fall every frame.

    A fall's cooldown key is (camera_id, domain, event_type) with no identity,
    so every frame collapses onto the same key. That is precisely the shape in
    which a consumed-but-unstaged decision is unrecoverable.
    """

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        return (
            BusinessEvent(
                domain="fall",
                event_type="fall.detected",
                identity=f"frame-{input_value.frame_index}",
                camera_id=self._camera_id,
                facility_id=f"facility-{self._camera_id}",
                time_sec=input_value.time_sec or 0.0,
                probability=0.99,
            ),
        )


@final
class _EpisodeFallDecider:
    """Port adapter exercising the production episode authority lifecycle."""

    def __init__(self, camera_id: str) -> None:
        self._policy = FallPolicyDeciderV2(
            camera_id=camera_id,
            facility_id=f"facility-{camera_id}",
            boot_id="boot",
            stream_epoch="epoch",
            source_generation=0,
            policy=FallPolicyV2(transition_votes=1, transition_window=1),
        )

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        return self._policy.update(
            {7: FallV2Probabilities(0.0, 0.9, 0.0)},
            (7,),
            frame_index=input_value.frame_index,
            time_sec=input_value.time_sec or 0.0,
        )

    def release_onset(self, event: BusinessEvent) -> None:
        self._policy.release_onset(event)


def test_a_transient_staging_failure_does_not_destroy_the_fall() -> None:
    """A decision must not stay consumed unless its envelope is durable.

    The sink deliberately refuses to proceed when a decision envelope cannot be
    admitted, so nothing is left half-written. But admission happens *after* the
    decider consumed the rising edge and set its cooldown, so a single transient
    failure meant the next frame produced nothing at all: the fall was lost for
    good even though staging would have succeeded a frame later.
    """
    results = InferenceResultSlot()
    analytics = _blank_analytics("camera-a", task_intervals={"pose": 1})
    decision = EventAggregator(
        deciders=(_EpisodeFallDecider("camera-a"),),
        incidents=IncidentManager(cooldown_sec=300.0),
    )
    sink = _RecordingSink()
    failures: list[int] = []
    original_emit = sink.emit

    def _flaky_emit(event: object) -> None:
        if not failures:
            failures.append(1)
            # The slot is free now that this frame was consumed, so the next
            # frame can be delivered from here without deadlocking the
            # capacity-one handoff.
            _deliver(results, _packet("camera-a", 2))
            raise RuntimeError("event delivery admission failed: ENTRY_CAPACITY")
        original_emit(event)

    sink.emit = _flaky_emit  # type: ignore[method-assign]
    pump = CameraPipelinePump(
        "camera-a",
        results,
        analytics,
        decision,
        sink,
        poll_timeout_sec=0.02,
        max_frames=2,
    )

    _deliver(results, _packet("camera-a", 1))
    pump.run()

    assert failures, "the transient staging failure never happened"
    assert len(sink.events) == 1, (
        "the second frame produced no event; a transient staging failure "
        "permanently destroyed the resident's fall"
    )


@final
class _StableBedExitDecider:
    """Emits a bed exit whose cooldown key is stable across frames."""

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        return (
            BusinessEvent(
                domain="bed_exit",
                event_type="bed.exit",
                identity=f"frame-{input_value.frame_index}",
                camera_id=self._camera_id,
                facility_id=f"facility-{self._camera_id}",
                time_sec=input_value.time_sec or 0.0,
                probability=0.9,
                bed_id=7,
            ),
        )


def test_an_event_behind_a_failure_is_released_too() -> None:
    """The suffix of the emission loop was admitted but never attempted.

    The aggregator admits the whole tuple before the loop begins. When an
    earlier event raises, the ones behind it are never tried at all, yet their
    cooldowns are already spent -- so the next frame reports nothing for them
    and they are lost without ever having been attempted once.
    """
    results = InferenceResultSlot()
    analytics = _blank_analytics("camera-a", task_intervals={"pose": 1})
    decision = EventAggregator(
        deciders=(_FallDecider("camera-a"), _StableBedExitDecider("camera-a")),
        incidents=IncidentManager(cooldown_sec=300.0),
    )
    sink = _RecordingSink()
    failures: list[int] = []
    original_emit = sink.emit

    def _flaky_emit(event: object) -> None:
        # "fall" sorts before "bed_exit"? No: bed_exit < fall alphabetically, so
        # the bed exit is first. Fail on whichever arrives first, leaving the
        # second admitted but unattempted.
        if not failures:
            failures.append(1)
            _deliver(results, _packet("camera-a", 2))
            raise RuntimeError("event delivery admission failed: ENTRY_CAPACITY")
        original_emit(event)

    sink.emit = _flaky_emit  # type: ignore[method-assign]
    pump = CameraPipelinePump(
        "camera-a", results, analytics, decision, sink, poll_timeout_sec=0.02, max_frames=2
    )

    _deliver(results, _packet("camera-a", 1))
    pump.run()

    assert failures, "the staging failure never happened"
    kinds = {event.domain for event in sink.events}
    assert {"fall", "bed_exit"}.issubset(kinds), (
        f"only {sorted(kinds)} were emitted on the second frame; an event behind "
        f"the failure stayed consumed and was never attempted at all"
    )
