"""Per-camera result loop: coordinated pose -> analytics -> decision -> output.

Pure wiring stage. Business math stays where it already lives -- extraction
and tracking in ``analytics`` (:class:`CompositeExtractor`, itself gated by
the camera's :class:`~worker.pipeline.bus.Scheduler`), domain interpretation
in ``decision`` (:class:`EventAggregator`), audit/snapshot attachment in
``worker.pipeline.output.evidence_attacher`` (:class:`AlertEvidenceAttacher`).
This module only pumps taken packets through those calls, forwards each
admitted event through the attacher (if configured) to the sink, and records
the per-frame measured-fps sample when a diagnostics collaborator is wired.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Final, Protocol, final

from contracts.observation import FrameObservation
from worker.adapters.model.errors import FatalAcceleratorError
from worker.interfaces.output import EventSink
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.decision import EventAggregator
from worker.pipeline.inference_coordinator import InferenceResultSlot
from worker.pipeline.trace import BoundedTraceWriter, TraceCapture
from worker.types import BusinessEvent, FramePacket, ModuleResult

LOGGER: Final = logging.getLogger(__name__)
DEFAULT_POLL_TIMEOUT_SEC: Final = 0.5
MEASURED_FPS_WINDOW_SEC: Final = 10.0


class MeasuredFpsSink(Protocol):
    """Structural seam so this module never imports `worker.runtime` back.

    `worker.runtime.telemetry.runtime_diagnostics.WorkerDiagnostics` already
    satisfies this; any other object with the same methods works too.
    """

    def update_measured_fps(self, camera_id: str, measured_fps: float | None) -> None: ...

    def record_detection_completed(self, camera_id: str) -> None: ...


class EvidenceAttacher(Protocol):
    """Structural seam matched by `AlertEvidenceAttacher` (worker.pipeline.output)."""

    def attach(
        self,
        event: BusinessEvent,
        packet: FramePacket,
        observation: FrameObservation,
    ) -> BusinessEvent: ...


class ObservationRecorder(Protocol):
    """Structural seam matched by `LatestObservationStore` (worker.pipeline.output).

    Deliberately a Protocol, like `EvidenceAttacher` above: this module stays
    import-free of `worker.pipeline.output` so the live view stays a consumer
    of what this loop already computed, not a stage this loop waits on.
    """

    def record(
        self,
        camera_id: str,
        observation: FrameObservation,
        debug_snapshots: tuple[Any, ...] = (),
        *,
        frame_index: int,
    ) -> None: ...


@final
class CameraPipelinePump:
    """Drive one camera's bounded inference-result slot to output.

    The shared coordinator owns pose inference; this pump remains the sole
    owner of per-camera analytics, tracker, scene, and domain mutation.
    ``analytics.process`` gates remaining extraction by the camera's
    ``Scheduler`` (only due modules run; every packet still advances the
    tracker/scene state). This loop forwards every packet taken from
    ``subscription`` through it, then through ``decision.update``, emitting
    each admitted event to ``sink``.
    """

    def __init__(
        self,
        camera_id: str,
        results: InferenceResultSlot,
        analytics: CompositeExtractor,
        decision: EventAggregator,
        sink: EventSink,
        *,
        poll_timeout_sec: float = DEFAULT_POLL_TIMEOUT_SEC,
        evidence_attacher: EvidenceAttacher | None = None,
        diagnostics: MeasuredFpsSink | None = None,
        max_frames: int | None = None,
        observation_recorder: ObservationRecorder | None = None,
        debug_snapshots_provider: Callable[[int], tuple[Any, ...]] | None = None,
        trace_capture: TraceCapture | None = None,
        trace_writer: BoundedTraceWriter | None = None,
    ) -> None:
        self._camera_id = camera_id
        self._untraced_events = 0
        self._telemetry_failures = 0
        self._results = results
        self._analytics = analytics
        self._decision = decision
        self._sink = sink
        self._poll_timeout_sec = poll_timeout_sec
        self._evidence_attacher = evidence_attacher
        self._diagnostics = diagnostics
        self._max_frames = max_frames
        self._observation_recorder = observation_recorder
        self._debug_snapshots_provider = debug_snapshots_provider
        if (trace_capture is None) != (trace_writer is None):
            raise ValueError("trace capture and writer must be composed together")
        self._trace_capture = trace_capture
        self._trace_writer = trace_writer
        self._fps_timestamps: deque[float] = deque()
        self._stop_event = threading.Event()
        self.failure_count = 0
        self.processed_count = 0

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def run(self) -> None:
        while not self._stop_event.is_set() and not self._frames_exhausted():
            coordinated = self._results.take(timeout_sec=self._poll_timeout_sec)
            if coordinated is None:
                continue
            packet = coordinated.packet
            try:
                self._pump_one(packet, coordinated.pose)
            except FatalAcceleratorError:
                raise
            except Exception as error:  # noqa: BLE001 - per-camera boundary
                self._record_failure(error)
            finally:
                packet.release()
                # `processed_count` (and therefore `--max-frames-per-camera`'s
                # cap) counts frames *attempted*, not frames that succeeded --
                # a failed `_pump_one` still advances it. This matches edge's
                # reference semantics (edge/runtime/camera_worker.py:169,
                # `processed += 1` after the try/except around
                # `process_frame`) and guarantees termination even if every
                # frame in the run fails.
                self.processed_count += 1

    def stop(self) -> None:
        self._stop_event.set()

    def _frames_exhausted(self) -> bool:
        return self._max_frames is not None and self.processed_count >= self._max_frames

    def _observe(self, what: str, record: Callable[[], None]) -> None:
        """Run a telemetry callback that must never affect detection.

        Measured-fps sampling, observation recording and the detection-completed
        counter are all reporting. They sat unguarded on the frame path ahead of
        the emission loop, so any one of them raising meant the events computed
        from that frame were never emitted at all. Three other auxiliary
        capabilities in this runtime have already been found holding that same
        power over a resident alert; these are the same shape.
        """
        try:
            record()
        except Exception:  # noqa: BLE001 - telemetry never blocks detection
            self._telemetry_failures += 1
            LOGGER.warning(
                "camera %s telemetry callback %r failed; detection continues "
                "(telemetry_failures=%d)",
                self._camera_id,
                what,
                self._telemetry_failures,
                exc_info=True,
            )

    def _pump_one(self, packet: FramePacket, pose: ModuleResult) -> None:
        self._observe("measured-fps", self._record_measured_fps)
        result = self._analytics.process(packet, prefetched_results=(pose,))
        self._observe(
            "observation", lambda: self._record_observation(packet, result.observation)
        )
        events = self._decision.update(result.decision_input)
        if self._diagnostics is not None:
            diagnostics = self._diagnostics
            self._observe(
                "detection-completed",
                lambda: diagnostics.record_detection_completed(self._camera_id),
            )
        if self._trace_capture is not None and self._trace_writer is not None:
            # Analysis tracing is auxiliary. The decision basis an admitted event
            # needs travels in its delivery-queue EVENT envelope, exactly as
            # worker/pipeline/trace/store.py states: that cache exists only for
            # best-effort annotated derivatives while the process is alive.
            #
            # Emission was nonetheless conditional on it. A writer not yet
            # started, a full handoff queue, or a failing trace store raised
            # TracePersistenceError out of this method, and the resident's fall
            # event was never emitted at all. A QA capability must not be able to
            # do that, so a trace failure now costs the trace pointer and nothing
            # more.
            try:
                traced = self._trace_capture.capture(
                    self._trace_writer,
                    packet,
                    result,
                    events,
                    require_persisted=bool(events),
                )
                if events and not isinstance(traced, bool):
                    events = traced
            except Exception:  # noqa: BLE001 - tracing never blocks detection
                self._untraced_events += len(events)
                LOGGER.warning(
                    "camera %s emitting %d event(s) without an analysis trace: trace "
                    "capture failed, so QA replay will have no timeline for them "
                    "(untraced_events=%d)",
                    self._camera_id,
                    len(events),
                    self._untraced_events,
                    exc_info=True,
                )
        for position, event in enumerate(events):
            attached = self._attach_evidence(event, packet, result.observation)
            emit_for_frame = getattr(self._sink, "emit_for_frame", None)
            try:
                if emit_for_frame is None:
                    self._sink.emit(attached)
                else:
                    emit_for_frame(attached, packet)
            except Exception:
                # The sink deliberately refuses to proceed when a decision
                # envelope cannot be admitted durably, so nothing is left
                # half-written. But admission happens AFTER the decider has
                # consumed the rising edge and set its cooldown, so without
                # this the next frame produces nothing and the fall is lost
                # for good -- a transient fault permanently destroying a
                # resident event. Re-open the decision so a later frame can
                # emit it again.
                release = getattr(self._decision, "release", None)
                if release is not None:
                    # This event AND every one after it. The aggregator admitted
                    # the whole tuple before the loop began, so the ones behind
                    # this failure had their cooldowns consumed and will now
                    # never be attempted at all -- a fall sitting after a
                    # bed-exit that raised is lost without ever being tried.
                    for pending in events[position:]:
                        release(pending)
                raise

    def _record_observation(
        self, packet: FramePacket, observation: FrameObservation
    ) -> None:
        """Cache this frame's observation for the live-view pump (todo 10).

        The preview used to be *published* from here, after ``analytics``,
        which put every operator frame behind a pose forward. Now this loop
        only records what it already computed -- a dict write, no encode, no
        second inference -- and ``LiveViewPump`` (draining ``bus.live`` on its
        own thread) decides when to draw it and whether it is still fresh.
        Still a tap, not a stage: absent when nothing consumes it, and any
        failure is logged and swallowed rather than counted as a pipeline
        failure.
        """
        recorder = self._observation_recorder
        if recorder is None:
            return
        provider = self._debug_snapshots_provider
        try:
            snapshots = () if provider is None else provider(packet.frame.index)
            recorder.record(
                self._camera_id,
                observation,
                snapshots,
                frame_index=packet.frame.index,
            )
        except Exception:  # noqa: BLE001 - a debug view must not stop detection
            LOGGER.warning(
                "live view observation record failed: camera_id=%s",
                self._camera_id,
                extra={"camera_id": self._camera_id},
                exc_info=True,
            )

    def _attach_evidence(
        self,
        event: BusinessEvent,
        packet: FramePacket,
        observation: FrameObservation,
    ) -> BusinessEvent:
        if self._evidence_attacher is None:
            return event
        return self._evidence_attacher.attach(event, packet, observation)

    def _record_measured_fps(self) -> None:
        """Mirror edge's `_record_measured_fps`: 10s sliding-window rate."""
        if self._diagnostics is None:
            return
        now = time.monotonic()
        timestamps = self._fps_timestamps
        timestamps.append(now)
        while timestamps and now - timestamps[0] > MEASURED_FPS_WINDOW_SEC:
            timestamps.popleft()
        if len(timestamps) >= 2:
            elapsed = timestamps[-1] - timestamps[0]
            self._diagnostics.update_measured_fps(
                self._camera_id, None if elapsed <= 0 else (len(timestamps) - 1) / elapsed
            )

    def _record_failure(self, error: Exception) -> None:
        self.failure_count += 1
        LOGGER.warning(
            "camera pipeline pump failed processing a frame: camera_id=%s error=%s",
            self._camera_id,
            type(error).__name__,
            extra={"camera_id": self._camera_id, "error": type(error).__name__},
        )


__all__ = ["CameraPipelinePump", "DEFAULT_POLL_TIMEOUT_SEC"]
