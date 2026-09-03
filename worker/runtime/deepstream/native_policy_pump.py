"""Image-free native perception to the existing CPU policy and evidence plane."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol, final, runtime_checkable

from contracts.observation import BoundingBox
from worker.native.deepstream.control import ChildControlError, DeepStreamControlClient
from worker.native.deepstream.ipc import MetadataFrame
from worker.native.deepstream.metadata import AcceptanceToken, LatestMetadataSlot, SourceBinding
from worker.pipeline.decision import EventAggregator
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.perception import SceneState, build_decision_input, build_frame_observation
from worker.runtime.deepstream.canary_telemetry import NativeCanaryTelemetry
from worker.types import BusinessEvent, ChannelState, NativeEvidenceTrigger

LOGGER = logging.getLogger(__name__)
_FPS_WINDOW_SEC = 10.0


@runtime_checkable
class NativeEventSink(Protocol):
    def emit_for_frame(self, event: BusinessEvent, trigger: NativeEvidenceTrigger) -> None: ...


class NativeDiagnostics(Protocol):
    def update_measured_fps(self, camera_id: str, measured_fps: float | None) -> None: ...
    def record_detection_completed(self, camera_id: str) -> None: ...
    def record_native_detection_attempt(self, camera_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class NativePolicyContext:
    metadata: LatestMetadataSlot
    control: DeepStreamControlClient
    scene_state: SceneState
    decision: EventAggregator
    sink: NativeEventSink
    attacher: AlertEvidenceAttacher
    diagnostics: NativeDiagnostics
    bed_interval: int
    canary_telemetry: NativeCanaryTelemetry | None = None


@final
class NativePolicyPump:
    """Consume exact accepted metadata without host frames or Python tracking."""

    def __init__(self, binding: SourceBinding, context: NativePolicyContext) -> None:
        self._binding = binding
        self._metadata = context.metadata
        self._control = context.control
        self._scene = context.scene_state
        self._decision = context.decision
        self._sink = context.sink
        self._attacher = context.attacher
        self._diagnostics = context.diagnostics
        self._bed_interval = context.bed_interval
        self._canary_telemetry = context.canary_telemetry
        self._stop = threading.Event()
        self._fps: deque[float] = deque()
        self.processed_count = 0
        self.failure_count = 0

    @property
    def camera_id(self) -> str:
        return self._binding.camera_id

    def _rebind_if_source_was_rebuilt(self) -> None:
        """Adopt the slot's current binding after a source rebuild.

        ``SourceLifecycle.rebuild`` asks the child for a fresh binding and
        re-registers it on the slot, which advances ``source_generation`` and
        ``stream_epoch``. The slot then keeps accepting frames against that new
        binding, but this pump was constructed with the pre-rebuild binding and
        ``wait_accepted`` matches against whatever binding its token carries --
        so without this the pump silently starves forever while the slot
        reports a clean accept tally and every frame is overwritten unread.

        Nothing counts the pump-side refusal (``wait_accepted`` is a predicate,
        not a publish path), which is why this failure presented as "the child
        never publishes" for a long time.
        """
        current = self._metadata.expected_binding(self.camera_id)
        if current is None or current == self._binding:
            return
        previous = self._binding
        self._binding = current
        LOGGER.info(
            "native policy pump rebound after source rebuild: camera_id=%s "
            "generation %d->%d epoch %d->%d",
            current.camera_id,
            previous.source_generation,
            current.source_generation,
            previous.stream_epoch,
            current.stream_epoch,
        )

    def run(self) -> None:
        token = self._metadata.subscribe(self._binding)
        while not self._stop.is_set():
            try:
                frame = self._metadata.wait_accepted(token, timeout_sec=0.5)
            except TimeoutError:
                self._rebind_if_source_was_rebuilt()
                token = AcceptanceToken(self._binding, token.native_publish_sequence)
                continue
            token = AcceptanceToken(self._binding, frame.native_publish_sequence)
            if self._canary_telemetry is not None:
                self._canary_telemetry.record(
                    frame.frame.identity.source_pts or 0,
                    frame.source_time_ns,
                    frame.native_publish_sequence,
                )
            self._diagnostics.record_native_detection_attempt(self.camera_id)
            try:
                self._process(frame)
            except (ChildControlError, OSError, ValueError, RuntimeError):
                self.failure_count += 1
                LOGGER.warning(
                    "native policy frame failed: camera_id=%s",
                    self.camera_id,
                    exc_info=True,
                )
            finally:
                self.processed_count += 1

    def stop(self) -> None:
        self._stop.set()

    def _process(self, metadata: MetadataFrame) -> None:
        frame = metadata.frame
        association = frame.association
        if association is None or metadata.source_width <= 0 or metadata.source_height <= 0:
            raise ValueError("native perception frame is incomplete")
        boxes = tuple(
            BoundingBox(box.x1, box.y1, box.x2, box.y2, box.confidence)
            for box in frame.person_box.boxes
        )
        track_ids: list[int | None] = [None] * len(boxes)
        for track_id, cue_index in zip(
            association.track_ids,
            association.selected_cue_indexes,
            strict=True,
        ):
            track_ids[cue_index] = track_id
        resolved_track_ids = tuple(track_id for track_id in track_ids if track_id is not None)
        if len(resolved_track_ids) != len(boxes):
            raise ValueError("native association did not assign every person cue")
        observation = build_frame_observation(
            boxes=boxes,
            poses=tuple(
                tuple((point.x, point.y, point.score) for point in pose)
                for pose in frame.human_pose.poses
            ),
            bed_boxes=tuple(
                BoundingBox(
                    region.x1,
                    region.y1,
                    region.x2,
                    region.y2,
                    region.confidence,
                    region.polygon,
                )
                for region in frame.bed_region.regions
            ),
            track_ids=resolved_track_ids,
        )
        decision_input = build_decision_input(
            observation,
            frame_width=metadata.source_width,
            frame_height=metadata.source_height,
            live_track_ids=association.live_track_ids,
            time_sec=(frame.identity.source_pts or 0) / 1_000_000_000,
            frame_index=frame.identity.seq,
            scene_state=self._scene,
            bed_scheduled=frame.bed_region.state is not ChannelState.SKIPPED,
            bed_interval=self._bed_interval,
        )
        _ = self._scene.observe(
            decision_input.observation,
            track_ids=resolved_track_ids,
        )
        events = self._decision.update(decision_input)
        trigger = NativeEvidenceTrigger(
            self.camera_id,
            frame.identity.worker_boot_id,
            frame.identity.stream_epoch,
            metadata.source_generation,
            frame.identity.seq,
            frame.identity.source_pts or 0,
            metadata.source_time_ns / 1_000_000_000,
        )
        for event in events:
            try:
                snapshot = self._control.snapshot(self.camera_id)
            except ChildControlError:
                snapshot = None
            self._sink.emit_for_frame(self._attacher.attach_native(event, snapshot), trigger)
        self._diagnostics.record_detection_completed(self.camera_id)
        now = time.monotonic()
        self._fps.append(now)
        while self._fps and now - self._fps[0] > _FPS_WINDOW_SEC:
            self._fps.popleft()
        if len(self._fps) >= 2:
            elapsed = self._fps[-1] - self._fps[0]
            self._diagnostics.update_measured_fps(
                self.camera_id,
                None if elapsed <= 0 else (len(self._fps) - 1) / elapsed,
            )

__all__ = ["NativeEventSink", "NativePolicyContext", "NativePolicyPump"]
