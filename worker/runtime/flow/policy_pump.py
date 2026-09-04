"""Image-free native perception to the existing CPU policy and evidence plane."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, final, runtime_checkable

from contracts.observation import BoundingBox
from contracts.replay_trace import ReplayRow, ReplaySource, ReplayTrack
from worker.pipeline.decision import EventAggregator
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.perception import SceneState, build_decision_input, build_frame_observation
from worker.pipeline.trace.replay_trace_writer import ReplayTraceWriter
from worker.runtime.flow.metadata_slot import AcceptanceToken, LatestMetadataSlot
from worker.types import BusinessEvent, ChannelState, NativeEvidenceTrigger
from worker.types.metadata import MetadataFrame, SourceBinding

LOGGER = logging.getLogger(__name__)
_FPS_WINDOW_SEC = 10.0


@runtime_checkable
class NativeEventSink(Protocol):
    def emit_for_frame(self, event: BusinessEvent, trigger: NativeEvidenceTrigger) -> None: ...


class NativeSnapshotControl(Protocol):
    """The only media-plane control operation policy processing needs."""

    def snapshot(self, camera_id: str) -> bytes: ...


class NativeDiagnostics(Protocol):
    def update_measured_fps(self, camera_id: str, measured_fps: float | None) -> None: ...
    def record_detection_completed(self, camera_id: str) -> None: ...
    def record_native_detection_attempt(self, camera_id: str) -> None: ...
    def record_track_id_switch(self, camera_id: str) -> None: ...
    def record_track_id_switch_absorbed_total(self, camera_id: str, total: int) -> None: ...
    def record_bed_polygon_source(self, camera_id: str, source: str) -> None: ...
    def record_replay_trace_write_failure(self, camera_id: str) -> None: ...
    def record_resample_gap_rows(self, camera_id: str, count: int = 1) -> None: ...
    def record_fall_inference_device(self, camera_id: str, device: str) -> None: ...


@dataclass(frozen=True, slots=True)
class NativePolicyContext:
    metadata: LatestMetadataSlot
    control: NativeSnapshotControl
    scene_state: SceneState
    decision: EventAggregator
    sink: NativeEventSink
    attacher: AlertEvidenceAttacher
    diagnostics: NativeDiagnostics
    bed_interval: int
    replay_trace: ReplayTraceWriter | None = None
    # G7: composition supplies the evaluated bed-exit detection window.
    night_window_active: Callable[[], bool] | None = None
    recreate_decision: Callable[[SourceBinding], EventAggregator] | None = None
    track_id_switch_absorbed_total: Callable[[EventAggregator], int] | None = None


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
        self._replay_trace = context.replay_trace
        self._night_window_active = context.night_window_active
        self._recreate_decision = context.recreate_decision
        if context.track_id_switch_absorbed_total is None:
            # ADR-0002: a missing seam refuses at wiring time, never mid-stream.
            raise ValueError("native policy pump requires an absorbed-switch reader")
        self._track_id_switch_absorbed_total = context.track_id_switch_absorbed_total
        self._stop = threading.Event()
        self._fps: deque[float] = deque()
        self.processed_count = 0
        self.failure_count = 0
        self._trace_epoch: tuple[int, int] | None = None
        self._trace_tracks: dict[int, ReplayTrack] = {}
        self._live_track_misses: dict[int, int] = {}
        self._trace_source_lost = False
        self._trace_source: ReplaySource | None = None
        self._trace_dims: tuple[int, int] = (1, 1)
        self._trace_last_pts_ns = 0
        self._trace_seq = 0
        self._trace_write_failure_logged = False

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
        if self._recreate_decision is not None:
            # A source rebuild starts a distinct native epoch. Recreate the
            # camera-local V2 window/policy so no partial 30-row state crosses
            # the boundary and every onset identity names the new generation.
            self._decision = self._recreate_decision(current)
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
                self._capture_source_lost()
                self._rebind_if_source_was_rebuilt()
                token = AcceptanceToken(self._binding, token.native_publish_sequence)
                continue
            token = AcceptanceToken(self._binding, frame.native_publish_sequence)
            self._diagnostics.record_native_detection_attempt(self.camera_id)
            try:
                self._process(frame)
            except (OSError, ValueError, RuntimeError):
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
        self._record_track_id_switches(resolved_track_ids)
        self._record_bed_polygon_source(metadata)
        self._capture_replay_row(metadata, boxes, resolved_track_ids)
        gap_rows_before = _resample_gap_rows_total(self._decision)
        events = self._decision.update(decision_input)
        self._diagnostics.record_track_id_switch_absorbed_total(
            self.camera_id, self._track_id_switch_absorbed_total(self._decision)
        )
        gap_rows = _resample_gap_rows_total(self._decision) - gap_rows_before
        if gap_rows > 0:
            self._diagnostics.record_resample_gap_rows(self.camera_id, gap_rows)
        trigger = NativeEvidenceTrigger(
            self.camera_id,
            frame.identity.worker_boot_id,
            frame.identity.stream_epoch,
            metadata.source_generation,
            frame.identity.seq,
            frame.identity.source_pts or 0,
            metadata.source_time_ns / 1_000_000_000,
        )
        for position, event in enumerate(events):
            try:
                snapshot = self._control.snapshot(self.camera_id)
            except Exception as error:  # noqa: BLE001 - the alert outranks its thumbnail
                # A snapshot is optional evidence; an admitted safety event is
                # not. Admission has already consumed the onset, so a failure
                # here must degrade to "no snapshot", never drop the alert.
                snapshot = None
                LOGGER.warning(
                    "snapshot unavailable for camera_id=%s; staging the event without it: %s",
                    self.camera_id,
                    error,
                )
            try:
                self._sink.emit_for_frame(self._attacher.attach_native(event, snapshot), trigger)
            except Exception:
                # Admission has already consumed the onset and cooldown. A
                # durable staging failure must restore this event and every
                # admitted event that has not yet reached the sink.
                for pending in events[position:]:
                    self._decision.release(pending)
                raise
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

    def _record_track_id_switches(self, track_ids: tuple[int, ...]) -> None:
        current = set(track_ids)
        for track_id in current:
            if track_id not in self._live_track_misses and any(
                previous_id not in current and misses <= 30
                for previous_id, misses in self._live_track_misses.items()
            ):
                self._diagnostics.record_track_id_switch(self.camera_id)
            self._live_track_misses.pop(track_id, None)
        for track_id in tuple(self._live_track_misses):
            if track_id not in current:
                self._live_track_misses[track_id] += 1
                if self._live_track_misses[track_id] > 30:
                    self._live_track_misses.pop(track_id)
        for track_id in current:
            self._live_track_misses.setdefault(track_id, 0)

    def _capture_replay_row(
        self,
        metadata: MetadataFrame,
        boxes: tuple[BoundingBox, ...],
        track_ids: tuple[int, ...],
    ) -> None:
        """Keep best-effort trace persistence outside the frame-loop contract."""
        try:
            self._capture_replay_row_unchecked(metadata, boxes, track_ids)
        except (OSError, ValueError, RuntimeError):
            self._diagnostics.record_replay_trace_write_failure(self.camera_id)
            self._log_trace_write_failure()

    def _capture_replay_row_unchecked(
        self,
        metadata: MetadataFrame,
        boxes: tuple[BoundingBox, ...],
        track_ids: tuple[int, ...],
    ) -> None:
        if self._replay_trace is None:
            return
        frame = metadata.frame
        trace_source = _replay_trace_source(frame.association.strategy)
        self._trace_source = trace_source
        current_epoch = (frame.identity.stream_epoch, metadata.source_generation)
        source_event = (
            "open"
            if self._trace_epoch is None
            else "reconnect"
            if current_epoch != self._trace_epoch
            else "frame"
        )
        self._trace_epoch = current_epoch
        self._trace_source_lost = False
        if source_event != "frame":
            self._replay_trace.append(
                ReplayRow(
                    camera_id=self.camera_id,
                    seq=self._next_trace_seq(),
                    pts_ns=frame.identity.source_pts or 0,
                    epoch=frame.identity.stream_epoch,
                    source_event=source_event,
                    source=trace_source,
                    tracks=(),
                    bed_polygon_id=None,
                    bed_polygon=None,
                    bed_polygon_image_size=None,
                    night_window_active=False,
                    frame_width=metadata.source_width,
                    frame_height=metadata.source_height,
                )
            )
        poses = frame.human_pose.poses
        current: dict[int, ReplayTrack] = {}
        for index, (track_id, box) in enumerate(zip(track_ids, boxes, strict=True)):
            pose = poses[index]
            current[track_id] = ReplayTrack(
                track_id=track_id,
                lifecycle="new" if track_id not in self._trace_tracks else "tracked",
                bbox=_unit_bbox(box, metadata.source_width, metadata.source_height),
                keypoints=tuple(
                    (point.x / metadata.source_width, point.y / metadata.source_height, point.score)
                    for point in pose
                ),
            )
        non_observed: list[ReplayTrack] = []
        for track_id, previous in tuple(self._trace_tracks.items()):
            if track_id in current:
                continue
            lifecycle = "shadow" if track_id in frame.association.live_track_ids else "lost"
            non_observed.append(
                ReplayTrack(
                    track_id=track_id,
                    lifecycle=lifecycle,
                    bbox=previous.bbox,
                    keypoints=previous.keypoints,
                )
            )
            if lifecycle == "lost":
                self._trace_tracks.pop(track_id, None)
        self._trace_tracks.update(current)
        polygon = _persisted_polygon(
            self._scene,
            fallback_width=metadata.source_width,
            fallback_height=metadata.source_height,
        )
        polygon_image_size = _persisted_polygon_image_size(
            self._scene,
            fallback_width=metadata.source_width,
            fallback_height=metadata.source_height,
        )
        self._replay_trace.append(
            ReplayRow(
                camera_id=self.camera_id,
                seq=self._next_trace_seq(),
                pts_ns=frame.identity.source_pts or 0,
                epoch=frame.identity.stream_epoch,
                source_event="frame",
                source=trace_source,
                tracks=tuple(current.values()) + tuple(non_observed),
                bed_polygon_id="persisted" if polygon is not None else None,
                bed_polygon=polygon,
                bed_polygon_image_size=polygon_image_size,
                night_window_active=(
                    False if self._night_window_active is None else self._night_window_active()
                ),
                frame_width=metadata.source_width,
                frame_height=metadata.source_height,
            )
        )
        self._trace_dims = (metadata.source_width, metadata.source_height)
        self._trace_last_pts_ns = frame.identity.source_pts or 0

    def _capture_source_lost(self) -> None:
        if self._replay_trace is None or self._trace_epoch is None or self._trace_source_lost:
            return
        self._trace_source_lost = True
        try:
            self._replay_trace.append(
                ReplayRow(
                    camera_id=self.camera_id,
                    seq=self._next_trace_seq(),
                    pts_ns=self._trace_last_pts_ns,
                    epoch=self._trace_epoch[0],
                    source_event="lost",
                    source=_required_trace_source(self._trace_source),
                    tracks=(),
                    bed_polygon_id=None,
                    bed_polygon=None,
                    bed_polygon_image_size=None,
                    night_window_active=False,
                    frame_width=self._trace_dims[0],
                    frame_height=self._trace_dims[1],
                )
            )
        except (OSError, ValueError, RuntimeError):
            self._diagnostics.record_replay_trace_write_failure(self.camera_id)
            self._log_trace_write_failure()

    def _next_trace_seq(self) -> int:
        seq = self._trace_seq
        self._trace_seq += 1
        return seq

    def _log_trace_write_failure(self) -> None:
        if self._trace_write_failure_logged:
            return
        self._trace_write_failure_logged = True
        LOGGER.warning("replay trace write failed: camera_id=%s", self.camera_id, exc_info=True)

    def _record_bed_polygon_source(self, frame: MetadataFrame) -> None:
        self._diagnostics.record_bed_polygon_source(self.camera_id, self._scene.bed_polygon_source)


def _persisted_polygon(
    scene: SceneState, *, fallback_width: int, fallback_height: int
) -> tuple[tuple[float, float], ...] | None:
    if not scene.persisted_bed_regions:
        return None
    polygon = scene.persisted_bed_regions[0].polygon
    if polygon is None:
        return None
    width = scene.bed_zone_image_width or fallback_width
    height = scene.bed_zone_image_height or fallback_height
    return tuple((float(x) / width, float(y) / height) for x, y in polygon)


def _persisted_polygon_image_size(
    scene: SceneState, *, fallback_width: int, fallback_height: int
) -> tuple[int, int] | None:
    if not scene.persisted_bed_regions or scene.persisted_bed_regions[0].polygon is None:
        return None
    return (
        scene.bed_zone_image_width or fallback_width,
        scene.bed_zone_image_height or fallback_height,
    )


def _replay_trace_source(strategy: str) -> ReplaySource:
    if strategy == "nvdcf":
        return "nvdcf"
    if strategy == "legacy-greedy-bbox-iou.v1":
        return "legacy-association"
    raise ValueError(f"unsupported association strategy for replay trace: {strategy}")


def _required_trace_source(source: ReplaySource | None) -> ReplaySource:
    if source is None:
        raise RuntimeError("replay trace source is unavailable before an accepted frame")
    return source


def _unit_bbox(
    box: BoundingBox, width: int, height: int
) -> tuple[float, float, float, float, float]:
    return (
        box.x1 / width,
        box.y1 / height,
        box.x2 / width,
        box.y2 / height,
        box.confidence,
    )


def _resample_gap_rows_total(decision: EventAggregator) -> int:
    """Read the V2 adapter's cumulative count without coupling domains to telemetry."""
    total = 0
    for decider in decision.deciders:
        target: object = decider
        while hasattr(target, "decider"):
            target = target.decider
        count = getattr(target, "resample_gap_rows_total", None)
        if isinstance(count, int):
            total += count
    return total


__all__ = ["NativeEventSink", "NativePolicyContext", "NativePolicyPump"]
