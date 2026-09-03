"""Downstream fall/bed-exit event timelines through the typed seam.

Drives the SAME typed seam (`AssociationStrategy.observe(identity,
person_box) -> AssociationResult`) with both the Python oracle
(`GreedyIouTracker`) and the native candidate, and asserts the resulting
`BusinessEvent` timelines are byte-equal for a fall fixture and a bed-exit
fixture.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains.bed_exit import BedExitConfig, BedExitMonitor, NightWindow
from worker.domains.fall import FallPolicyDeciderV2, FallV2DomainDecider, FallWindowClassifierV2
from worker.interfaces.fall_model import FallV2Probabilities
from worker.native.deepstream.association import build_active_association_strategy
from worker.pipeline.perception.tracker import GreedyIouTracker
from worker.types import BusinessEvent, DecisionInput
from worker.types.perception_frame import (
    ChannelState,
    PerceptionFrameIdentity,
    PersonBox,
    PersonBoxChannel,
)

_IDENTITY = PerceptionFrameIdentity(
    worker_boot_id="boot-1", camera_id="camera-1", stream_epoch=0, seq=0
)


def _box(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


def _person_box_channel(boxes: tuple[BoundingBox, ...]) -> PersonBoxChannel:
    state = ChannelState.INFERRED if boxes else ChannelState.INFERRED_EMPTY
    cues = tuple(
        PersonBox(x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2, confidence=box.confidence)
        for box in boxes
    )
    return PersonBoxChannel(state=state, boxes=cues)


_FRAME_SEC = 1 / 15
# V2 needs a 30-row window and three stride-5 votes: the onset lands at row 40.
_FRAMES = 41


class _FallModel:
    def predict(self, features: object) -> FallV2Probabilities:
        del features
        return FallV2Probabilities(background=0.1, fall_transition=0.9, fallen=0.0)


def _fall_decider() -> FallV2DomainDecider:
    return FallV2DomainDecider(
        classifier=FallWindowClassifierV2(_FallModel()),
        policy=FallPolicyDeciderV2(
            camera_id="camera-1",
            facility_id="facility-1",
            boot_id="boot-1",
            stream_epoch="0",
            source_generation=0,
        ),
    )


def _fall_decision_input(
    track_ids: tuple[int, ...], boxes: tuple[BoundingBox, ...], frame_index: int
) -> DecisionInput:
    pose = tuple((90, 50, 0.9) for _ in range(17))
    return DecisionInput(
        observation=FrameObservation(detections=(boxes, ()), poses=(pose,), track_ids=track_ids),
        frame_width=200,
        frame_height=200,
        live_track_ids=track_ids,
        time_sec=frame_index * _FRAME_SEC,
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def test_fall_event_timeline_is_identical_whether_oracle_or_native_supplies_track_ids() -> None:
    boxes = (_box(70, 20, 110, 130),)
    oracle = GreedyIouTracker()
    native = build_active_association_strategy()
    oracle_decider = _fall_decider()
    native_decider = _fall_decider()

    oracle_events: list[tuple[BusinessEvent, ...]] = []
    native_events: list[tuple[BusinessEvent, ...]] = []
    for frame_index in range(_FRAMES):
        oracle_track_ids = oracle.observe(boxes)
        native_result = native.observe(_IDENTITY, _person_box_channel(boxes))
        oracle_events.append(
            oracle_decider.update(_fall_decision_input(oracle_track_ids, boxes, frame_index))
        )
        native_events.append(
            native_decider.update(_fall_decision_input(native_result.track_ids, boxes, frame_index))
        )

    assert tuple(oracle_events) == tuple(native_events)
    assert any(events for events in oracle_events), "fixture must actually raise a fall event"


def _bed_exit_monitor() -> BedExitMonitor:
    return BedExitMonitor(
        config=BedExitConfig(
            camera_id="camera-1",
            facility_id="facility-1",
            min_containment=0.5,
            hold_frames=1,
            grace_frames=0,
            night_window=NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=lambda: datetime(2026, 7, 31, 22, 0, tzinfo=ZoneInfo("Asia/Seoul")),
    )


def _bed_exit_decision_input(
    track_ids: tuple[int, ...],
    boxes: tuple[BoundingBox, ...],
    bed: BoundingBox,
    frame_index: int,
) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=(boxes, ()),
            regions=((bed,), ()),
            track_ids=track_ids,
        ),
        frame_width=300,
        frame_height=300,
        live_track_ids=track_ids,
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH),
    )


def test_bed_exit_event_timeline_is_identical_whether_oracle_or_native_supplies_track_ids() -> None:
    bed = _box(0, 0, 80, 100)
    in_bed = _box(10, 10, 70, 90)
    step = _box(10, 30, 70, 110)
    outside = _box(10, 70, 70, 150)
    frames = (in_bed, step, outside)

    oracle = GreedyIouTracker()
    native = build_active_association_strategy()
    oracle_monitor = _bed_exit_monitor()
    native_monitor = _bed_exit_monitor()

    oracle_events: list[tuple[BusinessEvent, ...]] = []
    native_events: list[tuple[BusinessEvent, ...]] = []
    for frame_index, box in enumerate(frames):
        oracle_track_ids = oracle.observe((box,))
        native_result = native.observe(_IDENTITY, _person_box_channel((box,)))
        oracle_events.append(
            oracle_monitor.update(
                _bed_exit_decision_input(oracle_track_ids, (box,), bed, frame_index)
            )
        )
        native_events.append(
            native_monitor.update(
                _bed_exit_decision_input(native_result.track_ids, (box,), bed, frame_index)
            )
        )

    assert tuple(oracle_events) == tuple(native_events)
    assert any(events for events in oracle_events), "fixture must actually raise a bed-exit event"
