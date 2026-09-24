from __future__ import annotations

from datetime import UTC, datetime

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from shared.detection_policies import FallPolicyV2
from tests_support.bed_pose_fixtures import frame_pose_features, lying_in_bed
from worker.domains.bed_exit import BedExitConfig, BedExitMonitor
from worker.domains.fall import (
    FallDomainDecider,
    FallPolicyDecider,
    FallProbabilities,
    FallWindowClassifier,
)
from worker.types import DecisionInput
from worker.types.bed_pose_features import EMPTY_FRAME_BED_POSE_FEATURES, FrameBedPoseFeatures
from worker.types.trace import DecisionTraceMissingReason


class _ImmediateClassifier:
    """Scores every live track on every call, so no track is ever "not scored"."""

    def __init__(self) -> None:
        self.current_call_missing_score_reasons: dict[int, str] = {}

    def update(
        self, _rows: object, live_track_ids: tuple[int, ...]
    ) -> dict[int, FallProbabilities]:
        return {
            track_id: FallProbabilities(background=0.1, fall_transition=0.8, fallen=0.1)
            for track_id in live_track_ids
        }


class _RecordingModel:
    def __init__(self) -> None:
        self.inputs: list[object] = []

    def predict(self, features: object) -> FallProbabilities:
        self.inputs.append(features)
        return FallProbabilities(background=0.1, fall_transition=0.8, fallen=0.1)


def _traceable_fall(*, camera_id: str, facility_id: str) -> FallDomainDecider:
    """The production decider records compiled-vocabulary trace snapshots itself."""
    return FallDomainDecider(
        classifier=_ImmediateClassifier(),
        policy=FallPolicyDecider(
            camera_id=camera_id,
            facility_id=facility_id,
            boot_id="boot-a",
            stream_epoch="1",
            source_generation=0,
            policy=FallPolicyV2(transition_votes=1),
        ),
    )


def _classifier_fall(model: _RecordingModel, *, camera_id: str) -> FallDomainDecider:
    return FallDomainDecider(
        classifier=FallWindowClassifier(model),
        policy=FallPolicyDecider(
            camera_id=camera_id,
            facility_id="facility-a",
            boot_id="boot-a",
            stream_epoch="1",
            source_generation=0,
            policy=FallPolicyV2(transition_votes=1),
        ),
    )


def _input(
    person: BoundingBox,
    *,
    frame_index: int,
    live: tuple[int, ...] = (9,),
    time_sec: float | None = None,
    bed_pose_features: FrameBedPoseFeatures = EMPTY_FRAME_BED_POSE_FEATURES,
) -> DecisionInput:
    bed = BoundingBox(0, 0, 80, 100, 0.9)
    pose = tuple((index + 1, index + 2, 0.9) for index in range(17))
    observation = FrameObservation(
        detections=((person,), ()),
        poses=(pose,),
        regions=((bed,), ()),
        track_ids=(9,),
    )
    return DecisionInput(
        observation=observation,
        frame_width=180,
        frame_height=120,
        live_track_ids=live,
        time_sec=float(frame_index) if time_sec is None else time_sec,
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        bed_pose_features=bed_pose_features,
    )


def test_fall_trace_records_transition_confirmation() -> None:
    detector = _traceable_fall(camera_id="camera-a", facility_id="facility-a")

    # transition_votes=1: the first qualifying frame confirms the transition.
    events = detector.update(_input(BoundingBox(10, 10, 70, 90, 0.9), frame_index=1))

    assert len(events) == 1
    trace = detector.last_trace_snapshots[0]
    assert trace.reason == "transition-confirmed"
    assert trace.previous_state == "clear"
    assert trace.current_state == "transition-confirmed"
    assert trace.track_id == 9
    assert trace.values == {
        "fall_transition_probability": 0.8,
        "fallen_probability": 0.1,
        "transition_threshold": 0.5,
        "transition_votes": 1,
        "transition_window": 5,
    }


def test_fall_trace_links_current_classifier_warmup_and_stride_dispositions() -> None:
    model = _RecordingModel()
    detector = _classifier_fall(model, camera_id="camera-a")
    person = BoundingBox(10, 10, 70, 90, 0.9)

    for frame_index in range(5):
        assert (
            detector.update(
                _input(
                    person,
                    frame_index=frame_index,
                    time_sec=frame_index * 0.0667,
                )
            )
            == ()
        )
    warmup = detector.last_trace_snapshots[0]
    assert warmup.reason == "score-missing"
    assert warmup.missing_values == {
        "fall_transition_probability": DecisionTraceMissingReason.CLASSIFIER_WARMUP
    }
    assert model.inputs == []

    events = ()
    for frame_index in range(5, 30):
        events = detector.update(
            _input(
                person,
                frame_index=frame_index,
                time_sec=frame_index * 0.0667,
            )
        )
    assert len(events) == 1
    assert events[0].probability == 0.8
    assert len(model.inputs) == 1
    scored = detector.last_trace_snapshots[0]
    assert scored.reason == "transition-confirmed"
    assert scored.values["fall_transition_probability"] == 0.8

    assert detector.update(_input(person, frame_index=30, time_sec=30 * 0.0667)) == ()
    stride = detector.last_trace_snapshots[0]
    assert stride.reason == "score-missing"
    assert stride.current_state == "transition-confirmed"
    assert stride.missing_values == {
        "fall_transition_probability": (DecisionTraceMissingReason.CLASSIFIER_STRIDE_NOT_DUE)
    }
    assert len(model.inputs) == 1


def test_fall_trace_uses_valid_row_disposition_across_gaps_and_pts_reset() -> None:
    model = _RecordingModel()
    detector = _classifier_fall(model, camera_id="camera-a")
    person = BoundingBox(10, 10, 70, 90, 0.9)

    assert detector.update(_input(person, frame_index=0, time_sec=0.0)) == ()
    assert detector.update(_input(person, frame_index=1, time_sec=4 * 0.0667)) == ()
    assert detector.resample_gap_rows_total == 3
    gap_then_valid = detector.last_trace_snapshots
    assert gap_then_valid[0].missing_values == {
        "fall_transition_probability": DecisionTraceMissingReason.CLASSIFIER_WARMUP
    }

    assert detector.update(_input(person, frame_index=2, time_sec=0.26681)) == ()
    assert detector.last_trace_snapshots is gap_then_valid
    assert detector.resample_gap_rows_total == 3
    assert model.inputs == []

    classifier_before_reset = detector.classifier
    assert detector.update(_input(person, frame_index=3, time_sec=0.0)) == ()
    assert detector.classifier is not classifier_before_reset
    reset = detector.last_trace_snapshots[0]
    assert reset.missing_values == {
        "fall_transition_probability": (DecisionTraceMissingReason.CLASSIFIER_STRIDE_NOT_DUE)
    }
    assert model.inputs == []


def test_fall_classifier_dispositions_are_camera_local_with_a_shared_model() -> None:
    model = _RecordingModel()
    camera_a = _classifier_fall(model, camera_id="camera-a")
    camera_b = _classifier_fall(model, camera_id="camera-b")
    person = BoundingBox(10, 10, 70, 90, 0.9)

    camera_a_events = ()
    for frame_index in range(30):
        camera_a_events = camera_a.update(
            _input(
                person,
                frame_index=frame_index,
                time_sec=frame_index * 0.0667,
            )
        )
    for frame_index in range(5):
        assert (
            camera_b.update(
                _input(
                    person,
                    frame_index=frame_index,
                    time_sec=frame_index * 0.0667,
                )
            )
            == ()
        )

    assert len(camera_a_events) == 1
    assert camera_a.last_trace_snapshots[0].reason == "transition-confirmed"
    assert camera_b.last_trace_snapshots[0].missing_values == {
        "fall_transition_probability": DecisionTraceMissingReason.CLASSIFIER_WARMUP
    }
    assert len(model.inputs) == 1


def test_bed_exit_trace_distinguishes_outside_dwell_exit_from_stale_track_clear() -> None:
    """Absence never emits; only a live, posture-confirmed dwell can arm and exit.

    Supersedes the deleted shadow state machine's assertion that a track
    disappearing one frame after leaving the bed ("live-grace" ->
    "stale-track-exit") should fire -- that was exactly the absence-emit bug
    (issue: bed-exit firehose). Under the dwell model: a track must be
    observed lying/sitting in its own bed for `in_bed_dwell_sec` before it is
    armed, then observed outside continuously for `outside_dwell_sec` on
    live frames before it fires; a track vanishing is retired silently
    ("stale-track-clear", triggered=False) regardless of how far its dwell
    timers had climbed.
    """
    detector = BedExitMonitor(
        config=BedExitConfig(
            camera_id="camera-bed",
            facility_id="facility-bed",
            min_containment=0.5,
            hold_frames=1,
            grace_frames=2,
            in_bed_dwell_sec=1.0,
            outside_dwell_sec=1.0,
        ),
        clock=lambda: datetime(2026, 8, 13, tzinfo=UTC),
        boot_id="test-boot",
        stream_epoch="test-epoch",
        source_generation=0,
    )
    inside = BoundingBox(10, 10, 70, 90, 0.9)
    outside = BoundingBox(100, 10, 160, 90, 0.9)
    lying = frame_pose_features(lying_in_bed(track_id=9, bed_id=0))

    # Frame 0: assigns to bed 0 (hold_frames=1); no dwell math on the
    # assignment frame itself.
    assert detector.update(_input(inside, frame_index=0, time_sec=0.0)) == ()
    # Frame 1: still inside, posture-confirmed for a full in_bed_dwell_sec ->
    # arms. Never triggers by itself.
    assert (
        detector.update(_input(inside, frame_index=1, time_sec=1.0, bed_pose_features=lying)) == ()
    )
    armed_trace = detector.last_trace_snapshots[0]
    assert armed_trace.reason == "contained"
    assert armed_trace.current_state == "armed"
    assert armed_trace.values["in_bed_dwell_sec"] == 1.0
    assert armed_trace.values["in_bed_dwell_threshold_sec"] == 1.0

    # Frame 2: outside for a full outside_dwell_sec while armed -> fires
    # exactly one event.
    events = detector.update(_input(outside, frame_index=2, time_sec=2.0))
    assert len(events) == 1
    exit_trace = detector.last_trace_snapshots[0]
    assert exit_trace.reason == "outside-dwell-exit"
    assert exit_trace.previous_state == "armed"
    assert exit_trace.current_state == "triggered"
    assert exit_trace.values["outside_dwell_sec"] == 1.0
    assert exit_trace.values["outside_dwell_threshold_sec"] == 1.0

    # The same track returns to bed but is never observed long enough to
    # re-arm (no posture evidence this time), then disappears entirely.
    # Absence must not fire -- the track is simply retired.
    assert detector.update(_input(inside, frame_index=3, time_sec=3.0)) == ()
    assert (
        detector.update(_input(outside, frame_index=4, time_sec=4.0)) == ()
    )
    disappeared = detector.update(_input(outside, frame_index=5, time_sec=5.0, live=()))
    assert disappeared == ()
    stale_trace = detector.last_trace_snapshots[0]
    assert stale_trace.reason == "stale-track-clear"
    assert stale_trace.current_state == "retired"
    assert stale_trace.missing_values == {"containment_ratio": "track-no-longer-live"}


def test_numeric_decision_trace_is_hardware_neutral_for_equal_inputs() -> None:
    cpu = _traceable_fall(camera_id="camera-a", facility_id="f")
    nvidia = _traceable_fall(camera_id="camera-a", facility_id="f")
    input_value = _input(BoundingBox(10, 10, 70, 90, 0.9), frame_index=1)

    assert cpu.update(input_value) == nvidia.update(input_value)
    assert cpu.last_trace_snapshots == nvidia.last_trace_snapshots
