from __future__ import annotations

from datetime import UTC, datetime

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from shared.detection_policies import FallPolicyV2
from worker.domains.bed_exit import BedExitConfig, BedExitMonitor
from worker.domains.fall import (
    FallDomainDecider,
    FallPolicyDecider,
    FallProbabilities,
    FallWindowClassifier,
)
from worker.types import DecisionInput
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


def test_bed_exit_trace_distinguishes_live_grace_from_stale_track_exit() -> None:
    detector = BedExitMonitor(
        config=BedExitConfig(
            camera_id="camera-bed",
            facility_id="facility-bed",
            min_containment=0.5,
            hold_frames=1,
            grace_frames=2,
        ),
        clock=lambda: datetime(2026, 8, 13, tzinfo=UTC),
        boot_id="test-boot",
        stream_epoch="test-epoch",
        source_generation=0,
    )
    inside = BoundingBox(10, 10, 70, 90, 0.9)
    outside = BoundingBox(100, 10, 160, 90, 0.9)
    assert detector.update(_input(inside, frame_index=0)) == ()
    assert detector.update(_input(outside, frame_index=1)) == ()
    live_trace = detector.last_trace_snapshots[0]

    events = detector.update(_input(outside, frame_index=2, live=()))

    assert live_trace.reason == "live-grace"
    assert live_trace.values["containment_ratio"] == 0.0
    assert live_trace.values["grace_frames_before"] == 0
    assert live_trace.values["grace_frames_after"] == 1
    assert len(events) == 1
    stale_trace = detector.last_trace_snapshots[0]
    assert stale_trace.reason == "stale-track-exit"
    assert stale_trace.previous_state == "live-grace"
    assert stale_trace.current_state == "triggered"
    assert stale_trace.values["grace_frames_before"] == 1
    assert stale_trace.values["grace_threshold"] == 2


def test_fall_gap_ticks_never_feed_a_zero_row_to_the_model() -> None:
    """A PTS gap must make a live track coast on its last real row, never the
    domain's all-zero "no detection" sentinel -- a gap is not a teleport."""
    model = _RecordingModel()
    detector = _classifier_fall(model, camera_id="camera-a")
    person = BoundingBox(10, 10, 70, 90, 0.9)

    for frame_index in range(30):
        detector.update(_input(person, frame_index=frame_index, time_sec=frame_index * 0.0667))
    assert len(model.inputs) == 1

    # Jump forward five cadence buckets in one call: four gap rows plus one
    # valid row lands exactly on the next stride-due tick.
    detector.update(_input(person, frame_index=30, time_sec=35 * 0.0667))

    assert detector.resample_gap_rows_total > 0
    assert len(model.inputs) == 2
    for window in model.inputs:
        for row in window:
            assert any(value != 0.0 for value in row)


def test_numeric_decision_trace_is_hardware_neutral_for_equal_inputs() -> None:
    cpu = _traceable_fall(camera_id="camera-a", facility_id="f")
    nvidia = _traceable_fall(camera_id="camera-a", facility_id="f")
    input_value = _input(BoundingBox(10, 10, 70, 90, 0.9), frame_index=1)

    assert cpu.update(input_value) == nvidia.update(input_value)
    assert cpu.last_trace_snapshots == nvidia.last_trace_snapshots
