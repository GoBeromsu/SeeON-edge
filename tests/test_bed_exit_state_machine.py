"""E1-E17 for the shadow bed-exit state machine (todo 7).

The new path records traces and ``would_trigger`` predicates. It emits
ZERO ``BusinessEvent``s. Legacy containment remains the only emitter.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import pytest

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains.bed_exit.detector import BedExitMonitor
from worker.domains.bed_exit.geometry import _bed_polygon_mask
from worker.domains.bed_exit.night_window import NightWindow
from worker.domains.bed_exit.schema import BedExitConfig
from worker.domains.bed_exit.state_machine import (
    BedExitState,
    BedExitStateMachine,
    classify_posture,
)
from worker.types import (
    BedPoseFeatures,
    DecisionInput,
    FrameBedPoseFeatures,
    TemporalProfile,
)

CAMERA_ID: Final = "camera-bed-exit-sm"
FACILITY_ID: Final = "facility-bed-exit-sm"
RESIDENT_ID: Final = 7
CAREGIVER_ID: Final = 8
PROFILE: Final = TemporalProfile(ingest_fps=5.0)

# Frame counts from seconds through TemporalProfile -- never hardcoded.
IN_BED_DWELL: Final = PROFILE.frames_for_seconds(2.0)
SITTING_UP_DWELL: Final = PROFILE.frames_for_seconds(0.6)
EDGE_DWELL: Final = PROFILE.frames_for_seconds(0.6)
OUT_DWELL: Final = PROFILE.frames_for_seconds(0.4)
UNCERTAIN_DWELL: Final = PROFILE.frames_for_seconds(1.0)
DEFAULT_DWELL: Final = PROFILE.frames_for_seconds(0.4)

BED: Final = BoundingBox(0, 0, 80, 100, 0.99)
IN_BED_BOX: Final = BoundingBox(10, 10, 70, 90, 0.95)
# Containment 0.50 against BED -- above production 0.35, so the legacy path
# never accumulates grace. Mirrors failure mode F1.
# Containment against BED (0,0)-(80,100): intersection 80x50 / person 80x100 = 0.50.
EDGE_BOX: Final = BoundingBox(0, 50, 80, 150, 0.94)
STANDING_BOX: Final = BoundingBox(120, 10, 180, 90, 0.94)

_REPO_ROOT: Final = Path(__file__).resolve().parents[1]


def _features(
    *,
    track_id: int = RESIDENT_ID,
    bed_id: int | None = 0,
    torso_in_frac: float = 1.0,
    lower_in_frac: float = 1.0,
    keypoint_in_frac: float = 1.0,
    hip_depth: float = 0.257,
    torso_angle: float = 1.571,
    centroid_displacement: float = 0.0,
    hip_x_rel: float = 0.5,
    hip_y_rel: float = 0.5,
    observability: float = 1.0,
    bed_polygon_valid: bool = True,
) -> BedPoseFeatures:
    return BedPoseFeatures(
        track_id=track_id,
        bed_id=bed_id,
        torso_in_frac=torso_in_frac,
        lower_in_frac=lower_in_frac,
        keypoint_in_frac=keypoint_in_frac,
        hip_depth=hip_depth,
        torso_angle=torso_angle,
        centroid_displacement=centroid_displacement,
        hip_x_rel=hip_x_rel,
        hip_y_rel=hip_y_rel,
        observability=observability,
        bed_polygon_valid=bed_polygon_valid,
    )


# Measured postures (todo 6 verifier). Rim-perched EDGE is THIS todo's fixture.
IN_BED_FEATURES: Final = _features(hip_depth=0.257, lower_in_frac=1.0, torso_in_frac=1.0)
SITTING_UP_FEATURES: Final = _features(hip_depth=0.236, lower_in_frac=1.0, torso_in_frac=1.0)
RIM_EDGE_FEATURES: Final = _features(
    torso_in_frac=1.0,
    lower_in_frac=0.0,
    keypoint_in_frac=0.5,
    hip_depth=0.043,
)
# Plan E1 / E2 numbers: EDGE vs SITTING_UP differ ONLY by lower_in_frac.
E1_EDGE_FEATURES: Final = _features(
    torso_in_frac=0.55,
    lower_in_frac=0.20,
    keypoint_in_frac=0.4,
    hip_depth=0.0,
)
E2_SITTING_FEATURES: Final = _features(
    torso_in_frac=0.55,
    lower_in_frac=0.80,
    keypoint_in_frac=0.7,
    hip_depth=0.0,
)
OUT_FEATURES: Final = _features(
    torso_in_frac=0.0,
    lower_in_frac=0.0,
    keypoint_in_frac=0.0,
    hip_depth=-0.289,
)
# Todo 6's mid-mattress "EDGE" fixture. Hips still inside; NOT rim-perched.
TODO6_MID_MATTRESS: Final = _features(
    torso_in_frac=0.5,
    lower_in_frac=0.0,
    keypoint_in_frac=0.25,
    hip_depth=0.233,
)
LOW_OBS_FEATURES: Final = _features(
    torso_in_frac=0.0,
    lower_in_frac=0.0,
    keypoint_in_frac=0.0,
    hip_depth=0.0,
    torso_angle=0.0,
    observability=0.18,
)
INVALID_POLYGON_FEATURES: Final = _features(
    bed_id=None,
    torso_in_frac=0.0,
    lower_in_frac=0.0,
    keypoint_in_frac=0.0,
    hip_depth=0.0,
    observability=1.0,
    bed_polygon_valid=False,
)
CAREGIVER_FEATURES: Final = _features(
    track_id=CAREGIVER_ID,
    bed_id=None,
    torso_in_frac=0.10,
    lower_in_frac=0.05,
    keypoint_in_frac=0.10,
    hip_depth=-0.10,
)


def _clock_at(hour: int = 22, minute: int = 0) -> Callable[[], datetime]:
    fixed = datetime(2026, 7, 31, hour, minute, tzinfo=ZoneInfo("Asia/Seoul"))
    return lambda: fixed


def _monitor(
    *,
    hold_frames: int = 1,
    grace_frames: int = 3,
    min_containment: float = 0.35,
    night_window: NightWindow | None = None,
    clock: Callable[[], datetime] | None = None,
    camera_id: str = CAMERA_ID,
) -> BedExitMonitor:
    return BedExitMonitor(
        config=BedExitConfig(
            camera_id=camera_id,
            facility_id=FACILITY_ID,
            min_containment=min_containment,
            hold_frames=hold_frames,
            grace_frames=grace_frames,
            night_window=night_window,
        ),
        clock=_clock_at() if clock is None else clock,
        temporal_profile=PROFILE,
    )


def _input(
    *,
    frame_index: int,
    features: tuple[BedPoseFeatures, ...] = (),
    person_boxes: tuple[BoundingBox, ...] | None = None,
    bed_boxes: tuple[BoundingBox, ...] = (BED,),
    track_ids: tuple[int, ...] | None = None,
    live_track_ids: tuple[int, ...] | None = None,
    source: BedRegionCacheState = BedRegionCacheState.FRESH,
) -> DecisionInput:
    if track_ids is None:
        track_ids = tuple(item.track_id for item in features) or (RESIDENT_ID,)
    if person_boxes is None:
        person_boxes = tuple(IN_BED_BOX for _ in track_ids)
    if live_track_ids is None:
        live_track_ids = track_ids
    return DecisionInput(
        observation=FrameObservation(
            detections=(person_boxes, ()),
            regions=(bed_boxes, ()),
            track_ids=track_ids,
        ),
        frame_width=200,
        frame_height=200,
        live_track_ids=live_track_ids,
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=source),
        bed_pose_features=FrameBedPoseFeatures(items=features),
    )


def _drive(
    monitor: BedExitMonitor,
    features: BedPoseFeatures,
    frames: int,
    *,
    start: int = 0,
    box: BoundingBox = IN_BED_BOX,
    live: bool = True,
) -> list[object]:
    events: list[object] = []
    for offset in range(frames):
        events.extend(
            monitor.update(
                _input(
                    frame_index=start + offset,
                    features=(features,),
                    person_boxes=(box,),
                    live_track_ids=(features.track_id,) if live else (),
                )
            )
        )
    return events


def _shadow_state(monitor: BedExitMonitor, track_id: int = RESIDENT_ID) -> BedExitState:
    return monitor.state_machine.track_state(track_id)


def _would_triggers(monitor: BedExitMonitor) -> tuple[bool, ...]:
    return tuple(item.would_trigger for item in monitor.last_shadow_decisions)


def test_dwell_seconds_convert_through_temporal_profile() -> None:
    assert IN_BED_DWELL == 10
    assert SITTING_UP_DWELL == 3
    assert EDGE_DWELL == 3
    assert OUT_DWELL == 2
    assert UNCERTAIN_DWELL == 5
    assert DEFAULT_DWELL == 2


def test_e1_edge_sitting_reaches_edge_with_zero_events_then_out_triggers() -> None:
    monitor = _monitor()
    events = _drive(monitor, E1_EDGE_FEATURES, 20, box=EDGE_BOX)
    assert events == []
    assert _shadow_state(monitor) is BedExitState.EDGE_SITTING
    assert all(item.snapshot.triggered is False for item in monitor.last_shadow_decisions)

    trigger_events = _drive(monitor, OUT_FEATURES, 2, start=20, box=EDGE_BOX)
    assert trigger_events == []
    assert _shadow_state(monitor) is BedExitState.OUT_OF_BED
    assert _would_triggers(monitor) == (True,)
    assert monitor.last_shadow_trace_snapshots[0].triggered is False


def test_e2_sitting_up_vs_edge_differ_only_by_lower_in_frac() -> None:
    assert E1_EDGE_FEATURES.torso_in_frac == E2_SITTING_FEATURES.torso_in_frac
    assert E1_EDGE_FEATURES.hip_depth == E2_SITTING_FEATURES.hip_depth
    assert E1_EDGE_FEATURES.lower_in_frac != E2_SITTING_FEATURES.lower_in_frac
    assert classify_posture(E1_EDGE_FEATURES) is BedExitState.EDGE_SITTING
    assert classify_posture(E2_SITTING_FEATURES) is BedExitState.SITTING_UP
    assert classify_posture(RIM_EDGE_FEATURES) is BedExitState.EDGE_SITTING
    assert classify_posture(IN_BED_FEATURES) is BedExitState.IN_BED
    assert classify_posture(SITTING_UP_FEATURES) is BedExitState.SITTING_UP
    assert classify_posture(OUT_FEATURES) is BedExitState.OUT_OF_BED


def test_e3_low_observability_is_uncertain_with_zero_events() -> None:
    monitor = _monitor()
    events = _drive(monitor, LOW_OBS_FEATURES, 60)
    assert events == []
    assert _shadow_state(monitor) is BedExitState.UNCERTAIN
    assert monitor.last_shadow_decisions
    decision = monitor.last_shadow_decisions[0]
    assert decision.current_state is BedExitState.UNCERTAIN
    assert decision.would_trigger is False
    assert RESIDENT_ID in monitor.state_machine.known_track_ids()
    assert monitor.last_debug_snapshot is not None
    assert monitor.last_debug_snapshot.statuses[0].occupancy == "occupied"
    assert monitor.last_debug_snapshot.statuses[0].person_id == RESIDENT_ID


def test_e4_brief_uncertain_does_not_leave_in_bed() -> None:
    monitor = _monitor()
    _drive(monitor, IN_BED_FEATURES, IN_BED_DWELL)
    assert _shadow_state(monitor) is BedExitState.IN_BED
    _drive(monitor, LOW_OBS_FEATURES, UNCERTAIN_DWELL - 1, start=IN_BED_DWELL)
    assert _shadow_state(monitor) is BedExitState.IN_BED
    _drive(
        monitor,
        IN_BED_FEATURES,
        2,
        start=IN_BED_DWELL + UNCERTAIN_DWELL - 1,
    )
    assert _shadow_state(monitor) is BedExitState.IN_BED
    assert monitor.last_shadow_decisions[0].would_trigger is False


def test_e5_caregiver_never_takes_resident_bed() -> None:
    monitor = _monitor()
    events: list[object] = []
    for frame_index in range(IN_BED_DWELL):
        events.extend(
            monitor.update(
                _input(
                    frame_index=frame_index,
                    features=(IN_BED_FEATURES, CAREGIVER_FEATURES),
                    person_boxes=(IN_BED_BOX, STANDING_BOX),
                    track_ids=(RESIDENT_ID, CAREGIVER_ID),
                )
            )
        )
    assert events == []
    assert monitor.last_debug_snapshot is not None
    assert monitor.last_debug_snapshot.statuses[0].person_id == RESIDENT_ID
    caregiver_decisions = [
        item for item in monitor.last_shadow_decisions if item.track_id == CAREGIVER_ID
    ]
    assert caregiver_decisions
    assert caregiver_decisions[0].bed_id is None
    assert _shadow_state(monitor, RESIDENT_ID) is BedExitState.IN_BED


def test_e6_resident_exit_with_moving_caregiver_triggers_once() -> None:
    monitor = _monitor()
    frame = 0
    triggers: list[int] = []
    collected: list[object] = []

    def step(features: tuple[BedPoseFeatures, ...], count: int) -> None:
        nonlocal frame
        for _ in range(count):
            caregiver = _features(
                track_id=CAREGIVER_ID,
                bed_id=None,
                torso_in_frac=0.10,
                lower_in_frac=0.05,
                keypoint_in_frac=0.10,
                hip_depth=-0.10 - 0.001 * (frame % 5),
            )
            collected.extend(
                monitor.update(
                    _input(
                        frame_index=frame,
                        features=(features[0], caregiver),
                        person_boxes=(IN_BED_BOX, STANDING_BOX),
                        track_ids=(RESIDENT_ID, CAREGIVER_ID),
                    )
                )
            )
            for decision in monitor.last_shadow_decisions:
                if decision.would_trigger:
                    triggers.append(decision.track_id)
            frame += 1

    step((IN_BED_FEATURES,), IN_BED_DWELL)
    step((SITTING_UP_FEATURES,), SITTING_UP_DWELL)
    step((RIM_EDGE_FEATURES,), EDGE_DWELL)
    step((OUT_FEATURES,), OUT_DWELL)
    assert collected == []
    assert triggers == [RESIDENT_ID]
    assert _shadow_state(monitor) is BedExitState.OUT_OF_BED


def test_e7_track_id_reuse_must_re_serve_in_bed_dwell() -> None:
    monitor = _monitor()
    _drive(monitor, IN_BED_FEATURES, IN_BED_DWELL)
    drop = monitor.update(
        _input(
            frame_index=IN_BED_DWELL,
            features=(),
            person_boxes=(IN_BED_BOX,),
            track_ids=(RESIDENT_ID,),
            live_track_ids=(),
        )
    )
    assert drop == ()
    assert _would_triggers(monitor) == (False,)
    assert _shadow_state(monitor) is BedExitState.ABSENT

    first_return = monitor.update(
        _input(frame_index=IN_BED_DWELL + 1, features=(IN_BED_FEATURES,))
    )
    assert first_return == ()
    assert _shadow_state(monitor) is BedExitState.ABSENT
    assert monitor.last_shadow_decisions[0].would_trigger is False
    _drive(monitor, IN_BED_FEATURES, IN_BED_DWELL - 2, start=IN_BED_DWELL + 2)
    assert _shadow_state(monitor) is BedExitState.ABSENT
    last = monitor.update(
        _input(
            frame_index=IN_BED_DWELL * 2,
            features=(IN_BED_FEATURES,),
        )
    )
    assert last == ()
    assert _shadow_state(monitor) is BedExitState.IN_BED


def test_e8_track_loss_from_edge_sitting_satisfies_trigger() -> None:
    monitor = _monitor()
    _drive(monitor, E1_EDGE_FEATURES, DEFAULT_DWELL, box=EDGE_BOX)
    assert _shadow_state(monitor) is BedExitState.EDGE_SITTING
    lost = monitor.update(
        _input(
            frame_index=DEFAULT_DWELL,
            features=(),
            person_boxes=(EDGE_BOX,),
            track_ids=(RESIDENT_ID,),
            live_track_ids=(),
        )
    )
    assert lost == ()
    assert _would_triggers(monitor) == (True,)
    assert monitor.last_shadow_decisions[0].previous_state is BedExitState.EDGE_SITTING
    assert monitor.last_shadow_decisions[0].current_state is BedExitState.ABSENT
    assert monitor.last_shadow_trace_snapshots[0].triggered is False


def test_e9_track_loss_from_in_bed_does_not_trigger() -> None:
    monitor = _monitor()
    _drive(monitor, IN_BED_FEATURES, IN_BED_DWELL)
    lost = monitor.update(
        _input(
            frame_index=IN_BED_DWELL,
            features=(),
            person_boxes=(IN_BED_BOX,),
            track_ids=(RESIDENT_ID,),
            live_track_ids=(),
        )
    )
    assert lost == ()
    assert _would_triggers(monitor) == (False,)


def test_e10_invalid_polygon_is_no_state_decision() -> None:
    assert classify_posture(INVALID_POLYGON_FEATURES) is None
    machine = BedExitStateMachine(temporal_profile=PROFILE)
    assert machine.observe(INVALID_POLYGON_FEATURES) is None
    assert machine.known_track_ids() == ()
    monitor = _monitor()
    events = _drive(monitor, INVALID_POLYGON_FEATURES, 10)
    assert events == []
    assert _shadow_state(monitor) is BedExitState.ABSENT
    assert monitor.last_shadow_decisions == ()
    assert monitor.last_shadow_trace_snapshots
    snapshot = monitor.last_shadow_trace_snapshots[0]
    assert snapshot.reason == "bed-polygon-invalid"
    assert snapshot.current_state == "no-decision"
    assert snapshot.triggered is False


def test_e11_degenerate_polygon_logs_once_across_100_calls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    polygon = ((13, 17), (31, 41), (49, 65))
    features = _features(bed_polygon_valid=False, hip_depth=0.0)
    machine = BedExitStateMachine(temporal_profile=PROFILE)
    with caplog.at_level(logging.WARNING, logger="worker.domains.bed_exit.geometry"):
        for _ in range(100):
            assert _bed_polygon_mask(polygon) is None
            assert machine.observe(features) is None
            assert classify_posture(features) is None
    records = [
        record
        for record in caplog.records
        if record.name == "worker.domains.bed_exit.geometry"
    ]
    assert len(records) == 1
    assert "falling back to AABB" in records[0].getMessage()


def test_e12_night_window_does_not_let_shadow_path_consume_latch() -> None:
    now = [datetime(2026, 7, 31, 13, 0, tzinfo=ZoneInfo("Asia/Seoul"))]
    monitor = _monitor(
        clock=lambda: now[0],
        night_window=NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        hold_frames=1,
        grace_frames=0,
        min_containment=0.5,
    )
    daytime = _drive(monitor, E1_EDGE_FEATURES, DEFAULT_DWELL, box=EDGE_BOX)
    daytime += _drive(monitor, OUT_FEATURES, OUT_DWELL, start=DEFAULT_DWELL, box=EDGE_BOX)
    assert daytime == []
    assert monitor._latch.event_count == 0  # noqa: SLF001
    assert _shadow_state(monitor) is BedExitState.OUT_OF_BED
    assert monitor.last_shadow_trace_snapshots

    now[0] = datetime(2026, 7, 31, 22, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    night_shadow = monitor.update(
        _input(frame_index=10, features=(OUT_FEATURES,), person_boxes=(EDGE_BOX,))
    )
    assert night_shadow == ()
    # Legacy path can still fire later: the shadow path did not consume the latch.
    legacy = monitor.update(
        _input(
            frame_index=11,
            features=(OUT_FEATURES,),
            person_boxes=(STANDING_BOX,),
        )
    )
    assert len(legacy) == 1
    assert legacy[0].event_type == "bed-exit"
    assert monitor.last_shadow_trace_snapshots[0].triggered is False


def test_e13_confirmed_out_of_bed_triggers_once_on_rising_edge() -> None:
    monitor = _monitor()
    _drive(monitor, E1_EDGE_FEATURES, DEFAULT_DWELL, box=EDGE_BOX)
    trigger_count = 0
    events: list[object] = []
    for offset in range(30):
        events.extend(
            monitor.update(
                _input(
                    frame_index=DEFAULT_DWELL + offset,
                    features=(OUT_FEATURES,),
                    person_boxes=(EDGE_BOX,),
                )
            )
        )
        trigger_count += sum(_would_triggers(monitor))
    assert events == []
    assert trigger_count == 1
    assert _shadow_state(monitor) is BedExitState.OUT_OF_BED


def test_e14_coast_returns_empty_and_holds_edge_sitting() -> None:
    monitor = _monitor()
    _drive(monitor, RIM_EDGE_FEATURES, DEFAULT_DWELL, box=EDGE_BOX)
    assert _shadow_state(monitor) is BedExitState.EDGE_SITTING
    for frame_index in range(10):
        assert monitor.coast(frame_index=100 + frame_index) == ()
    assert _shadow_state(monitor) is BedExitState.EDGE_SITTING
    assert monitor.state_machine.coast() == ()


def test_e15_monitor_instances_do_not_crosstalk() -> None:
    first = _monitor(camera_id="camera-a")
    second = _monitor(camera_id="camera-b")
    _drive(first, IN_BED_FEATURES, IN_BED_DWELL)
    _drive(second, E1_EDGE_FEATURES, DEFAULT_DWELL, box=EDGE_BOX)
    assert _shadow_state(first) is BedExitState.IN_BED
    assert _shadow_state(second) is BedExitState.EDGE_SITTING
    lost = second.update(
        _input(
            frame_index=20,
            features=(),
            person_boxes=(EDGE_BOX,),
            live_track_ids=(),
        )
    )
    assert lost == ()
    assert _would_triggers(second) == (True,)
    assert _shadow_state(first) is BedExitState.IN_BED
    assert first.state_machine.known_track_ids() == (RESIDENT_ID,)


def test_e16_bed_pose_features_are_plain_scalars_and_domains_have_no_numpy() -> None:
    allowed = {int, float, bool, type(None)}
    for item in fields(BedPoseFeatures):
        annotation = item.type
        names = str(annotation)
        assert any(
            candidate.__name__ in names for candidate in (int, float, bool)
        ), names
        sample = IN_BED_FEATURES
        value = getattr(sample, item.name)
        assert type(value) in allowed or (item.name == "bed_id" and value is None)
    offenders: list[str] = []
    for path in (_REPO_ROOT / "worker" / "domains").rglob("*.py"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith(("import numpy", "from numpy")):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line_no}")
    assert offenders == []


def test_e17_bed_exit_config_keeps_production_defaults() -> None:
    config = BedExitConfig(camera_id="c", facility_id="f")
    assert config.min_containment == 0.35
    assert config.hold_frames == 2
    assert config.grace_frames == 3


def test_new_path_emits_zero_business_events_even_when_trigger_predicate_fires() -> None:
    monitor = _monitor()
    events = _drive(monitor, RIM_EDGE_FEATURES, DEFAULT_DWELL, box=EDGE_BOX)
    events += _drive(monitor, OUT_FEATURES, OUT_DWELL, start=DEFAULT_DWELL, box=EDGE_BOX)
    events += monitor.update(
        _input(
            frame_index=20,
            features=(),
            person_boxes=(EDGE_BOX,),
            live_track_ids=(),
        )
    )
    assert events == []
    assert all(snapshot.triggered is False for snapshot in monitor.last_trace_snapshots)
    assert all(
        snapshot.triggered is False for snapshot in monitor.last_shadow_trace_snapshots
    )


def test_adversarial_todo6_mid_mattress_fixture_is_not_edge_sitting() -> None:
    assert classify_posture(TODO6_MID_MATTRESS) is not BedExitState.EDGE_SITTING
    machine = BedExitStateMachine(temporal_profile=PROFILE)
    for _ in range(20):
        decision = machine.observe(TODO6_MID_MATTRESS)
        assert decision is not None
        assert decision.current_state is not BedExitState.EDGE_SITTING
    assert machine.track_state(RESIDENT_ID) is BedExitState.ABSENT


def test_adversarial_reentry_after_absent_re_serves_in_bed_dwell() -> None:
    machine = BedExitStateMachine(temporal_profile=PROFILE)
    for _ in range(IN_BED_DWELL):
        machine.observe(IN_BED_FEATURES)
    assert machine.track_state(RESIDENT_ID) is BedExitState.IN_BED
    machine.mark_absent(RESIDENT_ID)
    assert machine.track_state(RESIDENT_ID) is BedExitState.ABSENT
    for _ in range(IN_BED_DWELL - 1):
        decision = machine.observe(IN_BED_FEATURES)
        assert decision is not None
        assert decision.current_state is BedExitState.ABSENT
        assert decision.would_trigger is False
    committed = machine.observe(IN_BED_FEATURES)
    assert committed is not None
    assert committed.current_state is BedExitState.IN_BED


def test_adversarial_malformed_input_does_not_crash_or_fabricate_state() -> None:
    machine = BedExitStateMachine(temporal_profile=PROFILE)
    assert machine.observe(INVALID_POLYGON_FEATURES) is None
    missing_pose = _features(
        observability=0.0,
        torso_in_frac=0.0,
        lower_in_frac=0.0,
        keypoint_in_frac=0.0,
        hip_depth=0.0,
        torso_angle=0.0,
        bed_id=None,
    )
    for _ in range(UNCERTAIN_DWELL):
        decision = machine.observe(missing_pose)
        assert decision is not None
    assert machine.track_state(RESIDENT_ID) is BedExitState.UNCERTAIN
    assert classify_posture(INVALID_POLYGON_FEATURES) is None
    assert classify_posture(missing_pose) is BedExitState.UNCERTAIN


def test_legacy_path_still_emits_when_containment_rule_fires() -> None:
    """Shadow isolation: disabling nothing, legacy still works; shadow stays silent."""
    monitor = _monitor(min_containment=0.5, hold_frames=1, grace_frames=0)
    inside = monitor.update(_input(frame_index=0, features=(), person_boxes=(IN_BED_BOX,)))
    outside = monitor.update(_input(frame_index=1, features=(), person_boxes=(STANDING_BOX,)))
    assert inside == ()
    assert len(outside) == 1
    assert outside[0].event_type == "bed-exit"
    assert monitor.last_shadow_decisions == ()
