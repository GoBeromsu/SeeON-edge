"""BedPoseFeatures contract, perception producer, and DecisionInput wiring.

Characterization of the pre-todo-6 DecisionInput constructor lives at the
top so it remains meaningful after the new field is added: existing call
sites must still build with the original seven arguments.
"""

from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import pytest

from contracts.model import DEFAULT_FALL_CONFIDENCE_THRESHOLD
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.pipeline.perception.decision_input import build_decision_input
from worker.pipeline.perception.features.bed_geometry import (
    compute_bed_pose_features,
    compute_frame_bed_pose_features,
)
from worker.pipeline.perception.scene_state import SceneState
from worker.types.bed_pose_features import BedPoseFeatures, FrameBedPoseFeatures
from worker.types.decision_input import DecisionInput

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Worked geometry in a single 1920x1080 frame. The bed is an axis-aligned
# rectangle so inside/outside is unambiguous for a human reading the QA
# numbers. Torso = shoulders 5/6 + hips 11/12; lower = knees 13/14 +
# ankles 15/16.
_BED_X1, _BED_Y1, _BED_X2, _BED_Y2 = 400, 300, 1200, 800
_BED_POLYGON_1080 = (
    (_BED_X1, _BED_Y1),
    (_BED_X2, _BED_Y1),
    (_BED_X2, _BED_Y2),
    (_BED_X1, _BED_Y2),
)
_FRAME_W, _FRAME_H = 1920, 1080
_POSE_W, _POSE_H = 3840, 2160  # 2x the bed-zone image, the mismatch case
_CONF = 0.9
_COCO17 = 17


def _seven_arg_decision_input() -> DecisionInput:
    """The construction shape every existing test builder uses today."""
    return DecisionInput(
        observation=FrameObservation(),
        frame_width=1920,
        frame_height=1080,
        live_track_ids=(1,),
        time_sec=1.0,
        frame_index=4,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.EMPTY),
    )


def test_decision_input_still_constructs_with_original_seven_arguments() -> None:
    """Characterization: pre-existing call sites must keep compiling unchanged."""
    decision_input = _seven_arg_decision_input()
    names = tuple(item.name for item in fields(DecisionInput))
    assert names[:7] == (
        "observation",
        "frame_width",
        "frame_height",
        "live_track_ids",
        "time_sec",
        "frame_index",
        "bed_region",
    )
    assert decision_input.frame_width == 1920
    assert decision_input.frame_height == 1080
    assert decision_input.live_track_ids == (1,)
    assert decision_input.time_sec == 1.0
    assert decision_input.frame_index == 4
    assert decision_input.bed_region.source is BedRegionCacheState.EMPTY


def test_build_decision_input_does_not_require_a_new_argument() -> None:
    """Characterization: the perception builder keeps its existing signature."""
    observation = FrameObservation()
    scene = SceneState(camera_id="cam-char")
    decision_input = build_decision_input(
        observation,
        frame_width=640,
        frame_height=480,
        live_track_ids=(),
        time_sec=None,
        frame_index=0,
        scene_state=scene,
        bed_scheduled=False,
        bed_interval=30,
    )
    assert decision_input.frame_width == 640
    assert decision_input.frame_height == 480
    assert decision_input.live_track_ids == ()
    assert decision_input.observation.bed_boxes == ()


def test_bed_pose_features_is_a_plain_scalar_contract() -> None:
    """Domain-facing fields stay stdlib scalars; numpy never crosses this type."""
    names = tuple(item.name for item in fields(BedPoseFeatures))
    assert names == (
        "track_id",
        "bed_id",
        "torso_in_frac",
        "lower_in_frac",
        "keypoint_in_frac",
        "hip_depth",
        "torso_angle",
        "centroid_displacement",
        "hip_x_rel",
        "hip_y_rel",
        "observability",
        "bed_polygon_valid",
    )
    features = BedPoseFeatures(
        track_id=7,
        bed_id=0,
        torso_in_frac=0.9,
        lower_in_frac=0.8,
        keypoint_in_frac=0.85,
        hip_depth=0.1,
        torso_angle=0.2,
        centroid_displacement=0.0,
        hip_x_rel=0.5,
        hip_y_rel=0.5,
        observability=1.0,
        bed_polygon_valid=True,
    )
    assert features.track_id == 7
    assert features.bed_id == 0
    assert features.bed_polygon_valid is True
    frame = FrameBedPoseFeatures(items=(features,))
    assert frame.items == (features,)
    assert FrameBedPoseFeatures().items == ()


def _pose(
    points: dict[int, tuple[float, float]],
    *,
    scale: float = 1.0,
    confidence: float = _CONF,
) -> tuple[tuple[int, int, float], ...]:
    keypoints: list[tuple[int, int, float]] = []
    for index in range(_COCO17):
        if index in points:
            x_coordinate, y_coordinate = points[index]
            keypoints.append(
                (int(round(x_coordinate * scale)), int(round(y_coordinate * scale)), confidence)
            )
        else:
            keypoints.append((0, 0, 0.0))
    return tuple(keypoints)


def _bed_box(
    polygon: tuple[tuple[int, int], ...] = _BED_POLYGON_1080,
) -> BoundingBox:
    xs = tuple(point[0] for point in polygon)
    ys = tuple(point[1] for point in polygon)
    return BoundingBox(
        x1=min(xs),
        y1=min(ys),
        x2=max(xs),
        y2=max(ys),
        confidence=1.0,
        polygon=polygon,
    )


# Lying fully inside the bed: shoulders and hips well inside the rectangle,
# knees and ankles still on the mattress.
_IN_BED_POINTS: dict[int, tuple[float, float]] = {
    5: (700.0, 420.0),
    6: (900.0, 420.0),
    11: (720.0, 560.0),
    12: (880.0, 560.0),
    13: (730.0, 680.0),
    14: (870.0, 680.0),
    15: (740.0, 760.0),
    16: (860.0, 760.0),
}
# Hips still on the mattress; shoulders above the headboard and legs on the
# floor. torso_in_frac and lower_in_frac both drop versus IN_BED.
_EDGE_SITTING_POINTS: dict[int, tuple[float, float]] = {
    5: (700.0, 220.0),
    6: (900.0, 220.0),
    11: (720.0, 520.0),
    12: (880.0, 520.0),
    13: (730.0, 920.0),
    14: (870.0, 920.0),
    15: (740.0, 1000.0),
    16: (860.0, 1000.0),
}
# Entire body to the right of the bed.
_OUT_OF_BED_POINTS: dict[int, tuple[float, float]] = {
    5: (1400.0, 360.0),
    6: (1600.0, 360.0),
    11: (1420.0, 520.0),
    12: (1580.0, 520.0),
    13: (1430.0, 700.0),
    14: (1570.0, 700.0),
    15: (1440.0, 880.0),
    16: (1560.0, 880.0),
}


def _features_for(
    points: dict[int, tuple[float, float]],
    *,
    scale: float = 1.0,
    frame_width: int = _FRAME_W,
    frame_height: int = _FRAME_H,
    polygon_image_width: int | None = None,
    polygon_image_height: int | None = None,
    bed_boxes: tuple[BoundingBox, ...] | None = None,
    confidence: float = _CONF,
) -> BedPoseFeatures:
    return compute_bed_pose_features(
        track_id=1,
        pose=_pose(points, scale=scale, confidence=confidence),
        bed_boxes=bed_boxes if bed_boxes is not None else (_bed_box(),),
        frame_width=frame_width,
        frame_height=frame_height,
        polygon_image_width=polygon_image_width,
        polygon_image_height=polygon_image_height,
    )


def _assert_bed_relative_zero(features: BedPoseFeatures) -> None:
    assert features.bed_polygon_valid is False
    assert features.bed_id is None
    assert features.torso_in_frac == 0.0
    assert features.lower_in_frac == 0.0
    assert features.keypoint_in_frac == 0.0
    assert features.hip_depth == 0.0
    assert features.centroid_displacement == 0.0
    assert features.hip_x_rel == 0.0
    assert features.hip_y_rel == 0.0


def test_in_bed_and_edge_sitting_separate_on_lower_body() -> None:
    in_bed = _features_for(_IN_BED_POINTS)
    edge = _features_for(_EDGE_SITTING_POINTS)
    out = _features_for(_OUT_OF_BED_POINTS)
    assert in_bed.bed_polygon_valid is True
    assert edge.bed_polygon_valid is True
    assert in_bed.torso_in_frac == pytest.approx(1.0)
    assert in_bed.lower_in_frac == pytest.approx(1.0)
    assert edge.torso_in_frac == pytest.approx(0.5)
    assert edge.lower_in_frac == pytest.approx(0.0)
    assert in_bed.torso_in_frac > edge.torso_in_frac
    assert in_bed.lower_in_frac > edge.lower_in_frac
    assert out.torso_in_frac == pytest.approx(0.0)
    assert out.lower_in_frac == pytest.approx(0.0)
    assert out.hip_depth < 0.0
    assert in_bed.hip_depth > 0.0
    assert edge.torso_angle > 0.87
    assert in_bed.observability == pytest.approx(8.0 / 17.0)


def test_polygon_frame_scale_mismatch_is_corrected() -> None:
    """1920x1080 polygon + 3840x2160 pose must equal the same geometry in one frame."""
    matched = _features_for(
        _IN_BED_POINTS,
        frame_width=_FRAME_W,
        frame_height=_FRAME_H,
    )
    mismatched = _features_for(
        _IN_BED_POINTS,
        scale=2.0,
        frame_width=_POSE_W,
        frame_height=_POSE_H,
        polygon_image_width=_FRAME_W,
        polygon_image_height=_FRAME_H,
    )
    unscaled = _features_for(
        _IN_BED_POINTS,
        scale=2.0,
        frame_width=_POSE_W,
        frame_height=_POSE_H,
    )
    assert mismatched.torso_in_frac == pytest.approx(matched.torso_in_frac)
    assert mismatched.lower_in_frac == pytest.approx(matched.lower_in_frac)
    assert mismatched.keypoint_in_frac == pytest.approx(matched.keypoint_in_frac)
    assert mismatched.hip_depth == pytest.approx(matched.hip_depth, abs=1e-6)
    assert mismatched.hip_x_rel == pytest.approx(matched.hip_x_rel, abs=1e-6)
    assert mismatched.hip_y_rel == pytest.approx(matched.hip_y_rel, abs=1e-6)
    assert mismatched.centroid_displacement == pytest.approx(
        matched.centroid_displacement, abs=1e-6
    )
    # Without the scale the 2x pose sits far outside a 1080p polygon.
    assert unscaled.torso_in_frac == pytest.approx(0.0)
    assert unscaled.torso_in_frac != pytest.approx(matched.torso_in_frac)


def test_unusable_polygon_forces_bed_relative_fields_to_zero() -> None:
    for polygon in (
        ((400, 300), (500, 400)),
        ((400, 300), (800, 300), (1200, 300)),
        ((400, 300), (400, 300), (400, 300), (400, 300)),
    ):
        features = _features_for(_IN_BED_POINTS, bed_boxes=(_bed_box(polygon),))
        _assert_bed_relative_zero(features)
        assert features.observability == pytest.approx(8.0 / 17.0)
        assert features.torso_angle > 0.0


def test_all_keypoints_below_confidence_gate_are_a_defined_missing_state() -> None:
    features = _features_for(
        _IN_BED_POINTS,
        confidence=DEFAULT_FALL_CONFIDENCE_THRESHOLD - 1e-6,
    )
    assert features.bed_polygon_valid is True
    assert features.torso_in_frac == 0.0
    assert features.lower_in_frac == 0.0
    assert features.keypoint_in_frac == 0.0
    assert features.hip_depth == 0.0
    assert features.hip_x_rel == 0.0
    assert features.hip_y_rel == 0.0
    assert features.centroid_displacement == 0.0
    assert features.observability == 0.0
    assert features.torso_angle == 0.0
    assert features.bed_id == 0


def test_no_bed_boxes_is_an_invalid_polygon_state() -> None:
    features = _features_for(_IN_BED_POINTS, bed_boxes=())
    _assert_bed_relative_zero(features)
    assert features.torso_angle > 0.0


def test_torso_angle_is_near_zero_when_lying_and_near_pi_over_two_when_upright() -> None:
    # Ceiling-view person along the bed's long (x) axis: shoulders share an x,
    # hips share a further-right x. Midpoint-to-midpoint is image-horizontal.
    lying = _features_for(
        {
            5: (600.0, 480.0),
            6: (600.0, 560.0),
            11: (1000.0, 490.0),
            12: (1000.0, 550.0),
        }
    )
    sitting = _features_for(
        {
            5: (780.0, 340.0),
            6: (820.0, 340.0),
            11: (780.0, 620.0),
            12: (820.0, 620.0),
        }
    )
    assert lying.torso_angle < 0.2
    assert sitting.torso_angle == pytest.approx(math.pi / 2.0, abs=1e-6)


def test_compute_frame_skips_untracked_poses() -> None:
    observation = FrameObservation(
        poses=(_pose(_IN_BED_POINTS), _pose(_OUT_OF_BED_POINTS)),
        regions=((_bed_box(),), ()),
        track_ids=(3, None),
    )
    frame = compute_frame_bed_pose_features(
        observation,
        frame_width=_FRAME_W,
        frame_height=_FRAME_H,
    )
    assert len(frame.items) == 1
    assert frame.items[0].track_id == 3
    assert frame.items[0].torso_in_frac == pytest.approx(1.0)


def test_seven_arg_construction_defaults_empty_bed_pose_features() -> None:
    decision_input = _seven_arg_decision_input()
    assert decision_input.bed_pose_features.items == ()


def test_build_decision_input_populates_bed_pose_features() -> None:
    observation = FrameObservation(
        poses=(_pose(_IN_BED_POINTS, scale=2.0),),
        regions=((_bed_box(),), ()),
        track_ids=(4,),
    )
    scene = SceneState(
        camera_id="cam-wire",
        persisted_bed_regions=(_bed_box(),),
        bed_zone_image_width=_FRAME_W,
        bed_zone_image_height=_FRAME_H,
    )
    decision_input = build_decision_input(
        observation,
        frame_width=_POSE_W,
        frame_height=_POSE_H,
        live_track_ids=(4,),
        time_sec=1.0,
        frame_index=1,
        scene_state=scene,
        bed_scheduled=False,
        bed_interval=30,
    )
    matched = _features_for(_IN_BED_POINTS)
    scaled = decision_input.bed_pose_features.items[0]
    assert scaled.track_id == 4
    assert scaled.torso_in_frac == pytest.approx(matched.torso_in_frac)
    assert scaled.lower_in_frac == pytest.approx(matched.lower_in_frac)
    assert scaled.hip_depth == pytest.approx(matched.hip_depth, abs=1e-6)


def test_domains_have_no_direct_numpy_import() -> None:
    """The scalar contract exists so domains never import numpy themselves.

    ``import worker.domains.bed_exit`` still pulls numpy transitively through
    ``contracts.observation``; that is pre-existing and not this boundary.
    """
    offenders: list[str] = []
    for path in (_REPO_ROOT / "worker" / "domains").rglob("*.py"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("import numpy") or stripped.startswith("from numpy"):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line_no}:{line.rstrip()}")
    assert offenders == []
