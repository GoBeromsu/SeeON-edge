"""BedPoseFeatures contract, perception producer, and DecisionInput wiring.

Characterization of the pre-todo-6 DecisionInput constructor lives at the
top so it remains meaningful after the new field is added: existing call
sites must still build with the original seven arguments.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.pipeline.perception.decision_input import build_decision_input
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
