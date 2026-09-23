"""Shared `BedPoseFeatures` builders for bed-exit posture-gate tests.

`worker/domains/bed_exit/detector.py` only arms an exit once a track is
observed lying/sitting in its own bed (`hip_depth >= 0.10`, ported from the
deleted shadow state machine's measured convention -- see the module
constants there). These builders keep every bed-exit test's pose fixture
consistent with that exact convention instead of each file re-deriving its
own numbers.
"""

from __future__ import annotations

from worker.types import BedPoseFeatures, FrameBedPoseFeatures

# Measured by the deleted shadow state machine (worker/domains/bed_exit/
# detector.py's module docstring for `_MIN_IN_BED_HIP_DEPTH`): IN_BED +0.257.
_LYING_HIP_DEPTH = 0.257
# OUT_OF_BED -0.289: a standing person's hips sit well below the mattress
# plane regardless of how much of their bbox overlaps the bed polygon.
_STANDING_HIP_DEPTH = -0.289


def lying_in_bed(track_id: int, bed_id: int | None = 0) -> BedPoseFeatures:
    """Posture that satisfies the in-bed dwell gate (lying/sitting, observed)."""
    return BedPoseFeatures(
        track_id=track_id,
        bed_id=bed_id,
        torso_in_frac=1.0,
        lower_in_frac=1.0,
        keypoint_in_frac=1.0,
        hip_depth=_LYING_HIP_DEPTH,
        torso_angle=1.4,
        centroid_displacement=0.0,
        hip_x_rel=0.5,
        hip_y_rel=0.5,
        observability=0.9,
        bed_polygon_valid=True,
    )


def standing(track_id: int, bed_id: int | None = 0) -> BedPoseFeatures:
    """Posture that must never satisfy the gate (a caregiver leaning/standing)."""
    return BedPoseFeatures(
        track_id=track_id,
        bed_id=bed_id,
        torso_in_frac=1.0,
        lower_in_frac=1.0,
        keypoint_in_frac=1.0,
        hip_depth=_STANDING_HIP_DEPTH,
        torso_angle=1.5,
        centroid_displacement=0.0,
        hip_x_rel=0.5,
        hip_y_rel=0.2,
        observability=0.9,
        bed_polygon_valid=True,
    )


def frame_pose_features(*features: BedPoseFeatures) -> FrameBedPoseFeatures:
    return FrameBedPoseFeatures(items=tuple(features))


__all__ = ["frame_pose_features", "lying_in_bed", "standing"]
