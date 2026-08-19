from __future__ import annotations

from dataclasses import dataclass

# Per-track bed-relation scalars for one frame. Plain Python numbers -- never
# an ndarray -- so `worker.domains.bed_exit` stays numeric/hardware-agnostic;
# the perception producer (`worker.pipeline.perception.features.bed_geometry`)
# is the only place that may touch keypoints, polygons, or numpy. The domain
# layer reads these fields and never re-derives them.


@dataclass(frozen=True, slots=True)
class BedPoseFeatures:
    """One track's bed-relative pose measurements for a single frame."""

    track_id: int
    bed_id: int | None
    torso_in_frac: float
    lower_in_frac: float
    keypoint_in_frac: float
    hip_depth: float
    torso_angle: float
    centroid_displacement: float
    hip_x_rel: float
    hip_y_rel: float
    observability: float
    bed_polygon_valid: bool


@dataclass(frozen=True, slots=True)
class FrameBedPoseFeatures:
    """All per-track :class:`BedPoseFeatures` computed for one frame."""

    items: tuple[BedPoseFeatures, ...] = ()


EMPTY_FRAME_BED_POSE_FEATURES = FrameBedPoseFeatures()


__all__ = [
    "BedPoseFeatures",
    "EMPTY_FRAME_BED_POSE_FEATURES",
    "FrameBedPoseFeatures",
]
