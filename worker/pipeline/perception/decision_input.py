from __future__ import annotations

from contracts.observation import FrameObservation
from worker.pipeline.perception.scene_state import SceneState
from worker.types import DecisionInput
from worker.types.bed_pose_features import FrameBedPoseFeatures


def bed_pose_features_for(
    observation: FrameObservation,
    *,
    frame_width: int,
    frame_height: int,
    scene_state: SceneState,
) -> FrameBedPoseFeatures:
    """Compute the additive DecisionInput field from a resolved observation."""
    # Imported here so `import worker.pipeline.perception` stays numpy-free;
    # only the compute path pays for the producer.
    from worker.pipeline.perception.features.bed_geometry import (
        compute_frame_bed_pose_features,
    )

    return compute_frame_bed_pose_features(
        observation,
        frame_width=frame_width,
        frame_height=frame_height,
        polygon_image_width=scene_state.bed_zone_image_width,
        polygon_image_height=scene_state.bed_zone_image_height,
    )


def build_decision_input(
    observation: FrameObservation,
    *,
    frame_width: int,
    frame_height: int,
    live_track_ids: tuple[int, ...],
    time_sec: float | None,
    frame_index: int,
    scene_state: SceneState,
    bed_scheduled: bool,
    bed_interval: int,
) -> DecisionInput:
    """Resolve scene provenance and return the image-free decision boundary."""
    resolved_observation, bed_region = scene_state.resolve_bed_regions(
        observation,
        frame_index=frame_index,
        bed_scheduled=bed_scheduled,
        bed_interval=bed_interval,
    )
    return DecisionInput(
        observation=resolved_observation,
        frame_width=frame_width,
        frame_height=frame_height,
        live_track_ids=live_track_ids,
        time_sec=time_sec,
        frame_index=frame_index,
        bed_region=bed_region,
        bed_pose_features=bed_pose_features_for(
            resolved_observation,
            frame_width=frame_width,
            frame_height=frame_height,
            scene_state=scene_state,
        ),
    )


__all__ = ["bed_pose_features_for", "build_decision_input"]
