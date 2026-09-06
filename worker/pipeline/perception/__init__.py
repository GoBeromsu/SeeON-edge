"""Perception stage: observation building, tracking, scene cache and windows."""

from __future__ import annotations

from worker.pipeline.perception.decision_input import build_decision_input
from worker.pipeline.perception.observation_builder import (
    build_frame_observation,
    observation_from_detection_result,
)
from worker.pipeline.perception.scene_state import BedRegionCacheCounters, SceneState
from worker.pipeline.perception.window_buffer import WindowBuffer

__all__ = [
    "BedRegionCacheCounters",
    "SceneState",
    "WindowBuffer",
    "build_decision_input",
    "build_frame_observation",
    "observation_from_detection_result",
]
