"""Worker-internal envelope types (image-free, frozen dataclasses)."""

from __future__ import annotations

from worker.types.bed_pose_features import (
    EMPTY_FRAME_BED_POSE_FEATURES,
    BedPoseFeatures,
    FrameBedPoseFeatures,
)
from worker.types.business_event import BusinessEvent
from worker.types.capabilities import (
    ConverterCapabilities,
    FrameCapability,
    PipelineProfile,
    StageCapabilities,
)
from worker.types.copy_metrics import CopyMetrics, CopyMetricsSnapshot
from worker.types.decision_input import DecisionInput
from worker.types.fall_model_input import FallModelInput
from worker.types.frame_memory import (
    FrameDescriptor,
    FrameLease,
    FrameLeaseReleasedError,
    MemoryKind,
    PixelFormat,
)
from worker.types.frame_packet import FrameKey, FramePacket
from worker.types.module_result import ModuleResult
from worker.types.perception_frame import (
    AssociationResult,
    BedRegion,
    BedRegionChannel,
    ChannelState,
    HumanPoseChannel,
    Keypoint,
    PerceptionFrameFailure,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBox,
    PersonBoxChannel,
)
from worker.types.temporal_profile import (
    CURRENT_TEMPORAL_PROFILE,
    TemporalProfile,
    TemporalProfileError,
)
from worker.types.trace import DecisionTraceSnapshot, NumericTraceValue

__all__ = [
    "BedPoseFeatures",
    "BusinessEvent",
    "ConverterCapabilities",
    "CopyMetrics",
    "CopyMetricsSnapshot",
    "DecisionInput",
    "EMPTY_FRAME_BED_POSE_FEATURES",
    "FallModelInput",
    "FrameBedPoseFeatures",
    "FrameCapability",
    "FrameDescriptor",
    "FrameKey",
    "FrameLease",
    "FrameLeaseReleasedError",
    "FramePacket",
    "MemoryKind",
    "ModuleResult",
    "AssociationResult",
    "BedRegion",
    "BedRegionChannel",
    "ChannelState",
    "HumanPoseChannel",
    "Keypoint",
    "PerceptionFrameFailure",
    "PerceptionFrameIdentity",
    "PerceptionFrameV1",
    "PersonBox",
    "PersonBoxChannel",
    "DecisionTraceSnapshot",
    "NumericTraceValue",
    "PipelineProfile",
    "PixelFormat",
    "StageCapabilities",
    "CURRENT_TEMPORAL_PROFILE",
    "TemporalProfile",
    "TemporalProfileError",
]
