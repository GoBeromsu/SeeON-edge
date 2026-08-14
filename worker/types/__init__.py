"""Worker-internal envelope types (image-free, frozen dataclasses)."""

from __future__ import annotations

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
from worker.types.trace import DecisionTraceSnapshot, NumericTraceValue

__all__ = [
    "BusinessEvent",
    "ConverterCapabilities",
    "CopyMetrics",
    "CopyMetricsSnapshot",
    "DecisionInput",
    "FallModelInput",
    "FrameCapability",
    "FrameDescriptor",
    "FrameKey",
    "FrameLease",
    "FrameLeaseReleasedError",
    "FramePacket",
    "MemoryKind",
    "ModuleResult",
    "DecisionTraceSnapshot",
    "NumericTraceValue",
    "PipelineProfile",
    "PixelFormat",
    "StageCapabilities",
]
