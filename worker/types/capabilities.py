from __future__ import annotations

from dataclasses import dataclass

from worker.types.frame_memory import MemoryKind, PixelFormat


@dataclass(frozen=True, slots=True)
class FrameCapability:
    memory_kind: MemoryKind
    pixel_format: PixelFormat


@dataclass(frozen=True, slots=True)
class StageCapabilities:
    name: str
    accepts: frozenset[FrameCapability]
    produces: FrameCapability

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage capability name must be non-empty")
        if not self.accepts:
            raise ValueError(f"stage {self.name!r} must accept at least one frame capability")


@dataclass(frozen=True, slots=True)
class ConverterCapabilities:
    name: str
    source: FrameCapability
    target: FrameCapability
    copies_frame: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("converter name must be non-empty")
        if self.requires_memory_copy and not self.copies_frame:
            raise ValueError(
                f"converter {self.name!r} crosses memory domains and must declare copies_frame=True"
            )

    @property
    def requires_memory_copy(self) -> bool:
        return self.source.memory_kind is not self.target.memory_kind

    @property
    def effective_copies_frame(self) -> bool:
        return self.copies_frame or self.requires_memory_copy


@dataclass(frozen=True, slots=True)
class PipelineProfile:
    name: str
    stages: tuple[StageCapabilities, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pipeline profile name must be non-empty")
        if not self.stages:
            raise ValueError(f"pipeline profile {self.name!r} must contain stages")
        names = tuple(stage.name for stage in self.stages)
        if len(names) != len(set(names)):
            raise ValueError(f"pipeline profile {self.name!r} contains duplicate stage names")


HOST_RGB = FrameCapability(MemoryKind.HOST, PixelFormat.RGB24)
HOST_PIPELINE_PROFILE = PipelineProfile(
    name="host-rgb24",
    stages=(
        StageCapabilities("decode", frozenset({HOST_RGB}), HOST_RGB),
        StageCapabilities("inference", frozenset({HOST_RGB}), HOST_RGB),
        StageCapabilities("output", frozenset({HOST_RGB}), HOST_RGB),
    ),
)


__all__ = [
    "HOST_PIPELINE_PROFILE",
    "HOST_RGB",
    "ConverterCapabilities",
    "FrameCapability",
    "PipelineProfile",
    "StageCapabilities",
]
