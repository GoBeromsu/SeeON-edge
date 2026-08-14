from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from worker.types import FrameCapability, MemoryKind, PixelFormat

ProfileStage: TypeAlias = Literal["decode", "preprocess", "inference", "overlay", "encode"]
CopyDirection: TypeAlias = Literal["h2d", "d2h", "none"]


@dataclass(frozen=True, slots=True)
class MemoryPathStep:
    stage: ProfileStage
    memory_kind: MemoryKind
    pixel_format: PixelFormat

    @property
    def capability(self) -> FrameCapability:
        return FrameCapability(self.memory_kind, self.pixel_format)

    @property
    def label(self) -> str:
        return f"{self.memory_kind.value}/{self.pixel_format.value}"


@dataclass(frozen=True, slots=True)
class ProfileConverter:
    name: str
    source: FrameCapability
    target: FrameCapability
    direction: CopyDirection

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile converter name must be non-empty")
        crosses_memory = self.source.memory_kind is not self.target.memory_kind
        if crosses_memory and self.direction == "none":
            raise ValueError(
                f"profile converter {self.name!r} crosses memory domains without a copy direction"
            )
        if self.direction == "h2d" and self.source.memory_kind is not MemoryKind.HOST:
            raise ValueError(f"profile converter {self.name!r} h2d source must be host memory")
        if self.direction == "d2h" and self.target.memory_kind is not MemoryKind.HOST:
            raise ValueError(f"profile converter {self.name!r} d2h target must be host memory")


@dataclass(frozen=True, slots=True)
class RuntimeProfileEdge:
    source_stage: ProfileStage
    target_stage: ProfileStage
    source: FrameCapability
    target: FrameCapability
    converter_name: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeProfileDescriptor:
    """Boot-frozen truth about the selected infrastructure path.

    This is the typed Todo 7 descriptor only. Persistence, relay provenance,
    and content hashing belong to Todo 10.
    """

    requested_profile: str
    canonical_profile: str
    requested_decode_backend: str
    effective_decode_backend: str
    requested_preprocess_backend: str
    effective_preprocess_backend: str
    requested_inference_backend: str
    effective_inference_backend: str
    requested_overlay_backend: str
    effective_overlay_backend: str
    requested_encode_backend: str
    effective_encode_backend: str
    requested_memory_steps: tuple[MemoryPathStep, ...]
    effective_memory_steps: tuple[MemoryPathStep, ...]
    requested_converters: tuple[ProfileConverter, ...]
    effective_converters: tuple[ProfileConverter, ...]
    effective_edges: tuple[RuntimeProfileEdge, ...]
    degraded_reasons: tuple[str, ...]
    device_resident_after_decode: bool
    concrete_stages_available: bool

    def __post_init__(self) -> None:
        if not self.requested_profile or not self.canonical_profile:
            raise ValueError("requested and canonical profile names must be non-empty")
        expected_stages = ("decode", "preprocess", "inference", "overlay", "encode")
        for label, steps in (
            ("requested", self.requested_memory_steps),
            ("effective", self.effective_memory_steps),
        ):
            if tuple(step.stage for step in steps) != expected_stages:
                raise ValueError(
                    f"{label} runtime profile must describe decode through encode exactly once"
                )
        for label, converters in (
            ("requested", self.requested_converters),
            ("effective", self.effective_converters),
        ):
            names = tuple(converter.name for converter in converters)
            if len(names) != len(set(names)):
                raise ValueError(f"{label} runtime profile converter names must be unique")
        stages = {step.stage for step in self.effective_memory_steps}
        if not self.effective_edges:
            raise ValueError("runtime profile must declare its validated edges")
        for edge in self.effective_edges:
            if edge.source_stage not in stages or edge.target_stage not in stages:
                raise ValueError("runtime profile edge references an unknown stage")

    @property
    def memory_steps(self) -> tuple[MemoryPathStep, ...]:
        return self.effective_memory_steps

    @property
    def converters(self) -> tuple[ProfileConverter, ...]:
        return self.effective_converters

    @property
    def requested_memory_domains(self) -> tuple[MemoryKind, ...]:
        return tuple(step.memory_kind for step in self.requested_memory_steps)

    @property
    def effective_memory_domains(self) -> tuple[MemoryKind, ...]:
        return tuple(step.memory_kind for step in self.effective_memory_steps)

    @property
    def requested_pixel_formats(self) -> tuple[PixelFormat, ...]:
        return tuple(step.pixel_format for step in self.requested_memory_steps)

    @property
    def effective_pixel_formats(self) -> tuple[PixelFormat, ...]:
        return tuple(step.pixel_format for step in self.effective_memory_steps)

    @property
    def requested_memory_path(self) -> tuple[str, ...]:
        return tuple(step.label for step in self.requested_memory_steps)

    @property
    def memory_path(self) -> tuple[str, ...]:
        return tuple(step.label for step in self.effective_memory_steps)

    @property
    def requested_converter_chain(self) -> tuple[str, ...]:
        return tuple(converter.name for converter in self.requested_converters)

    @property
    def converter_chain(self) -> tuple[str, ...]:
        return tuple(converter.name for converter in self.effective_converters)

    @property
    def fallback_reason(self) -> str | None:
        return self.degraded_reasons[0] if self.degraded_reasons else None

    @property
    def degraded_reason(self) -> str | None:
        return ";".join(self.degraded_reasons) if self.degraded_reasons else None

    @property
    def full_frame_h2d_count(self) -> int:
        return sum(converter.direction == "h2d" for converter in self.effective_converters)

    @property
    def full_frame_d2h_count(self) -> int:
        return sum(converter.direction == "d2h" for converter in self.effective_converters)


__all__ = [
    "CopyDirection",
    "MemoryPathStep",
    "ProfileConverter",
    "ProfileStage",
    "RuntimeProfileDescriptor",
    "RuntimeProfileEdge",
]
