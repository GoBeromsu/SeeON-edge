"""Image-free metadata values emitted by worker media planes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from worker.types.perception_frame import PerceptionFrameIdentity, PerceptionFrameV1


@dataclass(frozen=True, slots=True)
class SourceBinding:
    worker_boot_id: str
    child_instance_id: str
    camera_id: str
    source_generation: int
    stream_epoch: int
    transform_id: str


@dataclass(frozen=True, slots=True)
class MetadataFrame:
    frame: PerceptionFrameV1
    source_generation: int
    child_instance_id: uuid.UUID
    native_publish_sequence: int
    transform_id: str
    source_width: int = 0
    source_height: int = 0
    source_time_ns: int = 0

    @property
    def identity(self) -> PerceptionFrameIdentity:
        return self.frame.identity


@dataclass(frozen=True, slots=True)
class MetadataCounters:
    accepted: int = 0
    overwritten: int = 0
    late: int = 0
    unknown_source: int = 0
    generation_mismatch: int = 0
    epoch_mismatch: int = 0
    boot_mismatch: int = 0
    child_mismatch: int = 0
    transform_mismatch: int = 0
    malformed: int = 0
    pull_failures: int = 0


__all__ = ["MetadataCounters", "MetadataFrame", "SourceBinding"]
