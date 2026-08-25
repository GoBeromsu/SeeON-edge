"""Versioned IPC envelope and metadata value types."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import override

from worker.types.perception_frame import (
    BedRegionChannel,
    ChannelState,
    HumanPoseChannel,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBoxChannel,
)


class MessageKind(IntEnum):
    ADD_SOURCE = 1
    REMOVE_SOURCE = 2
    RECORD = 3
    SNAPSHOT = 4
    STATUS = 5
    SOURCE_FAILURE = 6
    FATAL = 7
    EMIT_METADATA = 8
    GET_LATEST = 9
    SHUTDOWN = 10
    WAIT_PUBLISH = 11
    GET_SOURCE_STATE = 12
    INJECT_SOURCE_EOS = 13
    SET_PREVIEW_DEMAND = 14
    GET_PREVIEW_STATUS = 15
    WAIT_PREVIEW = 16
    ACK = 64
    STATUS_REPLY = 65
    ERROR = 66
    EPOCH_STARTED = 67
    CAPABILITY_INACTIVE = 68
    METADATA = 128


@dataclass(frozen=True, slots=True)
class IpcProtocolError(Exception):
    code: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ControlMessage:
    kind: MessageKind
    worker_boot_id: uuid.UUID
    child_instance_id: uuid.UUID
    camera_id: str
    source_generation: int
    stream_epoch: int
    source_pts: int
    source_sequence: int
    native_publish_sequence: int
    request_id: int
    transform_id: str
    payload: bytes = b""


@dataclass(frozen=True, slots=True)
class MetadataFrame:
    frame: PerceptionFrameV1
    source_generation: int
    child_instance_id: uuid.UUID
    native_publish_sequence: int
    transform_id: str
    # Wire-v2 geometry/source-time carrier (0 == not provided by the child).
    source_width: int = 0
    source_height: int = 0
    source_time_ns: int = 0

    @property
    def identity(self) -> PerceptionFrameIdentity:
        return self.frame.identity

    @classmethod
    def empty(cls, envelope: ControlMessage) -> MetadataFrame:
        identity = PerceptionFrameIdentity(
            worker_boot_id=str(envelope.worker_boot_id),
            camera_id=envelope.camera_id,
            stream_epoch=envelope.stream_epoch,
            seq=envelope.source_sequence,
            source_pts=envelope.source_pts,
        )
        empty = ChannelState.INFERRED_EMPTY
        return cls(
            frame=PerceptionFrameV1(
                identity=identity,
                person_box=PersonBoxChannel(empty),
                human_pose=HumanPoseChannel(empty),
                bed_region=BedRegionChannel(empty),
            ),
            source_generation=envelope.source_generation,
            child_instance_id=envelope.child_instance_id,
            native_publish_sequence=envelope.native_publish_sequence,
            transform_id=envelope.transform_id,
        )


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


__all__ = [
    "ControlMessage",
    "IpcProtocolError",
    "MessageKind",
    "MetadataCounters",
    "MetadataFrame",
]
