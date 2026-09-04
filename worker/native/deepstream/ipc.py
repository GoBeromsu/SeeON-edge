"""Versioned binary IPC for the dark DeepStream child."""

from __future__ import annotations

import struct
import uuid
from typing import Final

from worker.native.deepstream.ipc_types import (
    ControlMessage,
    IpcProtocolError,
    MessageKind,
    MetadataCounters,
    MetadataFrame,
)
from worker.native.deepstream.perception_wire import (
    PerceptionWireError,
    decode_perception_wire,
)
from worker.types.perception_frame import PerceptionFrameIdentity

_MAGIC: Final = b"SDS1"
PROTOCOL_VERSION: Final = 1
_HEADER: Final = struct.Struct("<4sBBHIIQQQQQ16s16sHH")
_MAX_FRAME: Final = 65_535


def encode_message(message: ControlMessage) -> bytes:
    camera = message.camera_id.encode()
    transform = message.transform_id.encode()
    body = camera + transform + message.payload
    if len(camera) > 65_535 or len(transform) > 65_535 or _HEADER.size + len(body) > _MAX_FRAME:
        raise IpcProtocolError("frame_too_large", str(_HEADER.size + len(body)))
    return (
        _HEADER.pack(
            _MAGIC,
            PROTOCOL_VERSION,
            int(message.kind),
            0,
            len(body),
            message.source_generation,
            message.stream_epoch,
            message.source_pts,
            message.source_sequence,
            message.native_publish_sequence,
            message.request_id,
            message.worker_boot_id.bytes,
            message.child_instance_id.bytes,
            len(camera),
            len(transform),
        )
        + body
    )


def decode_control_message(data: bytes) -> ControlMessage:
    if len(data) < _HEADER.size:
        raise IpcProtocolError("short_header", str(len(data)))
    (
        magic,
        version,
        kind_value,
        _reserved,
        body_size,
        generation,
        epoch,
        pts,
        sequence,
        publish_sequence,
        request_id,
        boot_bytes,
        child_bytes,
        camera_size,
        transform_size,
    ) = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != PROTOCOL_VERSION:
        raise IpcProtocolError("protocol_mismatch", f"magic={magic!r} version={version}")
    if len(data) != _HEADER.size + body_size:
        raise IpcProtocolError("framing_mismatch", f"declared={body_size} actual={len(data)}")
    try:
        kind = MessageKind(kind_value)
    except ValueError as error:
        raise IpcProtocolError("unknown_kind", str(kind_value)) from error
    if camera_size + transform_size > body_size:
        raise IpcProtocolError("identity_overflow", str(body_size))
    body = memoryview(data)[_HEADER.size :]
    try:
        camera = bytes(body[:camera_size]).decode()
        transform = bytes(body[camera_size : camera_size + transform_size]).decode()
    except UnicodeDecodeError as error:
        raise IpcProtocolError("identity_encoding", str(error)) from error
    payload = bytes(body[camera_size + transform_size :])
    return ControlMessage(
        kind=kind,
        worker_boot_id=uuid.UUID(bytes=boot_bytes),
        child_instance_id=uuid.UUID(bytes=child_bytes),
        camera_id=camera,
        source_generation=generation,
        stream_epoch=epoch,
        source_pts=pts,
        source_sequence=sequence,
        native_publish_sequence=publish_sequence,
        request_id=request_id,
        transform_id=transform,
        payload=payload,
    )


def decode_metadata(data: bytes) -> MetadataFrame:
    message = decode_control_message(data)
    if message.kind is not MessageKind.METADATA:
        raise IpcProtocolError("metadata_kind", message.kind.name)
    expected = PerceptionFrameIdentity(
        str(message.worker_boot_id),
        message.camera_id,
        message.stream_epoch,
        message.source_sequence,
        message.source_pts,
    )
    try:
        decoded = decode_perception_wire(message.payload, expected)
    except PerceptionWireError as error:
        raise IpcProtocolError(error.code, error.detail) from error
    return MetadataFrame(
        decoded.frame,
        message.source_generation,
        message.child_instance_id,
        message.native_publish_sequence,
        message.transform_id,
        decoded.source_width,
        decoded.source_height,
        decoded.source_time_ns,
    )


__all__ = [
    "PROTOCOL_VERSION",
    "ControlMessage",
    "IpcProtocolError",
    "MessageKind",
    "MetadataCounters",
    "MetadataFrame",
    "decode_control_message",
    "decode_metadata",
    "encode_message",
]
