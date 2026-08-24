"""Correlated native control reply validation and status decoding."""

from __future__ import annotations

from typing import Final

from worker.native.deepstream.control_types import ChildControlError, NativeStatus
from worker.native.deepstream.ipc import (
    ControlMessage,
    IpcProtocolError,
    MessageKind,
    decode_control_message,
)

_ACCEPTED_REPLIES: Final = frozenset(
    {
        MessageKind.ACK,
        MessageKind.STATUS_REPLY,
        MessageKind.EPOCH_STARTED,
        MessageKind.CAPABILITY_INACTIVE,
        MessageKind.METADATA,
    }
)
_STATUS_SIZE: Final = 45


def validate_reply(message: ControlMessage, raw: bytes) -> ControlMessage:
    try:
        reply = decode_control_message(raw)
    except IpcProtocolError as error:
        raise ChildControlError("invalid_reply", str(error)) from error
    if reply.worker_boot_id != message.worker_boot_id:
        raise ChildControlError("boot_mismatch", str(reply.worker_boot_id))
    if reply.child_instance_id != message.child_instance_id:
        raise ChildControlError("child_mismatch", str(reply.child_instance_id))
    if reply.request_id != message.request_id:
        raise ChildControlError("correlation_mismatch", str(reply.request_id))
    if reply.kind in _ACCEPTED_REPLIES:
        return reply
    code = "native_error" if reply.kind is MessageKind.ERROR else "reply_kind"
    detail = (
        reply.payload.decode(errors="replace")
        if reply.kind is MessageKind.ERROR
        else reply.kind.name
    )
    raise ChildControlError(code, detail)


def decode_status(payload: bytes) -> NativeStatus:
    if len(payload) != _STATUS_SIZE:
        raise ChildControlError("status_size", str(len(payload)))
    return NativeStatus(
        metadata_published=int.from_bytes(payload[0:8], "little"),
        metadata_overwritten=int.from_bytes(payload[8:16], "little"),
        wake_dropped=int.from_bytes(payload[16:24], "little"),
        source_failures=int.from_bytes(payload[24:32], "little"),
        malformed_frames=int.from_bytes(payload[32:40], "little"),
        source_count=int.from_bytes(payload[40:44], "little"),
        custom_transform_available=bool(payload[44]),
    )


__all__ = ["decode_status", "validate_reply"]
