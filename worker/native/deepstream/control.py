"""Reliable long-lived AF_UNIX control session for the native child."""

from __future__ import annotations

import socket
import threading
from typing import Final, final

from worker.native.deepstream.control_reply import decode_status, validate_reply
from worker.native.deepstream.control_types import (
    ChildControlError,
    ControlIdentity,
    NativePreviewStatus,
    NativeStatus,
    parse_source_uri,
)
from worker.native.deepstream.ipc import (
    ControlMessage,
    MessageKind,
    MetadataFrame,
    decode_metadata,
    encode_message,
)
from worker.native.deepstream.metadata import (
    MetadataPullFailure,
    MetadataPullStopped,
    SourceBinding,
)

_MAX_REPLY: Final = 65_535
_MAX_SOURCES: Final = 64


def _require_socket(endpoint: object) -> socket.socket:
    if not isinstance(endpoint, socket.socket):
        raise TypeError("inherited SOCK_SEQPACKET control socket required")
    return endpoint


@final
class DeepStreamControlClient:
    """One correlated SOCK_SEQPACKET session; mutation tracks source generations."""

    def __init__(
        self,
        endpoint: socket.socket,
        identity: ControlIdentity,
        *,
        timeout_sec: float = 2.0,
    ) -> None:
        self._endpoint = _require_socket(endpoint)
        self._identity = identity
        self._timeout_sec = timeout_sec
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()
        self._request_sequence = 0
        self._generations: dict[str, int] = {}
        self._epochs: dict[str, int] = {}

    def connect(self) -> None:
        self._endpoint.settimeout(self._timeout_sec)
        self._socket = self._endpoint

    def close(self) -> None:
        client, self._socket = self._socket, None
        if client is not None:
            client.close()

    def _message(self, kind: MessageKind, camera_id: str, payload: bytes = b"") -> ControlMessage:
        self._request_sequence += 1
        return ControlMessage(
            kind=kind,
            worker_boot_id=self._identity.worker_boot_id,
            child_instance_id=self._identity.child_instance_id,
            camera_id=camera_id,
            source_generation=self._generations.get(camera_id, 0),
            stream_epoch=self._epochs.get(camera_id, 0),
            source_pts=0,
            source_sequence=self._request_sequence,
            native_publish_sequence=0,
            request_id=self._request_sequence,
            transform_id=self._identity.transform_id,
            payload=payload,
        )

    def request(self, message: ControlMessage) -> ControlMessage:
        with self._lock:
            client = self._socket
            if client is None:
                raise ChildControlError("control_closed", "closed")
            try:
                client.sendall(encode_message(message))
                raw = client.recv(_MAX_REPLY)
            except OSError as error:
                raise ChildControlError("control_io", str(error)) from error
            if raw == b"":
                raise ChildControlError("control_eof", "eof")
            return validate_reply(message, raw)

    def status(self) -> NativeStatus:
        reply = self.request(self._message(MessageKind.STATUS, "_worker"))
        return decode_status(reply.payload)

    def add_source(self, camera_id: str, uri: str) -> SourceBinding:
        if camera_id == "" or len(camera_id.encode()) > 128:
            raise ChildControlError("camera_id_invalid", "bounds")
        if camera_id not in self._generations and len(self._generations) >= _MAX_SOURCES:
            raise ChildControlError("source_capacity", str(_MAX_SOURCES))
        parsed_uri = parse_source_uri(uri)
        self._generations[camera_id] = self._generations.get(camera_id, 0) + 1
        self._epochs[camera_id] = 0
        reply = self.request(self._message(MessageKind.ADD_SOURCE, camera_id, parsed_uri.encode()))
        if reply.kind is not MessageKind.EPOCH_STARTED:
            raise ChildControlError("source_not_started", reply.kind.name)
        self._epochs[camera_id] = reply.stream_epoch
        return SourceBinding(
            worker_boot_id=str(self._identity.worker_boot_id),
            child_instance_id=str(self._identity.child_instance_id),
            camera_id=camera_id,
            source_generation=reply.source_generation,
            stream_epoch=reply.stream_epoch,
            transform_id=self._identity.transform_id,
        )

    def remove_source(self, camera_id: str) -> None:
        _ = self.request(self._message(MessageKind.REMOVE_SOURCE, camera_id))
        _ = self._epochs.pop(camera_id, None)

    def source_failure(self, camera_id: str, category: str) -> SourceBinding:
        reply = self.request(
            self._message(MessageKind.SOURCE_FAILURE, camera_id, category.encode())
        )
        if reply.kind is not MessageKind.EPOCH_STARTED:
            raise ChildControlError("epoch_not_started", reply.kind.name)
        self._epochs[camera_id] = reply.stream_epoch
        return SourceBinding(
            worker_boot_id=str(self._identity.worker_boot_id),
            child_instance_id=str(self._identity.child_instance_id),
            camera_id=camera_id,
            source_generation=reply.source_generation,
            stream_epoch=reply.stream_epoch,
            transform_id=self._identity.transform_id,
        )

    def inject_source_eos(self, camera_id: str) -> None:
        _ = self.request(self._message(MessageKind.INJECT_SOURCE_EOS, camera_id))

    def set_preview_viewers(
        self,
        camera_id: str,
        viewers: int,
        mode: str = "none",
    ) -> None:
        if viewers < 0 or viewers > 4_096:
            raise ChildControlError("preview_viewers_invalid", str(viewers))
        mode_values = {"none": 0, "fall": 1, "bedexit": 2}
        mode_value = mode_values.get(mode)
        if mode_value is None:
            raise ChildControlError("preview_mode_invalid", mode)
        _ = self.request(
            self._message(
                MessageKind.SET_PREVIEW_DEMAND,
                camera_id,
                viewers.to_bytes(4, "little") + bytes((mode_value,)),
            )
        )

    def wait_for_preview(self, camera_id: str, target: int) -> None:
        _ = self.request(
            self._message(
                MessageKind.WAIT_PREVIEW,
                camera_id,
                target.to_bytes(8, "little"),
            )
        )

    def preview_status(self, camera_id: str) -> NativePreviewStatus:
        reply = self.request(self._message(MessageKind.GET_PREVIEW_STATUS, camera_id))
        if len(reply.payload) != 16:
            raise ChildControlError("preview_status_size", str(len(reply.payload)))
        return NativePreviewStatus(
            int.from_bytes(reply.payload[:8], "little"),
            int.from_bytes(reply.payload[8:], "little"),
        )

    def pull_latest(self, camera_id: str) -> MetadataFrame | None:
        try:
            reply = self.request(self._message(MessageKind.GET_LATEST, camera_id))
        except ChildControlError as error:
            if error.code in {"control_eof", "control_closed"}:
                raise MetadataPullStopped from error
            raise MetadataPullFailure(error.code) from error
        if reply.kind is MessageKind.CAPABILITY_INACTIVE:
            return None
        return decode_metadata(encode_message(reply))

    def snapshot(self, camera_id: str) -> bytes:
        reply = self.request(self._message(MessageKind.SNAPSHOT, camera_id))
        if not reply.payload.startswith(b"\xff\xd8") or not reply.payload.endswith(b"\xff\xd9"):
            raise ChildControlError("snapshot_invalid", camera_id)
        return reply.payload

    def record_status(self, camera_id: str) -> int:
        reply = self.request(self._message(MessageKind.RECORD, camera_id))
        if len(reply.payload) != 8:
            raise ChildControlError("record_status_size", str(len(reply.payload)))
        return int.from_bytes(reply.payload, "little")

    def wait_for_publish(self, target: int) -> None:
        payload = target.to_bytes(8, "little")
        _ = self.request(self._message(MessageKind.WAIT_PUBLISH, "_worker", payload))

    def fatal(self, category: str) -> None:
        _ = self.request(self._message(MessageKind.FATAL, "_worker", category.encode()))

    def shutdown(self) -> None:
        _ = self.request(self._message(MessageKind.SHUTDOWN, "_worker"))


__all__ = [
    "ChildControlError",
    "ControlIdentity",
    "DeepStreamControlClient",
    "NativePreviewStatus",
    "NativeStatus",
    "parse_source_uri",
]
