"""Bounded native-AU wire decoding and immutable stream configuration."""

from __future__ import annotations

import hashlib
import socket
import struct
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from typing import Final, assert_never

from worker.adapters.decode.native_au_mux_template import (
    NativeAuTemplateInput,
    build_native_au_mux_template,
)
from worker.types.source_packet import (
    SourceStreamConfiguration,
    SourceStreamDescriptor,
)

_HEADER: Final = struct.Struct("<4sIBBBBIQQqqqiiIIHHII")
_MAGIC: Final = b"SAU1"
MAX_AU_FRAME_BYTES: Final = 32 * 1024 * 1024


class AuKind(IntEnum):
    ACCESS_UNIT = 1
    GAP = 2


class AuCodec(IntEnum):
    H264 = 1
    H265 = 2


class AuFraming(IntEnum):
    ANNEX_B = 1
    AVCC = 2


@dataclass(frozen=True, slots=True)
class AuEnvelope:
    kind: AuKind
    codec: AuCodec
    framing: AuFraming
    keyframe: bool
    generation: int
    epoch: int
    sequence: int
    pts: int
    dts: int
    duration: int
    time_base: Fraction
    width: int
    height: int
    camera_id: str
    parser_caps: str
    codec_data: bytes
    payload: bytes


class AuFrameError(ValueError):
    def __init__(self, camera_id: str, detail: str) -> None:
        super().__init__(detail)
        self.camera_id = camera_id


def receive_envelope(endpoint: socket.socket) -> AuEnvelope:
    header = _recv_exact(endpoint, _HEADER.size)
    if header[:4] != _MAGIC:
        raise AuFrameError("", "native AU header invalid")
    size = int.from_bytes(header[4:8], "little")
    if size > MAX_AU_FRAME_BYTES:
        raise AuFrameError("", "native AU frame exceeds bound")
    body = _recv_exact(endpoint, size)
    camera_size = int.from_bytes(header[72:74], "little")
    camera = _camera(body, camera_size)
    try:
        return _decode(header, body, camera)
    except (UnicodeDecodeError, ValueError, ZeroDivisionError) as error:
        raise AuFrameError(camera, "native AU frame invalid") from error


def stream_configuration(envelope: AuEnvelope) -> SourceStreamConfiguration:
    match envelope.codec:
        case AuCodec.H264:
            codec_name = "h264"
        case AuCodec.H265:
            codec_name = "hevc"
        case unreachable:
            assert_never(unreachable)
    match envelope.framing:
        case AuFraming.ANNEX_B:
            stream_format, nal_length_size = "byte-stream", 4
        case AuFraming.AVCC:
            stream_format = "avc" if codec_name == "h264" else "hvc1"
            nal_length_size = _nal_length_size(envelope.codec_data, codec_name)
        case unreachable:
            assert_never(unreachable)
    descriptor = SourceStreamDescriptor(
        0,
        "video",
        codec_name,
        "avc1" if codec_name == "h264" else "hvc1",
        envelope.time_base,
        envelope.codec_data,
        envelope.width,
        envelope.height,
        stream_format=stream_format,
        alignment="au",
        nal_length_size=nal_length_size,
        parser_caps_sha256=hashlib.sha256(envelope.parser_caps.encode()).hexdigest(),
    )
    return SourceStreamConfiguration.from_streams(
        (descriptor,),
        mux_template=build_native_au_mux_template(
            descriptor,
            NativeAuTemplateInput(envelope.payload, envelope.duration, envelope.keyframe),
        ),
    )


def _recv_exact(endpoint: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = endpoint.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("native AU stream closed")
        chunks.extend(chunk)
    return bytes(chunks)


def _camera(body: bytes, size: int) -> str:
    if size == 0 or size > len(body):
        return ""
    try:
        return body[:size].decode()
    except UnicodeDecodeError:
        return ""


def _decode(header: bytes, body: bytes, camera: str) -> AuEnvelope:
    kind, codec, framing = AuKind(header[8]), AuCodec(header[9]), AuFraming(header[10])
    camera_size = int.from_bytes(header[72:74], "little")
    caps_size = int.from_bytes(header[74:76], "little")
    codec_size = int.from_bytes(header[76:80], "little")
    payload_size = int.from_bytes(header[80:84], "little")
    if camera_size + caps_size + codec_size + payload_size != len(body):
        raise ValueError("native AU body framing invalid")
    caps_end, codec_end = camera_size + caps_size, camera_size + caps_size + codec_size
    return AuEnvelope(
        kind,
        codec,
        framing,
        bool(header[11]),
        int.from_bytes(header[12:16], "little"),
        int.from_bytes(header[16:24], "little"),
        int.from_bytes(header[24:32], "little"),
        int.from_bytes(header[32:40], "little", signed=True),
        int.from_bytes(header[40:48], "little", signed=True),
        int.from_bytes(header[48:56], "little", signed=True),
        Fraction(
            int.from_bytes(header[56:60], "little", signed=True),
            int.from_bytes(header[60:64], "little", signed=True),
        ),
        int.from_bytes(header[64:68], "little"),
        int.from_bytes(header[68:72], "little"),
        camera,
        body[camera_size:caps_end].decode(),
        body[caps_end:codec_end],
        body[codec_end:],
    )


def _nal_length_size(codec_data: bytes, codec_name: str) -> int:
    index = 4 if codec_name == "h264" else 21
    if len(codec_data) <= index:
        raise ValueError("native codec data lacks NAL length size")
    return (codec_data[index] & 0b11) + 1


__all__ = [
    "MAX_AU_FRAME_BYTES",
    "AuCodec",
    "AuEnvelope",
    "AuFrameError",
    "AuFraming",
    "AuKind",
    "receive_envelope",
    "stream_configuration",
]
