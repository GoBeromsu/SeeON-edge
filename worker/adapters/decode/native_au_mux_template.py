"""PyAV mux-template capsule for one parser-aligned native access unit."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from fractions import Fraction

import av

from worker.types.source_packet import SourceStreamDescriptor


@dataclass(frozen=True, slots=True)
class NativeAuTemplateInput:
    payload: bytes
    duration: int
    keyframe: bool


def native_configuration_signature(
    codec: int,
    framing: int,
    parser_caps: str,
    codec_data: bytes,
    width: int,
    height: int,
    time_base: Fraction,
) -> str:
    digest = hashlib.sha256()
    for payload in (
        bytes((codec, framing)),
        parser_caps.encode(),
        codec_data,
        width.to_bytes(4, "little"),
        height.to_bytes(4, "little"),
        time_base.numerator.to_bytes(4, "little", signed=True),
        time_base.denominator.to_bytes(4, "little", signed=True),
    ):
        digest.update(payload)
    return digest.hexdigest()


def build_native_au_mux_template(
    descriptor: SourceStreamDescriptor,
    source: NativeAuTemplateInput,
) -> bytes:
    output_bytes = io.BytesIO()
    output = av.open(output_bytes, mode="w", format="mp4")
    try:
        match descriptor.codec_name:
            case "h264":
                stream = output.add_stream("h264", rate=30)
            case "hevc":
                stream = output.add_stream("hevc", rate=30)
            case _:
                raise ValueError("native AU codec is unsupported")
        stream.width = descriptor.width or 0
        stream.height = descriptor.height or 0
        stream.codec_context.extradata = descriptor.extradata
        stream.time_base = descriptor.time_base
        packet = av.Packet(source.payload)
        packet.pts = 0
        packet.dts = 0
        packet.duration = source.duration
        packet.time_base = descriptor.time_base
        packet.is_keyframe = source.keyframe
        packet.stream = stream
        output.mux(packet)
    finally:
        output.close()
    payload = output_bytes.getvalue()
    if not payload:
        raise ValueError("native AU mux template is empty")
    return payload


__all__ = [
    "NativeAuTemplateInput",
    "build_native_au_mux_template",
    "native_configuration_signature",
]
