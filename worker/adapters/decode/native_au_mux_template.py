"""PyAV mux-template capsule for one parser-aligned native access unit."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from fractions import Fraction

import av

from worker.types.source_packet import SourceStreamDescriptor

# Annex B start codes are 3- or 4-byte; the 4-byte form is the 3-byte form
# with a leading zero, so one pattern covers both.
_ANNEX_B_START_CODE = b"\x00\x00\x01"


@dataclass(frozen=True, slots=True)
class NativeAuTemplateInput:
    payload: bytes
    duration: int
    keyframe: bool


def distinct_parameter_sets(codec_data: bytes) -> bytes:
    """Collapse a repeated parameter-set blob to its distinct units, in order.

    Cameras retransmit VPS/SPS/PPS periodically, and an access unit sometimes
    carries the same set twice. The blob then alternates between two byte
    strings that describe the identical configuration -- measured on the live
    fleet as codec_data_len flipping 210 to 105, exactly two copies versus one,
    with every other signature input unchanged. Hashing the raw bytes turned
    that benign retransmission into a configuration change, and each one
    requested a source rebuild: 241 of them in a four-minute window.

    Splitting on Annex B start codes is safe for a blob that is not Annex B:
    it simply yields one unit and the result is the original bytes.

    A four-byte start code is the three-byte form with a leading zero, and that
    zero lands at the END of the preceding unit when splitting on the three-byte
    pattern. Trailing zeros are therefore stripped before comparing, otherwise
    ``PPS\\x00`` and ``PPS`` read as different units and identical parameter
    sets fail to deduplicate -- which is exactly what left 49 of these gaps
    still firing on cameras that emit four-byte start codes.
    """
    units = [
        stripped
        for unit in codec_data.split(_ANNEX_B_START_CODE)
        if (stripped := unit.rstrip(b"\x00"))
    ]
    return _ANNEX_B_START_CODE.join(dict.fromkeys(units))


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
        distinct_parameter_sets(codec_data),
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
    "distinct_parameter_sets",
    "native_configuration_signature",
]
