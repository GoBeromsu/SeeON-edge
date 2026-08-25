"""Bounded and explicit encoded packet framing normalization."""

from __future__ import annotations

from worker.types.source_packet import SourcePacket

_ANNEXB_START_3 = b"\x00\x00\x01"
_ANNEXB_START_4 = b"\x00\x00\x00\x01"
_MAX_NAL_UNITS = 4_096


def annexb_to_length_prefixed(payload: bytes, length_size: int) -> bytes | None:
    if not payload.startswith((_ANNEXB_START_3, _ANNEXB_START_4)):
        return None
    starts: list[tuple[int, int]] = []
    index, limit = 0, len(payload)
    while index < limit - 2:
        if payload[index] == 0 and payload[index + 1] == 0:
            if payload[index + 2] == 1:
                starts.append((index, 3))
                if len(starts) > _MAX_NAL_UNITS:
                    raise ValueError("Annex-B NAL unit count exceeds bound")
                index += 3
                continue
            if index + 3 < limit and payload[index + 2] == 0 and payload[index + 3] == 1:
                starts.append((index, 4))
                if len(starts) > _MAX_NAL_UNITS:
                    raise ValueError("Annex-B NAL unit count exceeds bound")
                index += 4
                continue
        index += 1
    if not starts:
        return None
    units: list[bytes] = []
    maximum = (1 << (length_size * 8)) - 1
    for position, (offset, code) in enumerate(starts):
        begin = offset + code
        end = starts[position + 1][0] if position + 1 < len(starts) else limit
        unit = payload[begin:end]
        if unit:
            if len(unit) > maximum:
                raise ValueError("NAL unit exceeds configured length field")
            units.append(len(unit).to_bytes(length_size, "big") + unit)
    return b"".join(units) if units else None


def expected_mux_payload(source: SourcePacket) -> bytes:
    descriptor = source.stream
    if descriptor.stream_format == "byte-stream":
        if descriptor.nal_length_size is None:
            raise ValueError("Annex-B stream lacks declared NAL length size")
        normalized = annexb_to_length_prefixed(source.payload, descriptor.nal_length_size)
        if normalized is None:
            raise ValueError("declared Annex-B AU has no start code")
        return normalized
    if descriptor.stream_format in {"avc", "avc3", "hvc1", "hev1"}:
        return source.payload
    normalized = annexb_to_length_prefixed(source.payload, 4)
    return source.payload if normalized is None else normalized


__all__ = ["annexb_to_length_prefixed", "expected_mux_payload"]
