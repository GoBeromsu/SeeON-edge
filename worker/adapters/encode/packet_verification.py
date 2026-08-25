"""Fail-closed source timeline and post-remux packet verification."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from worker.adapters.encode.packet_normalization import expected_mux_payload
from worker.types.source_packet import SourcePacket, SourceStreamConfiguration


@dataclass(frozen=True, slots=True)
class MuxedPacketFact:
    stream_index: int
    pts: int | None
    dts: int | None
    duration: int | None
    time_base: Fraction | None
    is_keyframe: bool
    payload: bytes


def validate_source_timeline(
    packets: tuple[SourcePacket, ...], configuration: SourceStreamConfiguration
) -> None:
    identities = {(packet.epoch, packet.configuration.configuration_id) for packet in packets}
    if len(identities) != 1 or next(iter(identities))[1] != configuration.configuration_id:
        raise ValueError("packet selection mixes stream epochs or configurations")
    previous_arrival = -1
    previous_dts: dict[int, int] = {}
    for packet in packets:
        if packet.pts is None or packet.dts is None or packet.duration is None:
            raise ValueError("packet timestamps and duration must be present")
        if packet.duration < 0:
            raise ValueError("packet duration is negative")
        if packet.discontinuity is not None:
            raise ValueError("packet timeline contains a discontinuity")
        if packet.arrival_index <= previous_arrival:
            raise ValueError("packet demux order is not strictly increasing")
        previous_arrival = packet.arrival_index
        prior_dts = previous_dts.get(packet.stream_index)
        if prior_dts is not None and packet.dts <= prior_dts:
            raise ValueError("packet decode timeline is not strictly increasing")
        previous_dts[packet.stream_index] = packet.dts


def verify_packet_facts(
    expected: tuple[SourcePacket, ...],
    actual: tuple[MuxedPacketFact, ...],
    configuration: SourceStreamConfiguration,
    *,
    container_normalized_streams: set[int] | None = None,
    interior_loss: set[int] | None = None,
) -> tuple[dict[int, int], Fraction]:
    if len(actual) != len(expected):
        raise ValueError("remuxed packet count changed")
    del container_normalized_streams
    translations: dict[int, int] = {}
    previous_dts: dict[int, int] = {}
    for source, packet in zip(expected, actual, strict=True):
        if packet.pts is None or packet.dts is None or packet.duration is None:
            raise ValueError("remuxed packet timeline is incomplete")
        if source.pts is None or source.dts is None or source.duration is None:
            raise ValueError("source packet timeline is incomplete")
        if packet.stream_index != source.stream_index:
            raise ValueError("remuxed packet stream identity changed")
        if source.duration and packet.duration * 2 >= source.duration * 3:
            if interior_loss is None:
                raise ValueError(
                    "remuxed packet duration changed: stretched across missing packets "
                    f"(source={source.duration} remuxed={packet.duration} "
                    f"time_base={source.stream.time_base} stream={source.stream_index})"
                )
            interior_loss.add(source.stream_index)
        if packet.time_base != source.stream.time_base:
            raise ValueError("remuxed packet time base changed")
        if source.stream.media_type == "video" and source.is_keyframe and not packet.is_keyframe:
            raise ValueError("remuxed packet keyframe identity changed: keyframe lost")
        if packet.payload != source.payload and packet.payload != expected_mux_payload(source):
            raise ValueError("remuxed packet payload changed")
        pts_translation = packet.pts - source.pts
        dts_translation = packet.dts - source.dts
        if pts_translation != dts_translation:
            raise ValueError("remuxed PTS-DTS composition offset changed")
        prior_translation = translations.setdefault(source.stream_index, pts_translation)
        if prior_translation != pts_translation:
            raise ValueError("remuxed packet timestamps drift nonuniformly")
        if (source.pts >= 0 and packet.pts < 0) or (source.dts >= 0 and packet.dts < 0):
            raise ValueError("remuxed timestamp translation created a negative timeline")
        prior_dts = previous_dts.get(source.stream_index)
        if prior_dts is not None and packet.dts <= prior_dts:
            raise ValueError("remuxed decode order changed")
        previous_dts[source.stream_index] = packet.dts
    translation_seconds = {
        translation * configuration.stream(stream_index).time_base
        for stream_index, translation in translations.items()
    }
    if len(translation_seconds) != 1:
        raise ValueError("remuxed streams received nonuniform timestamp translations")
    return translations, next(iter(translation_seconds))


__all__ = ["MuxedPacketFact", "validate_source_timeline", "verify_packet_facts"]
