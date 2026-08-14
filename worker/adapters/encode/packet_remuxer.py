from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast, final

import av
from av.audio.codeccontext import AudioCodecContext
from av.video.codeccontext import VideoCodecContext

from worker.adapters.encode.adapter_errors import ClipRemuxError
from worker.adapters.encode.models import ClipArtifact, RemuxStreamFact
from worker.types.source_packet import (
    SourcePacket,
    SourceStreamConfiguration,
    SourceStreamDescriptor,
)

REMUX_METHOD = "pyav-packet-stream-copy"


@dataclass(frozen=True, slots=True)
class _MuxedPacketFact:
    stream_index: int
    pts: int | None
    dts: int | None
    duration: int | None
    time_base: Fraction | None
    is_keyframe: bool
    payload: bytes


@final
class PyAvPacketRemuxer:
    """Mux original encoded packet payloads without invoking any encoder."""

    def remux(
        self,
        packets: Sequence[SourcePacket],
        configuration: SourceStreamConfiguration,
        output_path: Path,
    ) -> ClipArtifact:
        if not packets:
            raise ClipRemuxError("cannot remux an empty packet selection")
        if not configuration.mux_template:
            raise ClipRemuxError("source codec template is unavailable")
        expected = tuple(packets)
        try:
            _validate_source_timeline(expected, configuration)
        except ValueError as exc:
            raise ClipRemuxError(
                f"source packet timeline is invalid ({type(exc).__name__})"
            ) from exc
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._write(expected, configuration, temporary)
            translations, translation_seconds = self._verify(
                expected,
                configuration,
                temporary,
            )
            os.replace(temporary, output_path)
        except (OSError, ValueError, av.FFmpegError) as exc:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise ClipRemuxError(f"source packet remux failed ({type(exc).__name__})") from exc
        first = packets[0]
        starts = [packet.presentation_time for packet in packets]
        duration = float(max(starts) - min(starts))
        return ClipArtifact(
            path=output_path,
            generation=first.epoch.stream_epoch,
            segment_count=1,
            duration_s=max(duration, 0.001),
            worker_boot_id=first.epoch.worker_boot_id,
            camera_id=first.epoch.camera_id,
            stream_epoch=first.epoch.stream_epoch,
            media_origin_pts_sec=float(min(starts)),
            selected_start_pts_sec=float(min(starts)),
            selected_end_pts_sec=float(max(starts)),
            packet_count=len(packets),
            configuration_id=configuration.configuration_id,
            streams=tuple(
                _stream_fact(
                    stream,
                    sum(packet.stream_index == stream.index for packet in expected),
                    translations.get(stream.index),
                )
                for stream in configuration.streams
            ),
            remux_method=REMUX_METHOD,
            remux_version=av.__version__,
            timestamp_translation_seconds=translation_seconds,
        )

    def _write(
        self,
        packets: tuple[SourcePacket, ...],
        configuration: SourceStreamConfiguration,
        path: Path,
    ) -> None:
        template = av.open(io.BytesIO(configuration.mux_template), mode="r")
        output = av.open(
            str(path),
            mode="w",
            format="mp4",
            options={"movflags": "+faststart", "avoid_negative_ts": "disabled"},
        )
        try:
            template_streams = tuple(
                stream for stream in template.streams if stream.type in {"video", "audio"}
            )
            if len(template_streams) != len(configuration.streams):
                raise ValueError("codec template stream count changed")
            output_streams = {}
            for descriptor, template_stream in zip(
                configuration.streams, template_streams, strict=True
            ):
                output_stream = output.add_stream_from_template(template_stream)
                output_stream.time_base = descriptor.time_base
                output_streams[descriptor.index] = output_stream
            for source in packets:
                packet = av.Packet(source.payload)
                packet.pts = source.pts
                packet.dts = source.dts
                packet.duration = source.duration
                packet.time_base = source.stream.time_base
                packet.stream = output_streams[source.stream_index]
                packet.is_keyframe = source.is_keyframe
                output.mux(packet)
        finally:
            output.close()
            template.close()

    def _verify(
        self,
        expected: tuple[SourcePacket, ...],
        configuration: SourceStreamConfiguration,
        path: Path,
    ) -> tuple[dict[int, int], Fraction]:
        container = av.open(str(path), mode="r")
        template = av.open(io.BytesIO(configuration.mux_template), mode="r")
        try:
            actual_streams = tuple(
                stream for stream in container.streams if stream.type in {"video", "audio"}
            )
            template_streams = tuple(
                stream for stream in template.streams if stream.type in {"video", "audio"}
            )
            if len(actual_streams) != len(configuration.streams) or len(template_streams) != len(
                configuration.streams
            ):
                raise ValueError("remuxed stream count changed")
            container_normalized_streams: set[int] = set()
            for descriptor, stream, template_stream in zip(
                configuration.streams,
                actual_streams,
                template_streams,
                strict=True,
            ):
                codec = stream.codec_context
                template_codec = template_stream.codec_context
                if codec.name != descriptor.codec_name:
                    raise ValueError("remuxed codec identity changed")
                if stream.time_base != descriptor.time_base:
                    raise ValueError("remuxed stream time base changed")
                if descriptor.codec_tag != str(
                    template_codec.codec_tag or ""
                ) or descriptor.extradata != bytes(template_codec.extradata or b""):
                    container_normalized_streams.add(descriptor.index)
                if str(codec.codec_tag or "") != str(template_codec.codec_tag or "") or bytes(
                    codec.extradata or b""
                ) != bytes(template_codec.extradata or b""):
                    raise ValueError("remuxed codec container configuration changed")
                if descriptor.media_type == "video":
                    video = cast("VideoCodecContext", codec)
                    if video.width != descriptor.width or video.height != descriptor.height:
                        raise ValueError("remuxed video geometry changed")
                if descriptor.media_type == "audio":
                    audio = cast("AudioCodecContext", codec)
                    if (
                        audio.sample_rate != descriptor.sample_rate
                        or audio.channels != descriptor.channels
                    ):
                        raise ValueError("remuxed audio configuration changed")
            output_to_source = {
                output.index: descriptor.index
                for descriptor, output in zip(configuration.streams, actual_streams, strict=True)
            }
            actual = tuple(
                _MuxedPacketFact(
                    stream_index=output_to_source[packet.stream.index],
                    pts=packet.pts,
                    dts=packet.dts,
                    duration=packet.duration,
                    time_base=None if packet.time_base is None else Fraction(packet.time_base),
                    is_keyframe=packet.is_keyframe,
                    payload=bytes(packet),
                )
                for packet in container.demux(actual_streams)
                if packet.dts is not None and bytes(packet)
            )
            return _verify_packet_facts(
                expected,
                actual,
                configuration,
                container_normalized_streams=container_normalized_streams,
            )
        finally:
            template.close()
            container.close()


def _validate_source_timeline(
    packets: tuple[SourcePacket, ...],
    configuration: SourceStreamConfiguration,
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


def _verify_packet_facts(
    expected: tuple[SourcePacket, ...],
    actual: tuple[_MuxedPacketFact, ...],
    configuration: SourceStreamConfiguration,
    *,
    container_normalized_streams: set[int] | None = None,
) -> tuple[dict[int, int], Fraction]:
    if len(actual) != len(expected):
        raise ValueError("remuxed packet count changed")
    translations: dict[int, int] = {}
    previous_dts: dict[int, int] = {}
    normalized = set() if container_normalized_streams is None else container_normalized_streams
    for source, packet in zip(expected, actual, strict=True):
        if packet.pts is None or packet.dts is None or packet.duration is None:
            raise ValueError("remuxed packet timeline is incomplete")
        if source.pts is None or source.dts is None or source.duration is None:
            raise ValueError("source packet timeline is incomplete")
        if packet.stream_index != source.stream_index:
            raise ValueError("remuxed packet stream identity changed")
        if packet.duration != source.duration:
            raise ValueError("remuxed packet duration changed")
        if packet.time_base != source.stream.time_base:
            raise ValueError("remuxed packet time base changed")
        if source.stream_index not in normalized:
            if source.stream.media_type == "video" and packet.is_keyframe != source.is_keyframe:
                raise ValueError("remuxed packet keyframe identity changed")
            if packet.payload != source.payload:
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


def _stream_fact(
    descriptor: SourceStreamDescriptor,
    packet_count: int,
    timestamp_translation_ticks: int | None,
) -> RemuxStreamFact:
    return RemuxStreamFact(
        index=descriptor.index,
        media_type=descriptor.media_type,
        codec_name=descriptor.codec_name,
        codec_tag=descriptor.codec_tag,
        time_base=descriptor.time_base,
        extradata_sha256=hashlib.sha256(descriptor.extradata).hexdigest(),
        width=descriptor.width,
        height=descriptor.height,
        sample_rate=descriptor.sample_rate,
        channels=descriptor.channels,
        packet_count=packet_count,
        timestamp_translation_ticks=timestamp_translation_ticks,
    )


__all__ = ["REMUX_METHOD", "PyAvPacketRemuxer"]
