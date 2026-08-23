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
            translations, translation_seconds, interior_loss = self._verify(
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
            truncation_reasons=(INTERIOR_PACKET_LOSS,) if interior_loss else (),
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
                reframed = _annexb_to_length_prefixed(source.payload)
                packet = av.Packet(source.payload if reframed is None else reframed)
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
    ) -> tuple[dict[int, int], Fraction, set[int]]:
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
            interior_loss: set[int] = set()
            translations, translation_seconds = _verify_packet_facts(
                expected,
                actual,
                configuration,
                container_normalized_streams=container_normalized_streams,
                interior_loss=interior_loss,
            )
            return translations, translation_seconds, interior_loss
        finally:
            template.close()
            container.close()


INTERIOR_PACKET_LOSS = "INTERIOR_PACKET_LOSS"

_ANNEXB_START_3 = b"\x00\x00\x01"
_ANNEXB_START_4 = b"\x00\x00\x00\x01"


def _annexb_to_length_prefixed(payload: bytes) -> bytes | None:
    """Reframe Annex-B NAL units as 4-byte length-prefixed ones, or None.

    RTSP delivers H.264/HEVC as Annex-B: NAL units separated by 00 00 01 or
    00 00 00 01 start codes. MP4 sample descriptions (avcC/hvcC) instead carry
    length-prefixed units, and the MOV muxer only converts when the *extradata*
    it was handed is itself Annex-B. This deployment's mux template is built by
    writing an MP4 header, so its extradata is already hvcC -- the muxer
    therefore assumed the samples were length-prefixed too and wrote the
    Annex-B bytes through verbatim.

    A decoder then reads the leading 00 00 00 01 as a NAL length of 1, consumes
    one byte, and reads the next four (01 30 00 00) as a length of 19922944 --
    the exact "Invalid NAL unit size (19922944 > 798)" this system produced for
    every clip it ever recorded, none of which decoded a single frame.

    Returns None when the payload is not Annex-B, so an already-conforming
    source stays a byte-true copy.
    """
    if not payload.startswith((_ANNEXB_START_3, _ANNEXB_START_4)):
        return None
    starts: list[tuple[int, int]] = []
    index = 0
    limit = len(payload)
    while index < limit - 2:
        if payload[index] == 0 and payload[index + 1] == 0:
            if payload[index + 2] == 1:
                starts.append((index, 3))
                index += 3
                continue
            if index + 3 < limit and payload[index + 2] == 0 and payload[index + 3] == 1:
                starts.append((index, 4))
                index += 4
                continue
        index += 1
    if not starts:
        return None
    units: list[bytes] = []
    for position, (offset, code) in enumerate(starts):
        begin = offset + code
        end = starts[position + 1][0] if position + 1 < len(starts) else limit
        unit = payload[begin:end]
        if unit:
            units.append(len(unit).to_bytes(4, "big") + unit)
    return b"".join(units) if units else None


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
    interior_loss: set[int] | None = None,
) -> tuple[dict[int, int], Fraction]:
    """Verify the remux.

    ``interior_loss`` turns the missing-packet check from a refusal into a
    report: the caller passes a sink, the affected stream indices land in it,
    and the clip is published carrying a truncation reason. Refusing the whole
    clip destroys 60 seconds of usable footage over one dropped frame, which is
    the loss ADR-0001 exists to prevent; recording the gap keeps the evidence
    and keeps it honest. Callers that pass no sink still get the refusal.
    """
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
        # Duration is refined by the container, so it cannot be compared for
        # equality -- but it must not stretch far enough to cover a packet that
        # went missing.
        #
        # Measured on this deployment's live RTSP cameras (time_base 1/90000):
        # the depacketizer declares a nominal 3000 ticks on every packet (an
        # exact 1/30s) while MP4 writes the true inter-frame delta -- 2880,
        # 2970, 3060 -- because the stream jitters. Demanding equality rejected
        # 100% of recorded clips as REMUX_FAILED and deleted them, with every
        # payload byte identical and every PTS preserved exactly.
        #
        # Equality cannot simply be dropped either: when the ring evicts a
        # packet from inside the selected window, the survivors keep their
        # exact PTS, the count matches the (already shortened) selection, and
        # payloads may be exempt on a normalized stream -- so the hole shows up
        # *only* as a duration stretched across it. Discontinuity marking does
        # not help; it triggers on a 60-second jump, not one dropped frame.
        #
        # A refinement moves the value by jitter (measured within 4%). A hole
        # multiplies it by the number of frames lost, so it starts at 2x. The
        # bound below sits between the two with an order of magnitude of
        # headroom on the jitter side.
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
        if source.stream_index not in normalized:
            # Losing a keyframe breaks seeking, so that still fails closed.
            # Gaining one does not: once the NAL units are correctly framed the
            # container parses their types instead of trusting the demuxer's
            # packet flag, and it is right to. Measured here on a real source
            # packet whose flag said False while its payload began 00 00 01 45
            # -- NAL type 5, an IDR. The clip is more seekable than the flags
            # claimed, which is the direction this guard exists to protect.
            if (
                source.stream.media_type == "video"
                and source.is_keyframe
                and not packet.is_keyframe
            ):
                raise ValueError("remuxed packet keyframe identity changed: keyframe lost")
            if packet.payload != source.payload:
                # The one legitimate rewrite: Annex-B start codes reframed as
                # the length prefixes an MP4 sample description requires. Any
                # other difference is still a corrupted copy.
                if packet.payload != _annexb_to_length_prefixed(source.payload):
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


__all__ = ["INTERIOR_PACKET_LOSS", "REMUX_METHOD", "PyAvPacketRemuxer"]
