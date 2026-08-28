from __future__ import annotations

import hashlib
import io
import os
import struct
from collections.abc import Sequence
from contextlib import suppress
from fractions import Fraction
from pathlib import Path
from typing import cast, final

import av
from av.audio.codeccontext import AudioCodecContext
from av.stream import Stream
from av.video.codeccontext import VideoCodecContext

from worker.adapters.encode.adapter_errors import ClipRemuxError
from worker.adapters.encode.models import ClipArtifact, RemuxStreamFact
from worker.adapters.encode.packet_normalization import (
    annexb_to_length_prefixed,
)
from worker.adapters.encode.packet_normalization import (
    expected_mux_payload as _expected_mux_payload,
)
from worker.adapters.encode.packet_verification import MuxedPacketFact as _MuxedPacketFact
from worker.adapters.encode.packet_verification import (
    validate_source_timeline as _validate_source_timeline,
)
from worker.adapters.encode.packet_verification import verify_packet_facts as _verify_packet_facts
from worker.types.source_packet import (
    SourcePacket,
    SourceStreamConfiguration,
    SourceStreamDescriptor,
)


def _annexb_to_length_prefixed(payload: bytes, length_size: int) -> bytes | None:
    return annexb_to_length_prefixed(payload, length_size)


REMUX_METHOD = "pyav-packet-stream-copy"
NORMALIZER_VERSION = "annexb-length-prefix.v1"
INTERIOR_PACKET_LOSS = "INTERIOR_PACKET_LOSS"


@final
class PyAvPacketRemuxer:
    def remux(
        self,
        packets: Sequence[SourcePacket],
        configuration: SourceStreamConfiguration,
        output_path: Path,
    ) -> ClipArtifact:
        if not packets or not configuration.mux_template:
            raise ClipRemuxError("source packets and codec template are required")
        expected = tuple(packets)
        try:
            _validate_source_timeline(expected, configuration)
        except ValueError as error:
            raise ClipRemuxError("source packet timeline is invalid") from error
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._write(expected, configuration, temporary)
            translations, translation_seconds, interior_loss = self._verify(
                expected, configuration, temporary
            )
            os.replace(temporary, output_path)
        except (OSError, ValueError, av.FFmpegError) as error:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise ClipRemuxError(
                f"source packet remux failed ({type(error).__name__})"
            ) from error
        au_hash, au_size = _write_au_index(output_path.with_name("au-index.cbor"), expected)
        starts = [packet.presentation_time for packet in expected]
        first = expected[0]
        return ClipArtifact(
            path=output_path,
            generation=first.epoch.stream_epoch,
            segment_count=1,
            duration_s=max(float(max(starts) - min(starts)), 0.001),
            worker_boot_id=first.epoch.worker_boot_id,
            camera_id=first.epoch.camera_id,
            stream_epoch=first.epoch.stream_epoch,
            media_origin_pts_sec=float(min(starts)),
            selected_start_pts_sec=float(min(starts)),
            selected_end_pts_sec=float(max(starts)),
            packet_count=len(expected),
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
            au_index_sha256=au_hash,
            au_index_size_bytes=au_size,
            au_index_schema=1,
            au_index_count=len(expected),
        )

    def _write(
        self,
        packets: tuple[SourcePacket, ...],
        configuration: SourceStreamConfiguration,
        path: Path,
    ) -> None:
        template = av.open(io.BytesIO(configuration.mux_template), mode="r")
        output = av.open(
            str(path), mode="w", format="mp4",
            options={"movflags": "+faststart", "avoid_negative_ts": "disabled"},
        )
        try:
            template_streams = tuple(
                stream for stream in template.streams if stream.type in {"video", "audio"}
            )
            if len(template_streams) != len(configuration.streams):
                raise ValueError("codec template stream count changed")
            output_streams: dict[int, Stream] = {}
            for descriptor, source_stream in zip(
                configuration.streams, template_streams, strict=True
            ):
                stream = output.add_stream_from_template(source_stream)
                stream.time_base = descriptor.time_base
                output_streams[descriptor.index] = stream
            for source in packets:
                packet = av.Packet(_expected_mux_payload(source))
                packet.pts, packet.dts, packet.duration = source.pts, source.dts, source.duration
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
            for descriptor, stream, template_stream in zip(
                configuration.streams, actual_streams, template_streams, strict=True
            ):
                _verify_stream(descriptor, stream, template_stream)
            output_to_source = {
                output.index: descriptor.index
                for descriptor, output in zip(configuration.streams, actual_streams, strict=True)
            }
            actual = tuple(
                _MuxedPacketFact(
                    output_to_source[packet.stream.index], packet.pts, packet.dts, packet.duration,
                    None if packet.time_base is None else Fraction(packet.time_base),
                    packet.is_keyframe, bytes(packet),
                )
                for packet in container.demux(actual_streams)
                if packet.dts is not None and bytes(packet)
            )
            interior_loss: set[int] = set()
            translations, seconds = _verify_packet_facts(
                expected, actual, configuration, interior_loss=interior_loss
            )
            return translations, seconds, interior_loss
        finally:
            template.close()
            container.close()


def _verify_stream(
    descriptor: SourceStreamDescriptor, stream: Stream, template: Stream
) -> None:
    codec, template_codec = stream.codec_context, template.codec_context
    if codec.name != descriptor.codec_name or stream.time_base != descriptor.time_base:
        raise ValueError("remuxed stream identity changed")
    if str(codec.codec_tag or "") != str(template_codec.codec_tag or "") or bytes(
        codec.extradata or b""
    ) != bytes(template_codec.extradata or b""):
        raise ValueError("remuxed codec container configuration changed")
    if descriptor.media_type == "video":
        video = cast("VideoCodecContext", codec)
        if video.width != descriptor.width or video.height != descriptor.height:
            raise ValueError("remuxed video geometry changed")
    elif descriptor.media_type == "audio":
        audio = cast("AudioCodecContext", codec)
        if audio.sample_rate != descriptor.sample_rate or audio.channels != descriptor.channels:
            raise ValueError("remuxed audio configuration changed")


def _write_au_index(path: Path, packets: tuple[SourcePacket, ...]) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with path.open("xb") as output:
        records = [b"SAUI1" + len(packets).to_bytes(4, "little")]
        records.extend(
            struct.pack(
                "<QIqqq?Q", packet.arrival_index, packet.stream_index, packet.pts,
                packet.dts, packet.duration, packet.is_keyframe, packet.epoch.stream_epoch,
            )
            + bytes.fromhex(packet.configuration.configuration_id)
            + hashlib.sha256(packet.payload).digest()
            for packet in packets
        )
        for record in records:
            _ = output.write(record)
            digest.update(record)
            size += len(record)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest(), size


def _stream_fact(
    descriptor: SourceStreamDescriptor, packet_count: int,
    timestamp_translation_ticks: int | None,
) -> RemuxStreamFact:
    framing = descriptor.stream_format
    return RemuxStreamFact(
        descriptor.index, descriptor.media_type, descriptor.codec_name, descriptor.codec_tag,
        descriptor.time_base, hashlib.sha256(descriptor.extradata).hexdigest(),
        descriptor.width, descriptor.height, descriptor.sample_rate, descriptor.channels,
        packet_count, timestamp_translation_ticks, framing,
        "length-prefixed" if framing == "byte-stream" else framing,
        NORMALIZER_VERSION if framing == "byte-stream" else "none",
        descriptor.parser_caps_sha256,
    )


__all__ = ["INTERIOR_PACKET_LOSS", "REMUX_METHOD", "PyAvPacketRemuxer"]
