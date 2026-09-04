from __future__ import annotations

import io
import logging
from collections.abc import Callable
from fractions import Fraction
from typing import Final, Protocol, cast, final

import av
from av.audio.codeccontext import AudioCodecContext
from av.container import InputContainer
from av.stream import Stream
from av.video.codeccontext import VideoCodecContext

from worker.interfaces.source_packet import SourcePacketSink
from worker.types.source_packet import (
    MediaType,
    SourcePacket,
    SourceStreamConfiguration,
    SourceStreamDescriptor,
    StreamEpoch,
)

LOGGER: Final = logging.getLogger(__name__)
_MAX_PRE_TEMPLATE_PACKETS: Final = 512


class DecodeConfig(Protocol):
    @property
    def camera_id(self) -> str: ...

    @property
    def url(self) -> str: ...

    @property
    def open_timeout_ms(self) -> int: ...

    @property
    def read_timeout_ms(self) -> int: ...


ConfigurationConsumer = Callable[[SourceStreamConfiguration, bool], None]
PacketConsumer = Callable[[av.Packet, SourcePacket, bool], None]


@final
class PyAvPacketDemuxer:
    """Demux one source and publish byte-identical compressed packets."""

    def __init__(
        self,
        config: DecodeConfig,
        container: InputContainer,
        sink: SourcePacketSink,
    ) -> None:
        self.config = config
        self.container = container
        self.sink = sink
        self.streams = tuple(
            stream for stream in container.streams if stream.type in {"video", "audio"}
        )
        videos = tuple(stream for stream in self.streams if stream.type == "video")
        if len(videos) != 1:
            raise RuntimeError("packet-preserving ingest requires exactly one video stream")
        self.video = videos[0]
        self._configuration: SourceStreamConfiguration | None = None
        self._observed_descriptors: tuple[SourceStreamDescriptor, ...] | None = None
        self._observed_id = ""
        self._arrival_index = 0
        self._last_dts: dict[int, Fraction] = {}
        self.packet_drop_count = 0

    def _observed_configuration_id(self) -> str:
        """Configuration id of the streams as they look right now.

        The id is a pure function of the stream descriptors, so it is only
        recomputed (sorting, hashing, JSON) when a descriptor actually changes
        rather than on every demuxed packet.
        """
        descriptors = tuple(_descriptor(stream) for stream in self.streams)
        if descriptors != self._observed_descriptors:
            self._observed_id = SourceStreamConfiguration.from_streams(
                list(descriptors)
            ).configuration_id
            self._observed_descriptors = descriptors
        return self._observed_id

    def run(
        self,
        epoch: StreamEpoch,
        *,
        stop_requested: Callable[[], bool],
        on_configuration: ConfigurationConsumer,
        on_packet: PacketConsumer,
    ) -> None:
        pending: list[tuple[av.Packet, bytes]] = []
        configuration_changed = False
        for packet in self.container.demux(self.streams):
            if stop_requested():
                break
            payload = bytes(packet)
            if packet.dts is None or not payload:
                continue
            observed_id = self._observed_configuration_id()
            if (
                self._configuration is not None
                and observed_id != self._configuration.configuration_id
            ):
                self._configuration = None
                pending.clear()
                configuration_changed = True
            if self._configuration is None:
                pending.append((packet, payload))
                _validate_pending_template_bound(pending)
                first_by_stream = {item.stream.index: item for item, _ in pending}
                if set(first_by_stream) != {stream.index for stream in self.streams}:
                    continue
                self._configuration = _configuration(
                    self.streams,
                    _template_capsule(
                        self.streams,
                        tuple(first_by_stream[stream.index] for stream in self.streams),
                    ),
                )
                on_configuration(self._configuration, configuration_changed)
                configuration_changed = False
                for buffered, buffered_payload in pending:
                    source = self._publish(buffered, buffered_payload, epoch)
                    on_packet(buffered, source, buffered is packet)
                pending.clear()
            else:
                source = self._publish(packet, payload, epoch)
                on_packet(packet, source, True)

    def _publish(self, packet: av.Packet, payload: bytes, epoch: StreamEpoch) -> SourcePacket:
        configuration = self._configuration
        if configuration is None:
            raise RuntimeError("packet stream configuration is unavailable")
        stream = configuration.stream(packet.stream.index)
        dts = packet.dts
        if dts is None:  # pragma: no cover - run excludes timestamp-less packets
            raise RuntimeError("source packet DTS disappeared")
        dts_time = dts * stream.time_base
        previous = self._last_dts.get(stream.index)
        discontinuity = None
        if previous is not None and (dts_time < previous or dts_time - previous > 60):
            discontinuity = "dts-backward-or-jump"
        self._last_dts[stream.index] = dts_time
        source = SourcePacket(
            epoch=epoch,
            configuration=configuration,
            stream_index=stream.index,
            pts=packet.pts,
            dts=dts,
            duration=packet.duration,
            is_keyframe=packet.is_keyframe,
            payload=payload,
            arrival_index=self._arrival_index,
            discontinuity=discontinuity,
        )
        self._arrival_index += 1
        if not self.sink.append(source):
            self.packet_drop_count += 1
            if self.packet_drop_count & (self.packet_drop_count - 1) == 0:
                LOGGER.warning(
                    "source packet ring dropped packet: camera_id=%s dropped=%s",
                    self.config.camera_id,
                    self.packet_drop_count,
                    extra={
                        "camera_id": self.config.camera_id,
                        "dropped_packets": self.packet_drop_count,
                    },
                )
        return source


def stream_descriptor(stream: Stream) -> SourceStreamDescriptor:
    return _descriptor(stream)


def _configuration(streams: tuple[Stream, ...], mux_template: bytes) -> SourceStreamConfiguration:
    return SourceStreamConfiguration.from_streams(
        [_descriptor(stream) for stream in streams],
        mux_template=mux_template,
    )


def _descriptor(stream: Stream) -> SourceStreamDescriptor:
    codec = stream.codec_context
    media_type = stream.type
    if media_type not in {"video", "audio"}:
        raise ValueError("unsupported source stream type")
    time_base = stream.time_base
    if time_base is None:
        raise ValueError("source stream time base is unavailable")
    video = cast("VideoCodecContext", codec) if media_type == "video" else None
    audio = cast("AudioCodecContext", codec) if media_type == "audio" else None
    return SourceStreamDescriptor(
        index=stream.index,
        media_type=cast("MediaType", media_type),
        codec_name=codec.name,
        codec_tag=str(codec.codec_tag or ""),
        time_base=time_base,
        extradata=bytes(codec.extradata or b""),
        width=None if video is None else video.width,
        height=None if video is None else video.height,
        sample_rate=None if audio is None else audio.sample_rate,
        channels=None if audio is None else getattr(audio, "channels", None),
        profile=(str(codec.profile) if getattr(codec, "profile", None) is not None else None),
        level=getattr(codec, "level", None),
    )


def _validate_pending_template_bound(pending: list[tuple[av.Packet, bytes]]) -> None:
    if len(pending) > _MAX_PRE_TEMPLATE_PACKETS:
        raise RuntimeError("codec template was not available within bounded history")


def _template_capsule(streams: tuple[Stream, ...], sources: tuple[av.Packet, ...]) -> bytes:
    buffer = io.BytesIO()
    output = av.open(buffer, mode="w", format="mp4")
    try:
        mapped = {stream.index: output.add_stream_from_template(stream) for stream in streams}
        for source in sources:
            packet = av.Packet(bytes(source))
            packet.pts = source.pts
            packet.dts = source.dts
            packet.duration = source.duration
            packet.time_base = source.time_base
            packet.is_keyframe = source.is_keyframe
            packet.stream = mapped[source.stream.index]
            output.mux(packet)
    finally:
        output.close()
    payload = buffer.getvalue()
    if not payload:
        raise RuntimeError("codec template capsule is empty")
    return payload


__all__ = ["DecodeConfig", "PyAvPacketDemuxer", "stream_descriptor"]
