from __future__ import annotations

import io
import logging
import threading
import time
from collections import deque
from fractions import Fraction
from typing import Final, Protocol, cast, final

import av
import numpy as np
from av.audio.codeccontext import AudioCodecContext
from av.codec.hwaccel import HWAccel
from av.container import InputContainer
from av.stream import Stream
from av.video.codeccontext import VideoCodecContext
from numpy.typing import NDArray

from contracts.frame import Frame
from worker.interfaces.source_packet import SourcePacketSink
from worker.types import FramePacket
from worker.types.source_packet import (
    MediaType,
    SourcePacket,
    SourceStreamConfiguration,
    SourceStreamDescriptor,
    StreamEpoch,
)

LOGGER: Final = logging.getLogger(__name__)
_MAX_PENDING_FRAMES: Final = 2
_MAX_PRE_TEMPLATE_PACKETS: Final = 512


class _DecodeConfig(Protocol):
    @property
    def camera_id(self) -> str: ...

    @property
    def url(self) -> str: ...

    @property
    def open_timeout_ms(self) -> int: ...

    @property
    def read_timeout_ms(self) -> int: ...


@final
class PyAvPreservingSession:
    def __init__(
        self,
        config: _DecodeConfig,
        container: InputContainer,
        sink: SourcePacketSink,
    ) -> None:
        self._config = config
        self._container = container
        self._sink = sink
        self._streams = tuple(
            stream for stream in container.streams if stream.type in {"video", "audio"}
        )
        videos = tuple(stream for stream in self._streams if stream.type == "video")
        if len(videos) != 1:
            container.close()
            raise RuntimeError("packet-preserving ingest requires exactly one video stream")
        self._video = videos[0]
        self._epoch: StreamEpoch | None = None
        self._configuration: SourceStreamConfiguration | None = None
        self._arrival_index = 0
        self._last_dts: dict[int, Fraction] = {}
        self._frames: deque[av.VideoFrame] = deque()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._demux_done = threading.Event()
        self._closed = False
        self._eof = False
        self._error: Exception | None = None
        self._seq = 0
        self.packet_drop_count = 0

    def set_stream_identity(self, worker_boot_id: str, stream_epoch: int) -> None:
        with self._condition:
            if self._epoch is not None:
                raise RuntimeError("stream identity is already assigned")
            self._epoch = StreamEpoch(worker_boot_id, self._config.camera_id, stream_epoch)
            self._thread = threading.Thread(
                target=self._demux,
                name=f"packet-demux-{self._config.camera_id}",
                daemon=True,
            )
            self._thread.start()

    def read(self) -> FramePacket | None:
        deadline = time.monotonic() + self._config.read_timeout_ms / 1000.0
        with self._condition:
            while not self._frames and not self._eof and self._error is None and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._error is not None:
                error = self._error
                self._error = None
                raise RuntimeError(
                    f"packet-preserving decode failed ({type(error).__name__})"
                ) from error
            if not self._frames:
                return None
            decoded = self._frames.popleft()
        image = decoded.to_ndarray(format="rgb24")
        if not isinstance(image, np.ndarray):  # pragma: no cover - PyAV guarantee
            raise TypeError("PyAV returned a non-array video frame")
        rgb_image = cast("NDArray[np.uint8]", image)
        height, width = rgb_image.shape[:2]
        source_time_base = Fraction(decoded.time_base) if decoded.time_base is not None else None
        source_pts = decoded.pts
        pts = (
            None
            if source_pts is None or source_time_base is None
            else float(source_pts * source_time_base)
        )
        if pts is None:
            raise RuntimeError("decoded source frame has no authoritative PTS")
        epoch = self._epoch
        if epoch is None:  # pragma: no cover - reads start only after identity assignment
            raise RuntimeError("packet session identity is unavailable")
        seq = self._seq
        self._seq += 1
        return FramePacket(
            camera_id=self._config.camera_id,
            frame=Frame(index=seq, time_sec=pts, image=rgb_image),
            pts=pts,
            seq=seq,
            width=width,
            height=height,
            decode_time_ms=0.0,
            worker_boot_id=epoch.worker_boot_id,
            stream_epoch=epoch.stream_epoch,
            source_pts=source_pts,
            source_dts=decoded.dts,
            source_time_base=source_time_base,
        )

    def wait_demux_complete(self, timeout: float) -> bool:
        return self._demux_done.wait(timeout)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        thread = self._thread
        if thread is None:
            self._container.close()
        elif thread is not threading.current_thread():
            # PyAV/FFmpeg containers are not safe to close from one thread while
            # another thread is inside demux/decode. The configured read timeout
            # releases blocked I/O; the demux owner closes the container below.
            thread.join(timeout=max(5.0, self._config.read_timeout_ms / 1000.0 + 1.0))

    def _demux(self) -> None:
        pending: list[av.Packet] = []
        try:
            for packet in self._container.demux(self._streams):
                if self._closed:
                    break
                if packet.dts is None or not bytes(packet):
                    continue
                observed_configuration_id = SourceStreamConfiguration.from_streams(
                    [_descriptor(stream) for stream in self._streams]
                ).configuration_id
                if (
                    self._configuration is not None
                    and observed_configuration_id != self._configuration.configuration_id
                ):
                    self._configuration = None
                    pending.clear()
                if self._configuration is None:
                    pending.append(packet)
                    _validate_pending_template_bound(pending)
                    first_by_stream = {buffered.stream.index: buffered for buffered in pending}
                    if set(first_by_stream) != {stream.index for stream in self._streams}:
                        continue
                    self._configuration = _configuration(
                        self._streams,
                        _template_capsule(
                            self._streams,
                            tuple(first_by_stream[stream.index] for stream in self._streams),
                        ),
                    )
                    for buffered in pending:
                        self._publish_packet(buffered)
                    pending.clear()
                else:
                    self._publish_packet(packet)
                if packet.stream.index == self._video.index:
                    for frame in packet.decode():
                        if not isinstance(frame, av.VideoFrame):
                            continue
                        with self._condition:
                            if len(self._frames) == _MAX_PENDING_FRAMES:
                                self._frames.popleft()
                            self._frames.append(frame)
                            self._condition.notify()
        except Exception as exc:  # noqa: BLE001 - demux thread boundary
            if not self._closed:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
        finally:
            self._container.close()
            with self._condition:
                self._eof = True
                self._condition.notify_all()
            self._demux_done.set()

    def _publish_packet(self, packet: av.Packet) -> None:
        epoch = self._epoch
        configuration = self._configuration
        if epoch is None or configuration is None:
            raise RuntimeError("packet session identity/configuration is unavailable")
        stream = configuration.stream(packet.stream.index)
        dts = packet.dts
        if dts is None:  # pragma: no cover - caller excludes timestamp-less packets
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
            dts=packet.dts,
            duration=packet.duration,
            is_keyframe=packet.is_keyframe,
            payload=bytes(packet),
            arrival_index=self._arrival_index,
            discontinuity=discontinuity,
        )
        self._arrival_index += 1
        if not self._sink.append(source):
            self.packet_drop_count += 1
            if self.packet_drop_count & (self.packet_drop_count - 1) == 0:
                LOGGER.warning(
                    "source packet ring dropped packet: camera_id=%s dropped=%s",
                    self._config.camera_id,
                    self.packet_drop_count,
                    extra={
                        "camera_id": self._config.camera_id,
                        "dropped_packets": self.packet_drop_count,
                    },
                )


@final
class PyAvPreservingAdapter:
    def __init__(self, sink: SourcePacketSink, *, decode_backend: str) -> None:
        self._sink = sink
        self._decode_backend = decode_backend

    def open(self, config: _DecodeConfig) -> PyAvPreservingSession:
        hwaccel = _hardware_acceleration(self._decode_backend, config)
        try:
            container = av.open(
                config.url,
                mode="r",
                options={"rtsp_transport": "tcp"},
                timeout=(config.open_timeout_ms / 1000.0, config.read_timeout_ms / 1000.0),
                hwaccel=hwaccel,
            )
        except (OSError, ValueError, av.FFmpegError) as exc:
            raise RuntimeError(
                f"packet-preserving source open failed for camera {config.camera_id}"
            ) from exc
        return PyAvPreservingSession(config, container, self._sink)


def _hardware_acceleration(backend: str, config: _DecodeConfig) -> HWAccel | None:
    if backend in {"opencv", "cpu"}:
        return None
    if backend == "nvdec":
        return HWAccel("cuda", allow_software_fallback=False)
    if backend == "vaapi":
        device = getattr(config, "render_device", None)
        return HWAccel("vaapi", device=device, allow_software_fallback=False)
    raise RuntimeError(f"unsupported packet-preserving decode backend: {backend!r}")


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


def _validate_pending_template_bound(pending: list[av.Packet]) -> None:
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


__all__ = ["PyAvPreservingAdapter", "PyAvPreservingSession"]
