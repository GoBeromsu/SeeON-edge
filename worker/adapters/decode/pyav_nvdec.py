from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, NoReturn, final

import av
from av.container import InputContainer

from contracts.frame import Frame
from worker.adapters.decode.nvdec_cuvid.adapter import ffmpeg_decode_args
from worker.adapters.decode.nvdec_cuvid.errors import sanitized_nvdec_error
from worker.adapters.decode.nvdec_cuvid.input_queue import DecoderInputQueue
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig, StreamMetadata
from worker.adapters.decode.nvdec_cuvid.probe import cuvid_decoder_for
from worker.adapters.decode.nvdec_cuvid.process import DecoderProcess, ProcessSpawner
from worker.adapters.decode.nvdec_cuvid.raw_frame import raw_frame_size, rgb24_image
from worker.adapters.decode.pyav_demux import (
    DecodeConfig,
    PyAvPacketDemuxer,
    stream_descriptor,
)
from worker.interfaces.source_packet import EpochRollingSourcePacketSink
from worker.types import FramePacket
from worker.types.source_packet import (
    SourcePacket,
    SourceStreamConfiguration,
    SourceStreamDescriptor,
    StreamEpoch,
)

_PROCESS_REAP_TIMEOUT_SEC: Final = 5.0


@dataclass(slots=True)
class _DecoderState:
    process: DecoderProcess
    metadata: StreamMetadata
    decoder_input: DecoderInputQueue


@final
class NvdecPacketTeeSession:
    """Demux in-process while all video decode runs in one ffmpeg child."""

    def __init__(
        self,
        config: DecodeConfig,
        container: InputContainer,
        sink: EpochRollingSourcePacketSink,
        *,
        process_spawner: ProcessSpawner,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._container = container
        self._sink = sink
        self._demuxer = PyAvPacketDemuxer(config, container, sink)
        self._process_spawner = process_spawner
        self._clock = clock
        self._decoder_config = NvdecCuvidConfig(
            camera_id=config.camera_id,
            url=config.url,
            open_timeout_ms=config.open_timeout_ms,
            read_timeout_ms=config.read_timeout_ms,
            ffmpeg_bin=str(getattr(config, "ffmpeg_bin", "ffmpeg")),
        )
        self._condition = threading.Condition()
        self._epoch: StreamEpoch | None = None
        self._thread: threading.Thread | None = None
        self._demux_done = threading.Event()
        self._closed = False
        self._eof = False
        self._error: Exception | None = None
        self._decoder_input_overflow_count = 0
        self._seq = 0
        metadata = _metadata_from_configuration_stream(stream_descriptor(self._demuxer.video))
        self._state = self._spawn_state(metadata)

    @property
    def packet_drop_count(self) -> int:
        return self._demuxer.packet_drop_count

    @property
    def decoder_input_overflow_count(self) -> int:
        with self._condition:
            return self._decoder_input_overflow_count + self._state.decoder_input.overflow_count

    def set_stream_identity(self, worker_boot_id: str, stream_epoch: int) -> None:
        with self._condition:
            if self._epoch is not None:
                raise RuntimeError("stream identity is already assigned")
            epoch = StreamEpoch(worker_boot_id, self._config.camera_id, stream_epoch)
            self._sink.roll_epoch(epoch)
            self._epoch = epoch
            self._thread = threading.Thread(
                target=self._demux,
                name=f"packet-demux-{self._config.camera_id}",
                daemon=True,
            )
            self._thread.start()

    def read(self) -> FramePacket | None:
        while True:
            with self._condition:
                if self._closed:
                    return None
                self._raise_demux_error()
                state = self._state
                state.decoder_input.raise_if_failed()
            started_at = self._clock()
            payload = state.process.read_frame(self._config.read_timeout_ms / 1000.0)
            finished_at = self._clock()
            with self._condition:
                self._raise_demux_error()
                if state is not self._state:
                    continue
                state.decoder_input.raise_if_failed()
                if payload is None:
                    return None
                source = state.decoder_input.pop_timing()
                epoch = self._epoch
                if epoch is None:  # pragma: no cover - identity precedes reads
                    _raise_identity_unavailable()
                seq = self._seq
                self._seq += 1
            image = rgb24_image(payload, state.metadata)
            pts = float(source.presentation_time)
            return FramePacket(
                camera_id=self._config.camera_id,
                frame=Frame(index=seq, time_sec=pts, image=image),
                pts=pts,
                seq=seq,
                width=state.metadata.width,
                height=state.metadata.height,
                decode_time_ms=max(0.0, (finished_at - started_at) * 1000.0),
                worker_boot_id=epoch.worker_boot_id,
                stream_epoch=epoch.stream_epoch,
                source_pts=source.pts,
                source_dts=source.dts,
                source_time_base=source.stream.time_base,
            )

    def wait_demux_complete(self, timeout: float) -> bool:
        return self._demux_done.wait(timeout)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            state = self._state
            self._condition.notify_all()
        state.decoder_input.abort()
        _ = state.process.reap(timeout_sec=_PROCESS_REAP_TIMEOUT_SEC)
        state.decoder_input.join(_PROCESS_REAP_TIMEOUT_SEC)
        thread = self._thread
        if thread is None:
            self._container.close()
        elif thread is not threading.current_thread():
            thread.join(timeout=max(5.0, self._config.read_timeout_ms / 1000.0 + 1.0))

    def _spawn_state(self, metadata: StreamMetadata) -> _DecoderState:
        decoder = cuvid_decoder_for(metadata.codec_name)
        try:
            process = self._process_spawner(
                ffmpeg_decode_args(self._decoder_config, decoder),
                raw_frame_size(metadata),
            )
        except OSError as error:
            raise sanitized_nvdec_error("ffmpeg spawn failed", error) from None
        return _DecoderState(process, metadata, DecoderInputQueue(process))

    def _demux(self) -> None:
        try:
            epoch = self._epoch
            if epoch is None:  # pragma: no cover - thread starts after assignment
                _raise_identity_unavailable()
            self._demuxer.run(
                epoch,
                stop_requested=lambda: self._closed,
                on_configuration=self._on_configuration,
                on_packet=self._on_packet,
            )
        except Exception as exc:  # noqa: BLE001 - demux thread boundary
            if not self._closed:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
        finally:
            with self._condition:
                state = self._state
            state.decoder_input.finish()
            self._container.close()
            with self._condition:
                self._eof = True
                self._condition.notify_all()
            self._demux_done.set()

    def _on_configuration(
        self,
        configuration: SourceStreamConfiguration,
        changed: bool,
    ) -> None:
        if not changed:
            return
        epoch = self._epoch
        if epoch is None:  # pragma: no cover - callback runs after assignment
            raise RuntimeError("packet session identity is unavailable")
        with self._condition:
            if self._closed:
                return
            old_state = self._state
            self._decoder_input_overflow_count += old_state.decoder_input.overflow_count
            old_state.decoder_input.abort()
            _ = old_state.process.reap(timeout_sec=_PROCESS_REAP_TIMEOUT_SEC)
            old_state.decoder_input.join(_PROCESS_REAP_TIMEOUT_SEC)
            self._sink.roll_epoch(epoch)
            new_state = self._spawn_state(
                _metadata_from_configuration_stream(configuration.video_streams[0])
            )
            self._state = new_state
            self._condition.notify_all()

    def _on_packet(
        self,
        _packet: av.Packet,
        source: SourcePacket,
        _current_packet: bool,
    ) -> None:
        if source.stream.media_type != "video":
            return
        with self._condition:
            state = self._state
        state.decoder_input.offer(source)

    def _raise_demux_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"packet-preserving NVDEC decode failed ({type(self._error).__name__})"
            ) from self._error


def _metadata_from_configuration_stream(
    stream: SourceStreamDescriptor,
) -> StreamMetadata:
    if stream.width is None or stream.height is None:
        raise RuntimeError("NVDEC source video dimensions are unavailable")
    return StreamMetadata(stream.width, stream.height, stream.codec_name)


def _raise_identity_unavailable() -> NoReturn:
    raise RuntimeError("packet session identity is unavailable")


__all__ = ["NvdecPacketTeeSession"]
