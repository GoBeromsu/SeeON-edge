from __future__ import annotations

import threading
import time
from collections import deque
from fractions import Fraction
from typing import NoReturn, cast, final

import av
import numpy as np
from av.codec.hwaccel import HWAccel
from av.container import InputContainer
from numpy.typing import NDArray

from contracts.frame import Frame
from worker.adapters.decode.nvdec_cuvid.process import (
    ProcessSpawner,
    spawn_decoder_process,
)
from worker.adapters.decode.pyav_demux import DecodeConfig, PyAvPacketDemuxer
from worker.adapters.decode.pyav_nvdec import NvdecPacketTeeSession
from worker.interfaces.source_packet import (
    EpochRollingSourcePacketSink,
    SourcePacketSink,
)
from worker.types import FramePacket
from worker.types.source_packet import SourcePacket, SourceStreamConfiguration, StreamEpoch

_MAX_PENDING_FRAMES = 2


@final
class PyAvPreservingSession:
    """Existing CPU/VAAPI preserving path; decode remains in PyAV."""

    def __init__(
        self,
        config: DecodeConfig,
        container: InputContainer,
        sink: SourcePacketSink,
    ) -> None:
        self._config = config
        self._container = container
        try:
            self._demuxer = PyAvPacketDemuxer(config, container, sink)
        except Exception:
            container.close()
            raise
        self._epoch: StreamEpoch | None = None
        self._frames: deque[av.VideoFrame] = deque()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._demux_done = threading.Event()
        self._closed = False
        self._eof = False
        self._error: Exception | None = None
        self._seq = 0

    @property
    def packet_drop_count(self) -> int:
        return self._demuxer.packet_drop_count

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
                raise RuntimeError(
                    f"packet-preserving decode failed ({type(self._error).__name__})"
                ) from self._error
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
        if epoch is None:  # pragma: no cover - reads start only after assignment
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
            thread.join(timeout=max(5.0, self._config.read_timeout_ms / 1000.0 + 1.0))

    def _demux(self) -> None:
        try:
            epoch = self._epoch
            if epoch is None:  # pragma: no cover - thread starts after assignment
                _raise_identity_unavailable()
            self._demuxer.run(
                epoch,
                stop_requested=lambda: self._closed,
                on_configuration=self._ignore_configuration,
                on_packet=self._decode_packet,
            )
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

    def _ignore_configuration(
        self,
        _configuration: SourceStreamConfiguration,
        _changed: bool,
    ) -> None:
        return None

    def _decode_packet(
        self,
        packet: av.Packet,
        source: SourcePacket,
        current_packet: bool,
    ) -> None:
        if source.stream_index != self._demuxer.video.index or not current_packet:
            return
        for frame in packet.decode():
            if not isinstance(frame, av.VideoFrame):
                continue
            with self._condition:
                if len(self._frames) == _MAX_PENDING_FRAMES:
                    self._frames.popleft()
                self._frames.append(frame)
                self._condition.notify()


@final
class PyAvPreservingAdapter:
    def __init__(
        self,
        sink: SourcePacketSink,
        *,
        decode_backend: str,
        process_spawner: ProcessSpawner = spawn_decoder_process,
    ) -> None:
        self._sink = sink
        self._decode_backend = decode_backend
        self._process_spawner = process_spawner

    def open(
        self,
        config: DecodeConfig,
    ) -> PyAvPreservingSession | NvdecPacketTeeSession:
        hwaccel = _hardware_acceleration(self._decode_backend, config)
        if self._decode_backend == "nvdec" and not isinstance(
            self._sink, EpochRollingSourcePacketSink
        ):
            raise RuntimeError("NVDEC packet preservation requires an epoch-rolling sink")
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
        if self._decode_backend != "nvdec":
            return PyAvPreservingSession(config, container, self._sink)
        try:
            return NvdecPacketTeeSession(
                config,
                container,
                cast("EpochRollingSourcePacketSink", self._sink),
                process_spawner=self._process_spawner,
            )
        except Exception:
            container.close()
            raise


def _raise_identity_unavailable() -> NoReturn:
    raise RuntimeError("packet session identity is unavailable")


def _hardware_acceleration(backend: str, config: DecodeConfig) -> HWAccel | None:
    if backend in {"opencv", "cpu", "nvdec"}:
        return None
    if backend == "vaapi":
        device = getattr(config, "render_device", None)
        return HWAccel("vaapi", device=device, allow_software_fallback=False)
    raise RuntimeError(f"unsupported packet-preserving decode backend: {backend!r}")


__all__ = [
    "NvdecPacketTeeSession",
    "PyAvPreservingAdapter",
    "PyAvPreservingSession",
]
