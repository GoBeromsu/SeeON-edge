from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Final, final

import numpy as np

from contracts.frame import Frame
from worker.adapters.decode.vaapi.errors import (
    VaapiReadError,
    sanitized_vaapi_error,
)
from worker.adapters.decode.vaapi.models import StreamDimensions, VaapiConfig
from worker.adapters.decode.vaapi.probe import (
    ProbeRunner,
    probe_stream_dimensions,
    run_ffprobe,
)
from worker.adapters.decode.vaapi.process import (
    DecoderProcess,
    ProcessSpawner,
    spawn_decoder_process,
)
from worker.types import FramePacket

_RGB_CHANNELS: Final = 3
_PROCESS_REAP_TIMEOUT_SEC: Final = 5.0


def ffmpeg_decode_args(config: VaapiConfig) -> tuple[str, ...]:
    """Build the ffmpeg argv for VAAPI-decoded, CPU-side rgb24 output.

    Frame-acquisition choice (see PR body for the full rationale): a
    subprocess ``ffmpeg -hwaccel vaapi`` pipeline, matching the pattern
    already proven by ``nvdec_cuvid`` in this repo, rather than PyAV or
    OpenCV. OpenCV's ``VideoCapture`` has no clean VAAPI path at all (ruled
    out per the task). PyAV would add a new binary dependency and a second,
    divergent frame-acquisition code path to maintain alongside the existing
    ffmpeg-subprocess one for zero real benefit here, since decode-only
    VAAPI output has to be downloaded to system memory anyway before this
    adapter can hand a numpy array to CPU inference.

    No explicit ``-c:v <codec>_vaapi`` decoder is selected (unlike
    ``nvdec_cuvid``'s ``cuvid_decoder_for``): ``-hwaccel vaapi`` lets ffmpeg
    auto-negotiate the hardware decoder for whatever codec the stream
    actually uses, falling back to attempting software decode internally if
    the codec has no VAAPI decoder -- per-stream codec support is therefore
    a per-camera concern, not a boot-time one (see the boot-level fallback
    docstring on ``resolve_decode_or_fallback`` for the host-capability half
    of graceful degradation this PR implements).

    ``-hwaccel_output_format`` is deliberately omitted: without it, ffmpeg's
    VAAPI hwaccel downloads decoded frames to system memory automatically
    (nv12), exactly mirroring how ``nvdec_cuvid``'s ``-hwaccel cuda`` (also
    without ``-hwaccel_output_format cuda``) already yields host-memory
    frames in this codebase -- no explicit ``hwdownload`` filter is needed,
    and the ``-pix_fmt rgb24`` output triggers ffmpeg's own swscale
    nv12->rgb24 conversion, same as the existing nvdec pipeline's implicit
    conversion.
    """
    return (
        config.ffmpeg_bin,
        "-nostdin",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-hwaccel",
        "vaapi",
        "-hwaccel_device",
        config.render_device,
        "-i",
        config.url,
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    )


@final
class VaapiSession:
    """Own mutable frame timing and one decoder process for a camera."""

    def __init__(
        self,
        config: VaapiConfig,
        dimensions: StreamDimensions,
        process: DecoderProcess,
        clock: Callable[[], float],
    ) -> None:
        self._config = config
        self._dimensions = dimensions
        self._process = process
        self._clock = clock
        self._frame_size = dimensions.width * dimensions.height * _RGB_CHANNELS
        self._seq = 0
        self._first_frame_at: float | None = None
        self._closed = False
        self._close_lock = threading.Lock()

    def read(self) -> FramePacket | None:
        if self._closed:
            return None
        started_at = self._clock()
        payload = self._process.read_frame(self._config.read_timeout_ms / 1000.0)
        finished_at = self._clock()
        if payload is None:
            return None
        if len(payload) != self._frame_size:
            self.close()
            raise VaapiReadError(self._frame_size, len(payload))
        image = (
            np.frombuffer(payload, dtype=np.uint8)
            .reshape(
                self._dimensions.height,
                self._dimensions.width,
                _RGB_CHANNELS,
            )
            .copy()
        )
        if self._first_frame_at is None:
            self._first_frame_at = finished_at
        pts = max(0.0, finished_at - self._first_frame_at)
        decode_time_ms = max(0.0, (finished_at - started_at) * 1000.0)
        seq = self._seq
        self._seq += 1
        frame = Frame(index=seq, time_sec=pts, image=image)
        return FramePacket(
            camera_id=self._config.camera_id,
            frame=frame,
            pts=pts,
            seq=seq,
            width=self._dimensions.width,
            height=self._dimensions.height,
            decode_time_ms=decode_time_ms,
        )

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        _ = self._process.reap(timeout_sec=_PROCESS_REAP_TIMEOUT_SEC)


@final
class VaapiAdapter:
    def __init__(
        self,
        *,
        probe_runner: ProbeRunner = run_ffprobe,
        process_spawner: ProcessSpawner = spawn_decoder_process,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probe_runner = probe_runner
        self._process_spawner = process_spawner
        self._clock = clock

    def open(self, config: VaapiConfig) -> VaapiSession:
        dimensions = probe_stream_dimensions(config, runner=self._probe_runner)
        frame_size = dimensions.width * dimensions.height * _RGB_CHANNELS
        try:
            process = self._process_spawner(
                ffmpeg_decode_args(config),
                frame_size,
            )
        except OSError as error:
            raise sanitized_vaapi_error("ffmpeg spawn failed", error) from None
        return VaapiSession(config, dimensions, process, self._clock)


__all__ = [
    "VaapiAdapter",
    "VaapiSession",
    "ffmpeg_decode_args",
]
