from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Final, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from contracts.frame import Frame
from worker.adapters.decode.cpu_av.models import CpuAvConfig
from worker.interfaces.decode import DecodeSession
from worker.types import FramePacket

_RTSP_CAPTURE_OPTIONS: Final = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_RTSP_OVER_TCP: Final = "rtsp_transport;tcp"
Clock = Callable[[], float]


class Capture(Protocol):
    def isOpened(self) -> bool: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]: ...

    def release(self) -> None: ...


class CaptureFactory(Protocol):
    def __call__(
        self,
        url: str,
        backend: int | None = None,
        params: list[int] | None = None,
    ) -> Capture: ...


class CpuAvOpenError(RuntimeError):
    pass


class _OpenCvCaptureFactory:
    def __call__(
        self,
        url: str,
        backend: int | None = None,
        params: list[int] | None = None,
    ) -> Capture:
        if backend is None:
            return cv2.VideoCapture(url)
        if params is None:
            return cv2.VideoCapture(url, backend)
        return cv2.VideoCapture(url, backend, params)


class _CpuAvSession:
    def __init__(self, config: CpuAvConfig, capture: Capture, clock: Clock) -> None:
        self._config: CpuAvConfig = config
        self._capture: Capture | None = capture
        self._clock: Clock = clock
        self._seq: int = 0
        self._stream_origin: float | None = None

    def read(self) -> FramePacket | None:
        capture = self._capture
        if capture is None:
            return None

        started_at = self._clock()
        read_ok, image_bgr = capture.read()
        finished_at = self._clock()
        if not read_ok or image_bgr is None:
            self.close()
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self._stream_origin is None:
            self._stream_origin = finished_at
        time_sec = max(0.0, finished_at - self._stream_origin)
        decode_time_ms = max(0.0, (finished_at - started_at) * 1000.0)
        height, width = image_rgb.shape[:2]
        seq = self._seq
        self._seq += 1
        return FramePacket(
            camera_id=self._config.camera_id,
            frame=Frame(index=seq, time_sec=time_sec, image=image_rgb),
            pts=time_sec,
            seq=seq,
            width=width,
            height=height,
            decode_time_ms=decode_time_ms,
        )

    def close(self) -> None:
        capture = self._capture
        if capture is None:
            return
        self._capture = None
        capture.release()


class CpuAvAdapter:
    def __init__(
        self,
        *,
        capture_factory: CaptureFactory | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self._capture_factory: CaptureFactory = capture_factory or _OpenCvCaptureFactory()
        self._clock: Clock = clock

    def open(self, config: CpuAvConfig) -> DecodeSession:
        _ = os.environ.setdefault(_RTSP_CAPTURE_OPTIONS, _RTSP_OVER_TCP)
        params: list[int] = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            config.open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            config.read_timeout_ms,
        ]
        capture = self._open_capture(config, params)
        if not capture.isOpened():
            capture.release()
            raise CpuAvOpenError("OpenCV could not open the RTSP capture")
        return _CpuAvSession(config, capture, self._clock)

    # No post-open property application. Measured on this repo's real
    # MediaMTX + ffmpeg e2e fixture: OpenCV's FFMPEG backend reports
    # ``False`` from ``set(CAP_PROP_READ_TIMEOUT_MSEC, ...)`` after the
    # capture is open, because that timeout is honoured from the constructor
    # params instead. The previous code issued the call and discarded the
    # result, which is exactly the "looks configured, proves nothing" pattern
    # ADR-0003 removes. Treating that discarded result as a required property
    # instead rejects working captures, so the honest contract is: the open
    # and read timeouts travel as constructor params, and there is no
    # best-effort post-open surface at all.
    #
    # ``CAP_PROP_BUFFERSIZE`` is likewise not set. It was an unverified
    # latency knob (its ``set()`` result was also discarded) and the prime
    # suspect in the reconnect loop.

    def _open_capture(self, config: CpuAvConfig, params: list[int]) -> Capture:
        """Open the capture with exactly one explicit signature.

        ADR-0003 (explicit fallback only): the previous implementation caught
        ``TypeError`` and retried as ``(url, backend)`` and then ``(url)``,
        which silently dropped the required open/read timeouts and the explicit
        FFmpeg backend. A capture that succeeded that way had neither the
        timeout contract nor the backend this adapter claims to require, and
        the operator saw a working camera instead of a configuration error.

        The supported boundary is ``opencv-python-headless>=4.10`` with the
        lockfile pin. Builds outside it are reported, not accommodated.
        """
        try:
            return self._capture_factory(config.url, cv2.CAP_FFMPEG, params)
        except TypeError as exc:
            raise CpuAvOpenError(
                "OpenCV does not accept the required (url, CAP_FFMPEG, params) "
                "capture signature; this build is outside the supported "
                "boundary (opencv-python-headless>=4.10)"
            ) from exc
        except cv2.error as exc:
            # `cv2.error` text routinely echoes the URL it was given, which for
            # RTSP carries credentials. Report the failure class only; the
            # original exception stays on __cause__ for local debugging and is
            # never formatted into this message.
            raise CpuAvOpenError(
                f"OpenCV failed to open the RTSP capture for camera {config.camera_id}"
            ) from exc


__all__ = ["CpuAvAdapter"]
