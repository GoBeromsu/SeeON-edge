from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NoReturn, TypeVar

from worker.interfaces.decode import DecodeAdapter, DecodeSession
from worker.pipeline.ingest.rtsp_url import mask_rtsp_url
from worker.types import FramePacket

_DecodeConfigT = TypeVar("_DecodeConfigT")
ProbeErrorClass = Literal["timeout", "decode", "auth"]
ProbePayloadValue = bool | str | int


@dataclass(frozen=True, slots=True)
class RTSPProbeResult:
    masked_url: str
    requested_backend: str
    backend: str
    width: int
    height: int
    channels: int

    def as_dict(self) -> dict[str, ProbePayloadValue]:
        return {
            "ok": True,
            "url": self.masked_url,
            "requested_backend": self.requested_backend,
            "backend": self.backend,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
        }


class RTSPProbeError(RuntimeError):
    __slots__ = ("error_class", "masked_url")

    def __init__(
        self,
        error_class: ProbeErrorClass,
        message: str,
        masked_url: str,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.masked_url = masked_url

    def as_dict(self) -> dict[str, ProbePayloadValue]:
        return {
            "ok": False,
            "url": self.masked_url,
            "error_class": self.error_class,
            "message": str(self),
        }


def probe_first_frame(
    url: str,
    *,
    decoder: DecodeAdapter[_DecodeConfigT],
    config: _DecodeConfigT,
    requested_backend: str = "injected",
    selected_backend: str = "injected",
    timeout_ms: int = 5000,
    monotonic: Callable[[], float] = time.monotonic,
) -> RTSPProbeResult:
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")

    masked_url = mask_rtsp_url(url)
    deadline = monotonic() + (timeout_ms / 1000.0)
    session: DecodeSession | None = None
    try:
        session = decoder.open(config)
        while True:
            if monotonic() >= deadline:
                _raise_timeout(masked_url, timeout_ms)
            try:
                packet = session.read()
            except Exception as exc:  # noqa: BLE001 - decoder errors are normalized at this boundary.
                raise _probe_error(_classify_exception(exc), masked_url, timeout_ms) from exc
            if packet is None:
                if monotonic() >= deadline:
                    _raise_timeout(masked_url, timeout_ms)
                continue
            try:
                height, width, channels = _frame_dimensions(packet, masked_url)
                return RTSPProbeResult(
                    masked_url=masked_url,
                    requested_backend=requested_backend,
                    backend=selected_backend,
                    width=width,
                    height=height,
                    channels=channels,
                )
            finally:
                packet.release()
    except RTSPProbeError:
        raise
    except Exception as exc:  # noqa: BLE001 - decoder errors are normalized at this boundary.
        raise _probe_error(_classify_exception(exc), masked_url, timeout_ms) from exc
    finally:
        if session is not None:
            session.close()


def _frame_dimensions(
    packet: FramePacket,
    masked_url: str,
) -> tuple[int, int, int]:
    descriptor = packet.descriptor
    if descriptor.width != packet.width or descriptor.height != packet.height:
        raise _probe_error("decode", masked_url, 0)

    shape = tuple(int(dimension) for dimension in packet.borrow_host_frame().image.shape)
    if len(shape) == 2:
        height, width = shape
        channels = 1
    elif len(shape) == 3:
        height, width, channels = shape
    else:
        raise _probe_error("decode", masked_url, 0)
    if (
        height <= 0
        or width <= 0
        or channels <= 0
        or width != descriptor.width
        or height != descriptor.height
    ):
        raise _probe_error("decode", masked_url, 0)
    return int(height), int(width), int(channels)


def _classify_exception(exc: Exception) -> ProbeErrorClass:
    message = str(exc).lower()
    if any(term in message for term in ("401", "403", "unauthorized", "forbidden", "auth")):
        return "auth"
    if any(term in message for term in ("timeout", "timed out")):
        return "timeout"
    return "decode"


def _raise_timeout(masked_url: str, timeout_ms: int) -> NoReturn:
    raise _probe_error("timeout", masked_url, timeout_ms)


def _probe_error(
    error_class: ProbeErrorClass,
    masked_url: str,
    timeout_ms: int,
) -> RTSPProbeError:
    if error_class == "timeout":
        return RTSPProbeError(
            "timeout",
            "Timed out waiting for first decodable RTSP frame "
            f"from {masked_url} within {timeout_ms} ms.",
            masked_url,
        )
    if error_class == "auth":
        return RTSPProbeError(
            "auth",
            f"RTSP authentication failed for {masked_url}; "
            "verify username, password, and NVR permissions.",
            masked_url,
        )
    return RTSPProbeError(
        "decode",
        f"Could not decode a first RTSP frame from {masked_url}; "
        "verify codec, stream profile, and URL path.",
        masked_url,
    )


__all__ = [
    "ProbeErrorClass",
    "RTSPProbeError",
    "RTSPProbeResult",
    "mask_rtsp_url",
    "probe_first_frame",
]
