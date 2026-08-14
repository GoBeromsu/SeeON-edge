from __future__ import annotations

from typing import final


class DeviceEncoderUnavailableError(RuntimeError):
    """The experimental device-input NVENC encoder cannot open on this host.

    Mirrors ``worker.adapters.decode.nvdec_device.errors.DeviceResidentUnavailableError``:
    a distinct type from the production ``EncoderStartError``
    (``worker.adapters.encode.adapter_errors``), because a device-input NVENC
    open failure must never be silently absorbed into the existing nvenc ->
    libx264 clip-encoder fallback (#53) -- that fallback is sanctioned only
    for the host-bridge clip encoder path, never for this profile's
    zero-host-readback contract.
    """

    __slots__: tuple[str, ...] = ("reason",)

    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@final
class DeviceEncoderPoolExhaustedError(RuntimeError):
    """Every bounded device-input encoder session slot is outstanding (backpressure)."""

    __slots__: tuple[str, ...] = ("capacity", "outstanding")

    capacity: int
    outstanding: int

    def __init__(self, capacity: int, outstanding: int) -> None:
        self.capacity = capacity
        self.outstanding = outstanding
        super().__init__(
            f"device-input NVENC pool exhausted: {outstanding}/{capacity} sessions outstanding"
        )


@final
class DeviceEncoderRejectedInputError(RuntimeError):
    """A submitted frame is not an owned, synchronized device-resident surface.

    Raised instead of silently reading the surface back to host memory --
    the encoder seam accepts only explicit device surfaces with declared
    ownership/synchronization metadata (Todo 18's "no host readback"
    requirement); anything else is a caller contract violation, not a
    software-fallback opportunity.
    """

    __slots__: tuple[str, ...] = ("reason",)

    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


__all__ = [
    "DeviceEncoderPoolExhaustedError",
    "DeviceEncoderRejectedInputError",
    "DeviceEncoderUnavailableError",
]
