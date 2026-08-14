from __future__ import annotations

from typing import final


class DeviceResidentUnavailableError(RuntimeError):
    """The experimental NVIDIA device-resident path cannot boot on this host.

    Carries the same fail-closed contract as
    ``worker.adapters.decode.nvdec_cuvid.errors.NvdecUnavailableError``, but is
    a distinct type: an unavailable *concrete-stage* capability (this
    package) is a different failure than an unavailable *plain-cuda*
    capability (``worker.adapters.device.cuda.probe``) -- the profile boot
    gate must never conflate the production ``nvidia-host-bridge`` profile's
    device check with this experimental profile's stricter one.
    """

    __slots__: tuple[str, ...] = ("reason",)

    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@final
class DevicePoolExhaustedError(RuntimeError):
    """Every bounded device-resident pool slot is outstanding (backpressure)."""

    __slots__: tuple[str, ...] = ("capacity", "outstanding")

    capacity: int
    outstanding: int

    def __init__(self, capacity: int, outstanding: int) -> None:
        self.capacity = capacity
        self.outstanding = outstanding
        super().__init__(
            f"device-resident pool exhausted: {outstanding}/{capacity} slots outstanding"
        )


__all__ = ["DevicePoolExhaustedError", "DeviceResidentUnavailableError"]
