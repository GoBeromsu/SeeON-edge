from __future__ import annotations

from typing import final


class VaapiUnavailableError(RuntimeError):
    __slots__: tuple[str, ...] = ("reason", "returncode")

    reason: str
    returncode: int | None

    def __init__(self, reason: str, *, returncode: int | None = None) -> None:
        self.reason = reason
        self.returncode = returncode
        super().__init__(reason)


@final
class VaapiConfigError(VaapiUnavailableError):
    pass


@final
class VaapiReadError(VaapiUnavailableError):
    __slots__ = ("actual_size", "expected_size")

    def __init__(self, expected_size: int, actual_size: int) -> None:
        self.expected_size = expected_size
        self.actual_size = actual_size
        reason = (
            f"ffmpeg produced an incomplete rgb24 frame (expected={expected_size}, "
            f"actual={actual_size})"
        )
        super().__init__(reason)


def sanitized_vaapi_error(stage: str, error: Exception) -> VaapiUnavailableError:
    returncode = getattr(error, "returncode", None)
    bounded_returncode = returncode if isinstance(returncode, int) else None
    reason = f"{stage}: {type(error).__name__} (returncode={bounded_returncode})"
    return VaapiUnavailableError(reason, returncode=bounded_returncode)


__all__ = [
    "VaapiConfigError",
    "VaapiReadError",
    "VaapiUnavailableError",
    "sanitized_vaapi_error",
]
