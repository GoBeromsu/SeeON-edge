from __future__ import annotations

from typing import final


class NvdecUnavailableError(RuntimeError):
    __slots__: tuple[str, ...] = ("reason", "returncode")

    reason: str
    returncode: int | None

    def __init__(self, reason: str, *, returncode: int | None = None) -> None:
        self.reason = reason
        self.returncode = returncode
        super().__init__(reason)

    @property
    def safe_log_detail(self) -> str:
        return self.reason


@final
class NvdecConfigError(NvdecUnavailableError):
    pass


@final
class UnsupportedCodecError(NvdecUnavailableError):
    __slots__ = ("codec_name",)

    def __init__(self, codec_name: str) -> None:
        self.codec_name = codec_name
        super().__init__(f"unsupported NVDEC codec: {codec_name or '<blank>'}")


@final
class NvdecReadError(NvdecUnavailableError):
    __slots__ = ("actual_size", "expected_size")

    def __init__(self, expected_size: int, actual_size: int) -> None:
        self.expected_size = expected_size
        self.actual_size = actual_size
        reason = (
            f"ffmpeg produced an incomplete rgb24 frame (expected={expected_size}, "
            f"actual={actual_size})"
        )
        super().__init__(reason)


def sanitized_nvdec_error(stage: str, error: Exception) -> NvdecUnavailableError:
    returncode = getattr(error, "returncode", None)
    bounded_returncode = returncode if isinstance(returncode, int) else None
    reason = f"{stage}: {type(error).__name__} (returncode={bounded_returncode})"
    return NvdecUnavailableError(reason, returncode=bounded_returncode)


__all__ = [
    "NvdecConfigError",
    "NvdecReadError",
    "NvdecUnavailableError",
    "UnsupportedCodecError",
    "sanitized_nvdec_error",
]
