from __future__ import annotations

from typing import final


@final
class EncoderPolicyError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@final
class EncoderStartError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@final
class EncoderWriteError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@final
class ClipRemuxError(RuntimeError):
    def __init__(self, reason: str, returncode: int | None = None) -> None:
        self.reason = reason
        self.returncode = returncode
        super().__init__(reason)


@final
class CrossGenerationSegmentError(RuntimeError):
    def __init__(self, generations: tuple[int, ...]) -> None:
        self.generations = generations
        values = ", ".join(str(generation) for generation in generations)
        super().__init__(f"cannot finalize segments from multiple generations: {values}")


class ThumbnailGenerationError(RuntimeError):
    def __init__(self, reason: str, returncode: int | None = None) -> None:
        self.reason = reason
        self.returncode = returncode
        super().__init__(reason)


@final
class ThumbnailTimeoutError(ThumbnailGenerationError):
    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        super().__init__("ffmpeg thumbnail extraction timed out")


@final
class ThumbnailPayloadError(ThumbnailGenerationError):
    def __init__(self, byte_count: int) -> None:
        self.byte_count = byte_count
        super().__init__("thumbnail JPEG payload is invalid")


@final
class ThumbnailSecurityError(ThumbnailGenerationError):
    def __init__(self, operation: str, error_type: str) -> None:
        self.operation = operation
        self.error_type = error_type
        super().__init__(f"secure thumbnail {operation} failed ({error_type})")


__all__ = [
    "ClipRemuxError",
    "CrossGenerationSegmentError",
    "EncoderPolicyError",
    "EncoderStartError",
    "EncoderWriteError",
    "ThumbnailGenerationError",
    "ThumbnailPayloadError",
    "ThumbnailSecurityError",
    "ThumbnailTimeoutError",
]
