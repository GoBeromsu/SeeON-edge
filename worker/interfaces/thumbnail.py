from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ThumbnailGenerator(Protocol):
    def generate(
        self,
        video_path: Path,
        thumbnail_path: Path,
        duration_s: float,
    ) -> Path: ...


__all__ = ["ThumbnailGenerator"]
