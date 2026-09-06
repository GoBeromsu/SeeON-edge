"""Image-free envelopes for the operator live preview (issue #490)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FallPreviewStatus = Literal["normal", "suspected"]


@dataclass(frozen=True, slots=True)
class OverlaySelection:
    """Which tracked subjects the operator wants drawn on the live preview."""

    person: bool = True
    bed: bool = True


@dataclass(frozen=True, slots=True)
class FallPreviewState:
    """Latest CPU fall-policy verdict for one live track, for display only."""

    track_id: int
    status: FallPreviewStatus
    probability: float | None = None


__all__ = ["FallPreviewState", "FallPreviewStatus", "OverlaySelection"]
