"""Privacy-bounded values for operator review state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ReviewDisposition(StrEnum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"


@dataclass(frozen=True, slots=True)
class EvidenceReview:
    review_id: str
    incident_id: str
    clip_id: str | None
    version: int
    actor_id: str
    reviewed_at: str
    disposition: ReviewDisposition
    notes: str | None


__all__ = ["EvidenceReview", "ReviewDisposition"]
