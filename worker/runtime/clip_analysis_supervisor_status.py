"""Stored clip-analysis supervisor status value."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClipAnalysisStatus:
    state: str
    reason: str | None = None
