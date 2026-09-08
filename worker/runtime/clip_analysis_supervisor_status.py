"""Stored clip-analysis supervisor status value."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClipAnalysisStatus:
    state: str
    reason: str | None = None


class StatusLedger:
    """Bounded status storage retaining all non-terminal work."""

    def __init__(self) -> None:
        self._values: OrderedDict[str, ClipAnalysisStatus] = OrderedDict()

    def get(self, clip_id: str) -> ClipAnalysisStatus:
        return self._values.get(clip_id, ClipAnalysisStatus("idle"))

    def record(self, clip_id: str, status: ClipAnalysisStatus) -> None:
        self._values[clip_id] = status
        self._values.move_to_end(clip_id)
        terminal = [
            key for key, value in self._values.items() if value.state not in ("queued", "running")
        ]
        while len(terminal) > 256:
            self._values.pop(terminal.pop(0))
