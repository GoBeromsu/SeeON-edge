"""Bed-exit domain: assignment, latching and night-window decisions."""

from __future__ import annotations

from worker.domains.bed_exit.detector import BedExitMonitor, BedExitScoringRecorder
from worker.domains.bed_exit.latch import BedExitLatch, BedExitLatchStatus
from worker.domains.bed_exit.night_window import NightWindow
from worker.domains.bed_exit.schema import (
    BedExitConfig,
    BedExitDebugSnapshot,
    BedExitEvent,
    BedExitFrame,
    BedOccupancy,
    BedStatus,
)

__all__ = [
    "BedExitConfig",
    "BedExitDebugSnapshot",
    "BedExitEvent",
    "BedExitFrame",
    "BedExitLatch",
    "BedExitLatchStatus",
    "BedExitMonitor",
    "BedOccupancy",
    "BedExitScoringRecorder",
    "BedStatus",
    "NightWindow",
]
