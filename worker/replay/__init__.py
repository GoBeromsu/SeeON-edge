"""Deterministic replay primitives for backend-owned ML QA.

Composes ``worker.domains`` and ``worker.pipeline.trace`` to re-run a compiled
detection module's camera-local decider against frozen, image-free inputs
already captured by the real pipeline -- never extraction, never GPU, never
network. The inference slot has no replay CLI or local trace reader; an
authenticated backend API must supply recovered inputs to these primitives.
"""

from __future__ import annotations

from worker.replay.comparison import FrameMismatch, MismatchReason, ReplayComparison, compare_runs
from worker.replay.engine import (
    ReplayConfigurationError,
    ReplayFrameResult,
    ReplayRun,
    assess_reproducibility,
    replay_camera,
    replay_recovered,
)

__all__ = [
    "FrameMismatch",
    "MismatchReason",
    "ReplayComparison",
    "ReplayConfigurationError",
    "ReplayFrameResult",
    "ReplayRun",
    "assess_reproducibility",
    "compare_runs",
    "replay_camera",
    "replay_recovered",
]
