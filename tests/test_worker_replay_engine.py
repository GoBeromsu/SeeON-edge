from __future__ import annotations

from worker.replay.engine import ReplayConfigurationError, assess_reproducibility


def test_replay_engine_exports_surviving_v2_replay_surface() -> None:
    assert issubclass(ReplayConfigurationError, ValueError)
    assert callable(assess_reproducibility)
