from __future__ import annotations

import importlib.util


def test_inference_slot_does_not_ship_a_replay_cli() -> None:
    """Replay input retrieval is backend-owned, never a slot-side database read."""
    assert importlib.util.find_spec("worker.replay.cli") is None
