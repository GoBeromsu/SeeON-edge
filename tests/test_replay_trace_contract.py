import pytest

from contracts.replay_trace import ReplayTraceHeader, ReplayTraceRow, decode_jsonl, encode_jsonl


def test_jsonl_round_trip_and_float32_validation() -> None:
    row = ReplayTraceRow("legacy", "cam", 0, 1, 2, 3, "new", (0.1,) * 56)
    assert decode_jsonl(encode_jsonl(ReplayTraceHeader(), [row]))[1] == (row,)


def test_rejects_bad_lifecycle_and_shape() -> None:
    with pytest.raises(ValueError):
        ReplayTraceRow("legacy", "cam", 0, 1, 2, 3, "gone", (0.0,) * 56)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ReplayTraceRow("legacy", "cam", 0, 1, 2, 3, "new", ())
