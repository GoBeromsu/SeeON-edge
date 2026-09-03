import pytest

from contracts.replay_trace import (
    ReplayRow,
    ReplayTraceHeader,
    ReplayTrack,
    decode_document,
    decode_jsonl,
    encode_document,
    encode_jsonl,
)


def _row() -> ReplayRow:
    return ReplayRow(
        camera_id="cam",
        pts_ns=2,
        epoch=0,
        source_event="frame",
        source="legacy-association",
        tracks=(ReplayTrack(3, "new", (1, 2, 3, 4, 0.9), ((1, 2, 0.9),) * 17),),
        bed_polygon_id=None,
        bed_polygon=None,
        night_window_active=False,
    )


def test_jsonl_round_trip() -> None:
    assert decode_jsonl(encode_jsonl(ReplayTraceHeader(), [_row()]))[1] == (_row(),)


def test_document_round_trip() -> None:
    assert decode_document(encode_document(ReplayTraceHeader(), [_row()]))[1] == (_row(),)


def test_rejects_bad_track_shape() -> None:
    with pytest.raises(ValueError):
        ReplayTrack(3, "new", (), ())
