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
        seq=0,
        pts_ns=2,
        epoch=0,
        source_event="frame",
        source="legacy-association",
        tracks=(
            ReplayTrack(
                3,
                "new",
                (0.1, 0.2, 0.3, 0.4, 0.9),
                ((0.1, 0.2, 0.9),) * 17,
            ),
        ),
        bed_polygon_id=None,
        bed_polygon=None,
        bed_polygon_image_size=None,
        night_window_active=False,
        frame_width=1000,
        frame_height=1000,
    )


def test_jsonl_round_trip() -> None:
    assert decode_jsonl(encode_jsonl(ReplayTraceHeader(), [_row()]))[1] == (_row(),)


def test_document_round_trip() -> None:
    assert decode_document(encode_document(ReplayTraceHeader(), [_row()]))[1] == (_row(),)


def test_rejects_bad_track_shape() -> None:
    with pytest.raises(ValueError):
        ReplayTrack(3, "new", (), ())


@pytest.mark.parametrize(
    "payload",
    (
        (
            '{"version":"replay-trace-v2"}\n'
            '{"camera_id":"cam","seq":0,"pts_ns":true,"epoch":0,"source_event":"frame",'
            '"source":"legacy-association","tracks":[],"bed_polygon_id":null,'
            '"bed_polygon":null,"night_window_active":false,"frame_width":640,"frame_height":360}'
        ),
        (
            '{"version":"replay-trace-v2"}\n'
            '{"camera_id":"cam","seq":0,"pts_ns":1,"epoch":0,"source_event":"unknown",'
            '"source":"legacy-association","tracks":[],"bed_polygon_id":null,'
            '"bed_polygon":null,"night_window_active":false,"frame_width":640,"frame_height":360}'
        ),
        (
            '{"version":"replay-trace-v2"}\n'
            '{"camera_id":"cam","seq":0,"pts_ns":1,"epoch":0,"source_event":"frame",'
            '"source":"legacy-association","tracks":[{"track_id":1,"lifecycle":"new",'
            '"bbox":[0.0,0.0,1.0,1.0,NaN],"keypoints":[[0.0,0.0,1.0]]}],'
            '"bed_polygon_id":null,"bed_polygon":null,"night_window_active":false,"frame_width":640,"frame_height":360}'
        ),
    ),
)
def test_codec_rejects_adversarial_values(payload: str) -> None:
    with pytest.raises(ValueError, match="invalid replay trace"):
        decode_jsonl(payload)


def test_rejects_invalid_geometry_and_track_identity() -> None:
    with pytest.raises(ValueError, match="bbox corners"):
        ReplayTrack(3, "new", (0.3, 0.2, 0.1, 0.4, 0.9), ((0.1, 0.2, 0.9),) * 17)
    track = ReplayTrack(3, "new", (0.1, 0.2, 0.3, 0.4, 0.9), ((0.1, 0.2, 0.9),) * 17)
    with pytest.raises(ValueError, match="unique"):
        ReplayRow(
            "cam",
            0,
            2,
            0,
            "frame",
            "legacy-association",
            (track, track),
            None,
            None,
            None,
            False,
            640,
            360,
        )


def test_rejects_tracks_on_control_rows() -> None:
    with pytest.raises(ValueError, match="control rows"):
        ReplayRow(
            "cam",
            0,
            2,
            0,
            "open",
            "legacy-association",
            (
                ReplayTrack(3, "new", (0.1, 0.2, 0.3, 0.4, 0.9), ((0.1, 0.2, 0.9),) * 17),
            ),
            None,
            None,
            None,
            False,
            640,
            360,
        )
    with pytest.raises(ValueError, match="exactly"):
        ReplayRow(
            "cam", 0, 2, 0, "frame", "legacy-association", (), "bed", None, None, False, 640, 360
        )


@pytest.mark.parametrize("seq", (True, -1))
def test_rejects_invalid_seq(seq: object) -> None:
    with pytest.raises(ValueError):
        ReplayRow(
            camera_id="cam",
            seq=seq,  # type: ignore[arg-type]
            pts_ns=2,
            epoch=0,
            source_event="frame",
            source="legacy-association",
            tracks=(),
            bed_polygon_id=None,
            bed_polygon=None,
            bed_polygon_image_size=None,
            night_window_active=False,
            frame_width=640,
            frame_height=360,
        )
