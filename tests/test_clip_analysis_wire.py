import json
from dataclasses import replace

import pytest

from shared.events.clip_analysis_wire import (
    MAX_CLIP_ANALYSIS_FRAMES,
    MAX_CLIP_ANALYSIS_OUTPUT_BYTES,
    ClipAnalysisBedGeometry,
    ClipAnalysisBox,
    ClipAnalysisFrame,
    ClipAnalysisResult,
    ClipAnalysisTimeBase,
    ClipAnalysisWireError,
    decode_clip_analysis,
    encode_clip_analysis,
)

HASH = "a" * 64


def _result() -> ClipAnalysisResult:
    return ClipAnalysisResult(
        source="clip_reanalysis",
        clip_id="clip-500",
        clip_sha256=HASH,
        pose_model_sha256="b" * 64,
        bed_model_sha256="c" * 64,
        decoder_identity="pyav-16.1.0:h264",
        analysis_profile_sha256="d" * 64,
        image_width=1920,
        image_height=1080,
        time_base=ClipAnalysisTimeBase(1, 12_000),
        frames=(
            ClipAnalysisFrame(
                2**40 + 123,
                "available",
                (
                    ClipAnalysisBox(12.5, 20.0, 220.0, 400.0, 0.91),
                    ClipAnalysisBox(400.0, 100.0, 700.0, 600.0, 0.72),
                ),
            ),
            ClipAnalysisFrame(2**40 + 777, "no_evidence"),
            ClipAnalysisFrame(2**40 + 999, "ambiguous_timestamp"),
        ),
        bed_geometries=(
            ClipAnalysisBedGeometry(
                ((10.0, 20.0), (800.0, 20.0), (800.0, 500.0), (10.0, 500.0)),
                2**40 + 123,
            ),
            ClipAnalysisBedGeometry(
                ((100.0, 600.0), (900.0, 600.0), (900.0, 900.0)),
                2**40 + 123,
            ),
        ),
    )


def _payload() -> dict[str, object]:
    return json.loads(encode_clip_analysis(_result()))


def test_nontrivial_serialization_roundtrip_preserves_pts_and_geometry() -> None:
    encoded = encode_clip_analysis(_result())
    decoded = decode_clip_analysis(encoded)
    assert decoded == _result()
    assert decoded.frames[0].pts == 2**40 + 123
    assert len(decoded.bed_geometries) == 2


@pytest.mark.parametrize(
    "field", ("clip_sha256", "pose_model_sha256", "bed_model_sha256", "analysis_profile_sha256")
)
def test_rejects_malformed_hashes(field: str) -> None:
    payload = _payload()
    payload[field] = "not-a-sha"
    with pytest.raises(ClipAnalysisWireError, match="SHA-256"):
        decode_clip_analysis(payload)


def test_rejects_duplicate_pts_and_boxes_on_ambiguous_frame() -> None:
    payload = _payload()
    frames = payload["frames"]
    assert isinstance(frames, list)
    frames[1]["pts"] = frames[0]["pts"]
    with pytest.raises(ClipAnalysisWireError, match="unique"):
        decode_clip_analysis(payload)

    payload = _payload()
    frames = payload["frames"]
    assert isinstance(frames, list)
    frames[0]["status"] = "ambiguous_timestamp"
    with pytest.raises(ClipAnalysisWireError, match="cannot carry boxes"):
        decode_clip_analysis(payload)


@pytest.mark.parametrize(
    "field,value",
    (("x1", float("nan")), ("x2", 1921.0), ("y1", -1.0), ("confidence", float("inf"))),
)
def test_rejects_nonfinite_or_out_of_bounds_boxes(field: str, value: float) -> None:
    payload = _payload()
    frames = payload["frames"]
    assert isinstance(frames, list)
    boxes = frames[0]["boxes"]
    assert isinstance(boxes, list)
    boxes[0][field] = value
    with pytest.raises(ClipAnalysisWireError):
        decode_clip_analysis(payload)


@pytest.mark.parametrize("forbidden", ("track_id", "fall_state", "bed_exit_state"))
def test_rejects_forbidden_fields(forbidden: str) -> None:
    payload = _payload()
    payload[forbidden] = 1
    with pytest.raises(ClipAnalysisWireError, match="forbidden"):
        decode_clip_analysis(payload)

    payload = _payload()
    frames = payload["frames"]
    assert isinstance(frames, list)
    boxes = frames[0]["boxes"]
    assert isinstance(boxes, list)
    boxes[0][forbidden] = 1
    with pytest.raises(ClipAnalysisWireError, match="forbidden"):
        decode_clip_analysis(payload)


def test_rejects_more_than_bounded_frame_count() -> None:
    payload = _payload()
    payload["frames"] = [
        {"pts": pts, "status": "no_evidence", "boxes": []}
        for pts in range(MAX_CLIP_ANALYSIS_FRAMES + 1)
    ]
    with pytest.raises(ClipAnalysisWireError, match="5400"):
        decode_clip_analysis(payload)


def test_rejects_oversized_output() -> None:
    oversized = replace(
        _result(),
        decoder_identity="x" * (MAX_CLIP_ANALYSIS_OUTPUT_BYTES + 1),
    )
    with pytest.raises(ClipAnalysisWireError, match="32 MiB"):
        encode_clip_analysis(oversized)


def test_rejects_bed_provenance_that_is_not_an_available_clip_frame() -> None:
    with pytest.raises(ClipAnalysisWireError, match="available frame PTS"):
        replace(
            _result(),
            bed_geometries=(ClipAnalysisBedGeometry(((1.0, 1.0), (2.0, 1.0), (2.0, 2.0)), 999),),
        )


def test_time_base_must_be_lossless_reduced_positive_rational() -> None:
    with pytest.raises(ClipAnalysisWireError):
        ClipAnalysisTimeBase(2, 4)
    with pytest.raises(ClipAnalysisWireError):
        ClipAnalysisTimeBase(0, 1)


def test_rejects_pts_that_a_json_consumer_cannot_represent_losslessly() -> None:
    with pytest.raises(ClipAnalysisWireError, match="lossless"):
        ClipAnalysisFrame(2**53, "no_evidence")
