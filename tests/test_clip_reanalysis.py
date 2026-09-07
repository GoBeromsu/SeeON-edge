from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from shared.events.clip_analysis_wire import decode_clip_analysis, encode_clip_analysis
from worker.adapters.model import clip_reanalysis


class _Stream:
    width = 64
    height = 48
    duration = 20
    time_base = Fraction(1, 1)
    thread_type = ""
    thread_count = 0


class _Frame:
    def __init__(self, pts: int | None) -> None:
        self.pts = pts

    def to_ndarray(self, format: str):
        assert format == "rgb24"
        return np.zeros((48, 64, 3), dtype=np.uint8)


class _Container:
    def __init__(self, pts: list[int | None]) -> None:
        self.streams = type("Streams", (), {"video": [_Stream()]})()
        self._frames = [_Frame(value) for value in pts]

    def decode(self, stream):
        return iter(self._frames)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _Pose:
    artifact_digest = "1" * 64

    def __init__(self, *args, **kwargs) -> None:
        pass

    def detect_persons(self, image):
        return ((1.0, 2.0, 3.0, 4.0, 0.9),)


class _Bed:
    artifact_digest = "2" * 64

    def __init__(self, *args, **kwargs) -> None:
        pass

    def detect_beds(self, image):
        return type("Result", (), {"boxes": ()})()


def _request(tmp_path: Path, digest: str) -> clip_reanalysis.ClipAnalysisRequest:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    return clip_reanalysis.ClipAnalysisRequest(
        clip_id="event-1",
        clip_path=clip,
        clip_sha256=digest,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        analysis_profile=clip_reanalysis.ClipAnalysisProfile(),
    )


def test_sha_mismatch_rejected(tmp_path: Path) -> None:
    with pytest.raises(clip_reanalysis.ClipAnalysisRejected, match="clip_sha256"):
        clip_reanalysis.analyze_clip(
            _request(tmp_path, "0" * 64), decoder_identity="pyav-test/mpeg4"
        )


def test_input_byte_cap_is_rejected_before_opening(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"clip").hexdigest()
    request = _request(tmp_path, digest)
    constrained = clip_reanalysis.ClipAnalysisRequest(
        clip_id=request.clip_id,
        clip_path=request.clip_path,
        clip_sha256=request.clip_sha256,
        pose_model_path=request.pose_model_path,
        bed_model_path=request.bed_model_path,
        analysis_profile=clip_reanalysis.ClipAnalysisProfile(max_input_bytes=3),
    )
    with pytest.raises(clip_reanalysis.ClipAnalysisRejected, match="input_bytes"):
        clip_reanalysis.analyze_clip(constrained, decoder_identity="pyav-test/mpeg4")


def test_duplicate_pts_is_ambiguous_and_result_roundtrips(tmp_path: Path, monkeypatch) -> None:
    digest = hashlib.sha256(b"clip").hexdigest()
    calls = iter([_Container([0, 1, 1, 20]), _Container([20])])
    monkeypatch.setattr(clip_reanalysis.av, "open", lambda path: next(calls))
    monkeypatch.setattr(clip_reanalysis, "OrtClipPoseRunner", _Pose)
    monkeypatch.setattr(clip_reanalysis, "OrtBedSegRunner", _Bed)
    result = clip_reanalysis.analyze_clip(
        _request(tmp_path, digest), decoder_identity="pyav-test/mpeg4"
    )
    assert [(frame.pts, frame.status) for frame in result.frames] == [
        (0, "available"),
        (1, "ambiguous_timestamp"),
        (20, "available"),
    ]
    assert result.clip_id == "event-1"
    assert result.frames[1].boxes == ()
    assert decode_clip_analysis(encode_clip_analysis(result)) == result
