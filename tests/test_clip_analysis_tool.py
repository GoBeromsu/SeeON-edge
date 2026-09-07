from __future__ import annotations

import json
from pathlib import Path

from shared.events.clip_analysis_wire import (
    ClipAnalysisFrame,
    ClipAnalysisResult,
    ClipAnalysisTimeBase,
)
from worker.tools import clip_analysis


def _payload(tmp_path: Path) -> Path:
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "clip_path": "clip.mp4",
                "clip_sha256": "0" * 64,
                "pose_model_path": "pose.onnx",
                "bed_model_path": "bed.onnx",
                "analysis_profile": {
                    "person_threshold": 0.25,
                    "bed_confidence": 0.25,
                    "max_frames": 20,
                    "max_duration_s": 10,
                    "max_pixels": 100,
                    "max_input_bytes": 100,
                },
            }
        )
    )
    return path


def _result() -> ClipAnalysisResult:
    return ClipAnalysisResult(
        "clip_reanalysis",
        "clip.mp4",
        "0" * 64,
        "1" * 64,
        "2" * 64,
        "pyav-test/mpeg4",
        "3" * 64,
        64,
        48,
        ClipAnalysisTimeBase(1, 1),
        (ClipAnalysisFrame(0, "available"),),
    )


def test_tool_writes_atomically(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "result.json"
    monkeypatch.setattr(clip_analysis, "_decoder_identity", lambda path: "pyav-test/mpeg4")
    monkeypatch.setattr(clip_analysis, "analyze_clip", lambda request, decoder_identity: _result())
    assert clip_analysis.main(["--request", str(_payload(tmp_path)), "--out", str(output)]) == 0
    assert output.exists()
    assert not list(tmp_path.glob(".result.json.tmp"))


def test_tool_rejection_exit_code(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(clip_analysis, "_decoder_identity", lambda path: "pyav-test/mpeg4")

    def reject(request, decoder_identity):
        raise clip_analysis.ClipAnalysisRejected("clip_sha256")

    monkeypatch.setattr(clip_analysis, "analyze_clip", reject)
    assert (
        clip_analysis.main(["--request", str(_payload(tmp_path)), "--out", str(tmp_path / "out")])
        == 2
    )
    assert json.loads(capsys.readouterr().out) == {
        "error": "ClipAnalysisRejected",
        "reason": "clip_sha256",
    }
