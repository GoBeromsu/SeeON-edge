from __future__ import annotations

import base64
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import monotonic

import pytest

from shared.events.clip_analysis_wire import (
    ClipAnalysisFrame,
    ClipAnalysisResult,
    ClipAnalysisTimeBase,
    encode_clip_analysis,
)
from worker.runtime.clip_analysis_supervisor import ClipAnalysisSupervisor


@dataclass(frozen=True, slots=True)
class _Profile:
    person_threshold: float = 0.25
    bed_confidence: float = 0.5
    max_frames: int = 20
    max_duration_s: float = 2.0
    max_pixels: int = 100
    max_input_bytes: int = 100


def _wait(supervisor: ClipAnalysisSupervisor, clip_id: str) -> str:
    end = monotonic() + 3
    while monotonic() < end:
        state = supervisor.status(clip_id).state
        if state != "running":
            return state
        threading.Event().wait(0.01)
    raise AssertionError("supervisor did not finish")


def _wait_for_child(pids: list[int]) -> None:
    end = monotonic() + 3
    while monotonic() < end:
        if pids:
            return
        threading.Event().wait(0.01)
    raise AssertionError("child was not launched")


def _model(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    digest = sha256(content).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(f"{digest}\n")
    return digest


def _launch_for(payload: bytes, pids: list[int], delay: float = 0.0):
    encoded = base64.b64encode(payload).decode("ascii")
    code = (
        "import base64,pathlib,sys,time;"
        f"time.sleep({delay});"
        f"pathlib.Path(sys.argv[1]).write_bytes(base64.b64decode('{encoded}'))"
    )

    def launch(_command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        out = _command[-1]
        process = subprocess.Popen([sys.executable, "-c", code, out], **kwargs)
        pids.append(process.pid)
        return process

    return launch


def _result(clip_id: str, clip_sha: str, pose_sha: str, bed_sha: str) -> bytes:
    profile_sha = sha256(
        b'{"bed_confidence":0.5,"max_duration_s":2.0,"max_frames":20,"max_input_bytes":100,"max_pixels":100,"person_threshold":0.25}'
    ).hexdigest()
    return encode_clip_analysis(
        ClipAnalysisResult(
            source="clip_reanalysis",
            clip_id=clip_id,
            clip_sha256=clip_sha,
            pose_model_sha256=pose_sha,
            bed_model_sha256=bed_sha,
            decoder_identity="pyav-16/h264",
            analysis_profile_sha256=profile_sha,
            image_width=10,
            image_height=10,
            time_base=ClipAnalysisTimeBase(1, 1),
            frames=(ClipAnalysisFrame(0, "no_evidence"),),
        )
    )


def test_second_trigger_is_refused_and_slot_releases_after_publish(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    pose_sha = _model(tmp_path / "pose.onnx", b"pose")
    bed_sha = _model(tmp_path / "bed.onnx", b"bed")
    clip_sha = "a" * 64
    pids: list[int] = []
    supervisor = ClipAnalysisSupervisor(
        tmp_path,
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=None,
        launch=_launch_for(_result("event-1", clip_sha, pose_sha, bed_sha), pids, 0.05),
    )
    try:
        assert supervisor.trigger("event-1", clip, clip_sha)
        assert not supervisor.trigger("event-2", clip, "e" * 64)
        assert _wait(supervisor, "event-1") == "available"
        assert supervisor.trigger("event-2", clip, "e" * 64)
        assert _wait(supervisor, "event-2") == "failed"
        assert not list(tmp_path.glob(".clip-analysis-*"))
    finally:
        supervisor.shutdown()


@pytest.mark.heavy
def test_timeout_kills_and_reaps_child(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    _model(tmp_path / "pose.onnx", b"pose")
    _model(tmp_path / "bed.onnx", b"bed")
    pids: list[int] = []
    supervisor = ClipAnalysisSupervisor(
        tmp_path,
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=None,
        deadline_s=0.05,
        launch=_launch_for(b"{}", pids, 10),
    )
    try:
        assert supervisor.trigger("event-1", clip, "a" * 64)
        assert _wait(supervisor, "event-1") == "failed"
        assert supervisor.status("event-1").reason == "timeout"
        with pytest.raises(ProcessLookupError):
            os.kill(pids[0], 0)
        assert supervisor.trigger("event-2", clip, "b" * 64)
    finally:
        supervisor.shutdown()


def test_cancel_kills_before_artifact_publish(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    _model(tmp_path / "pose.onnx", b"pose")
    _model(tmp_path / "bed.onnx", b"bed")
    pids: list[int] = []
    supervisor = ClipAnalysisSupervisor(
        tmp_path,
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=None,
        launch=_launch_for(b"{}", pids, 10),
    )
    try:
        assert supervisor.trigger("event-1", clip, "a" * 64)
        _wait_for_child(pids)
        assert supervisor.cancel("event-1")
        assert _wait(supervisor, "event-1") == "failed"
        assert supervisor.status("event-1").reason == "cancelled"
        assert not (tmp_path / "clip.analysis.json").exists()
    finally:
        supervisor.shutdown()
