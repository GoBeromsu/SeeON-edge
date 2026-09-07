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
from worker.adapters.model.clip_reanalysis import profile_sha256
from worker.runtime.clip_analysis_supervisor import ClipAnalysisSupervisor
from worker.tools.clip_analysis import _load_request


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


def _wait_gone(pid: int) -> None:
    deadline = monotonic() + 5
    while monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        threading.Event().wait(0.01)
    raise AssertionError(f"process {pid} survived supervision")


def _model(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    digest = sha256(content).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(f"{digest}\n")
    return digest


def _cpu() -> int:
    return next(iter(os.sched_getaffinity(0)))


def _trigger(supervisor: ClipAnalysisSupervisor, clip_id: str, clip: Path, digest: str) -> bool:
    return supervisor.trigger(
        clip_id, clip, digest, size_bytes=4, duration_ms=100, width=10, height=10
    )


def _launch_for(payload: bytes, pids: list[int], delay: float = 0.0):
    encoded = base64.b64encode(payload).decode("ascii")
    code = (
        f"import base64,pathlib,sys,time;time.sleep({delay});"
        f"pathlib.Path(sys.argv[1]).write_bytes(base64.b64decode('{encoded}'))"
    )

    def launch(command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        out = command[command.index("--out") + 1]
        process = subprocess.Popen([sys.executable, "-c", code, out], **kwargs)
        pids.append(process.pid)
        return process

    return launch


def _result(clip_id: str, clip_sha: str, pose_sha: str, bed_sha: str) -> bytes:
    return encode_clip_analysis(
        ClipAnalysisResult(
            source="clip_reanalysis",
            clip_id=clip_id,
            clip_sha256=clip_sha,
            pose_model_sha256=pose_sha,
            bed_model_sha256=bed_sha,
            decoder_identity="pyav-16/h264",
            analysis_profile_sha256=profile_sha256(_Profile()),
            image_width=10,
            image_height=10,
            time_base=ClipAnalysisTimeBase(1, 1),
            frames=(ClipAnalysisFrame(0, "no_evidence"),),
        )
    )


def test_real_request_round_trip_preserves_canonical_clip_id(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    _model(tmp_path / "pose.onnx", b"pose")
    _model(tmp_path / "bed.onnx", b"bed")
    paths: list[Path] = []

    def launch(command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        request = Path(command[command.index("--request") + 1])
        paths.append(request)
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], **kwargs)

    supervisor = ClipAnalysisSupervisor(
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=_cpu(),
        launch=launch,
    )
    try:
        assert _trigger(supervisor, "canonical-id", clip, "a" * 64)
        end = monotonic() + 3
        while not paths and monotonic() < end:
            threading.Event().wait(0.01)
        assert paths and _load_request(paths[0]).clip_id == "canonical-id"
        assert _load_request(paths[0]).clip_id != clip.name
        supervisor.cancel("canonical-id")
        _wait(supervisor, "canonical-id")
    finally:
        supervisor.shutdown()


def test_second_trigger_is_refused_and_slot_releases_after_publish(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    pose_sha = _model(tmp_path / "pose.onnx", b"pose")
    bed_sha = _model(tmp_path / "bed.onnx", b"bed")
    pids: list[int] = []
    supervisor = ClipAnalysisSupervisor(
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=_cpu(),
        launch=_launch_for(_result("event-1", "a" * 64, pose_sha, bed_sha), pids, 0.05),
    )
    try:
        assert _trigger(supervisor, "event-1", clip, "a" * 64)
        assert not _trigger(supervisor, "event-2", clip, "e" * 64)
        assert _wait(supervisor, "event-1") == "available"
        assert _trigger(supervisor, "event-2", clip, "e" * 64)
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
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=_cpu(),
        deadline_s=0.05,
        launch=_launch_for(b"{}", pids, 10),
    )
    try:
        assert _trigger(supervisor, "event-1", clip, "a" * 64)
        assert _wait(supervisor, "event-1") == "failed"
        assert supervisor.status("event-1").reason == "timeout"
        with pytest.raises(ProcessLookupError):
            os.kill(pids[0], 0)
    finally:
        supervisor.shutdown()


@pytest.mark.heavy
def test_timeout_kills_group_and_grandchild_with_zero_survivors(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    _model(tmp_path / "pose.onnx", b"pose")
    _model(tmp_path / "bed.onnx", b"bed")
    grandchild_file = tmp_path / "grandchild.pid"

    def launch(command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        code = (
            "import pathlib,subprocess,sys;"
            "grandchild=subprocess.Popen([sys.executable, '-c', 'while True: pass']);"
            f"pathlib.Path({str(grandchild_file)!r}).write_text(str(grandchild.pid));"
            "\nwhile True: pass"
        )
        return subprocess.Popen([sys.executable, "-c", code], **kwargs)

    supervisor = ClipAnalysisSupervisor(
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=_cpu(),
        deadline_s=0.05,
        launch=launch,
    )
    try:
        assert _trigger(supervisor, "event-1", clip, "a" * 64)
        assert _wait(supervisor, "event-1") == "failed"
        assert supervisor.status("event-1").reason == "timeout"
        deadline = monotonic() + 3
        while not grandchild_file.exists() and monotonic() < deadline:
            threading.Event().wait(0.01)
        _wait_gone(int(grandchild_file.read_text()))
    finally:
        supervisor.shutdown()


@pytest.mark.heavy
def test_cancel_kills_group_reaps_and_releases_slot_with_zero_survivors(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    _model(tmp_path / "pose.onnx", b"pose")
    _model(tmp_path / "bed.onnx", b"bed")
    pids: list[int] = []
    supervisor = ClipAnalysisSupervisor(
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=_cpu(),
        launch=_launch_for(b"{}", pids, 10),
    )
    try:
        assert _trigger(supervisor, "event-1", clip, "a" * 64)
        deadline = monotonic() + 3
        while not pids and monotonic() < deadline:
            threading.Event().wait(0.01)
        assert supervisor.cancel("event-1")
        assert _wait(supervisor, "event-1") == "failed"
        assert supervisor.status("event-1").reason == "cancelled"
        _wait_gone(pids[0])
        assert _trigger(supervisor, "event-2", clip, "e" * 64)
    finally:
        supervisor.shutdown()


@pytest.mark.heavy
def test_completion_after_deadline_is_timeout_and_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    pose_sha = _model(tmp_path / "pose.onnx", b"pose")
    bed_sha = _model(tmp_path / "bed.onnx", b"bed")
    pids: list[int] = []
    times = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr("worker.runtime.clip_analysis_supervisor.monotonic", lambda: next(times))
    supervisor = ClipAnalysisSupervisor(
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=_cpu(),
        deadline_s=1.0,
        launch=_launch_for(_result("event-1", "a" * 64, pose_sha, bed_sha), pids),
    )
    try:
        assert _trigger(supervisor, "event-1", clip, "a" * 64)
        assert _wait(supervisor, "event-1") == "failed"
        assert supervisor.status("event-1").reason == "timeout"
        assert tuple(tmp_path.glob("clip.analysis.*.json")) == ()
    finally:
        supervisor.shutdown()


def test_launch_failure_keeps_supervisor_alive_and_releases_slot(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    _model(tmp_path / "pose.onnx", b"pose")
    _model(tmp_path / "bed.onnx", b"bed")
    attempts = 0

    def launch(_command: list[str], **_kwargs: object) -> subprocess.Popen[bytes]:
        nonlocal attempts
        attempts += 1
        raise OSError("launch_failed")

    supervisor = ClipAnalysisSupervisor(
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=_cpu(),
        launch=launch,
    )
    try:
        assert _trigger(supervisor, "event-1", clip, "a" * 64)
        assert _wait(supervisor, "event-1") == "failed"
        assert _trigger(supervisor, "event-2", clip, "e" * 64)
        assert _wait(supervisor, "event-2") == "failed"
        assert attempts == 2
    finally:
        supervisor.shutdown()


def test_teardown_failure_closes_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    pose_sha = _model(tmp_path / "pose.onnx", b"pose")
    bed_sha = _model(tmp_path / "bed.onnx", b"bed")
    pids: list[int] = []
    supervisor = ClipAnalysisSupervisor(
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=_cpu(),
        launch=_launch_for(_result("event-1", "a" * 64, pose_sha, bed_sha), pids),
    )
    monkeypatch.setattr(
        supervisor, "_terminate_group", lambda _process: (_ for _ in ()).throw(OSError("teardown"))
    )
    try:
        assert _trigger(supervisor, "event-1", clip, "a" * 64)
        assert _wait(supervisor, "event-1") == "failed"
        assert not _trigger(supervisor, "event-2", clip, "e" * 64)
    finally:
        monkeypatch.setattr(supervisor, "_terminate_group", lambda process: process.kill())
        supervisor.shutdown()


@pytest.mark.heavy
def test_no_job_state_marker_while_active_or_after_teardown(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    _model(tmp_path / "pose.onnx", b"pose")
    _model(tmp_path / "bed.onnx", b"bed")
    pids: list[int] = []
    supervisor = ClipAnalysisSupervisor(
        python_executable=sys.executable,
        pose_model_path=tmp_path / "pose.onnx",
        bed_model_path=tmp_path / "bed.onnx",
        profile=_Profile(),
        cpu_index=_cpu(),
        launch=_launch_for(b"{}", pids, 10),
    )
    try:
        assert _trigger(supervisor, "event-1", clip, "a" * 64)
        deadline = monotonic() + 3
        while not pids and monotonic() < deadline:
            threading.Event().wait(0.01)
        assert tuple(tmp_path.glob("*.state")) == ()
        assert supervisor.cancel("event-1")
        assert _wait(supervisor, "event-1") == "failed"
        assert tuple(tmp_path.glob("*.state")) == ()
    finally:
        supervisor.shutdown()
