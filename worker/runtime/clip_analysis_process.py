"""Child-process primitives for clip re-analysis."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from tempfile import mkstemp
from typing import Final

from shared.events.clip_analysis_wire import ClipAnalysisResult

_TOOL_MODULE: Final = "worker.tools.clip_analysis"
_GROUP_EMPTY_RETRIES: Final = 20


@dataclass(frozen=True, slots=True)
class ClipAnalysisJob:
    clip_id: str
    clip_path: Path
    clip_sha256: str
    pose_model_path: Path
    bed_model_path: Path
    profile: object
    profile_sha256: str
    decoder_identity: str
    front: bool
    size_bytes: int = 0
    duration_ms: int = 0
    width: int = 0
    height: int = 0
    pose_model_sha256: str = ""
    bed_model_sha256: str = ""
    generation: int = 0


@dataclass(frozen=True, slots=True)
class ClipMediaFacts:
    width: int
    height: int
    decoder_identity: str


@dataclass(slots=True)
class ClipAnalysisChild:
    process: subprocess.Popen[bytes]
    scratch: Path
    request: Path
    control_write_fd: int

    def cleanup(self) -> None:
        os.close(self.control_write_fd)
        self.scratch.unlink(missing_ok=True)
        self.request.unlink(missing_ok=True)


def terminate_group(
    process: subprocess.Popen[bytes], *, failure: Callable[[str], BaseException]
) -> None:
    """Kill, reap, and prove this job's process group is gone."""
    pgid = process.pid
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    process.wait()
    for _ in range(_GROUP_EMPTY_RETRIES):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        threading.Event().wait(0.01)
    raise failure("process_group_not_empty")


def launch_child(
    launcher: Callable[..., subprocess.Popen[bytes]],
    *,
    python_executable: str,
    job: ClipAnalysisJob,
    cpu_index: int,
) -> ClipAnalysisChild:
    scratch = scratch_path(job.clip_path)
    request: Path | None = None
    read_fd = write_fd = -1
    try:
        request = request_path(
            clip_id=job.clip_id,
            clip_path=job.clip_path,
            clip_sha256=job.clip_sha256,
            pose_model_path=job.pose_model_path,
            bed_model_path=job.bed_model_path,
            profile=job.profile,
            scratch=scratch,
        )
        read_fd, write_fd = os.pipe()
        process = launch(
            launcher,
            python_executable=python_executable,
            request=request,
            scratch=scratch,
            cpu_index=cpu_index,
            control_fd=read_fd,
        )
        os.close(read_fd)
        read_fd = -1
        return ClipAnalysisChild(process, scratch, request, write_fd)
    except Exception:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        scratch.unlink(missing_ok=True)
        if request is not None:
            request.unlink(missing_ok=True)
        raise


def launch(
    launcher: Callable[..., subprocess.Popen[bytes]],
    *,
    python_executable: str,
    request: Path,
    scratch: Path,
    cpu_index: int,
    control_fd: int,
) -> subprocess.Popen[bytes]:
    return launcher(
        [
            python_executable,
            "-m",
            _TOOL_MODULE,
            "--request",
            str(request),
            "--out",
            str(scratch),
            "--expected-parent",
            str(os.getpid()),
            "--cpu",
            str(cpu_index),
            "--control-fd",
            str(control_fd),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        pass_fds=(control_fd,),
        env=os.environ.copy(),
    )


def profile_digest(profile: object) -> str:
    from worker.adapters.model.clip_reanalysis import profile_sha256

    if not is_dataclass(profile):
        raise TypeError("profile must be a dataclass")
    return profile_sha256(profile)


def validate_sha(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("invalid_clip_sha256")


def model_digest(path: Path) -> str:
    from worker.adapters.model.artifact import read_artifact_digest_sidecar

    return read_artifact_digest_sidecar(path)


def identity_matches(job: ClipAnalysisJob, result: ClipAnalysisResult) -> bool:
    return (
        result.clip_id == job.clip_id
        and result.clip_sha256 == job.clip_sha256
        and result.pose_model_sha256 == job.pose_model_sha256
        and result.bed_model_sha256 == job.bed_model_sha256
        and result.analysis_profile_sha256 == job.profile_sha256
        and result.decoder_identity == job.decoder_identity
    )


def probe_media_facts(clip_path: Path) -> ClipMediaFacts:
    """Probe media using the same single-thread decoder configuration as analysis."""
    import av

    with av.open(str(clip_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.thread_count = 1
        if stream.width <= 0 or stream.height <= 0:
            raise ValueError("invalid_dimensions")
        codec_name = stream.codec_context.name
        if not codec_name:
            raise ValueError("missing_codec")
        return ClipMediaFacts(
            width=stream.width,
            height=stream.height,
            decoder_identity=f"pyav-{av.__version__}/{codec_name}",
        )


def scratch_path(clip_path: Path) -> Path:
    fd, name = mkstemp(prefix=".clip-analysis-", suffix=".json", dir=clip_path.parent)
    os.close(fd)
    path = Path(name)
    path.unlink()
    return path


def request_path(
    *,
    clip_id: str,
    clip_path: Path,
    clip_sha256: str,
    pose_model_path: Path,
    bed_model_path: Path,
    profile: object,
    scratch: Path,
) -> Path:
    fd, name = mkstemp(prefix=".clip-analysis-request-", suffix=".json", dir=scratch.parent)
    payload = {
        "clip_id": clip_id,
        "clip_path": str(clip_path),
        "clip_sha256": clip_sha256,
        "pose_model_path": str(pose_model_path),
        "bed_model_path": str(bed_model_path),
        "analysis_profile": asdict(profile),
    }
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump(payload, file, sort_keys=True, separators=(",", ":"))
    return Path(name)


def reason(exc: BaseException) -> str:
    text = str(exc)
    return text if text and "rtsp" not in text.lower() else type(exc).__name__.lower()
