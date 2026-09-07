"""One-at-a-time, bounded subprocess supervision for clip re-analysis."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import mkstemp
from time import monotonic
from typing import Final

from shared.events.clip_analysis_wire import MAX_CLIP_ANALYSIS_OUTPUT_BYTES, decode_clip_analysis
from worker.pipeline.output.evidence.clip_analysis_artifact import (
    ClipAnalysisArtifactError,
    ClipAnalysisArtifactIdentity,
    publish_clip_analysis,
)
from worker.runtime.clip_analysis_subprocess import (
    ClipAnalysisPdeathsigUnavailable,
    child_setup,
    require_pdeathsig,
)

_TOOL_MODULE: Final = "worker.tools.clip_analysis"
_POLL_SECONDS: Final = 0.02


class ClipAnalysisSupervisorError(RuntimeError):
    """A supervisor operation could not safely start."""


class ClipAnalysisLaunchError(ClipAnalysisSupervisorError):
    """The child cannot be protected from orphaning."""


@dataclass(frozen=True, slots=True)
class ClipAnalysisStatus:
    state: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _Job:
    clip_id: str
    clip_path: Path
    clip_sha256: str
    pose_model_path: Path
    bed_model_path: Path
    profile: object
    profile_sha256: str


class ClipAnalysisSupervisor:
    def __init__(
        self,
        store_dir: Path,
        *,
        python_executable: str,
        pose_model_path: Path,
        bed_model_path: Path,
        profile: object,
        cpu_index: int | None,
        deadline_s: float = 600.0,
        launch: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        if deadline_s <= 0:
            raise ValueError("deadline_s must be positive")
        self._store_dir = store_dir
        self._python = python_executable
        self._pose_model = pose_model_path
        self._bed_model = bed_model_path
        self._profile = profile
        self._profile_sha = _profile_digest(profile)
        self._cpu_index = cpu_index
        self._deadline_s = deadline_s
        self._launch = launch
        self._condition = threading.Condition()
        self._pending: _Job | None = None
        self._active: _Job | None = None
        self._cancelled = False
        self._stopping = False
        self._statuses: dict[str, ClipAnalysisStatus] = {}
        self._thread = threading.Thread(target=self._serve, name="clip-analysis", daemon=True)
        self._thread.start()

    def status(self, clip_id: str) -> ClipAnalysisStatus:
        with self._condition:
            return self._statuses.get(clip_id, ClipAnalysisStatus("idle"))

    def trigger(self, clip_id: str, clip_path: Path, clip_sha256: str) -> bool:
        _validate_sha(clip_sha256)
        try:
            require_pdeathsig()
        except ClipAnalysisPdeathsigUnavailable as exc:
            raise ClipAnalysisLaunchError("pdeathsig_unavailable") from exc
        with self._condition:
            if self._active is not None or self._pending is not None:
                return False
            if self._stopping:
                raise ClipAnalysisSupervisorError("supervisor_stopped")
            job = _Job(
                clip_id,
                clip_path,
                clip_sha256,
                self._pose_model,
                self._bed_model,
                self._profile,
                self._profile_sha,
            )
            self._pending = job
            self._statuses[clip_id] = ClipAnalysisStatus("running")
            self._condition.notify()
            return True

    def cancel(self, clip_id: str) -> bool:
        with self._condition:
            if self._pending is not None and self._pending.clip_id == clip_id:
                self._pending = None
                self._statuses[clip_id] = ClipAnalysisStatus("failed", "cancelled")
                self._condition.notify_all()
                return True
            if self._active is None or self._active.clip_id != clip_id:
                return False
            self._cancelled = True
            return True

    def shutdown(self) -> None:
        with self._condition:
            self._stopping = True
            self._cancelled = True
            self._condition.notify()
        self._thread.join()

    def _serve(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._stopping)
                if self._stopping and self._pending is None:
                    return
                job = self._pending
                self._pending = None
                self._active = job
                self._cancelled = False
            assert job is not None
            status = self._run(job)
            with self._condition:
                self._statuses[job.clip_id] = status
                self._active = None
                self._condition.notify_all()
                if self._stopping:
                    return

    def _run(self, job: _Job) -> ClipAnalysisStatus:
        try:
            require_pdeathsig()
            scratch = _scratch_path(job.clip_path)
            request = _request_path(job, scratch)
        except (ClipAnalysisPdeathsigUnavailable, OSError, ValueError) as exc:
            return ClipAnalysisStatus("failed", _reason(exc))
        process: subprocess.Popen[bytes] | None = None
        read_fd = -1
        write_fd = -1
        try:
            read_fd, write_fd = os.pipe()
            process = self._launch(
                [
                    self._python,
                    "-m",
                    _TOOL_MODULE,
                    "--request",
                    str(request),
                    "--out",
                    str(scratch),
                ],
                stdin=read_fd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                preexec_fn=child_setup(os.getpid(), self._cpu_index),
                env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
            os.close(read_fd)
            read_fd = -1
            deadline = monotonic() + self._deadline_s
            reason: str | None = None
            while process.poll() is None:
                with self._condition:
                    cancelled = self._cancelled or self._stopping
                if cancelled:
                    reason = "cancelled"
                    break
                if monotonic() >= deadline:
                    reason = "timeout"
                    break
                threading.Event().wait(_POLL_SECONDS)
            if reason is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            exit_code = process.wait()
            if reason is not None:
                return ClipAnalysisStatus("failed", reason)
            with self._condition:
                if self._cancelled or self._stopping:
                    return ClipAnalysisStatus("failed", "cancelled")
            if exit_code != 0:
                return ClipAnalysisStatus("failed", f"child_exit_{exit_code}")
            if scratch.stat().st_size > MAX_CLIP_ANALYSIS_OUTPUT_BYTES:
                return ClipAnalysisStatus("failed", "output_too_large")
            result = decode_clip_analysis(scratch.read_bytes())
            if (
                result.clip_id != job.clip_id
                or result.clip_sha256 != job.clip_sha256
                or result.pose_model_sha256 != _model_digest(job.pose_model_path)
                or result.bed_model_sha256 != _model_digest(job.bed_model_path)
                or result.analysis_profile_sha256 != job.profile_sha256
            ):
                return ClipAnalysisStatus("failed", "identity_mismatch")
            identity = ClipAnalysisArtifactIdentity(
                job.clip_id,
                job.clip_sha256,
                result.pose_model_sha256,
                result.bed_model_sha256,
                job.profile_sha256,
                result.decoder_identity,
            )
            with self._condition:
                if self._cancelled or self._stopping:
                    return ClipAnalysisStatus("failed", "cancelled")
                publish_clip_analysis(job.clip_path, scratch, identity)
            return ClipAnalysisStatus("available")
        except (ClipAnalysisArtifactError, OSError, ValueError) as exc:
            return ClipAnalysisStatus("failed", _reason(exc))
        finally:
            if read_fd >= 0:
                os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
            scratch.unlink(missing_ok=True) if "scratch" in locals() else None
            request.unlink(missing_ok=True) if "request" in locals() else None


def _profile_digest(profile: object) -> str:
    if not is_dataclass(profile):
        raise TypeError("profile must be a dataclass")
    encoded = json.dumps(asdict(profile), sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _validate_sha(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("invalid_clip_sha256")


def _model_digest(path: Path) -> str:
    return path.with_name(f"{path.name}.sha256").read_text(encoding="ascii").strip()


def _scratch_path(clip_path: Path) -> Path:
    fd, name = mkstemp(prefix=".clip-analysis-", suffix=".json", dir=clip_path.parent)
    os.close(fd)
    path = Path(name)
    path.unlink()
    return path


def _request_path(job: _Job, scratch: Path) -> Path:
    fd, name = mkstemp(prefix=".clip-analysis-request-", suffix=".json", dir=scratch.parent)
    payload = {
        "clip_id": job.clip_id,
        "clip_path": str(job.clip_path),
        "clip_sha256": job.clip_sha256,
        "pose_model_path": str(job.pose_model_path),
        "bed_model_path": str(job.bed_model_path),
        "analysis_profile": asdict(job.profile),
    }
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump(payload, file, sort_keys=True, separators=(",", ":"))
    return Path(name)


def _reason(exc: BaseException) -> str:
    text = str(exc)
    return text if text and "rtsp" not in text.lower() else type(exc).__name__.lower()


__all__ = [
    "ClipAnalysisLaunchError",
    "ClipAnalysisStatus",
    "ClipAnalysisSupervisor",
    "ClipAnalysisSupervisorError",
]
