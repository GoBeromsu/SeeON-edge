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
from pathlib import Path
from tempfile import mkstemp
from time import monotonic
from typing import Final

from shared.events.clip_analysis_wire import MAX_CLIP_ANALYSIS_OUTPUT_BYTES, decode_clip_analysis
from worker.adapters.model.clip_reanalysis import ClipAnalysisRejected
from worker.pipeline.output.evidence.clip_analysis_artifact import (
    ClipAnalysisArtifactError,
    ClipAnalysisArtifactIdentity,
    publish_clip_analysis,
)
from worker.runtime.clip_analysis_subprocess import (
    ClipAnalysisPdeathsigUnavailable,
    require_pdeathsig,
)

_TOOL_MODULE: Final = "worker.tools.clip_analysis"
_GROUP_EMPTY_RETRIES: Final = 20


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
        if cpu_index is None:
            raise ClipAnalysisLaunchError("clip_analysis_cpu_required")
        available = os.sched_getaffinity(0)
        if len(available) <= 1:
            raise ClipAnalysisLaunchError("clip_analysis_cpu_unavailable")
        if cpu_index not in available:
            raise ClipAnalysisLaunchError("clip_analysis_cpu_invalid")
        try:
            require_pdeathsig()
        except ClipAnalysisPdeathsigUnavailable as exc:
            raise ClipAnalysisLaunchError("pdeathsig_unavailable") from exc
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
        self._process: subprocess.Popen[bytes] | None = None
        self._cancelled = False
        self._stopping = False
        self._statuses: dict[str, ClipAnalysisStatus] = {}
        self._thread = threading.Thread(target=self._serve, name="clip-analysis", daemon=True)
        self._thread.start()

    def status(self, clip_id: str) -> ClipAnalysisStatus:
        with self._condition:
            return self._statuses.get(clip_id, ClipAnalysisStatus("idle"))

    def trigger(
        self,
        clip_id: str,
        clip_path: Path,
        clip_sha256: str,
        *,
        size_bytes: int,
        duration_ms: int,
        width: int,
        height: int,
    ) -> bool:
        _validate_sha(clip_sha256)
        _pre_admission(self._profile, size_bytes, duration_ms, width, height)
        with self._condition:
            if self._active is not None or self._pending is not None:
                return False
            if self._stopping:
                raise ClipAnalysisSupervisorError("supervisor_stopped")
            self._pending = _Job(
                clip_id,
                clip_path,
                clip_sha256,
                self._pose_model,
                self._bed_model,
                self._profile,
                self._profile_sha,
            )
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
            process = self._process
        if process is not None:
            self._terminate_group(process)
        return True

    def shutdown(self) -> None:
        with self._condition:
            self._stopping = True
            self._cancelled = True
            process = self._process
            self._condition.notify_all()
        if process is not None:
            self._terminate_group(process)
        self._thread.join()

    def _serve(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._stopping)
                if self._stopping:
                    if self._pending is not None:
                        self._statuses[self._pending.clip_id] = ClipAnalysisStatus(
                            "failed", "cancelled"
                        )
                        self._pending = None
                    return
                job = self._pending
                self._pending = None
                self._active = job
                self._cancelled = False
            assert job is not None
            process: subprocess.Popen[bytes] | None = None
            try:
                status, process = self._run(job)
            except Exception as exc:  # noqa: BLE001 - keep the long-lived supervisor alive
                status = ClipAnalysisStatus("failed", _reason(exc))
            teardown_failed = False
            if process is not None:
                try:
                    self._terminate_group(process)
                except Exception as exc:  # noqa: BLE001 - never release an unproved process group
                    status = ClipAnalysisStatus("failed", _reason(exc))
                    teardown_failed = True
            with self._condition:
                self._process = None
                self._statuses[job.clip_id] = status
                if teardown_failed:
                    self._stopping = True
                    self._condition.notify_all()
                    return
                self._active = None
                self._condition.notify_all()
                if self._stopping:
                    return

    def _run(self, job: _Job) -> tuple[ClipAnalysisStatus, subprocess.Popen[bytes] | None]:
        scratch: Path | None = None
        request: Path | None = None
        read_fd = write_fd = -1
        process: subprocess.Popen[bytes] | None = None
        try:
            scratch = _scratch_path(job.clip_path)
            request = _request_path(job, scratch)
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
                    "--expected-parent",
                    str(os.getpid()),
                    "--cpu",
                    str(self._cpu_index),
                    "--control-fd",
                    str(read_fd),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                pass_fds=(read_fd,),
                env=os.environ.copy(),
            )
            os.close(read_fd)
            read_fd = -1
            with self._condition:
                self._process = process
            expires_at = monotonic() + self._deadline_s
            try:
                exit_code = process.wait(timeout=max(0.0, expires_at - monotonic()))
            except subprocess.TimeoutExpired:
                self._terminate_group(process)
                return ClipAnalysisStatus("failed", "timeout"), process
            with self._condition:
                cancelled = self._cancelled or self._stopping
            expired = monotonic() >= expires_at
            if cancelled or expired:
                self._terminate_group(process)
                return ClipAnalysisStatus(
                    "failed", "cancelled" if cancelled else "timeout"
                ), process
            if exit_code != 0:
                return ClipAnalysisStatus("failed", f"child_exit_{exit_code}"), process
            if scratch.stat().st_size > MAX_CLIP_ANALYSIS_OUTPUT_BYTES:
                return ClipAnalysisStatus("failed", "output_too_large"), process
            result = decode_clip_analysis(scratch.read_bytes())
            if (
                result.clip_id != job.clip_id
                or result.clip_sha256 != job.clip_sha256
                or result.pose_model_sha256 != _model_digest(job.pose_model_path)
                or result.bed_model_sha256 != _model_digest(job.bed_model_path)
                or result.analysis_profile_sha256 != job.profile_sha256
            ):
                return ClipAnalysisStatus("failed", "identity_mismatch"), process
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
                    return ClipAnalysisStatus("failed", "cancelled"), process
                if monotonic() >= expires_at:
                    return ClipAnalysisStatus("failed", "timeout"), process
                publish_clip_analysis(job.clip_path, scratch, identity)
            return ClipAnalysisStatus("available"), process
        except (ClipAnalysisArtifactError, OSError, ValueError, subprocess.SubprocessError) as exc:
            return ClipAnalysisStatus("failed", _reason(exc)), process
        finally:
            if read_fd >= 0:
                os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
            if scratch is not None:
                scratch.unlink(missing_ok=True)
            if request is not None:
                request.unlink(missing_ok=True)

    def _terminate_group(self, process: subprocess.Popen[bytes]) -> None:
        """Kill, reap, and positively prove this job's process group is gone."""
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
        raise ClipAnalysisSupervisorError("process_group_not_empty")


def _pre_admission(
    profile: object, size_bytes: int, duration_ms: int, width: int, height: int
) -> None:
    if any(
        not isinstance(value, int) or value < 0
        for value in (size_bytes, duration_ms, width, height)
    ):
        raise ClipAnalysisRejected("manifest_facts")
    if size_bytes > profile.max_input_bytes:
        raise ClipAnalysisRejected("input_bytes")
    if duration_ms > profile.max_duration_s * 1000:
        raise ClipAnalysisRejected("duration")
    if width == 0 or height == 0 or width * height > profile.max_pixels:
        raise ClipAnalysisRejected("resolution")


def _profile_digest(profile: object) -> str:
    from worker.adapters.model.clip_reanalysis import profile_sha256

    if not is_dataclass(profile):
        raise TypeError("profile must be a dataclass")
    return profile_sha256(profile)


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
