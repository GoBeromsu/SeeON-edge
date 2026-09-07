"""One-at-a-time, bounded subprocess supervision for clip re-analysis."""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from shared.events.clip_analysis_wire import MAX_CLIP_ANALYSIS_OUTPUT_BYTES, decode_clip_analysis
from worker.adapters.model.clip_reanalysis import ClipAnalysisRejected
from worker.pipeline.output.evidence.clip_analysis_artifact import (
    ClipAnalysisArtifactError,
    ClipAnalysisArtifactIdentity,
    publish_clip_analysis,
)
from worker.runtime import clip_analysis_process
from worker.runtime.clip_analysis_subprocess import (
    ClipAnalysisPdeathsigUnavailable,
    require_pdeathsig,
)


class ClipAnalysisSupervisorError(RuntimeError):
    """A supervisor operation could not safely start."""


class ClipAnalysisLaunchError(ClipAnalysisSupervisorError):
    """The child cannot be protected from orphaning."""


@dataclass(frozen=True, slots=True)
class ClipAnalysisStatus:
    state: str
    reason: str | None = None


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
        self._profile_sha = clip_analysis_process.profile_digest(profile)
        self._cpu_index = cpu_index
        self._deadline_s = deadline_s
        self._launch = launch
        self._condition = threading.Condition()
        self._pending: clip_analysis_process.ClipAnalysisJob | None = None
        self._active: clip_analysis_process.ClipAnalysisJob | None = None
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
        clip_analysis_process.validate_sha(clip_sha256)
        _pre_admission(self._profile, size_bytes, duration_ms, width, height)
        with self._condition:
            if self._active is not None or self._pending is not None:
                return False
            if self._stopping:
                raise ClipAnalysisSupervisorError("supervisor_stopped")
            self._pending = clip_analysis_process.ClipAnalysisJob(
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
                status = ClipAnalysisStatus("failed", clip_analysis_process.reason(exc))
            teardown_failed = False
            if process is not None:
                try:
                    self._terminate_group(process)
                except Exception as exc:  # noqa: BLE001 - never release an unproved process group
                    status = ClipAnalysisStatus("failed", clip_analysis_process.reason(exc))
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

    def _run(
        self, job: clip_analysis_process.ClipAnalysisJob
    ) -> tuple[ClipAnalysisStatus, subprocess.Popen[bytes] | None]:
        child: clip_analysis_process.ClipAnalysisChild | None = None
        process: subprocess.Popen[bytes] | None = None
        try:
            child = clip_analysis_process.launch_child(
                self._launch,
                python_executable=self._python,
                job=job,
                cpu_index=self._cpu_index,
            )
            process = child.process
            scratch = child.scratch
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
            if not clip_analysis_process.identity_matches(job, result):
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
            return ClipAnalysisStatus("failed", clip_analysis_process.reason(exc)), process
        finally:
            if child is not None:
                child.cleanup()

    def _terminate_group(self, process: subprocess.Popen[bytes]) -> None:
        clip_analysis_process.terminate_group(process, failure=ClipAnalysisSupervisorError)


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


__all__ = [
    "ClipAnalysisLaunchError",
    "ClipAnalysisStatus",
    "ClipAnalysisSupervisor",
    "ClipAnalysisSupervisorError",
]
