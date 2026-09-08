"""One-at-a-time, bounded subprocess supervision for clip re-analysis."""

from __future__ import annotations

import os
import subprocess
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from time import monotonic

from worker.adapters.model.clip_reanalysis import ClipAnalysisRejected
from worker.pipeline.output.evidence.clip_analysis_artifact import (
    ClipAnalysisArtifactIdentity,
    has_current_artifact,
    has_failed_outcome,
    publish_failed_outcome,
)
from worker.runtime import clip_analysis_process
from worker.runtime.clip_analysis_execution import run_job
from worker.runtime.clip_analysis_queue import Admission, ClipAnalysisQueue
from worker.runtime.clip_analysis_subprocess import (
    ClipAnalysisPdeathsigUnavailable,
    require_pdeathsig,
)
from worker.runtime.clip_analysis_supervisor_status import ClipAnalysisStatus


class ClipAnalysisSupervisorError(RuntimeError):
    """A supervisor operation could not safely start."""


class ClipAnalysisSupervisor:
    def __init__(
        self,
        *,
        python_executable: str,
        pose_model_path: Path,
        bed_model_path: Path,
        profile: object,
        cpu_index: int | None,
        deadline_s: float = 600.0,
        launch: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        probe: Callable[[Path], clip_analysis_process.ClipMediaFacts] = (
            clip_analysis_process.probe_media_facts
        ),
    ) -> None:
        if deadline_s <= 0:
            raise ValueError("deadline_s must be positive")
        if cpu_index is None:
            raise ClipAnalysisSupervisorError("clip_analysis_cpu_required")
        available = os.sched_getaffinity(0)
        if len(available) <= 1:
            raise ClipAnalysisSupervisorError("clip_analysis_cpu_unavailable")
        if cpu_index not in available:
            raise ClipAnalysisSupervisorError("clip_analysis_cpu_invalid")
        try:
            require_pdeathsig()
        except ClipAnalysisPdeathsigUnavailable as exc:
            raise ClipAnalysisSupervisorError("pdeathsig_unavailable") from exc
        self._python = python_executable
        self._pose_model = pose_model_path
        self._bed_model = bed_model_path
        self._profile = profile
        self._profile_sha = clip_analysis_process.profile_digest(profile)
        self._cpu_index = cpu_index
        self._deadline_s = deadline_s
        self._launch = launch
        self._probe = probe
        self._condition = threading.Condition()
        self._queue = ClipAnalysisQueue()
        self._notifications: list[tuple[str, Path, str, int, int]] = []
        self._active: clip_analysis_process.ClipAnalysisJob | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._cancelled = False
        self._stopping = False
        self._statuses: OrderedDict[str, ClipAnalysisStatus] = OrderedDict()
        self._thread = threading.Thread(target=self._serve, name="clip-analysis", daemon=True)
        self._thread.start()

    def status(self, clip_id: str) -> ClipAnalysisStatus:
        with self._condition:
            return self._statuses.get(clip_id, ClipAnalysisStatus("idle"))

    def notify(
        self, clip_id: str, clip_path: Path, clip_sha256: str, *, size_bytes: int, duration_ms: int
    ) -> None:
        """Constant-time publication hook; media I/O belongs to this thread."""
        with self._condition:
            if self._stopping:
                return
            self._notifications.append((clip_id, clip_path, clip_sha256, size_bytes, duration_ms))
            self._set_status(clip_id, ClipAnalysisStatus("queued"))
            self._condition.notify()

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
    ) -> Admission:
        return self.enqueue(
            clip_id,
            clip_path,
            clip_sha256,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            width=width,
            height=height,
            front=True,
        )

    def enqueue(
        self,
        clip_id: str,
        clip_path: Path,
        clip_sha256: str,
        *,
        size_bytes: int,
        duration_ms: int,
        width: int,
        height: int,
        front: bool = False,
    ) -> Admission:
        clip_analysis_process.validate_sha(clip_sha256)
        facts = self._probe(clip_path)
        _pre_admission(self._profile, size_bytes, duration_ms, facts.width, facts.height)
        with self._condition:
            if self._stopping:
                return Admission.STOPPED
            if self._active is not None and self._active.clip_id == clip_id:
                return Admission.ALREADY_RUNNING
            if has_current_artifact(
                clip_path.parent,
                clip_id=clip_id,
                clip_sha256=clip_sha256,
                pose_model_sha256=clip_analysis_process.model_digest(self._pose_model),
                bed_model_sha256=clip_analysis_process.model_digest(self._bed_model),
                analysis_profile_sha256=self._profile_sha,
                decoder_identity=facts.decoder_identity,
            ):
                self._set_status(clip_id, ClipAnalysisStatus("available"))
                return Admission.AVAILABLE
            job = clip_analysis_process.ClipAnalysisJob(
                clip_id,
                clip_path,
                clip_sha256,
                self._pose_model,
                self._bed_model,
                self._profile,
                self._profile_sha,
                facts.decoder_identity,
                front,
            )
            identity = ClipAnalysisArtifactIdentity(
                clip_id,
                clip_sha256,
                clip_analysis_process.model_digest(self._pose_model),
                clip_analysis_process.model_digest(self._bed_model),
                self._profile_sha,
                facts.decoder_identity,
            )
            if not front and has_failed_outcome(clip_path, identity):
                self._set_status(clip_id, ClipAnalysisStatus("failed", "previous_failure"))
                return Admission.REJECTED
            admission = self._queue.add(job, front=front)
            if admission == Admission.QUEUE_FULL:
                return admission
            self._set_status(clip_id, ClipAnalysisStatus("queued"))
            self._condition.notify()
            return admission

    def cancel(self, clip_id: str) -> bool:
        with self._condition:
            if self._queue.remove(clip_id):
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
                self._condition.wait_for(
                    lambda: self._queue or self._notifications or self._stopping
                )
                if self._stopping:
                    while (pending := self._queue.take()) is not None:
                        self._statuses[pending.clip_id] = ClipAnalysisStatus("failed", "cancelled")
                    return
                if self._notifications:
                    clip_id, clip_path, clip_sha256, size_bytes, duration_ms = (
                        self._notifications.pop(0)
                    )
                    try:
                        self.enqueue(
                            clip_id,
                            clip_path,
                            clip_sha256,
                            size_bytes=size_bytes,
                            duration_ms=duration_ms,
                            width=0,
                            height=0,
                        )
                    except (ClipAnalysisRejected, OSError, ValueError) as exc:
                        self._set_status(
                            clip_id, ClipAnalysisStatus("failed", clip_analysis_process.reason(exc))
                        )
                    continue
                job = self._queue.take()
                self._active = job
                self._cancelled = False
                self._statuses[job.clip_id] = ClipAnalysisStatus("running")
            assert job is not None
            status = ClipAnalysisStatus("failed", "unknown")
            try:
                status, _ = run_job(
                    job,
                    python_executable=self._python,
                    cpu_index=self._cpu_index,
                    deadline_s=self._deadline_s,
                    launch=self._launch,
                    set_process=self._set_process,
                    cancelled=self._cancelled_or_stopping,
                    terminate=self._terminate_group,
                    clock=monotonic,
                )
            except Exception as exc:  # noqa: BLE001 - keep the long-lived supervisor alive
                status = ClipAnalysisStatus("failed", clip_analysis_process.reason(exc))
            with self._condition:
                process = self._process
            teardown_failed = False
            if process is not None:
                try:
                    self._terminate_group(process)
                except Exception as exc:  # noqa: BLE001 - never release an unproved process group
                    status = ClipAnalysisStatus("failed", clip_analysis_process.reason(exc))
                    teardown_failed = True
            with self._condition:
                self._set_status(job.clip_id, status)
                if status.state == "failed" and status.reason not in ("timeout", "cancelled"):
                    publish_failed_outcome(
                        job.clip_path,
                        ClipAnalysisArtifactIdentity(
                            job.clip_id,
                            job.clip_sha256,
                            clip_analysis_process.model_digest(self._pose_model),
                            clip_analysis_process.model_digest(self._bed_model),
                            self._profile_sha,
                            job.decoder_identity,
                        ),
                        status.reason or "failed",
                    )
                if teardown_failed:
                    self._stopping = True
                    self._condition.notify_all()
                    return
                self._process = None
                self._active = None
                self._condition.notify_all()
                if self._stopping:
                    return

    def _set_process(self, process: subprocess.Popen[bytes]) -> bool:
        with self._condition:
            self._process = process
            return self._cancelled or self._stopping

    def _set_status(self, clip_id: str, status: ClipAnalysisStatus) -> None:
        self._statuses[clip_id] = status
        self._statuses.move_to_end(clip_id)
        terminal = [
            key for key, value in self._statuses.items() if value.state not in ("queued", "running")
        ]
        while len(terminal) > 256:
            self._statuses.pop(terminal.pop(0))

    def _cancelled_or_stopping(self) -> bool:
        with self._condition:
            return self._cancelled or self._stopping

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
    "ClipAnalysisStatus",
    "ClipAnalysisSupervisor",
    "ClipAnalysisSupervisorError",
]
