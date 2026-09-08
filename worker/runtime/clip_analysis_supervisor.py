"""One-at-a-time, bounded subprocess supervision for clip re-analysis."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from time import monotonic

from worker.runtime import clip_analysis_process
from worker.runtime.clip_analysis_admission import (
    ClipAnalysisSupervisorError,
    artifact_identity,
    build_job,
    log_eviction,
    prepare_job,
    validate_runtime,
)
from worker.runtime.clip_analysis_execution import execute_job
from worker.runtime.clip_analysis_lifecycle import retry_teardown, settle
from worker.runtime.clip_analysis_queue import Admission, ClipAnalysisQueue
from worker.runtime.clip_analysis_supervisor_status import ClipAnalysisStatus, StatusLedger


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
        probe: Callable[
            [Path], clip_analysis_process.ClipMediaFacts
        ] = clip_analysis_process.probe_media_facts,
    ) -> None:
        if deadline_s <= 0:
            raise ValueError("deadline_s must be positive")
        cpu_index = validate_runtime(cpu_index)
        self._python, self._pose_model, self._bed_model = (
            python_executable,
            pose_model_path,
            bed_model_path,
        )
        self._pose_model_sha = clip_analysis_process.model_digest(pose_model_path)
        self._bed_model_sha = clip_analysis_process.model_digest(bed_model_path)
        self._profile, self._profile_sha = profile, clip_analysis_process.profile_digest(profile)
        self._cpu_index, self._deadline_s, self._launch, self._probe = (
            cpu_index,
            deadline_s,
            launch,
            probe,
        )
        self._condition = threading.Condition()
        self._queue = ClipAnalysisQueue()
        self._active: clip_analysis_process.ClipAnalysisJob | None = None
        self._preparing: clip_analysis_process.ClipAnalysisJob | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._cancelled: set[int] = set()
        self._generation = 0
        self._stopping = False
        self._statuses = StatusLedger()
        self._set_status = self._statuses.record
        self._thread = threading.Thread(target=self._serve, name="clip-analysis", daemon=True)
        self._thread.start()

    def status(self, clip_id: str) -> ClipAnalysisStatus:
        with self._condition:
            return self._statuses.get(clip_id)

    def notify(
        self, clip_id: str, clip_path: Path, clip_sha256: str, *, size_bytes: int, duration_ms: int
    ) -> None:
        """Publication hook: only bounded, in-memory admission occurs here."""
        self._admit(clip_id, clip_path, clip_sha256, size_bytes, duration_ms, 0, 0, False)

    def trigger(self, clip_id: str, clip_path: Path, clip_sha256: str, **facts: int) -> Admission:
        """Dashboard trigger: manual jobs go to the head of the queue."""
        return self.enqueue(clip_id, clip_path, clip_sha256, front=True, **facts)

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
        return self._admit(
            clip_id, clip_path, clip_sha256, size_bytes, duration_ms, width, height, front
        )

    def wait_for_capacity(self, timeout: float) -> bool:
        with self._condition:
            return (
                self._condition.wait_for(
                    lambda: self._stopping or self._queue.has_capacity(), timeout
                )
                and not self._stopping
            )

    def _admit(
        self,
        clip_id: str,
        clip_path: Path,
        clip_sha256: str,
        size_bytes: int,
        duration_ms: int,
        width: int,
        height: int,
        front: bool,
    ) -> Admission:
        with self._condition:
            if self._stopping:
                return Admission.STOPPED
            self._generation += 1
            generation = self._generation
        job = build_job(
            clip_id,
            clip_path,
            clip_sha256,
            self._pose_model,
            self._bed_model,
            self._profile,
            self._profile_sha,
            size_bytes,
            duration_ms,
            width,
            height,
            front,
            self._pose_model_sha,
            self._bed_model_sha,
            generation,
        )
        with self._condition:
            if self._stopping:
                return Admission.STOPPED
            if self._active is not None and self._active.clip_id == clip_id:
                return Admission.ALREADY_RUNNING
            if self._preparing is not None and self._preparing.clip_id == clip_id:
                return Admission.ALREADY_QUEUED
            admitted = self._queue.push(job, front=front)
            if admitted.evicted is not None:
                self._set_status(
                    admitted.evicted.clip_id, ClipAnalysisStatus("failed", "evicted_for_manual")
                )
                log_eviction(admitted.evicted.clip_id)
            if admitted.kind in (Admission.QUEUED, Admission.ALREADY_QUEUED):
                self._set_status(clip_id, ClipAnalysisStatus("queued", "pre-probe"))
                self._condition.notify()
            return admitted.kind

    def cancel(self, clip_id: str) -> bool:
        with self._condition:
            if self._queue.remove(clip_id):
                self._set_status(clip_id, ClipAnalysisStatus("failed", "cancelled"))
                self._condition.notify_all()
                return True
            if self._preparing is not None and self._preparing.clip_id == clip_id:
                self._cancelled.add(self._preparing.generation)
                self._set_status(clip_id, ClipAnalysisStatus("failed", "cancelled"))
                self._condition.notify_all()
                return True
            if self._active is None or self._active.clip_id != clip_id:
                return False
            self._cancelled.add(self._active.generation)
            process = self._process
        if process is not None:
            self._terminate_group(process)
        return True

    def shutdown(self) -> None:
        with self._condition:
            self._stopping = True
            if self._active is not None:
                self._cancelled.add(self._active.generation)
            self._condition.notify_all()
        # Prove the child group empty before and after the serve thread exits:
        # a poisoned slot keeps its process handle until a teardown proof succeeds.
        self._release_after_teardown_proof()
        self._thread.join()
        self._release_after_teardown_proof()

    def _release_after_teardown_proof(self) -> None:
        with self._condition:
            process = self._process
        if process is not None and retry_teardown(process, self._terminate_group):
            with self._condition:
                self._active, self._process = None, None

    def _serve(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._stopping)
                if self._stopping:
                    self._cancel_pending()
                    return
                pending = self._queue.take()
                assert pending is not None
                self._preparing = pending
                self._condition.notify_all()
            job, status = prepare_job(
                pending,
                profile=self._profile,
                pose_model_path=self._pose_model,
                bed_model_path=self._bed_model,
                profile_sha256=self._profile_sha,
                probe=self._probe,
            )
            with self._condition:
                self._preparing = None
                if self._stopping or pending.generation in self._cancelled:
                    self._cancelled.discard(pending.generation)
                    self._set_status(pending.clip_id, ClipAnalysisStatus("failed", "cancelled"))
                    continue
                if job is None:
                    self._set_status(pending.clip_id, status)
                    continue
                self._active, self._process = job, None
                self._set_status(job.clip_id, ClipAnalysisStatus("running"))
            status = execute_job(
                job,
                self._python,
                self._cpu_index,
                self._deadline_s,
                self._launch,
                self._set_process,
                lambda generation=job.generation: self._is_cancelled(generation),
                self._terminate_group,
                monotonic,
            )
            self._settle(job, status)

    def _settle(
        self, job: clip_analysis_process.ClipAnalysisJob, status: ClipAnalysisStatus
    ) -> None:
        with self._condition:
            process = self._process
        status, teardown_failed = settle(
            job,
            status,
            process,
            identity=lambda queued: artifact_identity(queued, self._profile_sha),
            terminate=self._terminate_group,
        )
        with self._condition:
            self._set_status(job.clip_id, status)
            if teardown_failed:
                self._stopping = True
                self._set_status(job.clip_id, ClipAnalysisStatus("failed", "teardown_unproved"))
            else:
                self._active, self._process = None, None
            self._cancelled.discard(job.generation)
            self._condition.notify_all()

    def _set_process(self, process: subprocess.Popen[bytes]) -> bool:
        with self._condition:
            self._process = process
            return self._stopping or (
                self._active is not None and self._active.generation in self._cancelled
            )

    def _is_cancelled(self, generation: int) -> bool:
        with self._condition:
            return self._stopping or generation in self._cancelled

    def _cancel_pending(self) -> None:
        while (job := self._queue.take()) is not None:
            self._set_status(job.clip_id, ClipAnalysisStatus("failed", "cancelled"))

    def _terminate_group(self, process: subprocess.Popen[bytes]) -> None:
        clip_analysis_process.terminate_group(process, failure=ClipAnalysisSupervisorError)
