"""Bounded FIFO ownership for pending clip-analysis jobs."""

from __future__ import annotations

from collections import deque
from enum import StrEnum

from worker.runtime.clip_analysis_process import ClipAnalysisJob


class Admission(StrEnum):
    QUEUED = "queued"
    ALREADY_QUEUED = "already_queued"
    ALREADY_RUNNING = "already_running"
    AVAILABLE = "available"
    QUEUE_FULL = "queue_full"
    STOPPED = "stopped"
    REJECTED = "rejected"


class ClipAnalysisQueue:
    """A bounded, de-duplicated queue whose head can accept manual work."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._jobs: deque[ClipAnalysisJob] = deque()
        self._clip_ids: set[str] = set()

    def add(self, job: ClipAnalysisJob, *, front: bool) -> Admission:
        if job.clip_id in self._clip_ids:
            if front:
                for queued in self._jobs:
                    if queued.clip_id == job.clip_id:
                        self._jobs.remove(queued)
                        self._jobs.appendleft(queued)
                        break
            return Admission.ALREADY_QUEUED
        if len(self._jobs) >= self._capacity and (not front or not self._evict_tail_automatic()):
            return Admission.QUEUE_FULL
        if front:
            self._jobs.appendleft(job)
        else:
            self._jobs.append(job)
        self._clip_ids.add(job.clip_id)
        return Admission.QUEUED

    def _evict_tail_automatic(self) -> bool:
        for job in reversed(self._jobs):
            if not job.front:
                self._jobs.remove(job)
                self._clip_ids.remove(job.clip_id)
                return True
        return False

    def take(self) -> ClipAnalysisJob | None:
        if not self._jobs:
            return None
        job = self._jobs.popleft()
        self._clip_ids.remove(job.clip_id)
        return job

    def remove(self, clip_id: str) -> bool:
        for job in self._jobs:
            if job.clip_id == clip_id:
                self._jobs.remove(job)
                self._clip_ids.remove(clip_id)
                return True
        return False

    def __bool__(self) -> bool:
        return bool(self._jobs)

    def __len__(self) -> int:
        return len(self._jobs)
