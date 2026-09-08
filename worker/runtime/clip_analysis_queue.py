"""Bounded FIFO ownership for pending clip-analysis jobs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class Admitted:
    kind: Admission
    evicted: ClipAnalysisJob | None = None


class ClipAnalysisQueue:
    """A bounded, de-duplicated queue whose head can accept manual work."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._jobs: deque[ClipAnalysisJob] = deque()
        self._clip_ids: set[str] = set()

    def push(self, job: ClipAnalysisJob, *, front: bool) -> Admitted:
        if job.clip_id in self._clip_ids:
            if front:
                for queued in self._jobs:
                    if queued.clip_id == job.clip_id:
                        self._jobs.remove(queued)
                        self._jobs.appendleft(job)
                        break
            return Admitted(Admission.ALREADY_QUEUED)
        evicted = None
        if len(self._jobs) >= self._capacity:
            if not front:
                return Admitted(Admission.QUEUE_FULL)
            evicted = self._evict_tail_automatic()
            if evicted is None:
                return Admitted(Admission.QUEUE_FULL)
        if front:
            self._jobs.appendleft(job)
        else:
            self._jobs.append(job)
        self._clip_ids.add(job.clip_id)
        return Admitted(Admission.QUEUED, evicted)

    def add(self, job: ClipAnalysisJob, *, front: bool) -> Admission:
        """Return only the admission kind for legacy callers."""
        return self.push(job, front=front).kind

    def _evict_tail_automatic(self) -> ClipAnalysisJob | None:
        for job in reversed(self._jobs):
            if not job.front:
                self._jobs.remove(job)
                self._clip_ids.remove(job.clip_id)
                return job
        return None

    def take(self) -> ClipAnalysisJob | None:
        if not self._jobs:
            return None
        job = self._jobs.popleft()
        self._clip_ids.remove(job.clip_id)
        return job

    def has_capacity(self) -> bool:
        return len(self._jobs) < self._capacity

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
