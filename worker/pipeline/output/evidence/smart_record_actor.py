"""Serialized Smart Record extension policy for one camera."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Literal
from uuid import uuid4

from worker.interfaces.media_plane import MediaPlane, RecordingInfo, RecordingRefused

ClipBoundary = Literal["none", "extension_bounded", "extension_raced"]


class SmartRecordState(StrEnum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    STOPPING = "STOPPING"
    FINALIZING = "FINALIZING"


@dataclass(frozen=True, slots=True)
class ClipContributor:
    event_ref: str
    detected_at: str


@dataclass(frozen=True, slots=True)
class ClipSealed:
    clip_id: str
    path: str
    duration_ms: int
    contributors: tuple[ClipContributor, ...]
    boundary: ClipBoundary


class DuplicateRecordingSealedError(RuntimeError):
    """The media plane delivered a completion callback more than once."""

    def __init__(self, session_id: int) -> None:
        super().__init__(f"recording session {session_id} was sealed more than once")
        self.session_id = session_id


SmartRecordSinkItem = ClipSealed | DuplicateRecordingSealedError
SmartRecordSink = Callable[[SmartRecordSinkItem], None]


@dataclass(slots=True)
class _PendingAlert:
    event_ref: str
    detected_at: str


@dataclass(slots=True)
class _Recording:
    session_id: int
    clip_id: str
    contributors: list[ClipContributor]
    stop_due: float
    hard_deadline: float
    boundary: ClipBoundary = "none"


class SmartRecordActor:
    """Own the delayed-stop policy that gives Smart Record its extension semantics."""

    def __init__(
        self,
        *,
        camera_id: str,
        media_plane: MediaPlane,
        clock: Callable[[], float],
        sink: SmartRecordSink,
        lookback_sec: int,
        duration_sec: int = 180,
        extension_sec: int = 30,
        clip_id_factory: Callable[[], str] | None = None,
        max_pending_alerts: int = 128,
    ) -> None:
        if lookback_sec < 0:
            raise ValueError("lookback_sec must not be negative")
        if duration_sec <= 0 or extension_sec <= 0:
            raise ValueError("recording durations must be positive")
        if max_pending_alerts <= 0:
            raise ValueError("max_pending_alerts must be positive")
        self._camera_id = camera_id
        self._media_plane = media_plane
        self._clock = clock
        self._sink = sink
        self._lookback_sec = lookback_sec
        self._duration_sec = duration_sec
        self._extension_sec = extension_sec
        self._clip_id_factory = clip_id_factory or (lambda: str(uuid4()))
        self._max_pending_alerts = max_pending_alerts
        self._lock = RLock()
        self._state = SmartRecordState.IDLE
        self._recording: _Recording | None = None
        self._pending: deque[_PendingAlert] = deque()
        self._sealed_sessions: set[int] = set()
        self._sequence = 0
        self._smart_record_extended_total = 0
        self._smart_record_extension_raced_total = 0
        self._smart_record_start_refused_total = 0

    @property
    def state(self) -> SmartRecordState:
        with self._lock:
            return self._state

    @property
    def smart_record_extended_total(self) -> int:
        with self._lock:
            return self._smart_record_extended_total

    @property
    def smart_record_extension_raced_total(self) -> int:
        with self._lock:
            return self._smart_record_extension_raced_total

    @property
    def smart_record_start_refused_total(self) -> int:
        with self._lock:
            return self._smart_record_start_refused_total

    def admit(self, event_ref: str, detected_at: str) -> None:
        """Accept an admitted alert without ever issuing an overlapping start."""
        if not event_ref:
            raise ValueError("event_ref must not be empty")
        if not detected_at:
            raise ValueError("detected_at must not be empty")
        with self._lock:
            self._next_sequence()
            if self._state is SmartRecordState.RECORDING:
                self._extend(event_ref, detected_at)
                return
            if self._state in (SmartRecordState.STOPPING, SmartRecordState.FINALIZING):
                recording = self._require_recording()
                recording.boundary = "extension_raced"
                self._smart_record_extension_raced_total += 1
            self._enqueue(event_ref, detected_at)
            if self._state is SmartRecordState.IDLE:
                self._start_pending()

    def tick(self) -> None:
        """Retry refused starts and issue the single early-stop command when due."""
        with self._lock:
            self._next_sequence()
            if self._state is SmartRecordState.IDLE and self._pending:
                self._start_pending()
                return
            if self._state is not SmartRecordState.RECORDING:
                return
            recording = self._require_recording()
            if self._clock() < recording.stop_due:
                return
            self._state = SmartRecordState.STOPPING
            self._media_plane.stop_recording(self._camera_id, recording.session_id)

    def on_sealed(self, info: RecordingInfo) -> None:
        """Handle a media-plane completion callback exactly once by session id."""
        with self._lock:
            self._next_sequence()
            if info.camera_id != self._camera_id:
                raise ValueError(
                    f"recording callback belongs to {info.camera_id}, not {self._camera_id}"
                )
            if info.session_id in self._sealed_sessions:
                self._sink(DuplicateRecordingSealedError(info.session_id))
                return
            recording = self._require_recording()
            if info.session_id != recording.session_id:
                raise ValueError(f"unexpected recording session {info.session_id}")
            if self._state not in (SmartRecordState.STOPPING, SmartRecordState.FINALIZING):
                raise ValueError("recording sealed before stop was requested")
            self._state = SmartRecordState.FINALIZING
            sealed = ClipSealed(
                clip_id=recording.clip_id,
                path=info.path,
                duration_ms=info.duration_ms,
                contributors=tuple(
                    sorted(recording.contributors, key=lambda contributor: contributor.detected_at)
                ),
                boundary=recording.boundary,
            )
            self._sealed_sessions.add(info.session_id)
            self._recording = None
            self._state = SmartRecordState.IDLE
            self._sink(sealed)
            if self._pending:
                self._start_pending()

    def _start_pending(self) -> None:
        if not self._pending:
            return
        now = self._clock()
        try:
            session_id = self._media_plane.start_recording(
                self._camera_id,
                lookback_sec=self._lookback_sec,
                duration_sec=self._duration_sec,
                on_sealed=self.on_sealed,
            )
        except RecordingRefused:
            self._smart_record_start_refused_total += 1
            return
        contributors = [
            ClipContributor(event_ref=pending.event_ref, detected_at=pending.detected_at)
            for pending in self._pending
        ]
        self._pending.clear()
        hard_deadline = now + self._duration_sec
        self._recording = _Recording(
            session_id=session_id,
            clip_id=self._clip_id_factory(),
            contributors=contributors,
            stop_due=min(now + self._extension_sec, hard_deadline),
            hard_deadline=hard_deadline,
        )
        self._state = SmartRecordState.RECORDING

    def _extend(self, event_ref: str, detected_at: str) -> None:
        recording = self._require_recording()
        recording.contributors.append(ClipContributor(event_ref=event_ref, detected_at=detected_at))
        desired_stop = self._clock() + self._extension_sec
        recording.stop_due = min(desired_stop, recording.hard_deadline)
        if desired_stop >= recording.hard_deadline:
            recording.boundary = "extension_bounded"
        self._smart_record_extended_total += 1

    def _enqueue(self, event_ref: str, detected_at: str) -> None:
        if len(self._pending) >= self._max_pending_alerts:
            raise RuntimeError("smart record pending alert capacity exhausted")
        self._pending.append(_PendingAlert(event_ref=event_ref, detected_at=detected_at))

    def _require_recording(self) -> _Recording:
        if self._recording is None:
            raise RuntimeError("smart record state has no active recording")
        return self._recording

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence


__all__ = [
    "ClipBoundary",
    "ClipContributor",
    "ClipSealed",
    "DuplicateRecordingSealedError",
    "SmartRecordActor",
    "SmartRecordSink",
    "SmartRecordState",
]
