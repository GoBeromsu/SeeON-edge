"""Thread-safe camera lifecycle status storage."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from time import time
from typing import final

from worker.pipeline.ingest.lifecycle import IngestEvent


class CameraStatus(StrEnum):
    """Operator-visible camera lifecycle state."""

    STARTING = "STARTING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


@dataclass(frozen=True, slots=True)
class CameraStatusRecord:
    """Latest lifecycle state for one camera."""

    camera_id: str
    facility_id: str
    status: CameraStatus
    updated_at: float
    error_category: str | None = None


@dataclass(frozen=True, slots=True)
class OpsEvent:
    """Bounded operator event without exception or credential content."""

    event_type: str
    camera_id: str
    facility_id: str
    category: str
    timestamp: float
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    """Immutable camera lifecycle snapshot."""

    cameras: tuple[CameraStatusRecord, ...]
    ops_events: tuple[OpsEvent, ...]


@final
class StatusStore:
    """Mutable camera state protected for callbacks from ingest threads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._statuses: dict[str, CameraStatusRecord] = {}
        self._ops_events: list[OpsEvent] = []

    def set_status(
        self,
        camera_id: str,
        facility_id: str,
        status: CameraStatus | str,
        *,
        error_category: str | None = None,
        timestamp: float | None = None,
    ) -> CameraStatusRecord:
        record = CameraStatusRecord(
            camera_id=camera_id,
            facility_id=facility_id,
            status=CameraStatus(status),
            updated_at=_timestamp(timestamp),
            error_category=error_category,
        )
        with self._lock:
            self._statuses[camera_id] = record
        return record

    def get_status(self, camera_id: str) -> CameraStatusRecord | None:
        with self._lock:
            return self._statuses.get(camera_id)

    def record_ops_event(
        self,
        event_type: str,
        camera_id: str,
        facility_id: str,
        category: str,
        *,
        timestamp: float | None = None,
        detail: str | None = None,
    ) -> OpsEvent:
        event = OpsEvent(
            event_type=event_type,
            camera_id=camera_id,
            facility_id=facility_id,
            category=category,
            timestamp=_timestamp(timestamp),
            detail=detail,
        )
        with self._lock:
            self._ops_events.append(event)
        return event

    def record_camera_reconnecting(
        self,
        camera_id: str,
        facility_id: str,
        category: str,
        *,
        detail: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        with self._lock:
            _ = self.set_status(
                camera_id,
                facility_id,
                CameraStatus.DEGRADED,
                error_category=category,
                timestamp=timestamp,
            )
            _ = self.record_ops_event(
                "camera.offline",
                camera_id,
                facility_id,
                category,
                timestamp=timestamp,
                detail=detail,
            )

    def record_camera_recovered(
        self,
        camera_id: str,
        facility_id: str,
        *,
        detail: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        with self._lock:
            _ = self.set_status(
                camera_id,
                facility_id,
                CameraStatus.READY,
                timestamp=timestamp,
            )
            _ = self.record_ops_event(
                "camera.recovered",
                camera_id,
                facility_id,
                "rtsp_recovered",
                timestamp=timestamp,
                detail=detail,
            )

    def ops_events(self) -> tuple[OpsEvent, ...]:
        with self._lock:
            return tuple(self._ops_events)

    def snapshot(self) -> StatusSnapshot:
        with self._lock:
            return StatusSnapshot(
                cameras=tuple(self._statuses[camera_id] for camera_id in sorted(self._statuses)),
                ops_events=tuple(self._ops_events),
            )


@final
class IngestStatusReporter:
    """Bridge ingest callbacks into the worker-local status store."""

    def __init__(
        self,
        store: StatusStore,
        camera_facilities: Mapping[str, str],
    ) -> None:
        self._store = store
        self._camera_facilities = dict(camera_facilities)

    def mark_starting(self, camera_id: str) -> None:
        _ = self._store.set_status(
            camera_id,
            self._camera_facilities[camera_id],
            CameraStatus.STARTING,
        )

    def mark_ready(self, camera_id: str) -> None:
        _ = self._store.set_status(
            camera_id,
            self._camera_facilities[camera_id],
            CameraStatus.READY,
        )

    def mark_degraded(self, camera_id: str, *, category: str) -> None:
        _ = self._store.set_status(
            camera_id,
            self._camera_facilities[camera_id],
            CameraStatus.DEGRADED,
            error_category=category,
        )

    def emit(self, event: IngestEvent) -> None:
        _ = self._store.record_ops_event(
            event.event_type,
            event.camera_id,
            self._camera_facilities[event.camera_id],
            event.category,
            detail=event.detail,
        )


def _timestamp(timestamp: float | None) -> float:
    return time() if timestamp is None else float(timestamp)


__all__ = [
    "CameraStatus",
    "CameraStatusRecord",
    "IngestStatusReporter",
    "OpsEvent",
    "StatusSnapshot",
    "StatusStore",
]
