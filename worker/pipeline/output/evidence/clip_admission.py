from __future__ import annotations

import queue
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import final

from worker.pipeline.output.evidence.clip_identity import (
    ClipIdAllocator,
    ClipIdCollisionError,
    ClipReservation,
)
from worker.pipeline.output.evidence.clip_recorder_models import (
    ClipRecorderConfig,
    ClipRecorderStats,
    EventMessage,
    FlushMessage,
    FrameMessage,
    RecorderMessage,
)
from worker.pipeline.output.evidence.durability import fsync_directory
from worker.pipeline.output.evidence.evidence_metadata import (
    runtime_manifest_sha256_from_audit,
)
from worker.pipeline.output.evidence.evidence_outbox_types import ClipId
from worker.types import BusinessEvent, EvidenceTrigger, FramePacket


# allow: MUTABLE_OK - active union bounds expand on overlap.
@dataclass(slots=True)
class _ReservationWindow:
    """Mutable admission-owned bounds for one in-flight incident union."""

    reservation: ClipReservation
    worker_boot_id: str
    stream_epoch: int
    source_generation: int
    start_time_sec: float
    end_time_sec: float


@final
class ClipAdmission:
    def __init__(
        self,
        config: ClipRecorderConfig,
        stats: ClipRecorderStats,
        message_queue: queue.Queue[RecorderMessage],
    ) -> None:
        self._stats = stats
        self._queue = message_queue
        self._allocator = ClipIdAllocator(config.store_dir)
        self._pre_event_seconds = config.pre_event_seconds
        self._post_event_seconds = config.post_event_seconds
        self._lock = threading.RLock()
        self._reservations_by_camera: dict[str, list[_ReservationWindow]] = {}
        self._accepting = True

    def set_accepting(self, accepting: bool) -> None:
        with self._lock:
            self._accepting = accepting

    def accept_frame(self, packet: FramePacket) -> bool:
        with self._lock:
            if not self._accepting:
                return False
            queued_packet = packet.retain()
            try:
                self._queue.put_nowait(FrameMessage(queued_packet))
            except queue.Full:
                queued_packet.release()
                self._stats.dropped_frames += 1
                return False
        return True

    def accept_event(
        self,
        trigger_packet: EvidenceTrigger,
        event: BusinessEvent,
        *,
        allow_new_clip: bool,
        detected_at: datetime,
    ) -> str | None:
        camera_id = trigger_packet.camera_id
        if event.camera_id != camera_id:
            raise ValueError("event camera does not match trigger packet")
        _ = runtime_manifest_sha256_from_audit(event.audit)
        event_ref = str(event.identity)
        with self._lock:
            if not self._accepting:
                return None
            event_time = trigger_packet.trigger_time_sec
            start_time = event_time - self._pre_event_seconds
            end_time = event_time + self._post_event_seconds
            windows = self._reservations_by_camera.setdefault(camera_id, [])
            active = next(
                (
                    candidate
                    for candidate in windows
                    if candidate.worker_boot_id == trigger_packet.worker_boot_id
                    and candidate.stream_epoch == trigger_packet.stream_epoch
                    and candidate.source_generation == trigger_packet.source_generation
                    and start_time <= candidate.end_time_sec
                    and end_time >= candidate.start_time_sec
                ),
                None,
            )
            created = active is None
            if active is None:
                if self._stats.recording_suspended or not allow_new_clip:
                    if not allow_new_clip:
                        self._stats.attach_missed_events += 1
                    return None
                try:
                    reservation = self._allocator.reserve(camera_id)
                except ClipIdCollisionError:
                    self._stats.clip_id_collisions = self._allocator.collision_count
                    return None
                active = _ReservationWindow(
                    reservation,
                    trigger_packet.worker_boot_id,
                    trigger_packet.stream_epoch,
                    trigger_packet.source_generation,
                    start_time,
                    end_time,
                )
                windows.append(active)
                self._stats.clip_id_collisions = self._allocator.collision_count
            else:
                active.start_time_sec = min(active.start_time_sec, start_time)
                active.end_time_sec = max(active.end_time_sec, end_time)
            reservation = active.reservation
            queued_trigger = trigger_packet.retain()
            try:
                self._queue.put_nowait(
                    EventMessage(
                        reservation,
                        event_ref,
                        event.event_type,
                        event,
                        queued_trigger,
                        allow_new_clip,
                        detected_at,
                    )
                )
            except queue.Full:
                queued_trigger.release()
                self._stats.dropped_events += 1
                if created:
                    self.release(camera_id, reservation.clip_id)
                    self._cancel(reservation)
                return None
            return str(reservation.clip_id)

    def put_control(self, message: FlushMessage) -> bool:
        with self._lock:
            if not self._accepting:
                return False
            try:
                self._queue.put_nowait(message)
            except queue.Full:
                return False
        return True

    def close(self, camera_id: str, clip_id: ClipId) -> None:
        with self._lock:
            windows = self._reservations_by_camera.get(camera_id, [])
            remaining = [item for item in windows if item.reservation.clip_id != clip_id]
            if len(remaining) == len(windows):
                return
            if remaining:
                self._reservations_by_camera[camera_id] = remaining
            else:
                _ = self._reservations_by_camera.pop(camera_id, None)

    def release(self, camera_id: str, clip_id: ClipId) -> None:
        self.close(camera_id, clip_id)

    def cancel(self, reservation: ClipReservation) -> None:
        with self._lock:
            windows = self._reservations_by_camera.get(reservation.camera_id, [])
            remaining = [item for item in windows if item.reservation != reservation]
            if remaining:
                self._reservations_by_camera[reservation.camera_id] = remaining
            else:
                _ = self._reservations_by_camera.pop(reservation.camera_id, None)
            self._cancel(reservation)

    def _cancel(self, reservation: ClipReservation) -> None:
        if not reservation.staging_dir.exists():
            return
        try:
            shutil.rmtree(reservation.staging_dir)
            fsync_directory(reservation.staging_dir.parent)
        except OSError:
            self._stats.failed_writes += 1


__all__ = ["ClipAdmission"]
