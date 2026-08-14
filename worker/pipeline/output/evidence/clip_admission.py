from __future__ import annotations

import queue
import shutil
import threading

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
from worker.types import BusinessEvent, FramePacket


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
        self._lock = threading.RLock()
        self._reservations_by_camera: dict[str, ClipReservation] = {}
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
        trigger_packet: FramePacket,
        event: BusinessEvent,
        *,
        allow_new_clip: bool,
    ) -> str | None:
        camera_id = trigger_packet.camera_id
        if event.camera_id != camera_id:
            raise ValueError("event camera does not match trigger packet")
        _ = runtime_manifest_sha256_from_audit(event.audit)
        event_ref = str(event.identity)
        with self._lock:
            if not self._accepting:
                return None
            reservation = self._reservations_by_camera.get(camera_id)
            created = reservation is None
            if reservation is None:
                if self._stats.recording_suspended or not allow_new_clip:
                    if not allow_new_clip:
                        self._stats.attach_missed_events += 1
                    return None
                try:
                    reservation = self._allocator.reserve(camera_id)
                except ClipIdCollisionError:
                    self._stats.clip_id_collisions = self._allocator.collision_count
                    return None
                self._stats.clip_id_collisions = self._allocator.collision_count
                self._reservations_by_camera[camera_id] = reservation
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
            reservation = self._reservations_by_camera.get(camera_id)
            if reservation is None or reservation.clip_id != clip_id:
                return
            self._reservations_by_camera.pop(camera_id, None)

    def release(self, camera_id: str, clip_id: ClipId) -> None:
        self.close(camera_id, clip_id)

    def cancel(self, reservation: ClipReservation) -> None:
        with self._lock:
            registered = self._reservations_by_camera.get(reservation.camera_id)
            if registered == reservation:
                self._reservations_by_camera.pop(reservation.camera_id, None)
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
