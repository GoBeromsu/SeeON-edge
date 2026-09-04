from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import final

from worker.pipeline.output.evidence.clip_actor import (
    ClipActor,
    ClipActorDependencies,
)
from worker.pipeline.output.evidence.clip_actor_loop import release_pending, run_actor_loop
from worker.pipeline.output.evidence.clip_admission import ClipAdmission
from worker.pipeline.output.evidence.clip_maintenance import (
    ClipMaintenance,
    default_disk_usage,
    finalized_clips,
)
from worker.pipeline.output.evidence.clip_recorder_models import (
    ClipRecorderConfig,
    ClipRecorderStats,
    EpochRollMessage,
    EventMessage,
    FlushMessage,
    RecorderMessage,
)
from worker.pipeline.output.evidence.clip_recorder_services import ClipRecorderServices
from worker.pipeline.output.evidence.clip_recorder_services import (
    default_services as _default_services,
)
from worker.pipeline.output.evidence.clip_store_lock import ClipStoreLock
from worker.pipeline.output.evidence.evidence_outbox_types import ClipId
from worker.pipeline.output.evidence.evidence_retention import DiskUsage, PurgeResult
from worker.pipeline.output.evidence.packet_repository import PacketRingRepository
from worker.pipeline.output.evidence.packet_ring import PacketRingLimits
from worker.types import BusinessEvent, EvidenceTrigger, FramePacket
from worker.types.source_packet import StreamEpoch


@final
class ClipRecorder:
    def __init__(
        self,
        config: ClipRecorderConfig | None = None,
        services: ClipRecorderServices | None = None,
        *,
        disk_usage_provider: Callable[[Path], DiskUsage] | None = None,
        is_clip_held: Callable[[str], bool] | None = None,
        begin_clip_purge: Callable[[str], bool] | None = None,
        complete_clip_purge: Callable[[str], None] | None = None,
        fail_clip_purge: Callable[[str, str], None] | None = None,
        operator_delete_preflight: Callable[[str], PurgeResult | None] | None = None,
        startup_hook: Callable[[], None] | None = None,
        on_clip_finalized: Callable[[ClipId], None] | None = None,
    ) -> None:
        self.config = ClipRecorderConfig() if config is None else config
        self.stats = ClipRecorderStats()
        self._services = services
        self._queue: queue.Queue[RecorderMessage] = queue.Queue(maxsize=self.config.max_queue_size)
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._store_lock: ClipStoreLock | None = None
        self._actor: ClipActor | None = None
        self._fps_by_camera: dict[str, float] = {}
        self._admission = ClipAdmission(self.config, self.stats, self._queue)
        self._startup_hook = startup_hook
        self._on_clip_finalized = on_clip_finalized
        self._epoch_subscribed = False
        self._maintenance = ClipMaintenance(
            self.config,
            self.stats,
            is_clip_held=(lambda _clip_id: True) if is_clip_held is None else is_clip_held,
            disk_usage_provider=default_disk_usage
            if disk_usage_provider is None
            else disk_usage_provider,
            begin_clip_purge=begin_clip_purge,
            complete_clip_purge=complete_clip_purge,
            fail_clip_purge=fail_clip_purge,
            operator_delete_preflight=operator_delete_preflight,
        )

    @classmethod
    def from_env(
        cls,
        *,
        is_clip_held: Callable[[str], bool] | None = None,
        begin_clip_purge: Callable[[str], bool] | None = None,
        complete_clip_purge: Callable[[str], None] | None = None,
        fail_clip_purge: Callable[[str, str], None] | None = None,
        operator_delete_preflight: Callable[[str], PurgeResult | None] | None = None,
        startup_hook: Callable[[], None] | None = None,
        on_clip_finalized: Callable[[ClipId], None] | None = None,
    ) -> ClipRecorder:
        return cls(
            is_clip_held=is_clip_held,
            begin_clip_purge=begin_clip_purge,
            complete_clip_purge=complete_clip_purge,
            fail_clip_purge=fail_clip_purge,
            operator_delete_preflight=operator_delete_preflight,
            startup_hook=startup_hook,
            on_clip_finalized=on_clip_finalized,
        )

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._admission.set_accepting(False)
            store_lock = ClipStoreLock.acquire(self.config.store_dir)
            retained = False
            try:
                if self._startup_hook is not None:
                    self._startup_hook()
                (self.config.store_dir / "segments").mkdir(parents=True, exist_ok=True)
                (self.config.store_dir / "clips" / ".staging").mkdir(
                    parents=True,
                    exist_ok=True,
                )
                self._sweep_stale_staging()
                self._rotate(force=True)
                if self._services is None:
                    repository = PacketRingRepository(
                        (),
                        per_camera_limits=PacketRingLimits(
                            self.config.packet_ring_max_packets,
                            self.config.packet_ring_max_bytes_per_camera,
                            self.config.pre_event_seconds
                            + self.config.post_event_seconds
                            + self.config.finalize_grace_seconds,
                        ),
                        global_max_bytes=self.config.packet_ring_global_max_bytes,
                    )
                    self._services = _default_services(self.config, repository)
                for camera_id, fps in self._fps_by_camera.items():
                    self._services.coordinator.set_camera_fps(camera_id, fps)
                self.stats.encoder = self._services.encoder_name
                if self._services.repository is not None and not self._epoch_subscribed:
                    self._services.repository.subscribe_epoch_roll(self._on_epoch_roll)
                    self._epoch_subscribed = True
                self._actor = ClipActor(
                    self.config,
                    self.stats,
                    ClipActorDependencies(
                        coordinator=self._services.coordinator,
                        publisher=self._services.publisher,
                        close=self._admission.close,
                        release=self._admission.release,
                        cancel=self._admission.cancel,
                        finalized=self._on_clip_finalized,
                        encoder_name=self._services.encoder_name,
                    ),
                )
                self._store_lock = store_lock
                self._stop_event.clear()
                thread = threading.Thread(
                    target=self._run,
                    name="clip-recorder",
                    daemon=True,
                )
                self._admission.set_accepting(True)
                thread.start()
                self._thread = thread
                retained = True
            finally:
                if not retained:
                    self._store_lock = None
                    store_lock.close()

    def stop(self, *, timeout: float = 5.0) -> None:
        with self._lifecycle_lock:
            self._admission.set_accepting(False)
            self._stop_event.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lifecycle_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._release_pending_messages()
                if self._store_lock is not None:
                    self._store_lock.close()
                    self._store_lock = None

    def flush(self, *, timeout: float = 5.0) -> bool:
        thread = self._thread
        if thread is None or not thread.is_alive():
            return False
        done = threading.Event()
        if not self._put_control(FlushMessage(done)):
            return False
        return done.wait(timeout)

    def rotate_once(self, *, timeout: float = 5.0) -> bool:
        return self.flush(timeout=timeout)

    def preflight_clip_deletion(self, clip_id: str) -> PurgeResult | None:
        """Check hold and containment without deleting worker-owned bytes."""
        return self._maintenance.preflight_clip(clip_id)

    def delete_clip(self, clip_id: str) -> PurgeResult:
        return self._maintenance.purge_clip(clip_id)

    def set_camera_fps(self, camera_id: str, fps: float) -> None:
        self._fps_by_camera[camera_id] = fps
        if self._services is not None:
            self._services.coordinator.set_camera_fps(camera_id, fps)

    def on_frame(self, packet: FramePacket) -> bool:
        return self._admission.accept_frame(packet)

    def on_event(
        self,
        trigger_packet: EvidenceTrigger,
        event: BusinessEvent,
        *,
        allow_new_clip: bool = True,
        detected_at: datetime,
    ) -> str | None:
        return self._admission.accept_event(
            trigger_packet,
            event,
            allow_new_clip=allow_new_clip,
            detected_at=detected_at,
        )

    @property
    def dropped_frame_count(self) -> int:
        return self.stats.dropped_frames

    @property
    def dropped_event_count(self) -> int:
        return self.stats.dropped_events

    @property
    def active_clips(self) -> int:
        return self.stats.active_clips

    def _put_control(self, message: FlushMessage) -> bool:
        return self._admission.put_control(message)

    def _run(self) -> None:
        actor = self._actor
        if actor is not None:
            run_actor_loop(actor, self._queue, self._stop_event, self._rotate)

    def _on_epoch_roll(self, previous: StreamEpoch, current: StreamEpoch) -> None:
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        done = threading.Event()
        try:
            self._queue.put(EpochRollMessage(previous, current, done), timeout=2.0)
        except queue.Full as error:
            raise RuntimeError("epoch roll could not seal active clip") from error
        if not done.wait(timeout=5.0):
            raise RuntimeError("epoch roll active clip seal timed out")

    def _handle_event(self, message: EventMessage) -> None:
        if self._actor is not None:
            self._actor.handle_event(message)

    def _release_pending_messages(self) -> None:
        release_pending(self._queue)

    def _sweep_stale_staging(self) -> None:
        self._maintenance.sweep_stale_staging()

    def _rotate(self, *, force: bool = False) -> None:
        self._maintenance.rotate(force=force)


_finalized_clips = finalized_clips

__all__ = ["ClipRecorder", "ClipRecorderConfig", "ClipRecorderServices"]
