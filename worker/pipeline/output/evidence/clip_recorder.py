from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import assert_never, final

from contracts.frame import Frame
from worker.pipeline.output.evidence.clip_actor import (
    ClipActor,
    ClipActorDependencies,
)
from worker.pipeline.output.evidence.clip_admission import ClipAdmission
from worker.pipeline.output.evidence.clip_maintenance import (
    ClipMaintenance,
    default_disk_usage,
    finalized_clips,
)
from worker.pipeline.output.evidence.clip_recorder_models import (
    ActiveClip,
    ClipRecorderConfig,
    ClipRecorderStats,
    EventMessage,
    FlushMessage,
    FrameMessage,
    RecorderMessage,
)
from worker.pipeline.output.evidence.clip_recorder_services import (
    ClipRecorderServices,
)
from worker.pipeline.output.evidence.clip_recorder_services import (
    default_services as _default_services,
)
from worker.pipeline.output.evidence.clip_recorder_services import (
    resolve_encoder as _resolve_encoder,
)
from worker.pipeline.output.evidence.clip_store_lock import ClipStoreLock
from worker.pipeline.output.evidence.evidence_outbox_types import ClipId
from worker.pipeline.output.evidence.evidence_retention import DiskUsage


@final
class ClipRecorder:
    def __init__(
        self,
        config: ClipRecorderConfig | None = None,
        services: ClipRecorderServices | None = None,
        *,
        disk_usage_provider: Callable[[Path], DiskUsage] | None = None,
        is_clip_held: Callable[[str], bool] | None = None,
        startup_hook: Callable[[], None] | None = None,
        on_clip_finalized: Callable[[ClipId], None] | None = None,
    ) -> None:
        self.config = ClipRecorderConfig() if config is None else config
        self.stats = ClipRecorderStats()
        self._services = services
        self._queue: queue.Queue[RecorderMessage] = queue.Queue(
            maxsize=self.config.max_queue_size
        )
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._store_lock: ClipStoreLock | None = None
        self._actor: ClipActor | None = None
        self._fps_by_camera: dict[str, float] = {}
        self._admission = ClipAdmission(self.config, self.stats, self._queue)
        self._startup_hook = startup_hook
        self._on_clip_finalized = on_clip_finalized
        self._maintenance = ClipMaintenance(
            self.config,
            self.stats,
            is_clip_held=(lambda _clip_id: True)
            if is_clip_held is None
            else is_clip_held,
            disk_usage_provider=default_disk_usage
            if disk_usage_provider is None
            else disk_usage_provider,
        )

    @classmethod
    def from_env(
        cls,
        *,
        is_clip_held: Callable[[str], bool] | None = None,
        startup_hook: Callable[[], None] | None = None,
        on_clip_finalized: Callable[[ClipId], None] | None = None,
    ) -> ClipRecorder:
        return cls(
            is_clip_held=is_clip_held,
            startup_hook=startup_hook,
            on_clip_finalized=on_clip_finalized,
        )

    def start(self) -> None:
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
                self._services = _default_services(self.config, _resolve_encoder())
            for camera_id, fps in self._fps_by_camera.items():
                self._services.coordinator.set_camera_fps(camera_id, fps)
            self.stats.encoder = self._services.encoder_name
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
            self._thread = threading.Thread(
                target=self._run,
                name="clip-recorder",
                daemon=True,
            )
            self._admission.set_accepting(True)
            self._thread.start()
            retained = True
        finally:
            if not retained:
                self._store_lock = None
                store_lock.close()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._admission.set_accepting(False)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._thread is None or not self._thread.is_alive():
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

    def set_camera_fps(self, camera_id: str, fps: float) -> None:
        self._fps_by_camera[camera_id] = fps
        if self._services is not None:
            self._services.coordinator.set_camera_fps(camera_id, fps)

    def on_frame(self, camera_id: str, frame: Frame) -> bool:
        return self._admission.accept_frame(camera_id, frame)

    def on_event(
        self,
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        return self._admission.accept_event(
            camera_id,
            event_ref,
            event_type,
            allow_new_clip=allow_new_clip,
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
        if actor is None:
            return
        try:
            while True:
                try:
                    message = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._stop_event.is_set() and self._queue.empty():
                        break
                    actor.expire()
                    continue
                try:
                    match message:
                        case FrameMessage():
                            self._handle_frame(message)
                        case EventMessage():
                            self._handle_event(message)
                        case FlushMessage(done=done):
                            try:
                                actor.flush()
                                self._rotate(force=True)
                            finally:
                                done.set()
                        case unreachable:
                            assert_never(unreachable)
                finally:
                    self._queue.task_done()
        finally:
            actor.shutdown()

    def _handle_frame(self, message: FrameMessage) -> None:
        if self._actor is not None:
            self._actor.handle_frame(message)

    def _handle_event(self, message: EventMessage) -> None:
        if self._actor is not None:
            self._actor.handle_event(message)

    def _sweep_stale_staging(self) -> None:
        self._maintenance.sweep_stale_staging()

    def _rotate(self, *, force: bool = False) -> None:
        self._maintenance.rotate(force=force)

_ActiveClip = ActiveClip
_EventMessage = EventMessage
_FrameMessage = FrameMessage
_finalized_clips = finalized_clips

__all__ = ["ClipRecorder", "ClipRecorderConfig", "ClipRecorderServices"]
