"""Python PID-1 ownership and continuous containment of one dark native child."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Final, final

from worker.adapters.decode.native_au_receiver import NativeAuReceiver
from worker.adapters.decode.native_preview_receiver import NativePreviewReceiver
from worker.native.deepstream.control import ChildControlError, DeepStreamControlClient
from worker.native.deepstream.metadata import LatestMetadataSlot, MetadataReceiver
from worker.pipeline.output.evidence.packet_repository import PacketRingRepository
from worker.pipeline.output.evidence.packet_ring import PacketRingLimits
from worker.pipeline.output.evidence.scene_repository import SceneRingRepository
from worker.pipeline.output.live_view import LatestFrameStore
from worker.runtime.deepstream.canary_telemetry import NativeCanaryTelemetry
from worker.runtime.deepstream.child_monitor import ChildExitMonitor, monitor_metadata
from worker.runtime.deepstream.cleanup import ChildResources, stop_child_resources
from worker.runtime.deepstream.config import (
    ChildConfig,
    DarkChildConfigError,
    configured_dark_supervisors,
)
from worker.runtime.deepstream.errors import ChildFatalError, ChildStartupError
from worker.runtime.deepstream.failure_coordinator import NativeFailureCoordinator
from worker.runtime.deepstream.failure_receiver import NativeFailureReceiver
from worker.runtime.deepstream.fault import persist_child_fault
from worker.runtime.deepstream.path_security import (
    PrivatePathError,
    validate_private_directory,
)
from worker.runtime.deepstream.source_control import (
    DarkSourceController,
    SourceReadinessError,
)
from worker.runtime.deepstream.startup import connect_session
from worker.runtime.deepstream.transport import spawn_child
from worker.runtime.lease import GpuLease

FATAL_CHILD_EXIT_CODE: Final = 4
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SharedSupervisorResources:
    packet_repository: PacketRingRepository
    scene_repository: SceneRingRepository
    preview_frames: LatestFrameStore
    bootstrap_owns_lease: bool = True


@final
class DeepStreamChildSupervisor:
    """Own one inherited-IPC child and persist fatal exit without caller participation."""

    def __init__(
        self,
        config: ChildConfig,
        resources: SharedSupervisorResources | None = None,
    ) -> None:
        self._config: ChildConfig = config
        self._bootstrap_owns_lease = False if resources is None else resources.bootstrap_owns_lease
        self._process: subprocess.Popen[bytes] | None = None
        self._metadata: LatestMetadataSlot = LatestMetadataSlot()
        self._receiver: MetadataReceiver | None = None
        self._au_receiver: NativeAuReceiver | None = None
        self._preview_receiver: NativePreviewReceiver | None = None
        self._canary_au_telemetry: dict[str, tuple[int, NativeCanaryTelemetry | None]] = {}
        self._preview_frames = LatestFrameStore() if resources is None else resources.preview_frames
        self._packet_repository = (
            PacketRingRepository(
                (),
                per_camera_limits=PacketRingLimits(4_000, 64 * 1024 * 1024, 120.0),
                global_max_bytes=512 * 1024 * 1024,
            )
            if resources is None
            else resources.packet_repository
        )
        self._scene_repository = (
            SceneRingRepository() if resources is None else resources.scene_repository
        )
        self._failure_receiver: NativeFailureReceiver | None = None
        self._control: DeepStreamControlClient | None = None
        self._sources: DarkSourceController | None = None
        self._lease: GpuLease | None = None
        self._monitor: ChildExitMonitor | None = None
        self._metadata_monitor: threading.Thread | None = None
        self._stopping: threading.Event = threading.Event()
        self._fatal_received: threading.Event = threading.Event()
        self._fatal_category: str = "child_exit"

    @property
    def pid(self) -> int | None:
        process = self._process
        return None if process is None or process.poll() is not None else process.pid

    @property
    def metadata(self) -> LatestMetadataSlot:
        return self._metadata

    @property
    def packet_repository(self) -> PacketRingRepository:
        return self._packet_repository

    @property
    def scene_repository(self) -> SceneRingRepository:
        return self._scene_repository

    @property
    def au_receiver(self) -> NativeAuReceiver:
        if self._au_receiver is None:
            raise ChildStartupError("not_started", "gpu-0")
        return self._au_receiver

    @property
    def preview_frames(self) -> LatestFrameStore:
        return self._preview_frames

    @property
    def control(self) -> DeepStreamControlClient:
        if self._control is None:
            raise ChildStartupError("not_started", "gpu-0")
        return self._control

    @property
    def sources(self) -> DarkSourceController:
        if self._sources is None:
            raise ChildStartupError("not_started", "gpu-0")
        return self._sources

    def start(self) -> None:
        if self.pid is not None:
            raise ChildStartupError("already_started", "gpu-0")
        try:
            validate_private_directory(self._config.first_fault_path.parent)
            validate_private_directory(self._config.socket_dir)
        except PrivatePathError as error:
            raise ChildStartupError(error.code, error.detail) from error
        if not self._bootstrap_owns_lease:
            self._lease = GpuLease.acquire(self._config.lease_state_dir)
        try:
            transport = spawn_child(self._config)
        except OSError as error:
            self.stop()
            raise ChildStartupError("spawn_failed", "unavailable") from error
        self._process = transport.process
        self._monitor = ChildExitMonitor(
            transport.process,
            self._config,
            self._stopping,
            self._fatal_received,
            lambda: self._fatal_category,
        )
        self._monitor.start()
        try:
            session = connect_session(self._config, transport, self._metadata)
        except ChildStartupError:
            self.stop()
            raise
        self._control = session.control
        self._receiver = session.receiver
        self._sources = session.sources
        self._au_receiver = NativeAuReceiver(
            transport.access_units,
            str(self._config.worker_boot_id),
            self._packet_repository,
            self._handle_au_gap,
            self._fail_deadly,
            self._record_canary_au,
        )
        session.sources.set_retire_hook(self._au_receiver.retire_camera)
        self._au_receiver.start()
        self._preview_receiver = NativePreviewReceiver(
            transport.previews,
            self._preview_frames,
        )
        self._preview_receiver.start()
        failures = NativeFailureCoordinator(
            self._config,
            lambda: self._sources,
            lambda: self._process,
            self._fatal_received,
            self._set_fatal_category,
        )

        def source_failure(camera_id: str, category: str) -> None:
            LOGGER.warning(
                "native source failure: camera_id=%s category=%s",
                camera_id,
                category,
            )
            failures.source_failure(camera_id, category)

        self._failure_receiver = NativeFailureReceiver(
            transport.failures,
            self._config.worker_boot_id,
            self._config.child_instance_id,
            source_failure,
            failures.fatal,
        )
        self._failure_receiver.start()
        self._metadata_monitor = threading.Thread(
            target=monitor_metadata,
            args=(session.receiver, transport.process, self._stopping, self._mark_control_eof),
            name="deepstream-metadata-fatal",
            daemon=True,
        )
        self._metadata_monitor.start()

    def _set_fatal_category(self, category: str) -> None:
        self._fatal_category = category

    def _handle_au_gap(self, camera_id: str, category: str) -> None:
        sources = self._sources
        if sources is not None:
            try:
                _ = sources.rebuild(camera_id, category)
            except (ChildControlError, SourceReadinessError):
                self._fail_deadly("au_epoch_rebuild_failed")

    def _fail_deadly(self, category: str) -> None:
        process = self._process
        if category == "au_stream_closed" and process is not None:
            try:
                _ = process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
            else:
                category = "child_exit"
        self._set_fatal_category(category)
        self._fatal_received.set()
        _ = persist_child_fault(self._config, category)
        if process is not None and process.poll() is None:
            process.kill()

    def _mark_control_eof(self) -> None:
        if self._fatal_category == "child_exit":
            self._fatal_category = "control_eof"

    def fatal(self, category: str) -> None:
        self._fatal_category = category
        self._fatal_received.set()
        _ = persist_child_fault(self._config, category)
        self.control.fatal(category)

    def wait(self) -> int:
        monitor = self._monitor
        if monitor is None:
            raise ChildStartupError("not_started", "gpu-0")
        _ = monitor.exited.wait()
        code = monitor.exit_code if monitor.exit_code is not None else FATAL_CHILD_EXIT_CODE
        if code != 0:
            raise ChildFatalError(
                FATAL_CHILD_EXIT_CODE,
                self._fatal_category,
                self._config.first_fault_path,
            )
        return 0

    def _record_canary_au(
        self,
        camera_id: str,
        pts: int,
        sequence: int,
        generation: int,
    ) -> None:
        current = self._canary_au_telemetry.get(camera_id)
        if current is None or current[0] != generation:
            current = (generation, NativeCanaryTelemetry.from_environment(camera_id))
            self._canary_au_telemetry[camera_id] = current
        telemetry = current[1]
        if telemetry is not None:
            telemetry.record(pts, time.time_ns(), sequence)

    def stop(self) -> None:
        self._stopping.set()
        self._sources = None
        au_receiver, self._au_receiver = self._au_receiver, None
        if au_receiver is not None:
            au_receiver.close()
        preview_receiver, self._preview_receiver = self._preview_receiver, None
        if preview_receiver is not None:
            preview_receiver.close()
        resources = ChildResources(
            self._process,
            self._control,
            self._receiver,
            self._failure_receiver,
            self._monitor,
            self._lease,
            self._config.stop_timeout_sec,
        )
        self._process = None
        self._control = None
        self._receiver = None
        self._failure_receiver = None
        self._monitor = None
        self._lease = None
        stop_child_resources(resources)


__all__ = [
    "FATAL_CHILD_EXIT_CODE",
    "ChildConfig",
    "ChildFatalError",
    "ChildStartupError",
    "DarkChildConfigError",
    "DeepStreamChildSupervisor",
    "PrivatePathError",
    "SharedSupervisorResources",
    "configured_dark_supervisors",
    "validate_private_directory",
]
