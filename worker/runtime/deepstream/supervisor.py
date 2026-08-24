"""Python PID-1 ownership and continuous containment of one dark native child."""

from __future__ import annotations

import subprocess
import threading
from typing import Final

from worker.native.deepstream.control import DeepStreamControlClient
from worker.native.deepstream.metadata import LatestMetadataSlot, MetadataReceiver
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
from worker.runtime.deepstream.source_control import DarkSourceController
from worker.runtime.deepstream.startup import connect_session
from worker.runtime.deepstream.transport import spawn_child
from worker.runtime.lease import GpuLease

FATAL_CHILD_EXIT_CODE: Final = 4


class DeepStreamChildSupervisor:
    """Own one inherited-IPC child and persist fatal exit without caller participation."""

    def __init__(self, config: ChildConfig) -> None:
        self._config: ChildConfig = config
        self._process: subprocess.Popen[bytes] | None = None
        self._metadata: LatestMetadataSlot = LatestMetadataSlot()
        self._receiver: MetadataReceiver | None = None
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
        failures = NativeFailureCoordinator(
            self._config,
            lambda: self._sources,
            lambda: self._process,
            self._fatal_received,
            self._set_fatal_category,
        )
        self._failure_receiver = NativeFailureReceiver(
            transport.failures,
            self._config.worker_boot_id,
            self._config.child_instance_id,
            failures.source_failure,
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

    def stop(self) -> None:
        self._stopping.set()
        self._sources = None
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
    "ChildConfig",
    "ChildFatalError",
    "ChildStartupError",
    "DarkChildConfigError",
    "DeepStreamChildSupervisor",
    "FATAL_CHILD_EXIT_CODE",
    "configured_dark_supervisors",
    "PrivatePathError",
    "validate_private_directory",
]
