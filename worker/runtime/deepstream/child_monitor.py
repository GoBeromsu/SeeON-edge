"""Continuous native-child exit monitoring and first-fault persistence."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from typing import Final, final

from worker.native.deepstream.metadata import MetadataReceiver
from worker.runtime.deepstream.config import ChildConfig
from worker.runtime.deepstream.fault import persist_first_fault

_FATAL_EXIT_CODE: Final = 4


@final
class ChildExitMonitor:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        config: ChildConfig,
        stopping: threading.Event,
        fatal_received: threading.Event,
        category: Callable[[], str],
    ) -> None:
        self._process = process
        self._config = config
        self._stopping = stopping
        self._fatal_received = fatal_received
        self._category = category
        self.exited = threading.Event()
        self.exit_code: int | None = None
        self._thread = threading.Thread(target=self._run, name="deepstream-child", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout_sec: float) -> None:
        self._thread.join(timeout=timeout_sec)

    def _run(self) -> None:
        code = self._process.wait()
        self.exit_code = code
        if code != 0 and not self._stopping.is_set():
            if code == _FATAL_EXIT_CODE:
                _ = self._fatal_received.wait(timeout=0.5)
            _ = persist_first_fault(
                self._config.first_fault_path,
                category=self._category(),
                exit_code=_FATAL_EXIT_CODE,
                worker_boot_id=self._config.worker_boot_id,
                child_instance_id=self._config.child_instance_id,
            )
        self.exited.set()


def monitor_metadata(
    receiver: MetadataReceiver,
    process: subprocess.Popen[bytes],
    stopping: threading.Event,
    mark_control_eof: Callable[[], None],
) -> None:
    _ = receiver.fatal_event.wait()
    if stopping.is_set():
        return
    mark_control_eof()
    if process.poll() is None:
        process.kill()


__all__ = ["ChildExitMonitor", "monitor_metadata"]
