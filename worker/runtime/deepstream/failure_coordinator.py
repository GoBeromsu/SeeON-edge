"""Python-owned response to reliable native source and fatal failure events."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from typing import Final, final

from worker.native.deepstream.control import ChildControlError
from worker.runtime.deepstream.config import ChildConfig
from worker.runtime.deepstream.fault import persist_first_fault
from worker.runtime.deepstream.source_control import DarkSourceController

_FATAL_EXIT_CODE: Final = 4


@final
class NativeFailureCoordinator:
    def __init__(
        self,
        config: ChildConfig,
        sources: Callable[[], DarkSourceController | None],
        process: Callable[[], subprocess.Popen[bytes] | None],
        fatal_received: threading.Event,
        set_category: Callable[[str], None],
    ) -> None:
        self._config = config
        self._sources = sources
        self._process = process
        self._fatal_received = fatal_received
        self._set_category = set_category

    def source_failure(self, camera_id: str, category: str) -> None:
        sources = self._sources()
        if sources is None:
            self.fatal("source_failure_unbound")
            return
        try:
            _ = sources.rebuild(camera_id, category)
        except (ChildControlError, TimeoutError):
            self.fatal("source_rebuild_failed")

    def fatal(self, category: str) -> None:
        process = self._process()
        if category == "failure_ipc" and process is not None and process.poll() is not None:
            category = "child_exit"
        self._set_category(category)
        _ = persist_first_fault(
            self._config.first_fault_path,
            category=category,
            exit_code=_FATAL_EXIT_CODE,
            worker_boot_id=self._config.worker_boot_id,
            child_instance_id=self._config.child_instance_id,
        )
        self._fatal_received.set()
        if process is not None and process.poll() is None:
            process.kill()


__all__ = ["NativeFailureCoordinator"]
