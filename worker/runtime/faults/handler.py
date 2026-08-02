"""FaultHandler — stop all camera loops and exit on first fatal accelerator fault.

The handler is the composition root's fault boundary.  It:
1. Calls persist_first_fault() to write exactly one durable record.
2. Calls stop() on every registered IngestLoop so no frame N+1 is processed.
3. Calls _exit(4) — the injected exit function (os._exit in production).

todo 26 will add the watchdog on top of this; the watchdog triggers the same
exit boundary from an independent thread after a deadline.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Final, Protocol, final

from worker.adapters.model.errors import FatalAcceleratorError
from worker.runtime.faults.record import FirstFaultRecord, persist_first_fault
from worker.runtime.state_dir import resolve_state_dir

FATAL_ACCELERATOR_EXIT_CODE: Final = 4


class _Stoppable(Protocol):
    def stop(self) -> None: ...


@final
class FaultHandler:
    """Handles a FatalAcceleratorError by stopping all cameras and exiting.

    Parameters
    ----------
    profile:
        The active profile name (e.g. ``"cuda"``), for the fault record.
    hard_exit:
        Injected exit function.  Production code passes ``os._exit``; tests
        pass a recording callable so the process does not actually terminate.
    state_dir:
        Directory the first-fault record is written under.  Defaults to the
        resolved worker state directory.
    """

    def __init__(
        self,
        profile: str,
        *,
        hard_exit: Callable[[int], None] = os._exit,  # noqa: SLF001
        state_dir: Path | None = None,
    ) -> None:
        self._profile = profile
        self._hard_exit = hard_exit
        self._state_dir = state_dir if state_dir is not None else resolve_state_dir()
        self._loops: list[_Stoppable] = []
        self._handled = threading.Event()

    def register_loop(self, loop: _Stoppable) -> None:
        """Register an ingest loop that must be stopped on a fatal fault."""
        self._loops.append(loop)

    def handle(
        self,
        exc: FatalAcceleratorError,
        record: FirstFaultRecord,
    ) -> None:
        """Persist the fault record, stop all loops, and exit with code 4.

        Idempotent: if called more than once (race between camera threads), only
        the first call persists and exits; subsequent calls return immediately
        after the event is set.
        """
        if self._handled.is_set():
            return
        self._handled.set()

        # 1. Persist first-fault record (best-effort; never raises).
        persist_first_fault(record, state_dir=self._state_dir)

        # 2. Stop every camera loop so no further inference is scheduled.
        for loop in self._loops:
            try:
                loop.stop()
            except Exception:  # noqa: BLE001 S110
                pass

        # 3. Hard exit.  os._exit bypasses atexit handlers and Python shutdown
        #    so CUDA context cleanup cannot accidentally restart inference.
        self._hard_exit(FATAL_ACCELERATOR_EXIT_CODE)


__all__ = ["FATAL_ACCELERATOR_EXIT_CODE", "FaultHandler"]
