"""Container-only load-bearing PID-1 probe for the C5 dark runner."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import uuid
from pathlib import Path
from typing import override

from worker.runtime.deepstream import ChildConfig, DarkRunRequest, run_dark_child
from worker.runtime.deepstream.supervisor import DeepStreamChildSupervisor


def main() -> int:
    mode = sys.argv[1]
    state_dir = Path(sys.argv[2])
    if os.getpid() != 1:
        raise RuntimeError("dark runner probe must execute as PID 1")
    started = threading.Event()
    child_pid: list[int] = []

    class NotifyingSupervisor(DeepStreamChildSupervisor):
        @override
        def start(self) -> None:
            super().start()
            pid = self.pid
            if pid is None:
                raise RuntimeError("native child PID absent after startup")
            child_pid.append(pid)
            started.set()

    request = DarkRunRequest(
        ChildConfig(
            executable=Path("/usr/local/bin/seeon-deepstream-child"),
            worker_boot_id=uuid.uuid4(),
            socket_dir=state_dir / "ipc",
            first_fault_path=state_dir / "first-fault.json",
            lease_state_dir=state_dir,
            startup_timeout_sec=10.0,
            stop_timeout_sec=2.0,
        ),
        inject_fatal="cuda" if mode == "fatal" else None,
    )
    signal_thread: threading.Thread | None = None
    if mode == "graceful":
        def terminate_after_start() -> None:
            if not started.wait(timeout=10.0):
                raise RuntimeError("runner startup event absent")
            os.kill(1, signal.SIGTERM)

        signal_thread = threading.Thread(target=terminate_after_start)
        signal_thread.start()
    exit_code = run_dark_child(request, supervisor_factory=NotifyingSupervisor)
    if signal_thread is not None:
        signal_thread.join(timeout=10.0)
    expected = 4 if mode == "fatal" else 0
    if exit_code != expected:
        raise RuntimeError(f"runner exit mismatch: expected={expected} actual={exit_code}")
    fault_exists = request.child.first_fault_path.exists()
    if mode == "fatal" and not fault_exists:
        raise RuntimeError("fatal runner did not persist first fault")
    if mode == "graceful" and fault_exists:
        raise RuntimeError("graceful runner persisted a fault")
    print(
        json.dumps(
            {
                "child_pid": child_pid[0],
                "fault_exists": fault_exists,
                "mode": mode,
                "pid": os.getpid(),
                "runner_exit": exit_code,
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
