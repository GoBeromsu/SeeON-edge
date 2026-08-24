"""Bounded inherited-pipe readiness handshake."""

from __future__ import annotations

import os
import select
import subprocess


def wait_for_ready(
    ready_fd: int,
    process: subprocess.Popen[bytes],
    timeout_sec: float,
) -> bool:
    try:
        readable, _, _ = select.select([ready_fd], [], [], timeout_sec)
        if not readable or process.poll() is not None:
            return False
        return os.read(ready_fd, 1) == b"R"
    finally:
        os.close(ready_fd)


__all__ = ["wait_for_ready"]
