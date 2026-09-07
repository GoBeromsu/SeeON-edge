"""Linux child-process safety hooks for clip re-analysis."""

from __future__ import annotations

import ctypes
import os
import signal
from collections.abc import Callable


class ClipAnalysisPdeathsigUnavailable(RuntimeError):
    """The host cannot guarantee that orphaned analysis children die."""


def require_pdeathsig() -> None:
    try:
        _ = ctypes.CDLL(None).prctl
    except (AttributeError, OSError) as exc:
        raise ClipAnalysisPdeathsigUnavailable("pdeathsig_unavailable") from exc


def child_setup(expected_parent: int, cpu_index: int | None) -> Callable[[], None]:
    def setup() -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        if prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != expected_parent:
            os._exit(3)
        if cpu_index is not None:
            os.sched_setaffinity(0, {cpu_index})

    return setup


__all__ = ["ClipAnalysisPdeathsigUnavailable", "child_setup", "require_pdeathsig"]
