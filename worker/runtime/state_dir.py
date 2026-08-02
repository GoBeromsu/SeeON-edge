"""Resolve the worker's XDG-style state directory (issue #35).

``~/.local/state/<runtime>`` replaces the old ``/var/lib/ml-worker`` default
and its ``ML_WORKER_STATE_DIR`` env override: a root-owned ``/var/lib`` path
does not exist to write into on a macOS dev machine, crashing the very first
bootstrap stage (GPU lease acquisition) before any device probe runs. There is
deliberately no environment override here -- every caller that needs a
different directory (tests, alternate runtimes) passes one explicitly.
"""

from __future__ import annotations

from pathlib import Path


def resolve_state_dir(runtime: str = "ml-worker") -> Path:
    """Return the XDG state directory for *runtime* under the user's home."""
    return Path.home() / ".local" / "state" / runtime


__all__ = ["resolve_state_dir"]
