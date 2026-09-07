"""The ml-worker image ships no torch/ultralytics; the runtime import graph must not need them.

The CI boot smoke caught ``worker.runtime.worker`` importing ``yolo_pose`` (torch) through a
constant. This runs the same check hermetically: importing the composition root in a fresh
interpreter with torch and ultralytics blocked must succeed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_GUARD = """
import builtins, sys
real = builtins.__import__
BLOCKED = ("torch", "ultralytics")
def guard(name, *args, **kwargs):
    if name.split(".")[0] in BLOCKED:
        raise ImportError(f"blocked production import: {name}")
    return real(name, *args, **kwargs)
builtins.__import__ = guard
import worker.__main__  # noqa: F401 - the worker entrypoint
import worker.runtime.worker  # noqa: F401 - the composition root
print("ok")
"""


def test_worker_runtime_imports_without_torch_or_ultralytics() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _GUARD],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "ok"
