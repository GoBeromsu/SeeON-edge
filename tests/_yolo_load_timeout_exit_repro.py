"""Standalone subprocess helper for the process-exit regression test in
``tests/test_worker_yolo_adapters.py`` (PR #112 review finding from rv95).

Reproduces the real worker's fatal-stage exit sequence
(``bootstrap_or_exit`` -> ``sys.exit()``, not ``os._exit()``) around a
``construct`` callable that never returns, specifically to catch a hang at
*interpreter shutdown* rather than at ``load_yolo_model``'s own controlled
timeout. The bug this guards against: ``ThreadPoolExecutor``'s worker thread
is non-daemon, and CPython's ``concurrent.futures.thread`` module registers
an ``atexit`` hook that joins every such thread with no timeout of its own --
so a permanently stuck ``construct`` call would still raise a bounded
``YoloLoadError`` as designed, but then silently re-hang the process at exit.
A daemon thread (the actual fix) is never joined by the interpreter at exit,
so this process must exit promptly instead.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from worker.adapters.model.yolo_api import YoloLoadError, YoloModel, load_yolo_model


def _stuck_construct(_path: Path) -> YoloModel:
    threading.Event().wait()  # never set -- simulates a permanent stall
    raise AssertionError("unreachable")


def main() -> None:
    artifact = Path(sys.argv[1])
    try:
        load_yolo_model(artifact, "pose", timeout_seconds=0.05, construct=_stuck_construct)
    except YoloLoadError:
        pass  # expected: the bounded timeout fired as designed
    else:
        raise AssertionError("expected load_yolo_model to raise YoloLoadError")
    sys.exit(0)  # mirrors bootstrap_or_exit's real exit path on a fatal stage error


if __name__ == "__main__":
    main()
