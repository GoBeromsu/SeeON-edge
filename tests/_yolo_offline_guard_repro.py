"""Standalone subprocess helper for the issue #111 regression tests in
``tests/test_worker_yolo_adapters.py``.

Not a pytest test module (no ``test_`` prefix, not collected). It exists to
run in a *fresh* interpreter, because ultralytics' import-time connectivity
self-check (``ultralytics.utils.is_online()``) runs exactly once, at the
first ``import ultralytics`` anywhere in a process -- exercising it inside
the shared pytest process would give a false pass/fail depending on
whichever test happened to import ultralytics first. A subprocess gives
each invocation a clean, deterministic first import.

Simulates the exact machine condition from issue #111: DNS to
``one.one.one.one``/``dns.google`` (the two hosts ``is_online()`` probes) is
blackholed -- packets silently dropped, never refused -- rather than merely
unreachable. ``socket.getaddrinfo`` has no timeout for this, so a real
blackhole blocks forever; this script simulates that with a long sleep,
relying on the calling test's ``subprocess.run(..., timeout=...)`` to bound
the whole thing.

Two modes:
  * default: import ``worker.adapters.model.yolo_api`` the normal way. Its
    module-level ``YOLO_OFFLINE`` guard (issue #111 fix) must prevent the
    blackhole from ever reaching ``socket.getaddrinfo`` at all, so this
    prints ``SUBPROCESS_COMPLETED`` and exits quickly.
  * ``--skip-guard``: bare ``import ultralytics`` with no guard in place --
    a negative control proving the blackhole condition above is a real
    trigger for the hang (not a tautological test), by actually hanging.
"""

from __future__ import annotations

import argparse
import os
import socket
import time

_BLACKHOLED_HOSTS = ("one.one.one.one", "dns.google")
_real_getaddrinfo = socket.getaddrinfo


def _blackhole_getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
    if host in _BLACKHOLED_HOSTS:
        time.sleep(3600)  # the outer subprocess timeout kills us long before this returns
    return _real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-guard", action="store_true")
    args = parser.parse_args()

    socket.getaddrinfo = _blackhole_getaddrinfo  # type: ignore[assignment]

    if args.skip_guard:
        os.environ.pop("YOLO_OFFLINE", None)
        import ultralytics  # noqa: F401 -- first-ever import in this process
    else:
        import worker.adapters.model.yolo_api  # noqa: F401 -- sets YOLO_OFFLINE at import time

    print("SUBPROCESS_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
