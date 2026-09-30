"""Liveness contract of the observability load harness's worker thread.

The measurement is only valid while the in-process worker is alive, so the
harness must fail when the worker requests a hard exit, ends with an error or
outlives the stop join. These tests drive the harness's own stop path with
real threads and events; the real-stack smoke covers the WorkerRuntime wiring.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest
from observability_load_harness import (
    WorkerTerminated,
    _HardExitRequested,
    _stop_worker,
    _WorkerObserver,
)

# An exception escaping a worker thread would hide behind a warning; fail instead.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")

_JOIN_SEC = 5.0


def _worker_thread(observer: _WorkerObserver, target: Callable[[], object]) -> threading.Thread:
    thread = threading.Thread(target=observer.run, args=(target,), daemon=True)
    thread.start()
    return thread


def test_hard_exit_ends_the_worker_thread_and_fails_the_stop() -> None:
    observer = _WorkerObserver()
    stop_requested = threading.Event()
    ran_after_hard_exit = threading.Event()

    def worker() -> None:
        observer(4)
        ran_after_hard_exit.set()

    thread = _worker_thread(observer, worker)

    with pytest.raises(WorkerTerminated):
        _stop_worker(thread, observer, stop_requested, join_timeout_sec=_JOIN_SEC)
    assert not ran_after_hard_exit.is_set()


def test_hard_exit_on_another_thread_fails_an_otherwise_clean_stop() -> None:
    observer = _WorkerObserver()
    stop_requested = threading.Event()
    supervisor_ended = threading.Event()

    def supervisor() -> None:
        try:
            observer(4)
        except _HardExitRequested:
            supervisor_ended.set()

    thread = _worker_thread(observer, stop_requested.wait)
    side = threading.Thread(target=supervisor, daemon=True)
    side.start()
    side.join(timeout=_JOIN_SEC)
    assert supervisor_ended.is_set()

    with pytest.raises(WorkerTerminated):
        _stop_worker(thread, observer, stop_requested, join_timeout_sec=_JOIN_SEC)


def test_stop_request_ends_the_worker_thread_within_the_join() -> None:
    observer = _WorkerObserver()
    stop_requested = threading.Event()
    thread = _worker_thread(observer, stop_requested.wait)
    try:
        outcome = _stop_worker(thread, observer, stop_requested, join_timeout_sec=_JOIN_SEC)
    finally:
        stop_requested.set()
        thread.join(timeout=_JOIN_SEC)

    assert not thread.is_alive()
    assert outcome["thread_alive_after_join"] is False
    assert outcome["hard_exit_codes"] == []
    assert outcome["thread_error"] is None


def test_worker_thread_outliving_the_join_fails_the_stop() -> None:
    observer = _WorkerObserver()
    stop_requested = threading.Event()
    release = threading.Event()
    thread = _worker_thread(observer, release.wait)
    try:
        with pytest.raises(WorkerTerminated):
            _stop_worker(thread, observer, stop_requested, join_timeout_sec=0.05)
    finally:
        release.set()
        thread.join(timeout=_JOIN_SEC)


def test_worker_thread_error_fails_the_stop() -> None:
    observer = _WorkerObserver()
    stop_requested = threading.Event()

    def worker() -> None:
        raise SystemExit(1)

    thread = _worker_thread(observer, worker)

    with pytest.raises(WorkerTerminated):
        _stop_worker(thread, observer, stop_requested, join_timeout_sec=_JOIN_SEC)
