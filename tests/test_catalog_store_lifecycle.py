from __future__ import annotations

import os
import sqlite3
import subprocess
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import TracebackType
from typing import final

import pytest
from typing_extensions import override

from backend.app.features.clips.catalog import CatalogStore
from backend.app.features.clips.store import ClipStore

_PROC_FD = Path("/proc/self/fd")
_THREAD_TIMEOUT = 5.0


@final
class _TrackedConnection(sqlite3.Connection):
    close_calls = 0

    @override
    def close(self) -> None:
        self.close_calls += 1
        super().close()


@final
class _TraceGate:
    def __init__(self) -> None:
        self.enabled = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, statement: str) -> None:
        if (
            self.enabled
            and threading.current_thread().name == "admitted-operation"
            and statement == "PRAGMA integrity_check"
        ):
            self.entered.set()
            assert self.release.wait(timeout=_THREAD_TIMEOUT)


@final
class _ObservedLock:
    def __init__(self, timeline: list[str]) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition()
        self._attempted: dict[str, threading.Event] = {}
        self.timeline = timeline

    def attempted(self, thread_name: str) -> threading.Event:
        with self._condition:
            return self._attempted.setdefault(thread_name, threading.Event())

    def __enter__(self) -> _ObservedLock:
        name = threading.current_thread().name
        self.attempted(name).set()
        with self._condition:
            self._condition.notify_all()
        _ = self._lock.acquire()
        self.timeline.append(f"{name}-lock-acquired")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.timeline.append(f"{threading.current_thread().name}-lock-released")
        self._lock.release()


@pytest.fixture
def tracked_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[CatalogStore, list[_TrackedConnection], _TraceGate]]:
    connections: list[_TrackedConnection] = []
    connections_lock = threading.Lock()
    gate = _TraceGate()

    def connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(
            path,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
            factory=_TrackedConnection,
        )
        assert isinstance(connection, _TrackedConnection)
        connection.set_trace_callback(gate)
        for pragma in ("journal_mode = WAL", "synchronous = FULL", "busy_timeout = 5000"):
            _ = connection.execute(f"PRAGMA {pragma}")
        with connections_lock:
            connections.append(connection)
        return connection

    monkeypatch.setattr(CatalogStore, "_connect", staticmethod(connect))
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        yield store, connections, gate
    finally:
        store.close()


def _run_thread_wave(store: CatalogStore, width: int = 8) -> None:
    barrier = threading.Barrier(width + 1)
    results: list[str] = []

    def check() -> None:
        _ = barrier.wait(timeout=_THREAD_TIMEOUT)
        results.append(store.integrity_check())

    threads = [threading.Thread(target=check) for _ in range(width)]
    for thread in threads:
        thread.start()
    _ = barrier.wait(timeout=_THREAD_TIMEOUT)
    for thread in threads:
        thread.join(timeout=_THREAD_TIMEOUT)
    assert all(not thread.is_alive() for thread in threads)
    assert results == ["ok"] * width


def test_open_retains_one_connection_reused_by_thread_churn_and_closed_once(
    tracked_store: tuple[CatalogStore, list[_TrackedConnection], _TraceGate],
) -> None:
    store, connections, _gate = tracked_store

    assert store.integrity_check() == "ok"
    for _ in range(4):
        _run_thread_wave(store)
        assert sum(connection.close_calls == 0 for connection in connections) == 1
    assert len(connections) == 1

    store.close()
    store.close()
    assert connections[0].close_calls == 1
    with pytest.raises(RuntimeError, match="catalog store is closed"):
        _ = store.integrity_check()


def test_post_connect_migration_failure_closes_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[_TrackedConnection] = []

    def connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(path, factory=_TrackedConnection)
        assert isinstance(connection, _TrackedConnection)
        connections.append(connection)
        return connection

    def fail_migration(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("forced migration failure")

    monkeypatch.setattr(CatalogStore, "_connect", staticmethod(connect))
    monkeypatch.setattr(CatalogStore, "_migrate", staticmethod(fail_migration))

    with pytest.raises(sqlite3.OperationalError, match="forced migration failure"):
        _ = CatalogStore.open(tmp_path / "catalog.sqlite3")

    assert len(connections) == 1
    assert connections[0].close_calls == 1


def test_every_public_operation_holds_one_lock_for_its_complete_sql_work(
    tracked_store: tuple[CatalogStore, list[_TrackedConnection], _TraceGate],
    tmp_path: Path,
) -> None:
    store, connections, _gate = tracked_store
    timeline: list[str] = []
    lock = _ObservedLock(timeline)
    store.__dict__["_operation_lock"] = lock
    connections[0].set_trace_callback(lambda statement: timeline.append(f"sql:{statement}"))
    operations: tuple[Callable[[], object], ...] = (
        lambda: store.record("clips", "clip-1", {"clip_id": "clip-1"}),
        lambda: store.record_many((("events", "event-1", {"edge_event_id": "event-1"}),)),
        store.list_clips,
        lambda: store.records_with_columns("clips"),
        lambda: store.records("events"),
        store.integrity_check,
        lambda: store.backfill(ClipStore(tmp_path / "empty-clip-store")),
    )

    for operation in operations:
        timeline.clear()
        _ = operation()
        assert timeline[0].endswith("lock-acquired")
        assert timeline[-1].endswith("lock-released")
        assert timeline.count("MainThread-lock-acquired") == 1
        assert timeline.count("MainThread-lock-released") == 1


def test_close_marks_closing_before_waiting_and_rejects_queued_operation(
    tracked_store: tuple[CatalogStore, list[_TrackedConnection], _TraceGate],
) -> None:
    store, connections, gate = tracked_store
    timeline: list[str] = []
    operation_lock = _ObservedLock(timeline)
    store.__dict__["_operation_lock"] = operation_lock
    gate.enabled = True
    admitted_values: list[str] = []
    queued_errors: list[BaseException] = []
    closed_callers: list[str] = []

    def admitted() -> None:
        admitted_values.append(store.integrity_check())

    def queued() -> None:
        try:
            _ = store.records("clips")
        except RuntimeError as exc:
            queued_errors.append(exc)

    def close() -> None:
        store.close()
        closed_callers.append(threading.current_thread().name)

    admitted_thread = threading.Thread(name="admitted-operation", target=admitted)
    owner_thread = threading.Thread(name="close-owner", target=close)
    waiter_thread = threading.Thread(name="close-waiter", target=close)
    queued_thread = threading.Thread(name="queued-operation", target=queued)
    admitted_thread.start()
    assert gate.entered.wait(timeout=_THREAD_TIMEOUT)
    owner_thread.start()
    try:
        with store._state_condition:
            assert store._state_condition.wait_for(
                lambda: store._state.name == "CLOSING", timeout=_THREAD_TIMEOUT
            )
        queued_thread.start()
        waiter_thread.start()
        assert operation_lock.attempted("queued-operation").wait(timeout=_THREAD_TIMEOUT)
        assert not queued_errors
        assert owner_thread.is_alive()
        assert waiter_thread.is_alive()
    finally:
        gate.release.set()
    for thread in (admitted_thread, owner_thread, waiter_thread, queued_thread):
        thread.join(timeout=_THREAD_TIMEOUT)
        assert not thread.is_alive()

    assert admitted_values == ["ok"]
    assert sorted(closed_callers) == ["close-owner", "close-waiter"]
    assert [type(error) for error in queued_errors] == [RuntimeError]
    assert str(queued_errors[0]) == "catalog store is closed"
    assert connections[0].close_calls == 1


def _catalog_descriptor_tuple(path: Path) -> tuple[int, int, int]:
    resolved = path.resolve()
    targets = (resolved, Path(f"{resolved}-wal"), Path(f"{resolved}-shm"))
    counts = [0, 0, 0]
    descriptor_paths: list[Path] = []
    if _PROC_FD.is_dir():
        for entry in _PROC_FD.iterdir():
            try:
                descriptor_paths.append(Path(os.readlink(entry)).resolve())
            except OSError:
                continue
    else:
        output = subprocess.check_output(["lsof", "-a", "-p", str(os.getpid()), "-Fn"], text=True)
        lines = output.splitlines()
        descriptor_paths = [Path(line[1:]).resolve() for line in lines if line[:1] == "n"]
    for descriptor_path in descriptor_paths:
        for index, target in enumerate(targets):
            if descriptor_path == target:
                counts[index] += 1
    return counts[0], counts[1], counts[2]


def test_thread_churn_holds_warmed_catalog_fd_tuple_until_close(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    store = CatalogStore.open(path)
    try:
        assert store.integrity_check() == "ok"
        baseline = _catalog_descriptor_tuple(path)
        assert all(count > 0 for count in baseline)
        wave_counts: list[tuple[int, int, int]] = []
        for _ in range(4):
            _run_thread_wave(store, width=32)
            wave_counts.append(_catalog_descriptor_tuple(path))
        assert wave_counts == [baseline] * 4
    finally:
        store.close()
    assert _catalog_descriptor_tuple(path) == (0, 0, 0)
