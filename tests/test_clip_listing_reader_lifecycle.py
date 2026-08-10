from __future__ import annotations

import os
import sqlite3
import sys
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeAlias, final

import pytest
from typing_extensions import override

from backend.app.features.clips import listing_repository
from backend.app.features.clips.listing_generation import SqlParameter
from backend.app.features.clips.schemas import ClipListQuery

_PROC_FD = Path("/proc/self/fd")
_THREAD_TIMEOUT = 5.0


@final
class _TrackedConnection(sqlite3.Connection):
    statements: list[str] = []
    lifecycle_name = ""
    close_calls = 0
    close_events: list[str] = []

    def start_tracking(self, lifecycle_name: str, close_events: list[str]) -> None:
        self.statements = []
        self.lifecycle_name = lifecycle_name
        self.close_calls = 0
        self.close_events = close_events
        self.set_trace_callback(self.statements.append)

    @override
    def close(self) -> None:
        self.close_calls += 1
        self.close_events.append(self.lifecycle_name)
        super().close()


_TrackedRepository: TypeAlias = tuple[
    listing_repository.ListingRepository,
    list[_TrackedConnection],
    list[str],
]


@pytest.fixture
def tracked_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_TrackedRepository]:
    connections: list[_TrackedConnection] = []
    close_events: list[str] = []

    def connect_tracked(path: Path, create_statements: Iterable[str]) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            path,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
            factory=_TrackedConnection,
        )
        assert isinstance(connection, _TrackedConnection)
        lifecycle_name = "writer" if not connections else "reader"
        connection.start_tracking(lifecycle_name, close_events)
        _ = connection.execute("PRAGMA journal_mode = WAL")
        _ = connection.execute("PRAGMA synchronous = FULL")
        _ = connection.execute("PRAGMA busy_timeout = 5000")
        _ = connection.execute("PRAGMA foreign_keys = ON")
        for statement in create_statements:
            _ = connection.execute(statement)
        connections.append(connection)
        return connection

    monkeypatch.setattr(listing_repository, "connect_catalog_store", connect_tracked)
    repository = listing_repository.ListingRepository.open(tmp_path / "catalog.sqlite3")
    try:
        yield repository, connections, close_events
    finally:
        repository.close()


def _transactions(connection: _TrackedConnection) -> list[str]:
    return [
        statement
        for statement in connection.statements
        if statement in {"BEGIN", "COMMIT", "ROLLBACK"}
    ]


def test_open_owns_exactly_one_writer_and_one_reader_reused_by_successes(
    tracked_repository: _TrackedRepository,
) -> None:
    # Given: an open repository with one writer and one persistent reader.
    repository, connections, _close_events = tracked_repository

    # When: page and explain operations both succeed.
    page = repository.page(ClipListQuery(limit=48))
    plans = repository.explain(ClipListQuery(limit=48))

    # Then: both operations reused the same still-open reader.
    assert page.total == 0
    assert plans.page
    assert plans.summary
    assert len(connections) == 2
    writer, reader = connections
    assert [writer.lifecycle_name, reader.lifecycle_name] == ["writer", "reader"]
    assert _transactions(reader) == ["BEGIN", "COMMIT"]
    assert writer.close_calls == 0
    assert reader.close_calls == 0


def test_page_failure_rolls_back_then_reuses_the_same_reader(
    tracked_repository: _TrackedRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a page operation that fails after beginning its reader transaction.
    repository, connections, _close_events = tracked_repository

    def fail_active_generation(_connection: sqlite3.Connection) -> int:
        raise sqlite3.OperationalError("forced page failure")

    # When: the failed page is followed by a successful page.
    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(
            listing_repository.ListingRepository,
            "_active_generation",
            staticmethod(fail_active_generation),
        )
        with pytest.raises(sqlite3.OperationalError, match="forced page failure"):
            _ = repository.page(ClipListQuery(limit=48))
    page = repository.page(ClipListQuery(limit=48))

    # Then: rollback made the persistent reader reusable without replacing it.
    assert page.total == 0
    assert len(connections) == 2
    reader = connections[1]
    assert _transactions(reader) == ["BEGIN", "ROLLBACK", "BEGIN", "COMMIT"]
    assert reader.close_calls == 0
    assert connections[0].close_calls == 0


def test_explain_failure_keeps_reader_reusable_by_explain_and_page(
    tracked_repository: _TrackedRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an explain operation that fails while materializing its first plan.
    repository, connections, _close_events = tracked_repository

    def fail_plan(
        _connection: sqlite3.Connection,
        _sql: str,
        _values: tuple[SqlParameter, ...],
    ) -> tuple[str, ...]:
        raise sqlite3.OperationalError("forced explain failure")

    # When: successful explain and page operations follow the failure.
    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(listing_repository, "_plan", fail_plan)
        with pytest.raises(sqlite3.OperationalError, match="forced explain failure"):
            _ = repository.explain(ClipListQuery(limit=48))
    plans = repository.explain(ClipListQuery(limit=48))
    page = repository.page(ClipListQuery(limit=48))

    # Then: every operation used the original persistent reader.
    assert plans.page
    assert plans.summary
    assert page.total == 0
    assert len(connections) == 2
    assert connections[1].close_calls == 0
    assert connections[0].close_calls == 0


def test_open_closes_writer_when_persistent_reader_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: writer creation succeeds but the second connection cannot be opened.
    writers: list[_TrackedConnection] = []
    close_events: list[str] = []

    def fail_second_connection(
        path: Path,
        _create_statements: Iterable[str],
    ) -> sqlite3.Connection:
        if writers:
            raise sqlite3.OperationalError("forced reader open failure")
        connection = sqlite3.connect(
            path,
            isolation_level=None,
            check_same_thread=False,
            factory=_TrackedConnection,
        )
        assert isinstance(connection, _TrackedConnection)
        connection.start_tracking("writer", close_events)
        writers.append(connection)
        return connection

    monkeypatch.setattr(
        listing_repository,
        "connect_catalog_store",
        fail_second_connection,
    )

    # When: repository open attempts to create its persistent reader.
    with pytest.raises(sqlite3.OperationalError, match="forced reader open failure"):
        repository = listing_repository.ListingRepository.open(tmp_path / "catalog.sqlite3")
        repository.close()

    # Then: the already-created writer was physically closed exactly once.
    assert len(writers) == 1
    assert writers[0].close_calls == 1
    assert close_events == ["writer"]


def test_close_physically_closes_reader_before_writer_exactly_once(
    tracked_repository: _TrackedRepository,
) -> None:
    # Given: an open repository that owns exactly two physical connections.
    repository, connections, close_events = tracked_repository
    assert len(connections) == 2

    # When: two callers close it sequentially.
    repository.close()
    repository.close()

    # Then: the reader closes before the writer, with no repeated physical close.
    writer, reader = connections
    assert close_events == ["reader", "writer"]
    assert reader.close_calls == 1
    assert writer.close_calls == 1


def _catalog_descriptor_tuple(path: Path) -> tuple[int, int, int]:
    resolved = path.resolve()
    targets = (
        resolved,
        resolved.with_name(f"{resolved.name}-wal"),
        resolved.with_name(f"{resolved.name}-shm"),
    )
    counts = [0, 0, 0]
    for entry in _PROC_FD.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            descriptor_path = Path(os.readlink(entry)).resolve()
        except OSError:
            continue
        for index, target in enumerate(targets):
            if descriptor_path == target:
                counts[index] += 1
    return counts[0], counts[1], counts[2]


def _run_mixed_reader_wave(repository: listing_repository.ListingRepository) -> None:
    barrier = threading.Barrier(33)
    completed: list[int] = []

    def page() -> None:
        _ = barrier.wait(timeout=_THREAD_TIMEOUT)
        completed.append(repository.page(ClipListQuery(limit=48)).total)

    def explain() -> None:
        _ = barrier.wait(timeout=_THREAD_TIMEOUT)
        completed.append(len(repository.explain(ClipListQuery(limit=48)).page))

    threads = [threading.Thread(target=page) for _ in range(16)] + [
        threading.Thread(target=explain) for _ in range(16)
    ]
    for thread in threads:
        thread.start()
    _ = barrier.wait(timeout=_THREAD_TIMEOUT)
    for thread in threads:
        thread.join(timeout=_THREAD_TIMEOUT)
    assert all(not thread.is_alive() for thread in threads)
    assert len(completed) == 32


@pytest.mark.skipif(
    sys.platform != "linux" or not _PROC_FD.is_dir(),
    reason="requires Linux /proc/self/fd descriptor targets",
)
def test_mixed_reader_waves_hold_exact_warmed_fd_tuple_until_close(
    tmp_path: Path,
) -> None:
    # Given: a warmed writer/reader pair and its exact database/WAL/SHM FD tuple.
    path = tmp_path / "catalog.sqlite3"
    repository = listing_repository.ListingRepository.open(path)
    original_excepthook = threading.excepthook
    thread_errors: list[BaseException] = []

    def capture_thread_error(args: threading.ExceptHookArgs) -> None:
        assert args.exc_value is not None
        thread_errors.append(args.exc_value)

    threading.excepthook = capture_thread_error
    try:
        _ = repository.page(ClipListQuery(limit=48))
        _ = repository.explain(ClipListQuery(limit=48))
        baseline = _catalog_descriptor_tuple(path)
        assert all(count > 0 for count in baseline)

        # When: four 32-way waves each mix 16 page and 16 explain operations.
        wave_counts: list[tuple[int, int, int]] = []
        for _ in range(4):
            _run_mixed_reader_wave(repository)
            wave_counts.append(_catalog_descriptor_tuple(path))
    finally:
        threading.excepthook = original_excepthook
        repository.close()
    after_close = _catalog_descriptor_tuple(path)

    # Then: the warmed tuple never grows and close releases all three FD classes.
    assert wave_counts == [baseline] * 4
    assert after_close == (0, 0, 0)
    assert not thread_errors
