from __future__ import annotations

import fcntl
import os
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import final, override

import pytest

from backend.app.features.clips import listing_repository
from backend.app.features.clips.listing import ClipPage
from backend.app.features.clips.listing_generation import SqlParameter
from backend.app.features.clips.schemas import ClipListQuery

_FD_DIR = Path("/proc/self/fd") if Path("/proc/self/fd").exists() else Path("/dev/fd")


@final
class _TrackedConnection(sqlite3.Connection):
    statements: list[str] = []
    closed: bool = False

    def start_tracking(self) -> None:
        self.statements = []
        self.set_trace_callback(self.statements.append)

    @override
    def close(self) -> None:
        self.closed = True
        super().close()


def _catalog_descriptor_count(path: Path) -> int:
    resolved = path.resolve()
    catalog_paths = {
        resolved,
        resolved.with_name(f"{resolved.name}-wal"),
        resolved.with_name(f"{resolved.name}-shm"),
    }
    count = 0
    for entry in _FD_DIR.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if _FD_DIR == Path("/proc/self/fd"):
                descriptor_path = Path(os.readlink(entry)).resolve()
            else:
                raw: bytes = fcntl.fcntl(int(entry.name), fcntl.F_GETPATH, bytes(1024))
                descriptor_path = Path(raw.rstrip(b"\x00").decode()).resolve()
        except OSError:
            continue
        if descriptor_path in catalog_paths:
            count += 1
    return count


@pytest.fixture
def tracked_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[
    tuple[listing_repository.ListingRepository, list[_TrackedConnection]]
]:
    connections: list[_TrackedConnection] = []

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
        connection.start_tracking()
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
        yield repository, connections
    finally:
        repository.close()


def test_short_lived_page_threads_return_catalog_descriptors_to_baseline(
    tmp_path: Path,
) -> None:
    # Given: one completed page establishes the stable catalog descriptor baseline.
    path = tmp_path / "catalog.sqlite3"
    repository = listing_repository.ListingRepository.open(path)
    _ = repository.page(ClipListQuery(limit=48))
    baseline = _catalog_descriptor_count(path)
    assert baseline > 0
    pages: list[ClipPage] = []

    def read_page() -> None:
        pages.append(repository.page(ClipListQuery(limit=48)))

    threads = [threading.Thread(target=read_page) for _ in range(64)]

    # When: 64 independent page threads complete and terminate in sequence.
    for thread in threads:
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()

    # Then: operation readers are gone before repository shutdown.
    try:
        assert len(pages) == 64
        assert _catalog_descriptor_count(path) == baseline
    finally:
        repository.close()


def test_page_success_commits_and_closes_reader(
    tracked_repository: tuple[
        listing_repository.ListingRepository, list[_TrackedConnection]
    ],
) -> None:
    # Given: a repository whose writer and readers expose their lifecycle events.
    repository, connections = tracked_repository

    # When: a page read succeeds.
    page = repository.page(ClipListQuery(limit=48))

    # Then: its transaction commits and its operation-scoped reader closes.
    reader = connections[1]
    transactions = [
        statement for statement in reader.statements if statement in {"BEGIN", "COMMIT", "ROLLBACK"}
    ]
    assert page.total == 0
    assert transactions == ["BEGIN", "COMMIT"]
    assert reader.closed
    assert not connections[0].closed


def test_page_failure_rolls_back_closes_reader_and_keeps_writer_usable(
    tracked_repository: tuple[
        listing_repository.ListingRepository, list[_TrackedConnection]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a page reader that fails after opening its transaction.
    repository, connections = tracked_repository

    def fail_active_generation(_connection: sqlite3.Connection) -> int:
        raise sqlite3.OperationalError("forced page failure")

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(
            listing_repository.ListingRepository,
            "_active_generation",
            staticmethod(fail_active_generation),
        )

        # When: the page read raises.
        with pytest.raises(sqlite3.OperationalError, match="forced page failure"):
            _ = repository.page(ClipListQuery(limit=48))

    # Then: the reader rolls back and closes without affecting the writer.
    reader = connections[1]
    transactions = [
        statement for statement in reader.statements if statement in {"BEGIN", "COMMIT", "ROLLBACK"}
    ]
    assert transactions == ["BEGIN", "ROLLBACK"]
    assert reader.closed
    assert repository.active_clips() == {}
    assert not connections[0].closed


def test_explain_success_closes_its_single_reader(
    tracked_repository: tuple[
        listing_repository.ListingRepository, list[_TrackedConnection]
    ],
) -> None:
    # Given: a repository with tracked reader connections.
    repository, connections = tracked_repository

    # When: both page and summary plans are explained.
    plans = repository.explain(ClipListQuery(limit=48))

    # Then: both plans shared one reader and it is closed.
    assert plans.page
    assert plans.summary
    assert len(connections) == 2
    assert connections[1].closed
    assert not connections[0].closed


def test_explain_error_closes_reader(
    tracked_repository: tuple[
        listing_repository.ListingRepository, list[_TrackedConnection]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an explain operation that fails while reading its first plan.
    repository, connections = tracked_repository

    def fail_plan(
        _connection: sqlite3.Connection,
        _sql: str,
        _values: tuple[SqlParameter, ...],
    ) -> tuple[str, ...]:
        raise sqlite3.OperationalError("forced explain failure")

    monkeypatch.setattr(listing_repository, "_plan", fail_plan)

    # When: plan inspection raises.
    with pytest.raises(sqlite3.OperationalError, match="forced explain failure"):
        _ = repository.explain(ClipListQuery(limit=48))

    # Then: the operation reader still closes and the writer remains open.
    assert len(connections) == 2
    assert connections[1].closed
    assert not connections[0].closed
