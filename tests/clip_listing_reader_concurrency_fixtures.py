from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Generic, NamedTuple, TypeVar, final

import pytest
from typing_extensions import override

from backend.app.features.clips import listing_repository
from backend.app.features.clips.listing import ClipPage
from backend.app.features.clips.listing_queries import QueryPlans
from backend.app.features.clips.schemas import ClipListQuery
from backend.app.features.clips.store import ClipManifest

THREAD_TIMEOUT = 5.0
_Result = TypeVar("_Result")


@final
class _ObservedConnection(sqlite3.Connection):
    statements: list[tuple[str, str]] = []
    lifecycle_name = ""
    close_calls = 0
    timeline: list[str] = []

    def observe(self, lifecycle_name: str, timeline: list[str]) -> None:
        self.statements = []
        self.lifecycle_name = lifecycle_name
        self.close_calls = 0
        self.timeline = timeline
        self.set_trace_callback(
            lambda statement: self.statements.append(
                (threading.current_thread().name, statement)
            )
        )

    @override
    def close(self) -> None:
        self.close_calls += 1
        self.timeline.append(f"{self.lifecycle_name}-close")
        super().close()


@dataclass(frozen=True, slots=True)
class _LockEvents:
    attempted: threading.Event = field(default_factory=threading.Event)
    acquired: threading.Event = field(default_factory=threading.Event)
    released: threading.Event = field(default_factory=threading.Event)


@final
class _ObservedLock:
    def __init__(self, timeline: list[str]) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition()
        self._events: dict[str, _LockEvents] = {}
        self._timeline = timeline

    def events(self, thread_name: str) -> _LockEvents:
        with self._condition:
            return self._events.setdefault(thread_name, _LockEvents())

    def wait_for(self, predicate: Callable[[], bool]) -> None:
        with self._condition:
            assert self._condition.wait_for(predicate, timeout=THREAD_TIMEOUT)

    def notify_progress(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _emit(self, event: threading.Event, phase: str) -> None:
        with self._condition:
            self._timeline.append(f"{threading.current_thread().name}-lock-{phase}")
            event.set()
            self._condition.notify_all()

    def __enter__(self) -> _ObservedLock:
        events = self.events(threading.current_thread().name)
        self._emit(events.attempted, "attempted")
        _ = self._lock.acquire()
        self._emit(events.acquired, "acquired")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._emit(self.events(threading.current_thread().name).released, "released")
        self._lock.release()


@final
class _PageGate:
    def __init__(self, timeline: list[str]) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self._timeline = timeline

    def __call__(
        self,
        manifests: tuple[ClipManifest, ...],
        total: int,
        has_more: bool,
        event_type_counts: Mapping[str, int],
    ) -> ClipPage:
        self._timeline.append("page-materialized")
        self.entered.set()
        assert self.release.wait(timeout=THREAD_TIMEOUT)
        self._timeline.append("page-released")
        return ClipPage(manifests, total, has_more, event_type_counts)


@final
class Outcome(Generic[_Result]):
    def __init__(self, lock: _ObservedLock) -> None:
        self.values: list[_Result] = []
        self.errors: list[BaseException] = []
        self.done = threading.Event()
        self._lock = lock

    def start(self, name: str, operation: Callable[[], _Result]) -> threading.Thread:
        thread = threading.Thread(name=name, target=lambda: self._capture(operation))
        thread.start()
        return thread

    def _capture(self, operation: Callable[[], _Result]) -> None:
        try:
            self.values.append(operation())
        except BaseException as exc:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
            self.errors.append(exc)
        finally:
            self.done.set()
            self._lock.notify_progress()


class _ObservedRepository(NamedTuple):
    repository: listing_repository.ListingRepository
    connections: list[_ObservedConnection]
    timeline: list[str]
    lock: _ObservedLock


@pytest.fixture
def observed_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_ObservedRepository]:
    connections: list[_ObservedConnection] = []
    timeline: list[str] = []

    def connect(path: Path, _statements: Iterable[str]) -> sqlite3.Connection:
        connection = sqlite3.connect(
            path,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
            factory=_ObservedConnection,
        )
        assert isinstance(connection, _ObservedConnection)
        connection.observe("writer" if not connections else "reader", timeline)
        for pragma in ("journal_mode = WAL", "synchronous = FULL", "busy_timeout = 5000"):
            _ = connection.execute(f"PRAGMA {pragma}")
        connections.append(connection)
        return connection

    monkeypatch.setattr(listing_repository, "connect_catalog_store", connect)
    repository = listing_repository.ListingRepository.open(tmp_path / "catalog.sqlite3")
    lock = _ObservedLock(timeline)
    repository.__dict__["_reader_lock"] = lock
    try:
        yield _ObservedRepository(repository, connections, timeline, lock)
    finally:
        repository.close()


class BlockedPage(NamedTuple):
    observed: _ObservedRepository
    outcome: Outcome[ClipPage]
    thread: threading.Thread
    gate: _PageGate


@pytest.fixture
def blocked_page(
    observed_repository: _ObservedRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[BlockedPage]:
    gate = _PageGate(observed_repository.timeline)
    monkeypatch.setattr(listing_repository, "ClipPage", gate)
    outcome = Outcome[ClipPage](observed_repository.lock)
    thread = outcome.start(
        "admitted-page",
        lambda: observed_repository.repository.page(ClipListQuery(limit=48)),
    )
    assert gate.entered.wait(timeout=THREAD_TIMEOUT)
    try:
        yield BlockedPage(observed_repository, outcome, thread, gate)
    finally:
        gate.release.set()
        thread.join(timeout=THREAD_TIMEOUT)
        assert not thread.is_alive()


def thread_sql(observed: _ObservedRepository, thread_name: str) -> tuple[str, ...]:
    return tuple(
        statement
        for connection in observed.connections
        for observed_thread, statement in connection.statements
        if observed_thread == thread_name
    )


def join_thread(thread: threading.Thread) -> None:
    thread.join(timeout=THREAD_TIMEOUT)
    assert not thread.is_alive()


def closed_error(outcome: Outcome[ClipPage] | Outcome[QueryPlans]) -> None:
    assert not outcome.values
    assert [type(error) for error in outcome.errors] == [
        listing_repository.ListingRepositoryClosedError
    ]
