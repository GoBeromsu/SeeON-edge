from __future__ import annotations

import threading

from clip_listing_reader_concurrency_fixtures import (
    THREAD_TIMEOUT,
    BlockedPage,
    Outcome,
    closed_error,
    join_thread,
    thread_sql,
)

from backend.app.features.clips.listing import ClipPage
from backend.app.features.clips.listing_queries import QueryPlans
from backend.app.features.clips.schemas import ClipListQuery

# pyright: strict, reportImplicitRelativeImport=false

pytest_plugins = ("clip_listing_reader_concurrency_fixtures",)


def test_page_and_explain_serialize_the_complete_reader_operation(
    blocked_page: BlockedPage,
) -> None:
    # Given: a page owns the observed reader lock for its complete operation.
    observed = blocked_page.observed
    page_events = observed.lock.events("admitted-page")
    outcome = Outcome[QueryPlans](observed.lock)
    explain_events = observed.lock.events("queued-explain")
    thread = outcome.start(
        "queued-explain",
        lambda: observed.repository.explain(ClipListQuery(limit=48)),
    )
    observed.lock.wait_for(
        lambda: explain_events.attempted.is_set() or outcome.done.is_set()
    )
    before_release = (
        explain_events.attempted.is_set(),
        explain_events.acquired.is_set(),
        outcome.done.is_set(),
        thread_sql(observed, "queued-explain"),
    )

    # When: the admitted page releases the reader lock.
    blocked_page.gate.release.set()
    join_thread(blocked_page.thread)
    join_thread(thread)

    # Then: explain acquires only after page release and runs without worker errors.
    assert all(
        (
            page_events.attempted.is_set(),
            page_events.acquired.is_set(),
            page_events.released.is_set(),
        )
    )
    assert before_release == (True, False, False, ())
    assert page_events.acquired.is_set()
    assert blocked_page.gate.entered.is_set()
    assert not blocked_page.outcome.errors
    assert len(blocked_page.outcome.values) == 1
    assert not outcome.errors
    assert len(outcome.values) == 1
    assert explain_events.acquired.is_set()
    assert explain_events.released.is_set()
    assert thread_sql(observed, "queued-explain")
    assert observed.timeline.index("page-materialized") < observed.timeline.index(
        "page-released"
    )
    assert observed.timeline.index("page-released") < observed.timeline.index(
        "admitted-page-lock-released"
    )
    assert observed.timeline.index(
        "admitted-page-lock-released"
    ) < observed.timeline.index("queued-explain-lock-acquired")


def test_close_waits_for_an_admitted_read_before_physical_close(
    blocked_page: BlockedPage,
) -> None:
    # Given: a page owns the reader lock and close reaches that lock after CLOSING.
    observed = blocked_page.observed
    page_events = observed.lock.events("admitted-page")
    outcome = Outcome[None](observed.lock)
    close_events = observed.lock.events("close-owner")

    def close_repository() -> None:
        observed.repository.close()
        observed.timeline.append("close-returned")

    thread = outcome.start("close-owner", close_repository)
    observed.lock.wait_for(lambda: close_events.attempted.is_set() or outcome.done.is_set())
    before_release = (
        close_events.attempted.is_set(),
        close_events.acquired.is_set(),
        outcome.done.is_set(),
    )

    # When: the admitted page releases the reader lock.
    blocked_page.gate.release.set()
    join_thread(blocked_page.thread)
    join_thread(thread)

    # Then: physical close follows page release and precedes close return.
    assert page_events.acquired.is_set()
    assert blocked_page.gate.entered.is_set()
    assert before_release == (True, False, False)
    assert not blocked_page.outcome.errors
    assert not outcome.errors
    assert outcome.values == [None]
    assert close_events.acquired.is_set()
    assert close_events.released.is_set()
    assert observed.timeline.index(
        "admitted-page-lock-released"
    ) < observed.timeline.index("reader-close")
    assert observed.timeline.index("reader-close") < observed.timeline.index("writer-close")
    assert observed.timeline.index("writer-close") < observed.timeline.index("close-returned")


def test_close_rejects_queued_and_new_reads_without_sql(
    blocked_page: BlockedPage,
) -> None:
    # Given: page ownership, a queued read, closing lock contention, and a new read.
    observed = blocked_page.observed
    page_events = observed.lock.events("admitted-page")
    queue = Outcome[QueryPlans](observed.lock)
    close = Outcome[None](observed.lock)
    new = Outcome[ClipPage](observed.lock)
    queue_events = observed.lock.events("queued-explain")
    close_events = observed.lock.events("close-owner")
    new_events = observed.lock.events("new-page")
    queue_thread = queue.start(
        "queued-explain",
        lambda: observed.repository.explain(ClipListQuery(limit=48)),
    )
    observed.lock.wait_for(lambda: queue_events.attempted.is_set() or queue.done.is_set())
    close_thread = close.start("close-owner", observed.repository.close)
    observed.lock.wait_for(lambda: close_events.attempted.is_set() or close.done.is_set())
    new_thread = new.start(
        "new-page",
        lambda: observed.repository.page(ClipListQuery(limit=48)),
    )
    observed.lock.wait_for(lambda: new_events.attempted.is_set() or new.done.is_set())
    before_release = (
        page_events.acquired.is_set(),
        queue_events.attempted.is_set(),
        close_events.attempted.is_set(),
        new_events.attempted.is_set(),
        queue.done.is_set(),
        close.done.is_set(),
        new.done.is_set(),
        thread_sql(observed, "queued-explain"),
        thread_sql(observed, "new-page"),
    )

    # When: the admitted page releases after every contender has attempted the lock.
    blocked_page.gate.release.set()
    for thread in (blocked_page.thread, queue_thread, close_thread, new_thread):
        join_thread(thread)

    # Then: both reads reject post-lock without SQL and close drains once in order.
    assert before_release == (True, True, True, True, False, False, False, (), ())
    assert blocked_page.gate.entered.is_set()
    assert not blocked_page.outcome.errors
    closed_error(queue)
    closed_error(new)
    assert not close.errors
    assert close.values == [None]
    assert all((queue_events.acquired.is_set(), queue_events.released.is_set()))
    assert all((new_events.acquired.is_set(), new_events.released.is_set()))
    assert all((close_events.acquired.is_set(), close_events.released.is_set()))
    assert not thread_sql(observed, "queued-explain")
    assert not thread_sql(observed, "new-page")
    assert observed.timeline.count("reader-close") == 1
    assert observed.timeline.count("writer-close") == 1
    assert observed.timeline.index("reader-close") < observed.timeline.index("writer-close")


def test_concurrent_close_callers_wait_for_one_terminal_physical_close(
    blocked_page: BlockedPage,
) -> None:
    # Given: two close callers race while a page owns the reader lock.
    observed = blocked_page.observed
    page_events = observed.lock.events("admitted-page")
    barrier = threading.Barrier(3)
    outcomes = [Outcome[None](observed.lock), Outcome[None](observed.lock)]

    def close_repository(index: int) -> None:
        _ = barrier.wait(timeout=THREAD_TIMEOUT)
        observed.repository.close()
        observed.timeline.append(f"close-{index}-returned")

    threads = [
        outcome.start(f"close-{index}", lambda index=index: close_repository(index))
        for index, outcome in enumerate(outcomes)
    ]
    _ = barrier.wait(timeout=THREAD_TIMEOUT)
    events = [observed.lock.events(f"close-{index}") for index in range(2)]
    observed.lock.wait_for(
        lambda: any(event.attempted.is_set() for event in events)
        or all(outcome.done.is_set() for outcome in outcomes)
    )
    before_release = (
        page_events.acquired.is_set(),
        sum(event.attempted.is_set() for event in events),
        any(event.acquired.is_set() for event in events),
        any(outcome.done.is_set() for outcome in outcomes),
    )

    # When: the admitted page releases the reader lock.
    blocked_page.gate.release.set()
    join_thread(blocked_page.thread)
    for thread in threads:
        join_thread(thread)

    # Then: one owner closes physically and both callers return after terminal close.
    assert before_release == (True, 1, False, False)
    assert blocked_page.gate.entered.is_set()
    assert not blocked_page.outcome.errors
    assert all(not outcome.errors and outcome.values == [None] for outcome in outcomes)
    assert sum(event.acquired.is_set() for event in events) == 1
    assert sum(event.released.is_set() for event in events) == 1
    assert observed.timeline.count("reader-close") == 1
    assert observed.timeline.count("writer-close") == 1
    writer_close = observed.timeline.index("writer-close")
    assert writer_close < observed.timeline.index("close-0-returned")
    assert writer_close < observed.timeline.index("close-1-returned")
