"""A QA telemetry failure must never suppress a resident alert.

Analysis-trace publication ships traces to the backend so QA replay has
something to replay. It has no bearing on whether a fall happened.

It was nonetheless able to suppress alerts. `BoundedTraceWriter._persist`
invoked the publisher *outside* its exception handling, and `_run` had no
enclosing catch, so a relay outage or a non-2xx response propagated out, killed
the writer thread, and every later `submit(require_persisted=True)` failed.
Because an admitted event requires persistence before it is emitted, a dead QA
publisher silently stopped fall and bed-exit events from reaching anyone.

These tests pin the separation: publication may fail as loudly as it likes, and
persistence, and therefore detection, carries on.
"""

from __future__ import annotations

import itertools
import threading
from pathlib import Path
from typing import Any

import pytest

from worker.pipeline.trace import (
    AnalysisTrace,
    OptionalNumber,
    TraceFrame,
)
from worker.pipeline.trace.writer import BoundedTraceWriter


def _frame(sequence: int) -> TraceFrame:
    analysis = AnalysisTrace(
        trace_id=f"analysis-{sequence}",
        frame_key=("boot-a", "camera-a", 1, sequence),
        pts=OptionalNumber(float(sequence)),
        source_time=OptionalNumber(float(sequence)),
        frame_width=4,
        frame_height=4,
        bed_region_provenance="fresh",
        persons=(),
        beds=(),
        components=(),
    )
    return TraceFrame(analysis, ())


_SEQUENCE = itertools.count(1)


def _next() -> int:
    return next(_SEQUENCE)


class _ExplodingPublisher:
    """Fails exactly the way a dead relay or a 500 response does."""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.calls = 0
        self.called = threading.Event()

    def __call__(self, frames: tuple[Any, ...], truncation: Any) -> None:
        self.calls += 1
        self.called.set()
        raise self._error


def _writer(tmp_path: Path, publisher: Any) -> BoundedTraceWriter:
    return BoundedTraceWriter(tmp_path / "runtime-analysis", publisher=publisher)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("analysis trace relay delivery failed"),
        OSError("connection refused"),
        ValueError("non-2xx"),
    ],
    ids=["relay-refused", "network-down", "bad-response"],
)
def test_a_failing_publisher_does_not_kill_the_writer(
    tmp_path: Path, error: BaseException
) -> None:
    """The writer thread must survive any publisher failure."""
    publisher = _ExplodingPublisher(error)
    writer = _writer(tmp_path, publisher)
    writer.start()
    try:
        assert writer.submit(_frame(_next()), require_persisted=True), (
            "the first admitted event failed to persist"
        )
        assert publisher.called.wait(timeout=5.0), "the publisher was never invoked"

        # The decisive assertion: a second admitted event must still persist.
        # Under the defect the writer thread was already dead here and this
        # returned False, which raises TracePersistenceError upstream and drops
        # the resident alert.
        assert writer.submit(_frame(_next()), require_persisted=True), (
            "persistence failed after a publisher error, so an admitted fall "
            "event would have been suppressed by a QA telemetry outage"
        )
    finally:
        writer.stop()


def test_publication_failures_are_counted_not_silent(
    tmp_path: Path
) -> None:
    """Degradation must be observable, or a dead relay looks like success."""
    publisher = _ExplodingPublisher(RuntimeError("relay down"))
    writer = _writer(tmp_path, publisher)
    writer.start()
    try:
        assert writer.submit(_frame(_next()), require_persisted=True)
        assert publisher.called.wait(timeout=5.0)
        writer.stop()
    finally:
        pass

    assert publisher.calls >= 1
    assert writer._publication_failures >= 1, (  # noqa: SLF001 - observability point
        "publication failures are not counted, so an operator cannot tell a "
        "silently dead QA relay from a healthy one"
    )


class _FlakyPublisher:
    """Fails for the first N calls, then succeeds -- a relay restart."""

    def __init__(self, failures: int) -> None:
        self._remaining = failures
        self.published: list[tuple[object, ...]] = []
        self.succeeded = threading.Event()

    def __call__(self, frames: tuple[object, ...], _truncation: object = None) -> None:
        if self._remaining > 0:
            self._remaining -= 1
            raise RuntimeError("relay unreachable")
        self.published.append(frames)
        self.succeeded.set()


def test_frames_missed_during_an_outage_are_republished_on_recovery(
    tmp_path: Path,
) -> None:
    """A transient relay outage must not permanently thin the backend's copy.

    Publication was best-effort with no retry, so frames sent while the relay
    was down were never re-sent. The backend's timeline stayed permanently and
    silently less complete than the worker's local store, with nothing recording
    the difference. The frames persisted locally, so re-sending them is both
    possible and safe: backend ingest is byte-identical idempotent.
    """
    publisher = _FlakyPublisher(failures=2)
    writer = _writer(tmp_path, publisher)
    writer.start()
    try:
        for _ in range(3):
            assert writer.submit(_frame(_next()), require_persisted=True)
        assert publisher.succeeded.wait(timeout=5.0), "the publisher never recovered"
    finally:
        writer.stop()

    delivered = [frame for batch in publisher.published for frame in batch]
    assert len(delivered) >= 3, (
        f"only {len(delivered)} frame(s) reached the backend after recovery; "
        f"frames missed during the outage were never republished"
    )


def test_frames_shed_from_the_republication_buffer_are_counted(tmp_path: Path) -> None:
    """Bounded retry must not lose frames invisibly.

    The republication buffer is bounded so a relay that stays down cannot grow
    it without limit, which means a long outage sheds frames. If that shedding
    is not counted, the backend's copy is quietly less complete than the
    worker's and nothing anywhere records the difference.
    """
    import worker.pipeline.trace.writer as writer_module

    publisher = _ExplodingPublisher(RuntimeError("relay down"))
    writer = _writer(tmp_path, publisher)
    # A buffer far smaller than the number of frames submitted, so shedding is
    # certain rather than incidental.
    original = writer_module._MAX_UNPUBLISHED_FRAMES  # noqa: SLF001
    writer_module._MAX_UNPUBLISHED_FRAMES = 2  # noqa: SLF001
    writer.start()
    try:
        for _ in range(8):
            assert writer.submit(_frame(_next()), require_persisted=True)
        assert publisher.called.wait(timeout=5.0)
    finally:
        writer.stop()
        writer_module._MAX_UNPUBLISHED_FRAMES = original  # noqa: SLF001

    stats = writer.stats()
    assert stats.publication_failures > 0, "publication failures are not reported"
    assert stats.unpublished_shed_frames > 0, (
        "frames were shed from the bounded republication buffer without being "
        "counted, so the loss is invisible to any operator"
    )
