"""One failing entry must not halt delivery of everything behind it.

`entries()` returns a sorted, deterministic order and the sender always chose
the first EVENT. An entry failing with a retryable status was therefore
re-selected on every call, forever, and every other entry behind it -- including
newer fall evidence -- was never sent at all. On a live deployment that is a
silent, total halt of evidence delivery caused by a single bad row.

Entries now carry an attempt budget: while they still have attempts they are
retried, once exhausted they are passed over so the queue drains, and they are
retained for an operator rather than retried forever or deleted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shared.events.delivery_queue import DeliveryQueue, EventEntry
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from worker.pipeline.output.evidence.evidence_sender import (
    EvidenceSender,
    SenderConfig,
)


def _entry(index: int) -> EventEntry:
    return EventEntry(
        edge_event_id=f"11111111-1111-4111-8111-{index:012d}",
        event_type="fall",
        detected_at="2026-08-22T00:00:00Z",
        camera_id="camera-1",
        facility_id="facility-1",
        decision_trace=b"{}",
        values=b'{"probability": 0.9}',
    )


class _PoisonTransport:
    """Fails one specific event forever; delivers everything else."""

    def __init__(
        self,
        poisoned: str,
        disposition: DeliveryDisposition = DeliveryDisposition.PERMANENT,
    ) -> None:
        self._poisoned = poisoned
        self._disposition = disposition
        self.delivered: list[str] = []

    def send_event(self, payload: Any, edge_event_id: str) -> Any:
        if edge_event_id == self._poisoned:
            return DeliveryFailure(disposition=self._disposition, code="HTTP_500")
        self.delivered.append(edge_event_id)
        return _Receipt(edge_event_id)


class _Receipt:
    def __init__(self, edge_event_id: str) -> None:
        self.edge_event_id = edge_event_id
        self.event_id = f"backend-{edge_event_id}"


@pytest.fixture(name="queue_dir")
def _queue_dir(tmp_path: Path) -> Path:
    return tmp_path / "delivery-queue"


def test_a_permanently_failing_entry_does_not_block_the_ones_behind_it(
    queue_dir: Path,
) -> None:
    """The decisive property: newer evidence still gets delivered."""
    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    poisoned = entries[0].edge_event_id
    transport = _PoisonTransport(poisoned)
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )

    for _ in range(40):
        sender.run_once()

    assert poisoned not in transport.delivered, "the poisoned entry should never deliver"
    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} were delivered; a single failing entry "
        f"halted the evidence queue behind it"
    )


def test_the_exhausted_entry_is_retained_not_deleted(queue_dir: Path) -> None:
    """Unblocking the queue must not mean discarding the evidence."""
    queue = DeliveryQueue(queue_dir)
    entry = _entry(1)
    assert queue.try_admit(entry).accepted

    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=_PoisonTransport(entry.edge_event_id),
    )
    for _ in range(40):
        sender.run_once()

    assert queue.capacity_snapshot.accepted_count == 0, "the queue never drained"
    retained = sorted(queue.dead_letter_directory.iterdir())
    assert len(retained) == 1, "the exhausted entry was discarded rather than retained"


def test_a_transient_failure_is_never_dead_lettered(queue_dir: Path) -> None:
    """A relay outage is what the durable queue exists to survive.

    An attempt budget that counts transient failures turns an outage into mass
    dead-lettering of perfectly good evidence: every entry exhausts its budget
    while the relay is simply restarting. Transient failures must retry
    indefinitely; only a failure attributable to the entry itself may exhaust.
    """
    queue = DeliveryQueue(queue_dir)
    entry = _entry(1)
    assert queue.try_admit(entry).accepted

    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=_PoisonTransport(entry.edge_event_id, disposition=DeliveryDisposition.RETRY),
    )
    for _ in range(60):
        sender.run_once()

    snapshot = queue.capacity_snapshot
    assert snapshot.accepted_count == 1, "the entry was removed during an outage"
    assert snapshot.dead_lettered_count == 0, (
        "a transient relay failure dead-lettered live evidence; an outage would "
        "discard the entire queue instead of holding it"
    )


def test_a_transiently_failing_entry_does_not_block_the_ones_behind_it(
    queue_dir: Path,
) -> None:
    """The case only rotation covers.

    A transient failure never exhausts the attempt budget -- correctly, because
    a relay outage is what the durable queue exists to survive. So the budget
    cannot be what keeps a transiently-failing head from starving everything
    behind it; only rotating past it can. If one camera's entry is rejected by a
    flapping upstream while others are fine, the others must still deliver.
    """
    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    transport = _PoisonTransport(entries[0].edge_event_id, disposition=DeliveryDisposition.RETRY)
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )

    for _ in range(40):
        sender.run_once()

    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} delivered; a transiently failing entry "
        f"starved the queue behind it, and no attempt budget can rescue that "
        f"because transient failures must retry forever"
    )
    assert queue.capacity_snapshot.dead_lettered_count == 0


def test_a_full_retention_area_does_not_stall_the_live_queue(
    queue_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retention refusing must not become a new head-of-line stall.

    When retention is full, `dead_letter` returns False and the entry stays in
    the live queue. Ignoring that return value meant the entry was reselected on
    every call forever: the queue never drained, newer alerts behind it were
    never delivered, and admission itself would eventually start failing. The
    log also claimed the entry had been retained, which was false.
    """
    from shared.events import delivery_queue as module

    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    # Retention cannot accept anything at all.
    monkeypatch.setattr(module, "MAX_DEAD_LETTERED_ENTRIES", 0)

    transport = _PoisonTransport(
        entries[0].edge_event_id, disposition=DeliveryDisposition.PERMANENT
    )
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )
    for _ in range(60):
        sender.run_once()

    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} delivered; a full retention area turned the "
        f"undeliverable entry into a permanent stall of the whole queue"
    )
    # The undeliverable entry is still held, undelivered, and still counted.
    assert queue.capacity_snapshot.accepted_count == 1
    assert queue.capacity_snapshot.dead_lettered_count == 0


def test_a_422_with_retention_full_does_not_stall_the_queue(
    queue_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The branch where the deferral is genuinely load-bearing.

    A 422 returns before the attempt counter is incremented, so a refused entry
    sits at zero attempts forever and `_select`'s attempt filter never skips it.
    If the failed `dead_letter` is ignored there, the same entry is reselected on
    every call and nothing behind it is ever delivered. That is different from
    the exhausted-attempt branch, where the attempt filter already rotates past
    the entry and the deferral really is redundant.
    """
    from shared.events import delivery_queue as module

    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    monkeypatch.setattr(module, "MAX_DEAD_LETTERED_ENTRIES", 0)

    class _RefusingTransport(_PoisonTransport):
        def send_event(self, payload: Any, edge_event_id: str) -> Any:
            if edge_event_id == self._poisoned:
                return DeliveryFailure(
                    disposition=DeliveryDisposition.PERMANENT,
                    code="UNPROCESSABLE",
                    status_code=422,
                )
            self.delivered.append(edge_event_id)
            return _Receipt(edge_event_id)

    transport = _RefusingTransport(entries[0].edge_event_id)
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )
    for _ in range(60):
        sender.run_once()

    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} delivered; a 422 that could not be retained "
        f"was reselected forever and starved every newer alert behind it"
    )
    assert queue.capacity_snapshot.accepted_count == 1
    assert queue.capacity_snapshot.dead_lettered_count == 0


def test_an_entry_that_raises_does_not_starve_the_queue(queue_dir: Path) -> None:
    """An unexpected failure on one entry must not stop every other one.

    The sender loop caught exceptions, but silently and without moving on, so a
    corrupt or unserialisable entry was reselected on every iteration and newer
    evidence behind it was never delivered. Nothing was logged, so the outage
    looked like an idle queue.
    """

    class _RaisingTransport:
        def __init__(self, poisoned: str) -> None:
            self._poisoned = poisoned
            self.delivered: list[str] = []

        def send_event(self, payload: Any, edge_event_id: str) -> Any:
            if edge_event_id == self._poisoned:
                raise RuntimeError("entry payload is corrupt")
            self.delivered.append(edge_event_id)
            return _Receipt(edge_event_id)

    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    transport = _RaisingTransport(entries[0].edge_event_id)
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )
    for _ in range(40):
        sender.run_once()

    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} delivered; one raising entry starved the queue behind it"
    )


def test_unwritable_retention_does_not_stall_the_queue(queue_dir: Path) -> None:
    """Retention I/O can fail for the same reason delivery did.

    `dead_letter` writes to the filesystem, so on a full or failing disk it
    raises. Unguarded, that exception escaped the 422 branch entirely and the
    outer sender loop caught it silently without deferring, so the same head was
    reselected on every iteration and every newer resident event behind it was
    blocked indefinitely.

    A guard that depends on the resource it is guarding against is not a guard.
    """
    import errno
    from unittest.mock import patch

    import shared.events.delivery_queue as queue_module

    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    class _RefusingTransport(_PoisonTransport):
        def send_event(self, payload: Any, edge_event_id: str) -> Any:
            if edge_event_id == self._poisoned:
                return DeliveryFailure(
                    disposition=DeliveryDisposition.PERMANENT,
                    code="UNPROCESSABLE",
                    status_code=422,
                )
            self.delivered.append(edge_event_id)
            return _Receipt(edge_event_id)

    transport = _RefusingTransport(entries[0].edge_event_id)
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )

    swallowed: list[BaseException] = []
    with patch.object(queue_module.os, "link", side_effect=OSError(errno.ENOSPC, "no space")):
        for _ in range(40):
            # Exactly what EvidenceExportRuntime._run_sender does: it catches
            # and carries on. Collected rather than discarded so the test says
            # what it tolerated.
            try:
                sender.run_once()
            except Exception as caught:  # noqa: BLE001 - mirrors the production loop
                swallowed.append(caught)

    assert not swallowed, (
        f"run_once raised {swallowed[0]!r}; retention I/O failure must be handled "
        f"inside the sender, not left to the loop that cannot defer the entry"
    )

    others = {entry.edge_event_id for entry in entries[1:]}
    assert others.issubset(set(transport.delivered)), (
        f"only {transport.delivered} delivered; an unwritable retention area "
        f"blocked every newer resident event behind the refused one"
    )
    assert queue.capacity_snapshot.accepted_count == 1


def test_a_failing_acknowledge_does_not_monopolise_the_queue(queue_dir: Path) -> None:
    """Removal failing must not stop newer evidence being delivered.

    On the success path the backend already has the entry; only our removal
    failed. Unguarded, a filesystem fault there re-selected the same entry on
    every iteration: the backend was flooded with duplicates of one event while
    every newer resident event behind it never left the queue at all.
    """
    import errno
    from unittest.mock import patch

    import shared.events.delivery_queue as queue_module

    queue = DeliveryQueue(queue_dir)
    entries = [_entry(index) for index in range(1, 4)]
    for entry in entries:
        assert queue.try_admit(entry).accepted

    class _AlwaysDelivers:
        def __init__(self) -> None:
            self.sent: list[str] = []

        def send_event(self, payload: Any, edge_event_id: str) -> Any:
            self.sent.append(edge_event_id)
            return _Receipt(edge_event_id)

    transport = _AlwaysDelivers()
    sender = EvidenceSender(
        queue_dir,
        SenderConfig(relay_url="http://relay.test", relay_token="t", probe_camera_id="camera-1"),
        transport=transport,
    )

    real_unlink = queue_module.Path.unlink

    def _failing_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.parent.name == "delivery-queue":
            raise OSError(errno.EIO, "io error")
        real_unlink(self, *args, **kwargs)

    swallowed: list[BaseException] = []
    with patch.object(queue_module.Path, "unlink", _failing_unlink):
        for _ in range(30):
            try:
                sender.run_once()
            except Exception as caught:  # noqa: BLE001 - mirrors the production loop
                swallowed.append(caught)

    assert not swallowed, f"run_once raised {swallowed[0]!r} instead of handling it"
    assert set(transport.sent) == {entry.edge_event_id for entry in entries}, (
        f"only {sorted(set(transport.sent))} reached the backend; a failing "
        f"removal monopolised the queue and starved every newer event"
    )
