"""Evidence the backend refuses must be retained, never deleted as delivered.

`acknowledge_backend` unlinked the queue entry on HTTP 422 and the sender
reported it acknowledged. That is exactly how 41 real bed-exit events were
destroyed in this deployment: the cause was an undeclared field on an
`extra="forbid"` model, but the *mechanism* was this deletion. Fixing one
undeclared field does not make the next one safe -- and three further undeclared
fields were found in this same effort.

409 is different and stays a deletion: it means the backend already holds the
entry, so our copy is redundant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.events.delivery_queue import DeliveryQueue, EventEntry


def _entry(index: int = 1) -> EventEntry:
    return EventEntry(
        edge_event_id=f"11111111-1111-4111-8111-{index:012d}",
        event_type="fall",
        detected_at="2026-08-22T00:00:00Z",
        camera_id="camera-1",
        facility_id="facility-1",
        decision_trace=b"trace",
        values=b"values",
    )


def _queue(tmp_path: Path) -> DeliveryQueue:
    return DeliveryQueue(tmp_path / "delivery-queue")


def _live_entries(queue: DeliveryQueue) -> int:
    return queue.capacity_snapshot.accepted_count


def test_a_refused_entry_is_retained_outside_the_live_queue(tmp_path: Path) -> None:
    """422 must not destroy the evidence."""
    queue = _queue(tmp_path)
    entry = _entry()
    assert queue.try_admit(entry).accepted
    entry_id = next(iter(queue.entries()))["entry_id"]

    assert queue.dead_letter(str(entry_id), 422)

    assert _live_entries(queue) == 0, "the entry must leave the live queue"
    retained = sorted(queue.dead_letter_directory.iterdir())
    assert len(retained) == 1, "the refused evidence was not retained anywhere"
    assert retained[0].name.startswith("422."), (
        "the retained entry does not record why it was refused"
    )
    assert retained[0].read_bytes(), "the retained entry is empty"


def test_acknowledge_backend_refuses_to_treat_422_as_delivered(tmp_path: Path) -> None:
    """The deletion path must no longer accept a refusal status at all."""
    queue = _queue(tmp_path)
    assert queue.try_admit(_entry()).accepted
    entry_id = str(next(iter(queue.entries()))["entry_id"])

    assert queue.acknowledge_backend(entry_id, 422) is False, (
        "422 was accepted as an acknowledgement; refused evidence would be "
        "deleted and reported delivered"
    )
    assert _live_entries(queue) == 1, "the entry was removed despite the refusal"


@pytest.mark.parametrize("status", [200, 201, 202, 204, 409])
def test_genuinely_delivered_entries_are_still_removed(tmp_path: Path, status: int) -> None:
    """Guard the guard: retaining everything would stall the queue at its bound.

    409 means the backend already holds the entry, so our copy is redundant and
    deleting it is correct.
    """
    queue = _queue(tmp_path)
    assert queue.try_admit(_entry()).accepted
    entry_id = str(next(iter(queue.entries()))["entry_id"])

    assert queue.acknowledge_backend(entry_id, status) is True
    assert _live_entries(queue) == 0


def test_retained_evidence_survives_reopening_the_queue(tmp_path: Path) -> None:
    """Retention is only useful if it outlives the process that recorded it."""
    queue = _queue(tmp_path)
    assert queue.try_admit(_entry()).accepted
    entry_id = str(next(iter(queue.entries()))["entry_id"])
    queue.dead_letter(entry_id, 422)

    reopened = _queue(tmp_path)

    assert _live_entries(reopened) == 0
    assert len(sorted(reopened.dead_letter_directory.iterdir())) == 1


def test_a_second_refusal_of_the_same_id_does_not_clobber_the_first(
    tmp_path: Path,
) -> None:
    """Retention that overwrites is not retention.

    `os.replace` silently overwrites, so re-admitting the same entry id after an
    earlier refusal destroyed the first retained copy -- reintroducing exactly
    the evidence loss this directory exists to prevent.
    """
    queue = _queue(tmp_path)

    for _ in range(2):
        assert queue.try_admit(_entry()).accepted
        entry_id = str(next(iter(queue.entries()))["entry_id"])
        assert queue.dead_letter(entry_id, 422)

    retained = sorted(queue.dead_letter_directory.iterdir())
    assert len(retained) == 2, (
        f"only {len(retained)} retained after two refusals of the same id; the "
        f"first copy was overwritten"
    )


def test_retained_evidence_is_visible_in_the_capacity_snapshot(tmp_path: Path) -> None:
    """An operator cannot act on evidence the deployment never reports."""
    queue = _queue(tmp_path)
    assert queue.try_admit(_entry()).accepted
    entry_id = str(next(iter(queue.entries()))["entry_id"])
    queue.dead_letter(entry_id, 422)

    snapshot = _queue(tmp_path).capacity_snapshot

    assert snapshot.dead_lettered_count == 1, (
        "refused evidence is invisible to the status path, so nobody learns it needs review"
    )
    assert snapshot.dead_lettered_bytes > 0


def test_retention_refuses_rather_than_evicting_when_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full retention area must not make room by discarding older evidence.

    Evicting the oldest refused entry to retain the newest is the same deletion
    this directory exists to prevent, just slower. When full, the entry stays in
    the live queue where it is counted and visible, and the operator drains the
    retention area.
    """
    from shared.events import delivery_queue as module

    monkeypatch.setattr(module, "MAX_DEAD_LETTERED_ENTRIES", 1)
    queue = _queue(tmp_path)

    assert queue.try_admit(_entry(1)).accepted
    first = str(next(iter(queue.entries()))["entry_id"])
    assert queue.dead_letter(first, 422)

    assert queue.try_admit(_entry(2)).accepted
    second = str(next(iter(queue.entries()))["entry_id"])

    assert queue.dead_letter(second, 422) is False, "retention accepted past its bound"
    assert len(sorted(queue.dead_letter_directory.iterdir())) == 1, (
        "the earlier refused entry was evicted to make room"
    )
    assert queue.capacity_snapshot.accepted_count == 1, (
        "the entry vanished instead of staying in the live queue where it is counted and visible"
    )


def test_the_operator_command_inspects_and_requeues(tmp_path: Path) -> None:
    """A gate an operator cannot clear is worse than no gate."""
    import subprocess
    import sys

    state = tmp_path
    queue = DeliveryQueue(state / "delivery-queue")
    assert queue.try_admit(_entry(1)).accepted
    entry_id = str(next(iter(queue.entries()))["entry_id"])
    assert queue.dead_letter(entry_id, 422)

    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts/ops/review-refused-evidence.py"),
        "--state-dir",
        str(state),
    ]

    inspect = subprocess.run(command, capture_output=True, text=True, check=False)
    assert inspect.returncode == 1, "retained evidence must not report all-clear"
    assert '"422": 1' in inspect.stdout

    requeue = subprocess.run([*command, "--requeue"], capture_output=True, text=True, check=False)
    assert requeue.returncode == 0, requeue.stdout
    assert DeliveryQueue(state / "delivery-queue").capacity_snapshot.accepted_count == 1

    after = subprocess.run(command, capture_output=True, text=True, check=False)
    assert after.returncode == 0, "the retention area never reported clear"


def test_requeue_respects_the_live_queue_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requeue must not be a back door around admission control.

    Writing the retained file straight back into the queue directory bypasses
    the exclusive lock, the capacity bounds and atomic publication. An operator
    command repairing an evidence problem must not introduce a worse one.
    """
    from shared.events import delivery_queue as module

    queue = _queue(tmp_path)
    assert queue.try_admit(_entry(1)).accepted
    entry_id = str(next(iter(queue.entries()))["entry_id"])
    assert queue.dead_letter(entry_id, 422)
    retained = next(iter(queue.dead_letter_directory.iterdir()))

    monkeypatch.setattr(module, "MAX_ACCEPTED_ENTRIES", 0)
    assert queue.requeue_dead_lettered(retained) is False, (
        "requeue admitted past the live queue bound"
    )
    assert retained.exists(), "the retained copy was consumed by a failed requeue"


def test_requeue_of_an_already_present_identity_is_idempotent(tmp_path: Path) -> None:
    """A repeated requeue must not duplicate or destroy anything."""
    queue = _queue(tmp_path)
    assert queue.try_admit(_entry(1)).accepted
    entry_id = str(next(iter(queue.entries()))["entry_id"])
    assert queue.dead_letter(entry_id, 422)
    retained = next(iter(queue.dead_letter_directory.iterdir()))

    assert queue.requeue_dead_lettered(retained) is True
    assert queue.capacity_snapshot.accepted_count == 1

    # Retain and requeue the same identity again: byte-identical, so accepted.
    entry_id = str(next(iter(queue.entries()))["entry_id"])
    assert queue.dead_letter(entry_id, 422)
    retained = next(iter(queue.dead_letter_directory.iterdir()))
    assert queue.requeue_dead_lettered(retained) is True
    assert queue.capacity_snapshot.accepted_count == 1
    assert not sorted(queue.dead_letter_directory.iterdir())


def test_a_requeued_duplicate_is_visible_to_the_queue_again(tmp_path: Path) -> None:
    """A requeued entry the queue cannot see is evidence silently lost.

    Retained copies were disambiguated by appending to the end of the filename,
    producing names like `...json.1`. Requeueing one recovered `...json.1` as
    the entry identity and wrote a file the queue's own `*.json` scan does not
    match: present on disk, invisible to delivery. The disambiguator now sits
    ahead of the original name, so the recovered identity is always the one the
    queue admitted.
    """
    queue = _queue(tmp_path)

    # Two refusals of the same identity: the second must take a distinct name.
    for _ in range(2):
        assert queue.try_admit(_entry(1)).accepted
        entry_id = str(next(iter(queue.entries()))["entry_id"])
        assert queue.dead_letter(entry_id, 422)

    retained = sorted(queue.dead_letter_directory.iterdir())
    assert len(retained) == 2

    assert queue.requeue_dead_lettered(retained[1]) is True
    assert queue.capacity_snapshot.accepted_count == 1, (
        "the requeued entry is not visible to the queue that must deliver it"
    )
    # And it is genuinely readable as an entry, not just a file on disk.
    assert [entry["kind"] for entry in queue.entries()] == ["EVENT"]
