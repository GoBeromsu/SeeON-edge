from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from worker.pipeline.output.evidence.evidence_outbox import (
    ClaimedEvent,
    ClaimLease,
    ClipId,
    ClipLocalState,
    ClipOutcome,
    EdgeEventId,
    EventClipConflictError,
    EvidenceOutbox,
    NewerSchemaVersionError,
    StagedEvent,
    StagedEventConflictError,
)


def _event(index: int, queued_at: float = 100.0) -> StagedEvent:
    return StagedEvent(
        edge_event_id=EdgeEventId(f"00000000-0000-4000-8000-{index:012d}"),
        detected_at=f"2026-07-16T00:00:{index:02d}Z",
        payload_json=f'{{"sequence":{index}}}',
        queued_at=queued_at,
    )


def _stage_ready(outbox: EvidenceOutbox, index: int, clip_id: str = "clip-1") -> None:
    # Given: a stable event exists in durable STAGED state.
    outbox.stage(_event(index))

    # When: the recorder supplies a clip relation.
    outbox.bind_clip(_event(index).edge_event_id, ClipId(clip_id))


def _verified_clip(outbox: EvidenceOutbox, clip_id: str = "clip-1") -> None:
    _stage_ready(outbox, 1, clip_id)
    outbox.record_clip_outcome(
        ClipOutcome(
            clip_id=ClipId(clip_id),
            local_state=ClipLocalState.VERIFIED,
            manifest_path=f"clips/{clip_id}/manifest.json",
            state_version=1,
            media_relpath=f"clips/{clip_id}/clip.mp4",
            sha256="a" * 64,
            size_bytes=123,
            mime_type="video/mp4",
            codec="h264",
            duration_ms=1000,
            clip_start_at="2026-07-16T00:00:00Z",
            clip_end_at="2026-07-16T00:00:01Z",
            finalized_at="2026-07-16T00:00:02Z",
        )
    )
    event = outbox.claim(ClaimLease("event-sender", 100.0, 10.0))
    assert event is not None
    assert outbox.acknowledge(event, backend_event_id="backend-event")


def test_database_uses_wal_full_and_foreign_keys(tmp_path: Path) -> None:
    # Given: a new evidence database path.
    database_path = tmp_path / "evidence.sqlite3"

    # When: the outbox initializes the database.
    with EvidenceOutbox.open(database_path) as outbox:
        settings = outbox.database_settings()

    # Then: durability and referential-integrity settings are active.
    assert settings.journal_mode == "wal"
    assert settings.synchronous == 2
    assert settings.foreign_keys == 1


def test_exact_stage_replay_is_idempotent_and_preserves_original_queue_order(
    tmp_path: Path,
) -> None:
    # Given: one staged event and an exact replay with a later queue timestamp.
    database_path = tmp_path / "evidence.sqlite3"
    original = _event(1, queued_at=100.0)
    replay = StagedEvent(
        edge_event_id=original.edge_event_id,
        detected_at=original.detected_at,
        payload_json=original.payload_json,
        queued_at=999.0,
    )
    later_event = _event(2, queued_at=200.0)
    with EvidenceOutbox.open(database_path) as outbox:
        outbox.stage(original)

        # When: the exact event is replayed before another event is queued.
        outbox.stage(replay)
        outbox.bind_clip(original.edge_event_id, ClipId("clip-original"))
        outbox.stage(later_event)
        outbox.bind_clip(later_event.edge_event_id, ClipId("clip-later"))
        claimed = outbox.claim(ClaimLease(owner="sender", now=1000.0, duration=10.0))

    # Then: replay succeeds without replacing the original queue timestamp.
    assert claimed is not None
    assert claimed.edge_event_id == original.edge_event_id
    assert claimed.detected_at == original.detected_at
    assert claimed.payload_json == original.payload_json


@pytest.mark.parametrize(
    ("detected_at", "payload_json", "mismatched_field"),
    (
        ("2026-07-16T00:09:59Z", '{"sequence":1}', "detected_at"),
        ("2026-07-16T00:00:01Z", '{"sequence":999}', "payload_json"),
    ),
)
def test_stage_replay_conflict_refuses_mismatch_and_preserves_original(
    tmp_path: Path,
    detected_at: str,
    payload_json: str,
    mismatched_field: str,
) -> None:
    # Given: an immutable staged event and a replay that changes canonical data.
    database_path = tmp_path / "evidence.sqlite3"
    original = _event(1)
    conflicting = StagedEvent(
        edge_event_id=original.edge_event_id,
        detected_at=detected_at,
        payload_json=payload_json,
        queued_at=999.0,
    )
    with EvidenceOutbox.open(database_path) as outbox:
        outbox.stage(original)

        # When: the stable ID is reused with altered canonical evidence.
        with pytest.raises(StagedEventConflictError) as raised:
            outbox.stage(conflicting)

        outbox.bind_clip(original.edge_event_id, ClipId("clip-original"))
        claimed = outbox.claim(ClaimLease(owner="sender", now=1000.0, duration=10.0))

    # Then: only field names are disclosed and the original row remains intact.
    assert raised.value.edge_event_id == original.edge_event_id
    assert raised.value.mismatched_fields == (mismatched_field,)
    assert original.payload_json not in str(raised.value)
    assert conflicting.payload_json not in repr(raised.value)
    assert claimed is not None
    assert claimed.detected_at == original.detected_at
    assert claimed.payload_json == original.payload_json


def test_committed_event_survives_process_kill_and_reopen(tmp_path: Path) -> None:
    # Given: a child process that commits evidence and exits without cleanup.
    database_path = tmp_path / "evidence.sqlite3"
    script = """
import os
import sys
from pathlib import Path
from worker.pipeline.output.evidence.evidence_outbox import (
    ClipId,
    EdgeEventId,
    EvidenceOutbox,
    StagedEvent,
)

outbox = EvidenceOutbox.open(Path(sys.argv[1]))
event = StagedEvent(
    edge_event_id=EdgeEventId("00000000-0000-4000-8000-000000000001"),
    detected_at="2026-07-16T00:00:01Z",
    payload_json='{"sequence":1}',
    queued_at=100.0,
)
outbox.stage(event)
outbox.bind_clip(event.edge_event_id, ClipId("clip-kill"))
os._exit(137)
"""

    # When: the process is killed after commit and a new owner opens the DB.
    result = subprocess.run(
        [sys.executable, "-c", script, str(database_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    with EvidenceOutbox.open(database_path) as reopened:
        references = reopened.ordered_event_ids(ClipId("clip-kill"))

    # Then: the committed relation is recovered without a graceful close.
    assert result.returncode == 137
    assert references == (EdgeEventId("00000000-0000-4000-8000-000000000001"),)


def test_queue_saturation_does_not_drop_committed_rows(tmp_path: Path) -> None:
    # Given: more events than the legacy in-memory outbox capacity.
    database_path = tmp_path / "evidence.sqlite3"
    expected_count = 257
    with EvidenceOutbox.open(database_path) as outbox:
        for index in range(expected_count):
            _stage_ready(outbox, index, "clip-saturation")

    # When: a fresh process owner reopens the persistent queue.
    with EvidenceOutbox.open(database_path) as reopened:
        pending_count = reopened.pending_count()
        references = reopened.ordered_event_ids(ClipId("clip-saturation"))

    # Then: every committed event remains durable and ordered.
    assert pending_count == expected_count
    assert len(references) == expected_count
    assert references[0] == _event(0).edge_event_id
    assert references[-1] == _event(expected_count - 1).edge_event_id


def test_concurrent_claimers_receive_distinct_events(tmp_path: Path) -> None:
    # Given: two ready events and two independent SQLite connections.
    database_path = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database_path) as outbox:
        _stage_ready(outbox, 1)
        _stage_ready(outbox, 2)
    barrier = Barrier(2)

    def claim(owner: str) -> ClaimedEvent | None:
        with EvidenceOutbox.open(database_path) as outbox:
            barrier.wait()
            return outbox.claim(ClaimLease(owner=owner, now=100.0, duration=30.0))

    # When: both owners race to claim the oldest ready row.
    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = tuple(executor.map(claim, ("sender-a", "sender-b")))

    # Then: BEGIN IMMEDIATE claim serialization prevents double delivery.
    assert all(claim is not None for claim in claims)
    assert len({claim.edge_event_id for claim in claims if claim is not None}) == 2
    assert {claim.attempt_count for claim in claims if claim is not None} == {1}


def test_expired_lease_is_recovered_by_another_owner(tmp_path: Path) -> None:
    # Given: one event leased to an owner that disappears.
    database_path = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database_path) as outbox:
        _stage_ready(outbox, 1)
        first = outbox.claim(ClaimLease(owner="sender-a", now=100.0, duration=10.0))

        # When: another owner checks before and after lease expiry.
        before_expiry = outbox.claim(ClaimLease(owner="sender-b", now=109.0, duration=10.0))
        recovered = outbox.claim(ClaimLease(owner="sender-b", now=110.0, duration=10.0))

    # Then: the active lease is exclusive and the expired one is retryable.
    assert first is not None
    assert before_expiry is None
    assert recovered is not None
    assert recovered.edge_event_id == first.edge_event_id
    assert recovered.lease_owner == "sender-b"
    assert recovered.attempt_count == 2


def test_concurrent_clip_claimers_never_receive_the_same_clip(tmp_path: Path) -> None:
    database_path = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database_path) as outbox:
        _verified_clip(outbox)
    barrier = Barrier(2)

    def claim(owner: str):
        with EvidenceOutbox.open(database_path) as outbox:
            barrier.wait()
            return outbox.claim_clip(ClaimLease(owner, 100.0, 30.0))

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = tuple(executor.map(claim, ("sender-a", "sender-b")))

    claimed = tuple(claim for claim in claims if claim is not None)
    assert len(claimed) == 1
    assert claimed[0].clip_id == ClipId("clip-1")
    assert claimed[0].attempt_count == 1


def test_expired_clip_lease_is_recovered_by_another_owner(tmp_path: Path) -> None:
    database_path = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database_path) as outbox:
        _verified_clip(outbox)
        first = outbox.claim_clip(ClaimLease("sender-a", 100.0, 10.0))
        before_expiry = outbox.claim_clip(ClaimLease("sender-b", 109.0, 10.0))
        recovered = outbox.claim_clip(ClaimLease("sender-b", 110.0, 10.0))

    assert first is not None
    assert before_expiry is None
    assert recovered is not None
    assert recovered.clip_id == first.clip_id
    assert recovered.lease_owner == "sender-b"
    assert recovered.attempt_count == 2


def test_retry_schedule_defers_claim_until_due(tmp_path: Path) -> None:
    # Given: a claimed event that failed delivery.
    database_path = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database_path) as outbox:
        _stage_ready(outbox, 1)
        first = outbox.claim(ClaimLease(owner="sender-a", now=100.0, duration=10.0))
        assert first is not None

        # When: the owner records a deterministic next-attempt time.
        scheduled = outbox.schedule_retry(first, next_attempt_at=200.0)
        too_early = outbox.claim(ClaimLease(owner="sender-b", now=199.0, duration=10.0))
        due = outbox.claim(ClaimLease(owner="sender-b", now=200.0, duration=10.0))

    # Then: no sender can reclaim the row before its persisted deadline.
    assert scheduled is True
    assert too_early is None
    assert due is not None
    assert due.attempt_count == 2


def test_acknowledge_removes_claim_from_pending_delivery(tmp_path: Path) -> None:
    # Given: an event claimed under an exact lease token.
    database_path = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database_path) as outbox:
        _stage_ready(outbox, 1)
        claim = outbox.claim(ClaimLease(owner="sender-a", now=100.0, duration=10.0))
        assert claim is not None

        # When: the sender records the backend acknowledgment twice.
        acknowledged = outbox.acknowledge(claim)
        replay = outbox.acknowledge(claim)
        pending_count = outbox.pending_count()
        next_claim = outbox.claim(ClaimLease(owner="sender-b", now=200.0, duration=10.0))

    # Then: only the live lease transitions once and the row is terminal.
    assert acknowledged is True
    assert replay is False
    assert pending_count == 0
    assert next_claim is None


def test_clip_relations_are_atomic_ordered_and_idempotent(tmp_path: Path) -> None:
    # Given: three staged events for one coalesced clip.
    database_path = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database_path) as outbox:
        for index in (3, 1, 2):
            outbox.stage(_event(index))

        # When: the events are bound in recorder order, including a replay.
        ordinals = tuple(
            outbox.bind_clip(_event(index).edge_event_id, ClipId("clip-coalesced"))
            for index in (3, 1, 2)
        )
        replay_ordinal = outbox.bind_clip(_event(1).edge_event_id, ClipId("clip-coalesced"))
        references = outbox.ordered_event_ids(ClipId("clip-coalesced"))

    # Then: each relation commits once and preserves insertion order.
    assert ordinals == (0, 1, 2)
    assert replay_ordinal == 1
    assert references == tuple(_event(index).edge_event_id for index in (3, 1, 2))


def test_binding_event_to_different_clip_refuses_without_partial_relation(
    tmp_path: Path,
) -> None:
    # Given: an event already bound to an immutable clip identity.
    database_path = tmp_path / "evidence.sqlite3"
    with EvidenceOutbox.open(database_path) as outbox:
        _stage_ready(outbox, 1, "clip-original")

        # When: a caller attempts to rebind the event.
        with pytest.raises(EventClipConflictError):
            outbox.bind_clip(_event(1).edge_event_id, ClipId("clip-other"))

        original = outbox.ordered_event_ids(ClipId("clip-original"))
        other = outbox.ordered_event_ids(ClipId("clip-other"))

    # Then: rollback preserves the original relation and no empty clip leaks.
    assert original == (_event(1).edge_event_id,)
    assert other == ()


def test_newer_schema_is_refused_without_mutation(tmp_path: Path) -> None:
    # Given: a database created by a newer worker image.
    database_path = tmp_path / "evidence.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA user_version = 999")
    before = os.stat(database_path).st_size

    # When: this worker attempts to open it.
    with pytest.raises(NewerSchemaVersionError) as raised:
        EvidenceOutbox.open(database_path)

    # Then: startup refuses compatibility and leaves the DB untouched.
    assert raised.value.found == 999
    assert raised.value.supported == 6
    assert os.stat(database_path).st_size == before
