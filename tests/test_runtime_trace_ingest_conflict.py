"""Re-ingesting a trace must be byte-identical idempotent or a typed conflict.

`_insert_frame` used parent `ON CONFLICT(trace_id) DO NOTHING` and child
`INSERT OR IGNORE`. Under that shape a second submission carrying the same
`trace_id` but contradictory facts was silently accepted, and worse, a *new*
child ordinal merged into the previously stored frame. The result is a frame
that never existed on any camera, and replay over it would produce an
authoritative-looking output corresponding to no real decision.

That is the same class as the four evidence-loss defects in this effort: silent
acceptance where a typed refusal belongs. Idempotence is a legitimate
requirement, but it has to be byte-identical idempotence, not last-writer-merges.
"""

from __future__ import annotations

import copy
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from replay_fixtures import valid_trace_payload, valid_trace_payload_with_children

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.qa.runtime_trace_store import (
    RuntimeAnalysisStore,
    RuntimeTraceConflict,
)
from shared.events.replay_wire import MAX_TRACE_FRAMES, decode_replay_trace


def _store(tmp_path: Path) -> tuple[RuntimeAnalysisStore, Path]:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    return RuntimeAnalysisStore(database), database


def _person_rows(database: Path) -> int:
    connection = sqlite3.connect(database)
    try:
        return int(
            connection.execute("SELECT COUNT(*) FROM runtime_analysis_persons").fetchone()[0]
        )
    finally:
        connection.close()


def test_an_identical_re_ingest_is_accepted(tmp_path: Path) -> None:
    """A re-run must not fail; publish-once retries are ordinary."""
    store, _ = _store(tmp_path)
    payload = valid_trace_payload()

    store.ingest(decode_replay_trace(payload))
    store.ingest(decode_replay_trace(copy.deepcopy(payload)))


def test_a_contradictory_fact_raises_a_typed_conflict(tmp_path: Path) -> None:
    """Same trace_id, different content, must never be silently discarded."""
    store, _ = _store(tmp_path)
    payload = valid_trace_payload()
    store.ingest(decode_replay_trace(payload))

    contradictory: dict[str, Any] = copy.deepcopy(payload)
    frame = contradictory["frames"][0]
    frame["frame_width"] = int(frame["frame_width"]) + 1

    with pytest.raises(RuntimeTraceConflict):
        store.ingest(decode_replay_trace(contradictory))


def test_a_new_child_ordinal_does_not_merge_into_the_stored_frame(
    tmp_path: Path,
) -> None:
    """The merge hazard proper: extra children must not join an existing frame.

    A naive conflict check that compares only the parent row passes this
    submission, and the child `INSERT OR IGNORE` then adds the new ordinal
    alongside the original. The stored frame afterwards matches neither
    submission.
    """
    store, database = _store(tmp_path)
    payload = valid_trace_payload_with_children()
    store.ingest(decode_replay_trace(payload))
    before = _person_rows(database)

    extended: dict[str, Any] = copy.deepcopy(payload)
    persons = extended["frames"][0]["persons"]
    extra = copy.deepcopy(persons[0])
    extra["ordinal"] = max(int(person["ordinal"]) for person in persons) + 1
    persons.append(extra)

    with pytest.raises(RuntimeTraceConflict):
        store.ingest(decode_replay_trace(extended))

    assert _person_rows(database) == before, (
        "the extra child ordinal was written despite the conflict; the stored "
        "frame now matches neither submission and would replay as a decision "
        "basis that never existed"
    )


def _trace_rows(database: Path) -> int:
    connection = sqlite3.connect(database)
    try:
        return int(
            connection.execute("SELECT COUNT(*) FROM runtime_analysis_traces").fetchone()[0]
        )
    finally:
        connection.close()


def test_a_conflict_on_a_later_frame_rolls_back_the_earlier_ones(
    tmp_path: Path,
) -> None:
    """The real partial-write hazard needs more than one frame to appear.

    A single-frame payload cannot show it: the conflict is detected before that
    frame writes anything, so nothing is pending. With several frames, the
    earlier ones are already written inside the transaction when a later frame
    conflicts. If that transaction is not atomic, the database keeps a fragment
    of a submission that was rejected, and a later replay reads a timeline that
    was never accepted.
    """
    store, database = _store(tmp_path)

    first = valid_trace_payload_with_children()
    stored_frame = first["frames"][0]
    store.ingest(decode_replay_trace(first))
    traces_before = _trace_rows(database)
    persons_before = _person_rows(database)

    # A fresh frame that would ingest cleanly, followed by one that conflicts
    # with what is already stored.
    fresh: dict[str, Any] = copy.deepcopy(stored_frame)
    fresh["trace_id"] = "b" * 64
    fresh["frame_key"] = [*fresh["frame_key"][:3], int(fresh["frame_key"][3]) + 1]

    conflicting: dict[str, Any] = copy.deepcopy(stored_frame)
    conflicting["frame_width"] = int(conflicting["frame_width"]) + 1

    batch: dict[str, Any] = copy.deepcopy(first)
    batch["frames"] = [fresh, conflicting]

    with pytest.raises(RuntimeTraceConflict):
        store.ingest(decode_replay_trace(batch))

    assert _trace_rows(database) == traces_before, (
        "the accepted frame from a rejected submission was left behind; the "
        "stored timeline now contains a fragment nobody submitted successfully"
    )
    assert _person_rows(database) == persons_before


def test_recovery_is_explicitly_bounded_to_the_retention_window(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    payload = valid_trace_payload()
    frame = payload["frames"][0]
    for sequence in range(MAX_TRACE_FRAMES + 1):
        captured: dict[str, Any] = copy.deepcopy(payload)
        captured_frame = copy.deepcopy(frame)
        captured_frame["trace_id"] = f"{sequence:064x}"
        captured_frame["frame_key"][3] = sequence
        captured["frames"] = [captured_frame]
        captured["truncation"]["oldest_retained_seq"] = sequence
        captured["truncation"]["newest_retained_seq"] = sequence
        captured["truncation"]["oldest_retained_key"][3] = sequence
        captured["truncation"]["newest_retained_key"][3] = sequence
        store.ingest(decode_replay_trace(captured))

    recovered = store.recover(str(payload["camera_id"]))

    assert len(recovered.frames) == MAX_TRACE_FRAMES
    assert recovered.frames[0]["frame_key"][3] == 1
    assert recovered.frames[-1]["frame_key"][3] == MAX_TRACE_FRAMES
