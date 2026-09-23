"""Hermetic ExecutionRecordStore semantics against schema-19 execution_* tables."""

from __future__ import annotations

import hashlib
from pathlib import Path

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.features.diagnostics.records import (
    AvailabilityKind,
    CoverageKind,
    ExecutionRecordInput,
    GapReport,
    IngestBatch,
    Provenance,
    RecordKind,
    StorageState,
    late_ack_unit_id,
)
from backend.app.features.diagnostics.retention import RetentionBudget
from backend.app.features.diagnostics.store import ExecutionRecordStore

CAMERA = "cam-a"
BOOT = "boot-1"
PROVENANCE = Provenance(
    worker_build_revision="worker-rev",
    worker_image_digest="sha256:worker",
    model_digest="sha256:model",
    calibration_digest="sha256:cal",
    preprocessing_identity="pre-v1",
    config_digest="sha256:cfg",
    policy_identity="policy-v1",
    backend_build_revision="backend-rev",
)


class _Clock:
    def __init__(self, now: int = 1_000_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _database(tmp_path: Path) -> Path:
    path = tmp_path / "edge-state" / "edge.sqlite3"
    bootstrap_database(path)
    return path


def _factory(path: Path):
    return lambda: open_runtime_database(path, actor=RuntimeActor.API)


def _store(
    tmp_path: Path, *, total_bytes: int = 2**20, clock: _Clock | None = None
) -> tuple[ExecutionRecordStore, Path]:
    path = _database(tmp_path)
    store = ExecutionRecordStore(
        _factory(path),
        RetentionBudget(total_bytes=total_bytes),
        clock=clock or _Clock(),
    )
    return store, path


def _record(
    *,
    label: str,
    unit: str = "unit-a",
    kind: RecordKind = RecordKind.SDK_FRAME,
    seq: int = 0,
    observed: int = 100,
    payload: dict[str, object] | None = None,
    boot: str = BOOT,
    generation: int = 0,
    epoch: int = 0,
    producer: str = "sdk",
    outcome: str = "ok",
) -> ExecutionRecordInput:
    return ExecutionRecordInput(
        record_id=_hex(label),
        record_kind=kind,
        camera_id=CAMERA,
        worker_boot_id=boot,
        source_generation=generation,
        stream_epoch=epoch,
        producer=producer,
        producer_sequence=seq,
        observed_at_ns=observed,
        time_quality="trusted",
        causal_unit_id=unit,
        outcome=outcome,
        payload={} if payload is None else payload,
    )


def _batch(
    label: str,
    records: tuple[ExecutionRecordInput, ...],
    gaps: tuple[GapReport, ...] = (),
    *,
    boot: str = BOOT,
) -> IngestBatch:
    return IngestBatch(
        batch_id=_hex(label),
        camera_id=CAMERA,
        worker_boot_id=boot,
        provenance=PROVENANCE,
        records=records,
        gaps=gaps,
    )


def _count(path: Path, table: str) -> int:
    connection = open_runtime_database(path, actor=RuntimeActor.API)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def test_idempotent_batch_replay_returns_identical_receipt(tmp_path: Path) -> None:
    store, path = _store(tmp_path)
    batch = _batch("b1", (_record(label="r1"),))
    first = store.ingest_batch(batch)
    rows = _count(path, "execution_records")
    second = store.ingest_batch(batch)
    assert first == second
    assert first.storage_state is StorageState.COMMITTED
    assert first.accepted == 1
    assert _count(path, "execution_records") == rows
    assert _count(path, "execution_batches") == 1


def test_duplicate_and_conflict_record_dispositions(tmp_path: Path) -> None:
    store, path = _store(tmp_path)
    original = _record(label="same", seq=1, payload={"n": 1})
    store.ingest_batch(_batch("first", (original,)))
    duplicate = store.ingest_batch(_batch("second", (original,)))
    assert duplicate.accepted == 0
    assert duplicate.duplicates == 1
    assert duplicate.rejected == ()
    conflicted = _record(label="same", seq=1, payload={"n": 2})
    conflict = store.ingest_batch(_batch("third", (conflicted,)))
    assert conflict.accepted == 0
    assert conflict.duplicates == 0
    assert conflict.rejected == ((_hex("same"), "conflict"),)
    connection = open_runtime_database(path, actor=RuntimeActor.API)
    try:
        payload = connection.execute(
            "SELECT payload FROM execution_records WHERE record_id = ?", (_hex("same"),)
        ).fetchone()
    finally:
        connection.close()
    assert payload is not None and '"n":1' in str(payload[0])


def test_oversize_rejection_writes_coverage(tmp_path: Path) -> None:
    store, path = _store(tmp_path, total_bytes=512 * 1024)
    huge = _record(label="huge", payload={"blob": "x" * store.budget.max_record_bytes})
    receipt = store.ingest_batch(_batch("oversize", (huge,)))
    assert receipt.accepted == 0
    assert receipt.rejected == ((_hex("huge"), "oversize"),)
    assert receipt.storage_state is StorageState.COMMITTED
    assert _count(path, "execution_records") == 0
    connection = open_runtime_database(path, actor=RuntimeActor.API)
    try:
        kind = connection.execute("SELECT coverage_kind FROM execution_coverage").fetchone()
    finally:
        connection.close()
    assert kind == (CoverageKind.REJECTED_OVERSIZE,)


def test_query_availability_unknown_tails_and_cursor(tmp_path: Path) -> None:
    store, _path = _store(tmp_path)
    records = tuple(
        _record(label=f"q{index}", seq=index, observed=100 + index) for index in range(3)
    )
    store.ingest_batch(
        _batch(
            "query",
            records,
            gaps=(
                GapReport(
                    producer="sdk",
                    from_sequence=10,
                    to_sequence=11,
                    from_ns=200,
                    to_ns=210,
                    record_count=2,
                    cause="drop",
                ),
            ),
        )
    )
    page = store.query(CAMERA, 50, 300, limit=2)
    assert [item.producer_sequence for item in page.records] == [0, 1]
    assert page.next_cursor is not None
    rest = store.query(CAMERA, 50, 300, limit=2, cursor=page.next_cursor)
    assert [item.producer_sequence for item in rest.records] == [2]
    assert rest.next_cursor is None
    kinds = [item.kind for item in page.availability]
    assert AvailabilityKind.UNKNOWN in kinds
    assert AvailabilityKind.AVAILABLE in kinds
    assert AvailabilityKind.MISSING_NOT_RECORDED in kinds
    assert page.queryable_range.min_observed_at_ns == 100
    assert page.queryable_range.max_observed_at_ns == 102


def test_restart_reopens_same_data(tmp_path: Path) -> None:
    store, path = _store(tmp_path)
    store.ingest_batch(_batch("persist", (_record(label="p1", observed=50),)))
    restarted = ExecutionRecordStore(
        _factory(path), RetentionBudget(total_bytes=2**20), clock=_Clock()
    )
    result = restarted.query(CAMERA, 0, 100, limit=10)
    assert len(result.records) == 1
    assert result.records[0].record_id == _hex("p1")


def test_late_ack_uses_new_unit_and_ack_coverage(tmp_path: Path) -> None:
    store, path = _store(tmp_path)
    store.ingest_batch(
        _batch(
            "doomed",
            (
                _record(
                    label="doomed",
                    unit="unit-old",
                    seq=0,
                    observed=10,
                    payload={"k": "z"},
                ),
            ),
        )
    )
    connection = open_runtime_database(path, actor=RuntimeActor.API)
    try:
        from backend.app.edge_db.connection import write_transaction
        from backend.app.features.diagnostics.prune import prune_unit

        with write_transaction(connection):
            prune_unit(connection, "unit-old", 2)
    finally:
        connection.close()
    ack = _record(
        label="ack",
        unit="unit-old",
        kind=RecordKind.BACKEND_ACCEPTANCE,
        seq=99,
        observed=10,
        producer="backend",
        outcome="accepted",
    )
    receipt = store.ingest_batch(_batch("ack", (ack,)))
    assert receipt.storage_state is StorageState.COMMITTED
    expected_unit = late_ack_unit_id("unit-old", _hex("ack"))
    connection = open_runtime_database(path, actor=RuntimeActor.API)
    try:
        units = {
            str(row[0])
            for row in connection.execute("SELECT causal_unit_id FROM execution_units").fetchall()
        }
        kinds = {
            str(row[0])
            for row in connection.execute("SELECT coverage_kind FROM execution_coverage").fetchall()
        }
        stored_unit = connection.execute(
            "SELECT causal_unit_id FROM execution_records WHERE record_id = ?",
            (_hex("ack"),),
        ).fetchone()
    finally:
        connection.close()
    assert "unit-old" not in units
    assert expected_unit in units
    assert CoverageKind.ACK_OBSERVED_PARENT_DELETED in kinds or (
        CoverageKind.ACK_OBSERVED_PARENT_UNKNOWN_COARSENED in kinds
    )
    assert stored_unit == (expected_unit,)


def test_availability_is_a_span_between_contiguous_records_not_instants(tmp_path: Path) -> None:
    """Live rollout regression: a 120 s window with ~3,000 records painted
    10,241 ranges - each record a zero-length AVAILABLE with UNKNOWN between
    neighbours 3 ms apart. Adjacent producer_sequence in one lane proves nothing
    was lost between two records, so the interval is AVAILABLE. A sequence
    discontinuity without a gap row, or a lane boundary, ends the span."""
    store, _path = _store(tmp_path)
    contiguous = tuple(
        _record(label=f"s{index}", seq=index, observed=1_000 + index * 33) for index in range(5)
    )
    # seq 5 is missing and no gap row was reported: 6 starts a new span.
    resumed = tuple(
        _record(label=f"r{index}", seq=index, observed=1_000 + index * 33) for index in (6, 7)
    )
    store.ingest_batch(_batch("spans", contiguous + resumed))

    page = store.query(CAMERA, 900, 1_400, limit=10)
    painted = [(item.kind, item.from_ns, item.to_ns) for item in page.availability]
    assert len(painted) <= 5, painted
    assert painted[0][0] is AvailabilityKind.UNKNOWN  # before the earliest evidence
    available = [item for item in page.availability if item.kind is AvailabilityKind.AVAILABLE]
    assert [(item.from_ns, item.to_ns) for item in available] == [
        (1_000, 1_000 + 4 * 33),  # seq 0..4 as ONE span
        (1_000 + 6 * 33, 1_000 + 7 * 33),  # seq 6..7
    ]
    between = [
        item
        for item in page.availability
        if item.from_ns > 1_000 + 4 * 33 and item.to_ns < 1_000 + 6 * 33
    ]
    assert between and all(item.kind is AvailabilityKind.UNKNOWN for item in between)


def test_availability_lane_boundary_ends_a_span(tmp_path: Path) -> None:
    """A new boot restarts producer_sequence at 0; that boundary is not proof of
    continuity even when the timestamps abut, so the two boots are two spans
    (they may still merge if their time ranges touch, which is honest)."""
    store, _path = _store(tmp_path)
    first_boot = tuple(
        _record(label=f"a{index}", seq=index, observed=1_000 + index * 10, boot="boot-a")
        for index in range(3)
    )
    second_boot = tuple(
        _record(label=f"b{index}", seq=index, observed=5_000 + index * 10, boot="boot-b")
        for index in range(3)
    )
    store.ingest_batch(_batch("boot-a", first_boot, boot="boot-a"))
    store.ingest_batch(_batch("boot-b", second_boot, boot="boot-b"))
    page = store.query(CAMERA, 900, 5_100, limit=10)
    available = [
        (item.from_ns, item.to_ns)
        for item in page.availability
        if item.kind is AvailabilityKind.AVAILABLE
    ]
    assert available == [(1_000, 1_020), (5_000, 5_020)]
    gap = [item for item in page.availability if item.from_ns > 1_020 and item.to_ns < 5_000]
    assert gap and all(item.kind is AvailabilityKind.UNKNOWN for item in gap)
