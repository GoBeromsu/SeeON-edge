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
    store, path = _store(tmp_path, total_bytes=8192)
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
