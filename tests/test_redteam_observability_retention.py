"""Adversarial retention, prune, coarsening, and query-truth cases."""

from __future__ import annotations

import hashlib
from pathlib import Path

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from backend.app.features.diagnostics.coverage import insert_coverage
from backend.app.features.diagnostics.prune import coarsen_coverage, prune_unit
from backend.app.features.diagnostics.query import encode_cursor
from backend.app.features.diagnostics.records import (
    AvailabilityKind,
    CoverageKind,
    ExecutionRecordInput,
    IngestBatch,
    Provenance,
    RecordKind,
    UnitCausalState,
    late_ack_unit_id,
)
from backend.app.features.diagnostics.retention import (
    RetentionBudget,
    enforce_budget,
    used_bytes,
)
from backend.app.features.diagnostics.store import ExecutionRecordStore

CAMERA = "cam-a"
BOOT = "boot-1"
HORIZON = 1_000
DISK_BUDGET = 512 * 1024
PAYLOAD_BLOB = "x" * 200
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
    def __init__(self, now: int = 1) -> None:
        self.now = now

    def __call__(self) -> int:
        current = self.now
        self.now += 1
        return current


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _open(path: Path):
    return open_runtime_database(path, actor=RuntimeActor.API)


def _store(tmp_path: Path, budget: RetentionBudget) -> tuple[ExecutionRecordStore, Path]:
    path = tmp_path / "edge-state" / "edge.sqlite3"
    bootstrap_database(path)
    store = ExecutionRecordStore(lambda: _open(path), budget, clock=_Clock())
    return store, path


def _record(
    *,
    label: str,
    unit: str,
    seq: int,
    observed: int,
    payload: str,
    boot: str = BOOT,
    generation: int = 0,
    epoch: int = 0,
    producer: str = "sdk",
    kind: RecordKind = RecordKind.SDK_FRAME,
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
        payload={"blob": payload},
    )


def _batch(
    label: str, records: tuple[ExecutionRecordInput, ...], *, boot: str = BOOT
) -> IngestBatch:
    return IngestBatch(
        batch_id=_hex(label),
        camera_id=CAMERA,
        worker_boot_id=boot,
        provenance=PROVENANCE,
        records=records,
    )


def test_b1_forced_incomplete_unknown_converges_without_infinite_loop(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=DISK_BUDGET, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    blob = PAYLOAD_BLOB
    for index in range(500):
        store.ingest_batch(
            _batch(
                f"u{index}",
                (
                    _record(
                        label=f"r{index}",
                        unit=f"live-{index}",
                        seq=index,
                        observed=10 + index,
                        payload=blob,
                    ),
                ),
            )
        )
    connection = _open(path)
    try:
        remaining = {
            str(row[0])
            for row in connection.execute("SELECT causal_unit_id FROM execution_units").fetchall()
        }
        states = {
            str(row[0])
            for row in connection.execute("SELECT causal_state FROM execution_units").fetchall()
        }
        kinds = {
            str(row[0])
            for row in connection.execute("SELECT coverage_kind FROM execution_coverage").fetchall()
        }
        used = used_bytes(connection)
        # An exact enforce converges below high_water; between exact checks
        # usage may exceed high_water by one interval of accrual error but
        # never the total envelope.
        assert used <= budget.total_bytes
        # Each exact call prunes whole units and stops near low_water using
        # the measured ratio; it converges within a bounded number of calls.
        for _ in range(50):
            if used_bytes(connection) <= budget.high_water:
                break
            connection.execute("BEGIN IMMEDIATE")
            assert enforce_budget(connection, budget, 10_000_000) is True
            connection.execute("COMMIT")
        used = used_bytes(connection)
        counters = connection.execute(
            "SELECT MIN(record_count), MIN(payload_bytes) FROM execution_segments"
        ).fetchone()
    finally:
        connection.close()
    assert used <= budget.high_water
    assert "live-0" not in remaining
    assert CoverageKind.DELETED_BY_CAPACITY in kinds
    assert UnitCausalState.INCOMPLETE_UNKNOWN in states or "live-0" not in remaining
    assert counters is not None
    assert int(counters[0]) >= 0
    assert int(counters[1]) >= 0


def test_b2_prune_unit_straddling_segments_leaves_no_survivors(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=DISK_BUDGET, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    blob = PAYLOAD_BLOB
    old_records = tuple(
        _record(
            label=f"o{index}",
            unit="old",
            seq=index,
            observed=10 + index,
            payload=blob,
            producer="sdk" if index % 2 == 0 else "cpu",
        )
        for index in range(200)
    )
    store.ingest_batch(_batch("old", old_records))
    keep_records = tuple(
        _record(
            label=f"k{index}",
            unit="keep",
            seq=index,
            observed=10 + HORIZON + 5 + index,
            payload=blob,
        )
        for index in range(800)
    )
    store.ingest_batch(_batch("keep", keep_records))
    connection = _open(path)
    try:
        leftover = connection.execute(
            "SELECT COUNT(*) FROM execution_records WHERE causal_unit_id = 'old'"
        ).fetchone()
        units = {
            str(row[0])
            for row in connection.execute("SELECT causal_unit_id FROM execution_units").fetchall()
        }
        segments = connection.execute(
            "SELECT record_count, payload_bytes FROM execution_segments"
        ).fetchall()
        page = store.query(CAMERA, 0, 10_000, limit=500)
        deleted_ranges = [
            item for item in page.availability if item.kind is AvailabilityKind.DELETED_BY_CAPACITY
        ]
        exact_deleted = connection.execute(
            """
            SELECT from_ns, to_ns, exact FROM execution_coverage
            WHERE coverage_kind = ?
            """,
            (str(CoverageKind.DELETED_BY_CAPACITY),),
        ).fetchall()
    finally:
        connection.close()
    assert leftover == (0,)
    assert "old" not in units
    assert all(int(row[0]) >= 0 and int(row[1]) >= 0 for row in segments)
    for item in deleted_ranges:
        covered = [
            (int(row[0]), int(row[1]))
            for row in exact_deleted
            if int(row[2]) == 1 and not (int(row[1]) < item.from_ns or int(row[0]) > item.to_ns)
        ]
        assert covered
        cursor = item.from_ns
        for start, end in sorted(covered):
            if start > cursor:
                break
            cursor = max(cursor, end + 1)
        assert cursor > item.to_ns


def test_b3_late_ack_after_prune_creates_new_unit_never_resurrects(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    store.ingest_batch(
        _batch(
            "doomed",
            (
                _record(
                    label="doomed",
                    unit="unit-old",
                    seq=0,
                    observed=10,
                    payload="z",
                ),
            ),
        )
    )
    connection = _open(path)
    try:
        with write_transaction(connection):
            prune_unit(connection, "unit-old", 2)
    finally:
        connection.close()
    ack = _record(
        label="ack",
        unit="unit-old",
        seq=99,
        observed=10,
        payload="ack",
        producer="backend",
        kind=RecordKind.BACKEND_ACCEPTANCE,
        outcome="accepted",
    )
    receipt = store.ingest_batch(_batch("ack", (ack,)))
    assert receipt.accepted == 1
    expected_unit = late_ack_unit_id("unit-old", _hex("ack"))
    connection = _open(path)
    try:
        units = {
            str(row[0])
            for row in connection.execute("SELECT causal_unit_id FROM execution_units").fetchall()
        }
        stored = connection.execute(
            "SELECT causal_unit_id FROM execution_records WHERE record_id = ?",
            (_hex("ack"),),
        ).fetchone()
        resurrected = connection.execute(
            "SELECT COUNT(*) FROM execution_records WHERE causal_unit_id = 'unit-old'"
        ).fetchone()
        kinds = {
            str(row[0])
            for row in connection.execute("SELECT coverage_kind FROM execution_coverage")
        }
    finally:
        connection.close()
    assert "unit-old" not in units
    assert expected_unit in units
    assert stored == (expected_unit,)
    assert resurrected == (0,)
    assert CoverageKind.ACK_OBSERVED_PARENT_DELETED in kinds or (
        CoverageKind.ACK_OBSERVED_PARENT_UNKNOWN_COARSENED in kinds
    )


def test_b4_coarsening_does_not_widen_exact_deleted_ranges(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=HORIZON, coverage_rows_per_epoch=3)
    store, path = _store(tmp_path, budget)
    store.ingest_batch(
        _batch("seed", (_record(label="s", unit="u", seq=0, observed=1, payload="a"),))
    )
    connection = _open(path)
    try:
        with write_transaction(connection):
            insert_coverage(
                connection,
                camera_id=CAMERA,
                worker_boot_id=BOOT,
                source_generation=0,
                stream_epoch=0,
                kind=CoverageKind.DELETED_BY_CAPACITY,
                producer="sdk",
                from_sequence=0,
                to_sequence=0,
                from_ns=50,
                to_ns=50,
                record_count=1,
                exact=True,
                cause="capacity",
                recorded_at_ns=1,
            )
            for index in range(6):
                insert_coverage(
                    connection,
                    camera_id=CAMERA,
                    worker_boot_id=BOOT,
                    source_generation=0,
                    stream_epoch=0,
                    kind=CoverageKind.MISSING_NOT_RECORDED,
                    producer="sdk",
                    from_sequence=index + 1,
                    to_sequence=index + 1,
                    from_ns=100 + index,
                    to_ns=100 + index,
                    record_count=1,
                    exact=True,
                    cause="gap",
                    recorded_at_ns=index + 2,
                )
            coarsen_coverage(connection, budget.coverage_rows_per_epoch, 99)
        rows = connection.execute(
            """
            SELECT coverage_kind, exact, from_ns, to_ns
            FROM execution_coverage
            WHERE coverage_kind IN (?, ?, ?)
            ORDER BY from_ns
            """,
            (
                str(CoverageKind.UNKNOWN_COARSENED),
                str(CoverageKind.MISSING_NOT_RECORDED),
                str(CoverageKind.DELETED_BY_CAPACITY),
            ),
        ).fetchall()
        page = store.query(CAMERA, 0, 200, limit=10)
    finally:
        connection.close()
    coarsened = [row for row in rows if str(row[0]) == CoverageKind.UNKNOWN_COARSENED]
    exact_deleted = [
        row for row in rows if str(row[0]) == CoverageKind.DELETED_BY_CAPACITY and int(row[1]) == 1
    ]
    assert coarsened
    assert all(int(row[1]) == 0 for row in coarsened)
    for item in page.availability:
        if item.kind is AvailabilityKind.DELETED_BY_CAPACITY:
            assert any(
                int(row[2]) <= item.from_ns and int(row[3]) >= item.to_ns for row in exact_deleted
            )
            for coarse in coarsened:
                assert not (int(coarse[2]) <= item.from_ns and int(coarse[3]) >= item.to_ns)


def test_b5_availability_deleted_requires_exact_row(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    store.ingest_batch(
        _batch("seed", (_record(label="s", unit="u", seq=0, observed=1, payload="a"),))
    )
    connection = _open(path)
    try:
        with write_transaction(connection):
            insert_coverage(
                connection,
                camera_id=CAMERA,
                worker_boot_id=BOOT,
                source_generation=0,
                stream_epoch=0,
                kind=CoverageKind.UNKNOWN_COARSENED,
                producer=None,
                from_sequence=None,
                to_sequence=None,
                from_ns=20,
                to_ns=40,
                record_count=3,
                exact=False,
                cause="coarsened",
                recorded_at_ns=9,
            )
        page = store.query(CAMERA, 0, 50, limit=10)
    finally:
        connection.close()
    deleted = [
        item for item in page.availability if item.kind is AvailabilityKind.DELETED_BY_CAPACITY
    ]
    coarsened = [
        item for item in page.availability if item.kind is AvailabilityKind.UNKNOWN_COARSENED
    ]
    assert not deleted
    assert coarsened


def test_b9_foreign_cursor_does_not_leak_records(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=HORIZON)
    store, _path = _store(tmp_path, budget)
    store.ingest_batch(
        _batch(
            "cam-a",
            tuple(
                _record(label=f"a{index}", unit="ua", seq=index, observed=100 + index, payload="a")
                for index in range(3)
            ),
        )
    )
    other = ExecutionRecordInput(
        record_id=_hex("b0"),
        record_kind=RecordKind.SDK_FRAME,
        camera_id="cam-b",
        worker_boot_id=BOOT,
        source_generation=0,
        stream_epoch=0,
        producer="sdk",
        producer_sequence=0,
        observed_at_ns=100,
        time_quality="trusted",
        causal_unit_id="ub",
        outcome="ok",
        payload={"blob": "b"},
    )
    store.ingest_batch(
        IngestBatch(
            batch_id=_hex("cam-b"),
            camera_id="cam-b",
            worker_boot_id=BOOT,
            provenance=PROVENANCE,
            records=(other,),
        )
    )
    page = store.query(CAMERA, 0, 1_000, limit=1)
    assert page.next_cursor is not None
    foreign = store.query("cam-b", 0, 1_000, limit=10, cursor=page.next_cursor)
    leaked_ids = {row.record_id for row in foreign.records}
    assert _hex("a1") not in leaked_ids
    assert _hex("a2") not in leaked_ids
    assert all(row.camera_id == "cam-b" for row in foreign.records)
    forged = encode_cursor(100, 0, _hex("a0"))
    forged_page = store.query("cam-b", 0, 1_000, limit=10, cursor=forged)
    assert all(row.camera_id == "cam-b" for row in forged_page.records)
    assert _hex("a0") not in {row.record_id for row in forged_page.records}
