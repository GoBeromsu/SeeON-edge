"""Retention, prune, coarsening, and capacity receipts for execution records."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.features.diagnostics.coverage import insert_coverage
from backend.app.features.diagnostics.records import (
    CoverageKind,
    ExecutionRecordInput,
    IngestBatch,
    Provenance,
    RecordKind,
    SegmentStorageState,
    StorageState,
    UnitCausalState,
)
from backend.app.features.diagnostics.retention import (
    RetentionBudget,
    enforce_budget,
    used_bytes,
)
from backend.app.features.diagnostics.store import ExecutionRecordStore
from backend.app.features.diagnostics.terminals import (
    force_oldest_units_terminal,
    refresh_unit_terminals,
    seal_final_segments,
)

CAMERA = "cam-a"
BOOT = "boot-1"
HORIZON = 1_000
# Empty schema-19 execution_* b-trees already occupy ~70 KiB of pages; a
# 512 KiB envelope is the smallest budget that still admits a few hundred
# ~211 B payloads before high_water.
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
    camera: str = CAMERA,
    boot: str = BOOT,
    generation: int = 0,
    epoch: int = 0,
    producer: str = "sdk",
) -> ExecutionRecordInput:
    return ExecutionRecordInput(
        record_id=_hex(label),
        record_kind=RecordKind.SDK_FRAME,
        camera_id=camera,
        worker_boot_id=boot,
        source_generation=generation,
        stream_epoch=epoch,
        producer=producer,
        producer_sequence=seq,
        observed_at_ns=observed,
        time_quality="trusted",
        causal_unit_id=unit,
        outcome="ok",
        payload={"blob": payload},
    )


def _batch(
    label: str,
    records: tuple[ExecutionRecordInput, ...],
    *,
    camera: str = CAMERA,
    boot: str = BOOT,
) -> IngestBatch:
    return IngestBatch(
        batch_id=_hex(label),
        camera_id=camera,
        worker_boot_id=boot,
        provenance=PROVENANCE,
        records=records,
    )


def _dbstat_execution_sum(connection) -> int:
    names = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type IN ('table', 'index')
              AND (name LIKE 'execution_%' OR tbl_name LIKE 'execution_%')
            """
        ).fetchall()
    ]
    placeholders = ",".join("?" * len(names))
    row = connection.execute(
        f"SELECT COALESCE(SUM(pgsize), 0) FROM dbstat "
        f"WHERE aggregate = TRUE AND name IN ({placeholders})",
        names,
    ).fetchone()
    return int(row[0])


def test_budget_requires_explicit_total_bytes() -> None:
    with pytest.raises(ValueError, match="total_bytes"):
        RetentionBudget(total_bytes=255)
    budget = RetentionBudget(total_bytes=256)
    assert budget.max_record_bytes == 1
    assert budget.segment_bytes == 4
    assert budget.control_reserve == 16
    assert budget.high_water == 240
    assert budget.low_water == 210
    assert budget.coverage_rows_per_epoch == 512
    assert budget.unit_horizon_ns == 60_000_000_000


def test_segment_seals_at_segment_bytes(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=DISK_BUDGET, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    # max_record_bytes is segment_bytes / 4 by design, so one segment always
    # holds at least four admitted records; eight ~211 B records must therefore
    # spill from a sealed segment into a new OPEN one once the 8 KiB segment
    # envelope fills (eight records still fit one OPEN segment at 512 KiB —
    # ingest enough copies of the same-size payload to force a seal).
    blob = PAYLOAD_BLOB
    per_record = 211
    needed = (budget.segment_bytes // per_record) + 2
    store.ingest_batch(
        _batch(
            "s1",
            tuple(
                _record(label=f"r{index}", unit="u1", seq=index, observed=10 + index, payload=blob)
                for index in range(needed)
            ),
        )
    )
    connection = _open(path)
    try:
        rows = connection.execute(
            "SELECT storage_state, record_count, payload_bytes FROM execution_segments "
            "ORDER BY segment_ordinal"
        ).fetchall()
    finally:
        connection.close()
    states = [str(row[0]) for row in rows]
    assert states[0] == SegmentStorageState.SEALED_PENDING
    assert states[-1] == SegmentStorageState.OPEN
    assert len(rows) >= 2
    assert all(int(row[2]) <= budget.segment_bytes for row in rows)
    assert sum(int(row[1]) for row in rows) == needed


def test_unit_terminal_horizon_complete_and_known_gap(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    store.ingest_batch(
        _batch("u1", (_record(label="early", unit="early", seq=0, observed=10, payload="a"),))
    )
    store.ingest_batch(
        _batch(
            "u2",
            (_record(label="late", unit="late", seq=1, observed=10 + HORIZON + 1, payload="b"),),
        )
    )
    connection = _open(path)
    try:
        row = connection.execute(
            "SELECT terminal, causal_state FROM execution_units WHERE causal_unit_id = ?",
            ("early",),
        ).fetchone()
        assert row == (1, UnitCausalState.COMPLETE)
        insert_coverage(
            connection,
            camera_id=CAMERA,
            worker_boot_id=BOOT,
            source_generation=0,
            stream_epoch=0,
            kind=CoverageKind.MISSING_NOT_RECORDED,
            producer="sdk",
            from_sequence=0,
            to_sequence=0,
            from_ns=10,
            to_ns=10,
            record_count=1,
            exact=True,
            cause="drop",
            recorded_at_ns=1,
        )
        connection.execute(
            "UPDATE execution_units SET terminal = 0, causal_state = ? WHERE causal_unit_id = ?",
            (str(UnitCausalState.INCOMPLETE_UNKNOWN), "early"),
        )
        refresh_unit_terminals(connection, HORIZON)
        known = connection.execute(
            "SELECT terminal, causal_state FROM execution_units WHERE causal_unit_id = ?",
            ("early",),
        ).fetchone()
    finally:
        connection.close()
    assert known == (1, UnitCausalState.INCOMPLETE_KNOWN)


def test_unit_terminal_by_newer_epoch(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    store.ingest_batch(
        _batch("old", (_record(label="old", unit="old-u", seq=0, observed=10, payload="a"),))
    )
    store.ingest_batch(
        _batch(
            "new",
            (_record(label="new", unit="new-u", seq=0, observed=11, payload="b", epoch=1),),
        )
    )
    connection = _open(path)
    try:
        row = connection.execute(
            "SELECT terminal, causal_state FROM execution_units WHERE causal_unit_id = ?",
            ("old-u",),
        ).fetchone()
    finally:
        connection.close()
    assert row == (1, UnitCausalState.INCOMPLETE_UNKNOWN)


def test_prune_removes_whole_units_across_two_segments(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=DISK_BUDGET, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    # ~800 records (~211 B payload each) over many 8 KiB segments plus their
    # indexes exceed high_water (~480 KiB on disk); the terminal "old" unit
    # must be pruned coherently even though its records straddle segments.
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
        units = {
            str(row[0])
            for row in connection.execute("SELECT causal_unit_id FROM execution_units").fetchall()
        }
        leftover = connection.execute(
            "SELECT COUNT(*) FROM execution_records WHERE causal_unit_id = 'old'"
        ).fetchone()
        deleted = connection.execute(
            """
            SELECT producer, from_sequence, to_sequence, exact, coverage_kind
            FROM execution_coverage WHERE coverage_kind = ?
            ORDER BY producer
            """,
            (str(CoverageKind.DELETED_BY_CAPACITY),),
        ).fetchall()
        segments = connection.execute(
            "SELECT storage_state, record_count FROM execution_segments ORDER BY segment_ordinal"
        ).fetchall()
        assert used_bytes(connection) <= budget.low_water or "old" not in units
    finally:
        connection.close()
    assert "old" not in units
    assert leftover == (0,)
    assert deleted
    assert all(int(row[3]) == 1 for row in deleted)
    assert all(str(row[4]) == CoverageKind.DELETED_BY_CAPACITY for row in deleted)
    assert any(
        int(row[1]) > 0 or str(row[0]) == SegmentStorageState.PRUNED_SUMMARY for row in segments
    )


def test_forced_terminal_when_nothing_terminal(tmp_path: Path) -> None:
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
        kinds = {
            str(row[0])
            for row in connection.execute("SELECT coverage_kind FROM execution_coverage").fetchall()
        }
        # Hard invariant: never over the total envelope. high_water is the
        # prune trigger and may be exceeded by at most one interval of
        # accrual error (what control_reserve absorbs); an exact check
        # brings it back under.
        assert used_bytes(connection) <= budget.total_bytes
        # Each exact call prunes whole units and stops near low_water using
        # the measured ratio; it converges within a bounded number of calls.
        for _ in range(50):
            if used_bytes(connection) <= budget.high_water:
                break
            connection.execute("BEGIN IMMEDIATE")
            assert enforce_budget(connection, budget, 10_000_000) is True
            connection.execute("COMMIT")
        assert used_bytes(connection) <= budget.high_water
    finally:
        connection.close()
    assert "live-0" not in remaining
    assert CoverageKind.DELETED_BY_CAPACITY in kinds


def test_storage_unavailable_when_nothing_prunable(tmp_path: Path) -> None:
    # Empty execution_* b-trees already occupy more pages than a 256-byte
    # envelope's high_water, so the first ingest cannot prune its way under
    # the line and must refuse with STORAGE_UNAVAILABLE.
    budget = RetentionBudget(total_bytes=256, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    receipt = store.ingest_batch(
        _batch("tiny", (_record(label="r", unit="u", seq=0, observed=1, payload="x"),))
    )
    assert receipt.storage_state is StorageState.STORAGE_UNAVAILABLE
    assert receipt.accepted == 0
    connection = _open(path)
    try:
        records = connection.execute("SELECT COUNT(*) FROM execution_records").fetchone()
        kinds = {
            str(row[0])
            for row in connection.execute("SELECT coverage_kind FROM execution_coverage").fetchall()
        }
        batches = connection.execute("SELECT COUNT(*) FROM execution_batches").fetchone()
    finally:
        connection.close()
    assert records == (0,)
    assert CoverageKind.STORAGE_UNAVAILABLE in kinds
    assert batches == (1,)


def test_used_bytes_matches_dbstat_and_exceeds_payload(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    connection = _open(path)
    try:
        empty = used_bytes(connection)
        assert empty == _dbstat_execution_sum(connection)
        assert empty > 0
        assert empty < DISK_BUDGET
    finally:
        connection.close()

    count = 300
    receipt = store.ingest_batch(
        _batch(
            "pin",
            tuple(
                _record(
                    label=f"p{index}",
                    unit="u",
                    seq=index,
                    observed=10 + index,
                    payload=PAYLOAD_BLOB,
                )
                for index in range(count)
            ),
        )
    )
    assert receipt.accepted == count
    connection = _open(path)
    try:
        occupied = used_bytes(connection)
        payload = connection.execute(
            "SELECT COALESCE(SUM(payload_bytes), 0) FROM execution_records"
        ).fetchone()
        assert occupied == _dbstat_execution_sum(connection)
        assert int(payload[0]) > 0
        assert occupied >= int(payload[0]) * 2
    finally:
        connection.close()


def test_coarsening_yields_unknown_without_widening_exact_rows(tmp_path: Path) -> None:
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=HORIZON, coverage_rows_per_epoch=3)
    store, path = _store(tmp_path, budget)
    store.ingest_batch(
        _batch("seed", (_record(label="s", unit="u", seq=0, observed=1, payload="a"),))
    )
    connection = _open(path)
    try:
        from backend.app.edge_db.connection import write_transaction
        from backend.app.features.diagnostics.prune import coarsen_coverage

        with write_transaction(connection):
            for index in range(6):
                insert_coverage(
                    connection,
                    camera_id=CAMERA,
                    worker_boot_id=BOOT,
                    source_generation=0,
                    stream_epoch=0,
                    kind=CoverageKind.MISSING_NOT_RECORDED,
                    producer="sdk",
                    from_sequence=index,
                    to_sequence=index,
                    from_ns=100 + index,
                    to_ns=100 + index,
                    record_count=1,
                    exact=True,
                    cause="gap",
                    recorded_at_ns=index,
                )
            coarsen_coverage(connection, budget.coverage_rows_per_epoch, 99)
        rows = connection.execute(
            """
            SELECT coverage_kind, exact, from_ns, to_ns, record_count
            FROM execution_coverage
            WHERE coverage_kind IN (?, ?)
            ORDER BY from_ns
            """,
            (str(CoverageKind.UNKNOWN_COARSENED), str(CoverageKind.MISSING_NOT_RECORDED)),
        ).fetchall()
    finally:
        connection.close()
    coarsened = [row for row in rows if str(row[0]) == CoverageKind.UNKNOWN_COARSENED]
    exact = [row for row in rows if str(row[0]) == CoverageKind.MISSING_NOT_RECORDED]
    assert coarsened
    assert all(int(row[1]) == 0 for row in coarsened)
    assert all(int(row[1]) == 1 for row in exact)
    assert len(coarsened) + len(exact) <= 3


def test_enforce_prunes_a_bounded_number_of_units_per_call_and_converges(
    tmp_path: Path,
) -> None:
    """Live regression: a 4x-over backlog made one ingest request prune for
    minutes on the event loop. Each enforce call now prunes at most
    MAX_UNITS_PER_ENFORCE whole units, commits (progress was made), and the
    envelope converges over the following calls."""
    from backend.app.features.diagnostics.retention import (
        MAX_UNITS_PER_ENFORCE,
        enforce_budget,
    )

    budget = RetentionBudget(total_bytes=DISK_BUDGET, unit_horizon_ns=HORIZON)
    _store_unused, path = _store(tmp_path, budget)
    # Forty small terminal units, well over high_water together.
    for unit in range(40):
        records = tuple(
            _record(
                label=f"u{unit}-{index}",
                unit=f"unit-{unit:02d}",
                seq=unit * 100 + index,
                observed=10 + unit * 5 + index,
                payload=PAYLOAD_BLOB,
            )
            for index in range(30)
        )
        # Ingest with an enormous budget so nothing is pruned while filling.
        # The receive clock starts after every observed time: a batch is
        # received no earlier than its records were observed, as on a real edge.
        big = ExecutionRecordStore(
            lambda: _open(path),
            RetentionBudget(total_bytes=1 << 40),
            clock=_Clock(now=10_000 + unit),
        )
        big.ingest_batch(_batch(f"fill-{unit}", records))
    connection = _open(path)
    try:
        before = connection.execute("SELECT COUNT(*) FROM execution_units").fetchone()[0]
        assert used_bytes(connection) > budget.high_water
        now = 10 + 40 * 5 + HORIZON + 1
        connection.execute("BEGIN IMMEDIATE")
        ok = enforce_budget(connection, budget, now)
        connection.execute("COMMIT")
        after_one = connection.execute("SELECT COUNT(*) FROM execution_units").fetchone()[0]
        assert ok is True  # progress made, ingest may commit
        assert before - after_one <= MAX_UNITS_PER_ENFORCE
        assert before - after_one >= 1
        # Keep calling: it must converge to <= high_water without ever pruning
        # more than the bound in one call.
        calls = 0
        while used_bytes(connection) > budget.high_water and calls < 100:
            prior = connection.execute("SELECT COUNT(*) FROM execution_units").fetchone()[0]
            connection.execute("BEGIN IMMEDIATE")
            assert enforce_budget(connection, budget, now) is True
            connection.execute("COMMIT")
            now_units = connection.execute("SELECT COUNT(*) FROM execution_units").fetchone()[0]
            assert prior - now_units <= MAX_UNITS_PER_ENFORCE
            calls += 1
        assert used_bytes(connection) <= budget.high_water
        assert calls < 100  # converged, never by more than the bound per call
        # Batch receipts older than the oldest surviving record are gone;
        # receipts of surviving records are kept (exactness of the orphan drop).
        oldest_observed = connection.execute(
            "SELECT MIN(observed_at_ns) FROM execution_records WHERE camera_id = ?", (CAMERA,)
        ).fetchone()[0]
        stale = connection.execute(
            "SELECT COUNT(*) FROM execution_batches WHERE received_at_ns < ?", (oldest_observed,)
        ).fetchone()[0]
        assert stale == 0
        # Every receipt that still has a record is kept.
        live = connection.execute(
            """
            SELECT COUNT(*) FROM execution_batches b
            WHERE EXISTS (SELECT 1 FROM execution_records r
                          WHERE r.camera_id = b.camera_id AND r.committed_at_ns = b.received_at_ns)
            """
        ).fetchone()[0]
        assert live >= 1
    finally:
        connection.close()


def test_newer_boot_is_decided_by_observation_time_not_boot_id_text(tmp_path: Path) -> None:
    """Live regression: boot ids are UUIDs. The worker restarted with a boot id
    that sorted LOWER than the dead boot's, the string comparison called the
    live boot "older", every new unit was marked terminal on arrival and pruned
    first, and 5,300 dead-boot units were kept. Newest is by observation."""
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    dead_boot, live_boot = "f0437fa1-dead", "91a6a31d-live"  # live sorts lower
    assert live_boot < dead_boot
    store.ingest_batch(
        _batch(
            "dead",
            (_record(label="d0", unit="dead-u", seq=0, observed=10, payload="a", boot=dead_boot),),
            boot=dead_boot,
        )
    )
    store.ingest_batch(
        _batch(
            "live",
            (
                _record(
                    label="l0", unit="live-u", seq=0, observed=1_000, payload="b", boot=live_boot
                ),
            ),
            boot=live_boot,
        )
    )
    connection = _open(path)
    try:
        rows = dict(
            connection.execute("SELECT causal_unit_id, terminal FROM execution_units").fetchall()
        )
    finally:
        connection.close()
    assert rows["dead-u"] == 1, "the superseded boot's unit is terminal"
    assert rows["live-u"] == 0, "the live boot's unit must stay open"


def _unit_states(connection) -> dict[str, tuple[int, str]]:
    return {
        str(row[0]): (int(row[1]), str(row[2]))
        for row in connection.execute(
            "SELECT causal_unit_id, terminal, causal_state FROM execution_units"
        )
    }


def test_issue577_known_gap_survives_dense_watermark_closure(tmp_path: Path) -> None:
    """Dense units close from the lane watermark. An overlapping known gap is
    INCOMPLETE_KNOWN, and a unit already marked known is not rewritten COMPLETE."""
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    stamps = (0, 1, 2, 50_000)
    store.ingest_batch(
        _batch(
            "gaps",
            tuple(
                _record(
                    label=f"g{stamp}",
                    unit=f"gap-{stamp}",
                    seq=index,
                    observed=stamp,
                    payload="a",
                )
                for index, stamp in enumerate(stamps)
            ),
        )
    )
    connection = _open(path)
    try:
        connection.execute(
            "UPDATE execution_units SET terminal = 0, causal_state = ? WHERE causal_unit_id = ?",
            (str(UnitCausalState.INCOMPLETE_UNKNOWN), "gap-0"),
        )
        connection.execute(
            "UPDATE execution_units SET terminal = 0, causal_state = ? WHERE causal_unit_id = ?",
            (str(UnitCausalState.INCOMPLETE_KNOWN), "gap-1"),
        )
        insert_coverage(
            connection,
            camera_id=CAMERA,
            worker_boot_id=BOOT,
            source_generation=0,
            stream_epoch=0,
            kind=CoverageKind.MISSING_NOT_RECORDED,
            producer="sdk",
            from_sequence=0,
            to_sequence=0,
            from_ns=0,
            to_ns=0,
            record_count=1,
            exact=True,
            cause="drop",
            recorded_at_ns=1,
        )
        refresh_unit_terminals(connection, HORIZON)
        states = _unit_states(connection)
    finally:
        connection.close()
    assert states["gap-0"] == (1, UnitCausalState.INCOMPLETE_KNOWN)
    assert states["gap-1"] == (1, UnitCausalState.INCOMPLETE_KNOWN)
    assert states["gap-50000"][0] == 0
    assert states["gap-50000"][1] != UnitCausalState.COMPLETE


def test_issue577_epoch_closure_preserves_unfinished_uncertainty(tmp_path: Path) -> None:
    """A known gap does not exclude additional loss at an abrupt epoch end."""
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=HORIZON)
    store, path = _store(tmp_path, budget)
    store.ingest_batch(
        _batch("old", (_record(label="old", unit="old-known", seq=0, observed=10, payload="a"),))
    )
    store.ingest_batch(
        _batch(
            "new",
            (
                _record(
                    label="new",
                    unit="new-open",
                    seq=0,
                    observed=10 + HORIZON * 10,
                    payload="b",
                    epoch=1,
                ),
            ),
        )
    )
    connection = _open(path)
    try:
        connection.execute(
            "UPDATE execution_units SET terminal = 0, causal_state = ? WHERE causal_unit_id = ?",
            (str(UnitCausalState.INCOMPLETE_KNOWN), "old-known"),
        )
        refresh_unit_terminals(connection, HORIZON)
        states = _unit_states(connection)
    finally:
        connection.close()
    assert states["old-known"] == (1, UnitCausalState.INCOMPLETE_UNKNOWN)
    assert states["new-open"][0] == 0
    assert states["new-open"][1] != UnitCausalState.COMPLETE


def test_issue577_pressure_chooses_globally_oldest_not_camera_name(tmp_path: Path) -> None:
    """cam-a sorts first but was observed later. Pressure must terminal the
    quieter cam-z unit without claiming the forced closure is fully known."""
    horizon = 100_000
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=horizon)
    store, path = _store(tmp_path, budget)
    store.ingest_batch(
        _batch(
            "early-camera",
            (
                _record(
                    label="start",
                    unit="early-start",
                    seq=0,
                    observed=10,
                    payload="a",
                    camera="cam-a",
                ),
                _record(
                    label="recent",
                    unit="early-start",
                    seq=1,
                    observed=9_000,
                    payload="a",
                    camera="cam-a",
                ),
            ),
            camera="cam-a",
        )
    )
    store.ingest_batch(
        _batch(
            "quiet-camera",
            (
                _record(
                    label="quiet",
                    unit="quiet",
                    seq=0,
                    observed=5_000,
                    payload="b",
                    camera="cam-z",
                ),
            ),
            camera="cam-z",
        )
    )
    assert "cam-a" < "cam-z"
    connection = _open(path)
    try:
        refresh_unit_terminals(connection, horizon)
        opened = _unit_states(connection)
        assert opened["early-start"][0] == 0
        assert opened["quiet"][0] == 0
        connection.execute(
            "UPDATE execution_units SET causal_state = ? WHERE causal_unit_id = ?",
            (str(UnitCausalState.INCOMPLETE_KNOWN), "quiet"),
        )
        closed = force_oldest_units_terminal(connection, 1)
        states = _unit_states(connection)
    finally:
        connection.close()
    assert closed == 1
    assert states["quiet"] == (1, UnitCausalState.INCOMPLETE_UNKNOWN)
    assert states["early-start"][0] == 0


def test_issue577_pressure_tie_breaks_on_unit_id_not_camera_name(tmp_path: Path) -> None:
    """Equal observation times break ties by causal_unit_id. cam-a must not win
    just because its name sorts first."""
    horizon = 100_000
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=horizon)
    store, path = _store(tmp_path, budget)
    store.ingest_batch(
        _batch(
            "lexical",
            (
                _record(
                    label="m",
                    unit="unit-m",
                    seq=0,
                    observed=100,
                    payload="a",
                    camera="cam-a",
                ),
            ),
            camera="cam-a",
        )
    )
    store.ingest_batch(
        _batch(
            "older-id",
            (
                _record(
                    label="a",
                    unit="unit-a",
                    seq=0,
                    observed=100,
                    payload="b",
                    camera="cam-z",
                ),
            ),
            camera="cam-z",
        )
    )
    assert "cam-a" < "cam-z"
    assert "unit-a" < "unit-m"
    connection = _open(path)
    try:
        closed = force_oldest_units_terminal(connection, 1)
        states = _unit_states(connection)
    finally:
        connection.close()
    assert closed == 1
    assert states["unit-a"] == (1, UnitCausalState.INCOMPLETE_UNKNOWN)
    assert states["unit-m"][0] == 0


def _spanning_lane(tmp_path: Path) -> tuple[Path, int, int]:
    budget = RetentionBudget(total_bytes=2**20, unit_horizon_ns=10**15)
    store, path = _store(tmp_path, budget)
    old_count, live_count = 200, 10
    old_records = tuple(
        _record(
            label=f"o{index}",
            unit="old",
            seq=index,
            observed=10 + index,
            payload=PAYLOAD_BLOB,
        )
        for index in range(old_count)
    )
    live_records = tuple(
        _record(
            label=f"l{index}",
            unit="live",
            seq=old_count + index,
            observed=50_000 + index,
            payload=PAYLOAD_BLOB,
        )
        for index in range(live_count)
    )
    store.ingest_batch(_batch("span", old_records + live_records))
    return path, old_count, live_count


def _segment_members(connection) -> dict[int, tuple[str, dict[str, int]]]:
    rows = connection.execute(
        """
        SELECT s.segment_id, s.storage_state, r.causal_unit_id, COUNT(*)
        FROM execution_segments AS s
        JOIN execution_records AS r ON r.segment_id = s.segment_id
        GROUP BY s.segment_id, s.storage_state, r.causal_unit_id
        """
    ).fetchall()
    segments: dict[int, tuple[str, dict[str, int]]] = {}
    for segment_id, state, unit_id, count in rows:
        current = segments.get(int(segment_id))
        units = {} if current is None else current[1]
        units[str(unit_id)] = int(count)
        segments[int(segment_id)] = (str(state), units)
    return segments


def test_issue577_segment_seal_uses_membership_not_unrelated_open_unit(tmp_path: Path) -> None:
    """A pending segment whose own units are terminal becomes final even while
    a later segment on the lane still has an open unit. Sealing drops no rows."""
    path, old_count, live_count = _spanning_lane(tmp_path)
    connection = _open(path)
    try:
        before = _segment_members(connection)
        only_old = [
            segment_id for segment_id, (_state, units) in before.items() if set(units) == {"old"}
        ]
        with_live = [segment_id for segment_id, (state, units) in before.items() if "live" in units]
        assert only_old
        assert with_live
        assert sum(units.get("old", 0) for _state, units in before.values()) == old_count
        connection.execute(
            "UPDATE execution_units SET terminal = 1, causal_state = ? WHERE causal_unit_id = ?",
            (str(UnitCausalState.COMPLETE), "old"),
        )
        seal_final_segments(connection)
        after = _segment_members(connection)
        live = connection.execute(
            "SELECT terminal, causal_state FROM execution_units WHERE causal_unit_id = ?",
            ("live",),
        ).fetchone()
    finally:
        connection.close()
    for segment_id in only_old:
        assert after[segment_id][0] == SegmentStorageState.SEALED_FINAL
    for segment_id in with_live:
        assert after[segment_id][0] != SegmentStorageState.SEALED_FINAL
    assert sum(units.get("old", 0) for _state, units in after.values()) == old_count
    assert sum(units.get("live", 0) for _state, units in after.values()) == live_count
    assert live == (0, UnitCausalState.INCOMPLETE_UNKNOWN)


def test_issue577_spanning_unit_survives_seal_and_prunes_whole(tmp_path: Path) -> None:
    """Finalizing one segment of a causal unit must not delete that segment's
    rows while another segment still holds the unit. Prune removes every
    segment of the unit together."""
    path, old_count, live_count = _spanning_lane(tmp_path)
    connection = _open(path)
    try:
        connection.execute(
            "UPDATE execution_units SET terminal = 1, causal_state = ? WHERE causal_unit_id = ?",
            (str(UnitCausalState.COMPLETE), "old"),
        )
        seal_final_segments(connection)
        membership = _segment_members(connection)
        old_segments = [
            segment_id for segment_id, (_state, units) in membership.items() if "old" in units
        ]
        final_with_old = [
            segment_id
            for segment_id in old_segments
            if membership[segment_id][0] == SegmentStorageState.SEALED_FINAL
        ]
        still_open = [
            segment_id
            for segment_id in old_segments
            if membership[segment_id][0] != SegmentStorageState.SEALED_FINAL
        ]
        assert final_with_old
        assert still_open
        assert sum(membership[segment_id][1]["old"] for segment_id in old_segments) == old_count
        from backend.app.edge_db.connection import write_transaction
        from backend.app.features.diagnostics.prune import prune_unit

        with write_transaction(connection):
            prune_unit(connection, "old", 1)
        leftover = connection.execute(
            """
            SELECT segment_id, COUNT(*) FROM execution_records
            WHERE causal_unit_id = 'old'
            GROUP BY segment_id
            """
        ).fetchall()
        live = connection.execute(
            "SELECT COUNT(*) FROM execution_records WHERE causal_unit_id = ?",
            ("live",),
        ).fetchone()
        unit = connection.execute(
            "SELECT COUNT(*) FROM execution_units WHERE causal_unit_id = ?",
            ("old",),
        ).fetchone()
    finally:
        connection.close()
    assert leftover == []
    assert live == (live_count,)
    assert unit == (0,)


@pytest.mark.parametrize(
    "kind",
    (CoverageKind.UNKNOWN_COARSENED, CoverageKind.UNKNOWN, CoverageKind.DELETED_BY_CAPACITY),
)
@pytest.mark.parametrize(
    "prior", (UnitCausalState.INCOMPLETE_UNKNOWN, UnitCausalState.INCOMPLETE_KNOWN)
)
def test_issue577_watermark_cannot_erase_unknown_coverage(
    tmp_path: Path, kind: CoverageKind, prior: UnitCausalState
) -> None:
    store, path = _store(tmp_path, RetentionBudget(total_bytes=2**20, unit_horizon_ns=10**15))
    store.ingest_batch(
        _batch(
            "unknown",
            (
                _record(label="before", unit="before", seq=0, observed=10, payload="a"),
                _record(label="watermark", unit="watermark", seq=1, observed=50_000, payload="b"),
            ),
        )
    )
    connection = _open(path)
    try:
        connection.execute(
            "UPDATE execution_units SET causal_state = ? WHERE causal_unit_id = ?",
            (str(prior), "before"),
        )
        insert_coverage(
            connection,
            camera_id=CAMERA,
            worker_boot_id=BOOT,
            source_generation=0,
            stream_epoch=0,
            kind=kind,
            producer=None,
            from_sequence=None,
            to_sequence=None,
            from_ns=10,
            to_ns=10,
            record_count=1,
            exact=kind is CoverageKind.DELETED_BY_CAPACITY,
            cause="test-unknown",
            recorded_at_ns=50_000,
        )
        refresh_unit_terminals(connection, HORIZON)
        states = _unit_states(connection)
        retained = connection.execute(
            "SELECT COUNT(*) FROM execution_coverage WHERE coverage_kind = ?", (str(kind),)
        ).fetchone()
    finally:
        connection.close()
    assert states["before"] == (1, UnitCausalState.INCOMPLETE_UNKNOWN)
    assert states["watermark"][0] == 0
    assert retained == (1,)
