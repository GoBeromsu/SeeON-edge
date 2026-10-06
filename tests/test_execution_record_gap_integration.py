"""Exporter -> wire converter -> real store certainty across split commits."""

from __future__ import annotations

import json
from dataclasses import replace

import psycopg
import pytest
from psycopg import sql

from backend.app.features.diagnostics.prune import prune_unit
from backend.app.features.diagnostics.records import CoverageKind, StorageState, UnitCausalState
from backend.app.features.diagnostics.retention import RetentionBudget, used_bytes
from backend.app.features.diagnostics.store import ExecutionRecordStore
from backend.app.features.diagnostics.wire import ingest_batch_from_wire, wire_receipt_from_store
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.execution_records import (
    MAX_EXECUTION_RECORD_BODY_BYTES,
    WireBatch,
    WireGap,
    WireProvenance,
    WireRecord,
)
from tests_support.postgres_diagnostics_sandbox import DiagnosticsSandbox
from worker.pipeline.diagnostics.exporter import ExecutionRecordExporter
from worker.pipeline.diagnostics.lanes import ExecutionRecordLanes

pytest_plugins = ("tests_support.postgres_diagnostics_sandbox",)

_PROVENANCE = WireProvenance("rev", "image", "model", "cal", "pre", "config", "policy")


def _record(seq, unit, stamp, generation, epoch):
    return WireRecord(
        record_kind="policy.consume",
        camera_id="synthetic",
        worker_boot_id="opaque-boot",
        source_generation=generation,
        stream_epoch=epoch,
        producer="policy",
        producer_sequence=seq,
        observed_at_ns=stamp,
        time_quality="monotonic",
        causal_unit_id=unit,
        outcome="consumed",
        payload={"blob": ""},
    )


def _batch(records=(), gaps=()):
    return WireBatch("synthetic", "opaque-boot", _PROVENANCE, records, gaps)


def _execution_snapshot(diag: DiagnosticsSandbox):
    return {
        table: diag.admin.execute(
            sql.SQL("SELECT * FROM {} ORDER BY 1").format(sql.Identifier(table))
        ).fetchall()
        for table in (
            "execution_provenance",
            "execution_segments",
            "execution_units",
            "execution_records",
            "execution_coverage",
            "execution_batches",
        )
    }


def _unit_states(connection):
    return {
        unit: (terminal, state)
        for unit, terminal, state in connection.execute(
            "SELECT causal_unit_id, terminal, causal_state FROM execution_units"
        ).fetchall()
    }


class _StoreClient:
    def __init__(self, store, diag: DiagnosticsSandbox):
        self.store, self.diag = store, diag
        self.posted = []
        self.old_states = []
        self.fail_gap_once = False

    def post_batch(self, batch):
        # Reparse the actual serialized body rather than sharing DTO objects.
        decoded = WireBatch.from_json(json.loads(batch.encode()))
        self.posted.append(decoded)
        if decoded.gaps and self.fail_gap_once:
            self.fail_gap_once = False
            return DeliveryFailure(DeliveryDisposition.RETRY, "NETWORK")
        result = self.store.ingest_batch(
            ingest_batch_from_wire(decoded, backend_build_revision="api-rev")
        )
        self.old_states.append(
            self.diag.admin.execute(
                "SELECT terminal, causal_state FROM execution_units WHERE causal_unit_id = %s",
                ("old",),
            ).fetchone()
        )
        return wire_receipt_from_store(result)


@pytest.fixture
def stack(postgres_diagnostics_sandbox: DiagnosticsSandbox):
    diag = postgres_diagnostics_sandbox
    # This fixture must admit a cap-full record; it is NOT a deployment budget.
    store = ExecutionRecordStore(
        diag.database,
        RetentionBudget(
            total_bytes=512 * 1024 * 1024, unit_horizon_ns=100, coverage_rows_per_epoch=1
        ),
    )
    client = _StoreClient(store, diag)
    lanes = ExecutionRecordLanes(lane_capacity=8)
    exporter = ExecutionRecordExporter(
        lanes=lanes, client=client, provenance=_PROVENANCE, batch_max=8, flush_ms=50
    )
    return diag, client, lanes, exporter


@pytest.mark.parametrize("generation,epoch", [(0, 0), (7, 11)])
@pytest.mark.parametrize("fail_gap", [False, True])
def test_split_loss_commits_before_later_terminal_watermark(stack, generation, epoch, fail_gap):
    diag, client, lanes, exporter = stack
    assert lanes.try_emit(_record(0, "old", 10, generation, epoch))
    exporter.flush_once()
    assert lanes.try_emit(_record(1, "old", 10, generation, epoch))
    lost = lanes.drain_for("synthetic", "opaque-boot", limit=8)
    assert lost is not None
    lanes.note_export_failure(lost)
    future = _record(2, "future", 1000, generation, epoch)
    padding = MAX_EXECUTION_RECORD_BODY_BYTES - len(_batch((future,)).encode())
    future = replace(future, payload={"blob": "x" * padding})
    assert len(_batch((future,)).encode()) == MAX_EXECUTION_RECORD_BODY_BYTES
    client.fail_gap_once = fail_gap
    assert lanes.try_emit(future)
    before_gap_attempt = _execution_snapshot(diag)
    exporter.flush_once()
    split = client.posted[1:]
    assert split[0].gaps and not split[0].records
    if fail_gap:
        assert len(split) == 1
        assert _execution_snapshot(diag) == before_gap_attempt
        assert client.old_states[-1][0] == 0
        exporter.flush_once()
        split = client.posted[1:]
        assert len(split) == 3
        assert split[0] == split[1]
        committed_split = split[1:]
    else:
        assert len(split) == 2
        committed_split = split
    assert committed_split[0].gaps and not committed_split[0].records
    assert committed_split[1].records == (future,) and not committed_split[1].gaps
    assert not lanes.cameras_with_work()
    assert client.old_states[-1] == (1, UnitCausalState.INCOMPLETE_KNOWN)
    gaps = diag.admin.execute(
        "SELECT source_generation, stream_epoch, exact, record_count FROM execution_coverage"
    ).fetchall()
    assert gaps == [(generation, epoch, 1, 1)]
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


def test_legacy_unscoped_loss_is_unknown_even_after_coarsening(stack):
    diag, client, lanes, exporter = stack
    # Legacy wire has no scope keys and must never borrow the next record's scope.
    for seq in (0, 1):
        gap = WireGap("policy", seq, seq, 10, 10, 1, "export-failed")
        assert "source_generation" not in gap.to_json()
        client.post_batch(_batch(gaps=(gap,)))
    assert lanes.try_emit(_record(2, "old", 10, 7, 11))
    assert lanes.try_emit(_record(3, "future", 1000, 7, 11))
    exporter.flush_once()
    assert client.old_states[-1] == (1, UnitCausalState.INCOMPLETE_UNKNOWN)
    gaps = diag.admin.execute(
        "SELECT coverage_kind, exact, record_count, cause FROM execution_coverage"
    ).fetchall()
    assert gaps == [("UNKNOWN_COARSENED", 0, 2, "scope-unresolved")]


def test_late_loss_never_upgrades_forced_unknown_terminal(stack):
    diag, client, lanes, exporter = stack
    assert lanes.try_emit(_record(0, "old", 10, 7, 11))
    exporter.flush_once()
    diag.database.transact(
        lambda connection: (
            connection.execute(
                "UPDATE execution_units SET terminal = 1, causal_state = 'INCOMPLETE_UNKNOWN'"
            ).rowcount
        )
    )
    assert lanes.try_emit(_record(1, "old", 10, 7, 11))
    dropped = lanes.drain_for("synthetic", "opaque-boot", limit=8)
    assert dropped is not None
    lanes.note_export_failure(dropped)
    exporter.flush_once()
    assert client.old_states[-1] == (1, UnitCausalState.INCOMPLETE_UNKNOWN)


def test_delayed_loss_downgrades_previously_complete_unit(stack):
    _diag, client, lanes, exporter = stack
    assert lanes.try_emit(_record(0, "old", 10, 0, 0))
    exporter.flush_once()
    assert lanes.try_emit(_record(1, "old", 10, 0, 0))
    dropped = lanes.drain_for("synthetic", "opaque-boot", limit=8)
    assert dropped is not None
    lanes.note_export_failure(dropped)
    # The watermark committed on a different request while loss was in flight.
    client.post_batch(_batch((_record(2, "future", 1000, 0, 0),)))
    assert client.old_states[-1] == (1, UnitCausalState.COMPLETE)
    exporter.flush_once()
    assert client.old_states[-1] == (1, UnitCausalState.INCOMPLETE_KNOWN)


def test_storage_unavailable_receipt_retains_loss_until_committed():
    from shared.events.execution_records import WireBatchReceipt

    class RefuseFirst:
        def __init__(self):
            self.posted = []

        def post_batch(self, batch):
            self.posted.append(batch)
            if len(self.posted) == 1:
                return WireBatchReceipt(batch.batch_id, 0, 0, (), "STORAGE_UNAVAILABLE", 1)
            return WireBatchReceipt(batch.batch_id, 0, 0, (), "committed", 2)

    client = RefuseFirst()
    lanes = ExecutionRecordLanes(lane_capacity=4)
    exporter = ExecutionRecordExporter(
        lanes=lanes, client=client, provenance=_PROVENANCE, batch_max=4, flush_ms=50
    )
    assert lanes.try_emit(_record(0, "old", 10, 7, 11))
    exporter.flush_once()
    assert exporter.failures()[-1].code == "STORAGE_UNAVAILABLE"
    assert not exporter.receipts()
    exporter.flush_once()
    assert len(exporter.receipts()) == 1
    assert not client.posted[1].records
    assert sum(gap.record_count for gap in client.posted[1].gaps) == 1


@pytest.mark.parametrize("refusal_attempts", [2, 17])
def test_gap_only_capacity_refusal_recovers_same_id_exactly_once(stack, refusal_attempts):
    diag, client, _lanes, _exporter = stack
    sufficient_budget = client.store.budget
    refusal_budget = replace(sufficient_budget, total_bytes=256)
    committed_at_ns = (refusal_attempts + 1) * 100
    ticks = iter(range(100, committed_at_ns + 1, 100))
    client.store = ExecutionRecordStore(diag.database, refusal_budget, clock=lambda: next(ticks))
    batch = _batch(gaps=(WireGap("policy", 1, 1, 10, 10, 1, "export-failed", 7, 11),))

    for attempt in range(1, refusal_attempts + 1):
        refused_at_ns = attempt * 100
        refused = client.post_batch(batch)
        assert refused.batch_id == batch.batch_id
        assert refused.storage_state == StorageState.STORAGE_UNAVAILABLE
        assert (refused.accepted, refused.duplicates, refused.rejected) == (0, 0, ())
        assert refused.committed_at_ns == refused_at_ns
        refused_rows = _execution_snapshot(diag)
        for table in (
            "execution_provenance",
            "execution_segments",
            "execution_units",
            "execution_records",
        ):
            assert refused_rows[table] == []
        assert len(refused_rows["execution_batches"]) == 1
        assert len(refused_rows["execution_coverage"]) == refusal_budget.coverage_rows_per_epoch
        # The refusal envelope cannot fit in this deliberately tiny budget.
        assert used_bytes(diag.admin) > refusal_budget.high_water
        coverage = diag.admin.execute(
            """
            SELECT camera_id, worker_boot_id, source_generation, stream_epoch,
                   coverage_kind, producer, from_sequence, to_sequence,
                   from_ns, to_ns, record_count, exact, cause, recorded_at_ns
            FROM execution_coverage ORDER BY coverage_id
            """
        ).fetchall()
        assert coverage == [
            (
                batch.camera_id,
                batch.worker_boot_id,
                0,
                0,
                CoverageKind.STORAGE_UNAVAILABLE
                if attempt == 1
                else CoverageKind.UNKNOWN_COARSENED,
                None,
                None,
                None,
                100,
                refused_at_ns,
                0,
                0,
                "capacity" if attempt == 1 else "coarsened",
                refused_at_ns,
            )
        ]
        receipt_row = diag.admin.execute(
            """
            SELECT batch_id, received_at_ns, accepted_records, duplicate_records,
                   rejected_records, receipt FROM execution_batches
            """
        ).fetchone()
        assert receipt_row[:5] == (batch.batch_id, refused_at_ns, 0, 0, 0)
        assert json.loads(receipt_row[5]) == refused.to_json()

    client.store.budget = sufficient_budget
    committed = client.post_batch(batch)
    assert committed == replace(
        refused, storage_state=str(StorageState.COMMITTED), committed_at_ns=committed_at_ns
    )
    gaps = diag.admin.execute(
        """
        SELECT source_generation, stream_epoch, producer, from_sequence,
               to_sequence, from_ns, to_ns, record_count, exact, cause
        FROM execution_coverage WHERE coverage_kind = %s
        """,
        (str(CoverageKind.MISSING_NOT_RECORDED),),
    ).fetchall()
    assert gaps == [(7, 11, "policy", 1, 1, 10, 10, 1, 1, "export-failed")]
    receipt_row = diag.admin.execute(
        """
        SELECT batch_id, received_at_ns, accepted_records, duplicate_records,
               rejected_records, receipt FROM execution_batches
        """
    ).fetchone()
    assert receipt_row[:5] == (batch.batch_id, committed_at_ns, 0, 0, 0)
    assert json.loads(receipt_row[5]) == committed.to_json()
    committed_rows = _execution_snapshot(diag)
    assert len(committed_rows["execution_batches"]) == 1
    assert len(committed_rows["execution_provenance"]) == 1
    # One refusal control row and one actual scoped gap row, in separate lanes.
    assert len(committed_rows["execution_coverage"]) == 2
    for row in refused_rows["execution_coverage"]:
        assert row in committed_rows["execution_coverage"]
    for table in ("execution_segments", "execution_units", "execution_records"):
        assert committed_rows[table] == []

    # A new store must return the durable commit even under renewed pressure.
    client.store = ExecutionRecordStore(
        diag.database, refusal_budget, clock=lambda: committed_at_ns + 100
    )
    assert client.post_batch(batch) == committed
    assert client.posted == [batch] * (refusal_attempts + 2)
    assert _execution_snapshot(diag) == committed_rows


def test_gap_only_retry_error_rolls_back_to_durable_refusal(stack, monkeypatch):
    from backend.app.features.diagnostics import store as store_module

    diag, client, _lanes, _exporter = stack
    sufficient_budget = client.store.budget
    ticks = iter((100, 200, 300))
    client.store = ExecutionRecordStore(
        diag.database, replace(sufficient_budget, total_bytes=256), clock=lambda: next(ticks)
    )
    batch = _batch(gaps=(WireGap("policy", 1, 1, 10, 10, 1, "export-failed", 7, 11),))
    refused = client.post_batch(batch)
    assert refused.storage_state == StorageState.STORAGE_UNAVAILABLE
    before = _execution_snapshot(diag)
    client.store.budget = sufficient_budget
    real_enforce_budget = store_module.enforce_budget

    def fail_after_enforcement(connection, budget, now_ns, *, meter):
        assert real_enforce_budget(connection, budget, now_ns, meter=meter)
        receipt_text = connection.execute(
            "SELECT receipt FROM execution_batches WHERE batch_id = %s", (batch.batch_id,)
        ).fetchone()[0]
        assert json.loads(receipt_text)["storage_state"] == StorageState.COMMITTED
        assert connection.execute(
            "SELECT SUM(record_count) FROM execution_coverage WHERE coverage_kind = %s",
            (str(CoverageKind.MISSING_NOT_RECORDED),),
        ).fetchone() == (1,)
        raise psycopg.DataError("injected failure after capacity enforcement")

    with monkeypatch.context() as patch:
        patch.setattr(store_module, "enforce_budget", fail_after_enforcement)
        with pytest.raises(psycopg.DataError, match="injected failure"):
            client.post_batch(batch)
    # The refusal, including its original timestamp, and all six tables survive.
    assert _execution_snapshot(diag) == before
    committed = client.post_batch(batch)
    assert committed == replace(
        refused, storage_state=str(StorageState.COMMITTED), committed_at_ns=300
    )
    assert diag.admin.execute(
        """
        SELECT COUNT(*), SUM(record_count) FROM execution_coverage
        WHERE coverage_kind = %s
        """,
        (str(CoverageKind.MISSING_NOT_RECORDED),),
    ).fetchone() == (1, 1)


@pytest.mark.parametrize("survivor_span", [(20, 40), (10, 20)])
def test_contiguous_prune_extension_downgrades_terminal_in_same_transaction(stack, survivor_span):
    diag, client, _lanes, _exporter = stack
    first_ns, last_ns = survivor_span
    records = (
        replace(_record(0, "pruned-0", 0, 7, 11), producer="p"),
        replace(_record(1, "pruned-1", 30, 7, 11), producer="p"),
        _record(0, "survivor", first_ns, 7, 11),
        _record(1, "survivor", last_ns, 7, 11),
        _record(2, "previously-unknown", 20, 7, 11),
        _record(3, "previously-unknown", 40, 7, 11),
        _record(4, "outside-interval", 50, 7, 11),
        _record(5, "watermark", 1000, 7, 11),
    )
    controls = (("other-generation", 8, 11), ("other-epoch", 7, 12))
    for unit, generation, epoch in controls:
        records += (
            _record(0, unit, 20, generation, epoch),
            _record(1, unit, 40, generation, epoch),
            _record(2, f"{unit}-watermark", 1000, generation, epoch),
        )
    assert client.post_batch(_batch(records)).accepted == len(records)
    expected_states = dict.fromkeys(
        (
            "pruned-0",
            "pruned-1",
            "survivor",
            "previously-unknown",
            "outside-interval",
            "other-generation",
            "other-epoch",
        ),
        (1, UnitCausalState.COMPLETE),
    )
    expected_states.update(
        dict.fromkeys(
            ("watermark", "other-generation-watermark", "other-epoch-watermark"),
            (0, UnitCausalState.INCOMPLETE_UNKNOWN),
        )
    )
    assert _unit_states(diag.admin) == expected_states
    diag.database.transact(
        lambda connection: (
            connection.execute(
                "UPDATE execution_units SET causal_state = %s WHERE causal_unit_id = %s",
                (str(UnitCausalState.INCOMPLETE_UNKNOWN), "previously-unknown"),
            ).rowcount
        )
    )
    expected_states["previously-unknown"] = (1, UnitCausalState.INCOMPLETE_UNKNOWN)
    freed, lane = diag.database.transact(
        lambda connection: prune_unit(connection, "pruned-0", 1100)
    )
    assert freed > 0
    assert lane == ("synthetic", "opaque-boot", 7, 11)
    del expected_states["pruned-0"]
    assert _unit_states(diag.admin) == expected_states
    first_coverage = diag.admin.execute(
        """
        SELECT coverage_id, coverage_kind, producer, from_sequence, to_sequence,
               from_ns, to_ns, record_count, exact FROM execution_coverage
        """
    ).fetchall()
    assert len(first_coverage) == 1
    coverage_id = first_coverage[0][0]
    assert first_coverage[0][1:] == (CoverageKind.DELETED_BY_CAPACITY, "p", 0, 0, 0, 0, 1, 1)
    before_extension = _execution_snapshot(diag)
    extended_states = dict(expected_states)
    del extended_states["pruned-1"]
    extended_states["survivor"] = (1, UnitCausalState.INCOMPLETE_UNKNOWN)

    def prune_then_abort(connection: psycopg.Connection) -> None:
        prune_unit(connection, "pruned-1", 1200)
        assert _unit_states(connection) == extended_states
        raise RuntimeError("abort prune")

    with pytest.raises(RuntimeError, match="abort prune"):
        diag.database.transact(prune_then_abort)
    assert _execution_snapshot(diag) == before_extension

    def prune_and_check(connection: psycopg.Connection) -> None:
        prune_unit(connection, "pruned-1", 1200)
        # No refresh or coarsening is allowed to repair certainty afterward.
        assert _unit_states(connection) == extended_states
        coverage = connection.execute(
            """
            SELECT coverage_id, coverage_kind, producer, from_sequence, to_sequence,
                   from_ns, to_ns, record_count, exact FROM execution_coverage
            """
        ).fetchall()
        assert coverage == [(coverage_id, CoverageKind.DELETED_BY_CAPACITY, "p", 0, 1, 0, 30, 2, 1)]

    diag.database.transact(prune_and_check)
    assert _unit_states(diag.admin) == extended_states
