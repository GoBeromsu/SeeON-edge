"""ExecutionRecordStore semantics against the real diagnostics PostgreSQL schema."""

from __future__ import annotations

import hashlib
import json

from psycopg import sql

from backend.app.features.diagnostics.prune import prune_unit
from backend.app.features.diagnostics.records import (
    AvailabilityKind,
    CoverageKind,
    ExecutionRecordInput,
    GapReport,
    IngestBatch,
    Provenance,
    RecordKind,
    StorageState,
    UnitCausalState,
    late_ack_unit_id,
)
from backend.app.features.diagnostics.retention import RetentionBudget
from backend.app.features.diagnostics.store import ExecutionRecordStore
from tests_support.postgres_diagnostics_sandbox import DiagnosticsSandbox

pytest_plugins = ("tests_support.postgres_diagnostics_sandbox",)

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


def _store(
    diag: DiagnosticsSandbox,
    *,
    total_bytes: int = 2**20,
    clock: _Clock | None = None,
    horizon_ns: int | None = None,
) -> ExecutionRecordStore:
    budget = (
        RetentionBudget(total_bytes=total_bytes)
        if horizon_ns is None
        else RetentionBudget(total_bytes=total_bytes, unit_horizon_ns=horizon_ns)
    )
    return ExecutionRecordStore(diag.database, budget, clock=clock or _Clock())


def _record(
    *,
    label: str,
    unit: str = "unit-a",
    kind: RecordKind = RecordKind.SDK_FRAME,
    seq: int = 0,
    observed: int = 100,
    payload: dict[str, object] | None = None,
    camera: str = CAMERA,
    boot: str = BOOT,
    generation: int = 0,
    epoch: int = 0,
    producer: str = "sdk",
    outcome: str = "ok",
) -> ExecutionRecordInput:
    return ExecutionRecordInput(
        record_id=_hex(label),
        record_kind=kind,
        camera_id=camera,
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
    camera: str = CAMERA,
    boot: str = BOOT,
) -> IngestBatch:
    return IngestBatch(
        batch_id=_hex(label),
        camera_id=camera,
        worker_boot_id=boot,
        provenance=PROVENANCE,
        records=records,
        gaps=gaps,
    )


def _count(diag: DiagnosticsSandbox, table: str) -> int:
    row = diag.admin.execute(
        sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
    ).fetchone()
    assert row is not None
    return int(row[0])


def test_idempotent_batch_replay_returns_identical_receipt(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    diag = postgres_diagnostics_sandbox
    store = _store(diag)
    batch = _batch("b1", (_record(label="r1"),))
    first = store.ingest_batch(batch)
    rows = _count(diag, "execution_records")
    second = store.ingest_batch(batch)
    assert first == second
    assert first.storage_state is StorageState.COMMITTED
    assert first.accepted == 1
    assert _count(diag, "execution_records") == rows
    assert _count(diag, "execution_batches") == 1


def test_duplicate_and_conflict_record_dispositions(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    diag = postgres_diagnostics_sandbox
    store = _store(diag)
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
    payload = diag.admin.execute(
        "SELECT payload FROM execution_records WHERE record_id = %s", (_hex("same"),)
    ).fetchone()
    assert payload is not None and json.loads(str(payload[0])) == {"n": 1}


def test_repeated_record_id_within_batch_keeps_first(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    diag = postgres_diagnostics_sandbox
    store = _store(diag)
    original = _record(label="same", seq=1, payload={"n": 1})
    conflicted = _record(label="same", seq=1, payload={"n": 2})
    receipt = store.ingest_batch(_batch("repeats", (original, original, conflicted)))
    assert receipt.storage_state is StorageState.COMMITTED
    assert receipt.accepted == 1
    assert receipt.duplicates == 1
    assert receipt.rejected == ((_hex("same"), "conflict"),)
    rows = diag.admin.execute(
        "SELECT payload FROM execution_records WHERE record_id = %s", (_hex("same"),)
    ).fetchall()
    assert len(rows) == 1
    assert json.loads(str(rows[0][0])) == {"n": 1}


def test_oversize_rejection_writes_coverage(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    diag = postgres_diagnostics_sandbox
    store = _store(diag, total_bytes=512 * 1024)
    huge = _record(label="huge", payload={"blob": "x" * store.budget.max_record_bytes})
    receipt = store.ingest_batch(_batch("oversize", (huge,)))
    assert receipt.accepted == 0
    assert receipt.rejected == ((_hex("huge"), "oversize"),)
    assert receipt.storage_state is StorageState.COMMITTED
    assert _count(diag, "execution_records") == 0
    kind = diag.admin.execute("SELECT coverage_kind FROM execution_coverage").fetchone()
    assert kind == (CoverageKind.REJECTED_OVERSIZE,)


def test_query_availability_unknown_tails_and_cursor(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    store = _store(postgres_diagnostics_sandbox)
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
                    source_generation=0,
                    stream_epoch=0,
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


def test_restart_reopens_same_data(postgres_diagnostics_sandbox: DiagnosticsSandbox) -> None:
    diag = postgres_diagnostics_sandbox
    store = _store(diag)
    store.ingest_batch(_batch("persist", (_record(label="p1", observed=50),)))
    restarted = ExecutionRecordStore(
        diag.database, RetentionBudget(total_bytes=2**20), clock=_Clock()
    )
    result = restarted.query(CAMERA, 0, 100, limit=10)
    assert len(result.records) == 1
    assert result.records[0].record_id == _hex("p1")


def test_late_ack_uses_new_unit_and_ack_coverage(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    diag = postgres_diagnostics_sandbox
    store = _store(diag)
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
    diag.database.transact(lambda connection: prune_unit(connection, "unit-old", 2))
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
    admin = diag.admin
    units = {
        str(row[0])
        for row in admin.execute("SELECT causal_unit_id FROM execution_units").fetchall()
    }
    kinds = {
        str(row[0])
        for row in admin.execute("SELECT coverage_kind FROM execution_coverage").fetchall()
    }
    stored_unit = admin.execute(
        "SELECT causal_unit_id FROM execution_records WHERE record_id = %s",
        (_hex("ack"),),
    ).fetchone()
    assert "unit-old" not in units
    assert expected_unit in units
    assert CoverageKind.ACK_OBSERVED_PARENT_DELETED in kinds or (
        CoverageKind.ACK_OBSERVED_PARENT_UNKNOWN_COARSENED in kinds
    )
    assert stored_unit == (expected_unit,)


def test_two_late_acks_in_one_batch_share_the_late_unit(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    diag = postgres_diagnostics_sandbox
    store = _store(diag)
    store.ingest_batch(
        _batch(
            "doomed",
            (_record(label="doomed", unit="unit-old", seq=0, observed=10, payload={"k": "z"}),),
        )
    )
    diag.database.transact(lambda connection: prune_unit(connection, "unit-old", 2))
    acks = tuple(
        _record(
            label=f"ack{index}",
            unit="unit-old",
            kind=RecordKind.BACKEND_ACCEPTANCE,
            seq=99 + index,
            observed=10,
            producer="backend",
            outcome="accepted",
        )
        for index in range(2)
    )
    receipt = store.ingest_batch(_batch("acks", acks))
    assert receipt.storage_state is StorageState.COMMITTED
    assert receipt.accepted == 2
    late_unit = late_ack_unit_id("unit-old", _hex("acks"))
    admin = diag.admin
    stored_units = admin.execute(
        "SELECT causal_unit_id FROM execution_records WHERE record_id = ANY(%s)",
        ([_hex("ack0"), _hex("ack1")],),
    ).fetchall()
    assert stored_units == [(late_unit,), (late_unit,)]
    unit_count = admin.execute(
        "SELECT record_count FROM execution_units WHERE causal_unit_id = %s", (late_unit,)
    ).fetchone()
    assert unit_count == (2,)
    ack_rows = admin.execute(
        "SELECT COUNT(*) FROM execution_coverage WHERE coverage_kind = ANY(%s)",
        (
            [
                str(CoverageKind.ACK_OBSERVED_PARENT_DELETED),
                str(CoverageKind.ACK_OBSERVED_PARENT_UNKNOWN_COARSENED),
            ],
        ),
    ).fetchone()
    assert ack_rows == (2,)


def test_availability_is_a_span_between_contiguous_records_not_instants(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    """Live rollout regression: a 120 s window with ~3,000 records painted
    10,241 ranges - each record a zero-length AVAILABLE with UNKNOWN between
    neighbours 3 ms apart. Adjacent producer_sequence in one lane proves nothing
    was lost between two records, so the interval is AVAILABLE. A sequence
    discontinuity without a gap row, or a lane boundary, ends the span."""
    store = _store(postgres_diagnostics_sandbox)
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


def test_availability_lane_boundary_ends_a_span(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    """A new boot restarts producer_sequence at 0; that boundary is not proof of
    continuity even when the timestamps abut, so the two boots are two spans
    (they may still merge if their time ranges touch, which is honest)."""
    store = _store(postgres_diagnostics_sandbox)
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


def _unit_states(diag: DiagnosticsSandbox) -> dict[str, tuple[int, str]]:
    return {
        str(row[0]): (int(row[1]), str(row[2]))
        for row in diag.admin.execute(
            "SELECT causal_unit_id, terminal, causal_state FROM execution_units"
        )
    }


def test_issue577_dense_consecutive_units_close_on_observed_watermark(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    """Successor gaps of 1 ns must not keep a unit open after the lane watermark
    is more than a horizon past its last observation. The immediate-successor
    test leaves dense-0 open; the tail inside the horizon stays open."""
    horizon = 1_000
    diag = postgres_diagnostics_sandbox
    store = _store(diag, horizon_ns=horizon)
    stamps = (0, 1, 2, 3, 48_500, 49_200, 50_000)
    store.ingest_batch(
        _batch(
            "dense",
            tuple(
                _record(label=f"d{stamp}", unit=f"dense-{stamp}", seq=index, observed=stamp)
                for index, stamp in enumerate(stamps)
            ),
        )
    )
    states = _unit_states(diag)
    for stamp in (0, 1, 2, 3, 48_500):
        assert states[f"dense-{stamp}"] == (1, UnitCausalState.COMPLETE), states[f"dense-{stamp}"]
    for stamp in (49_200, 50_000):
        terminal, causal = states[f"dense-{stamp}"]
        assert terminal == 0, states[f"dense-{stamp}"]
        assert causal != UnitCausalState.COMPLETE


def test_issue577_active_unit_stays_open_until_last_observed_passes_horizon(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    """A unit whose first observation is older than the horizon is still in
    flight when its last observation is the lane watermark. Closing it from
    first_observed_ns would claim COMPLETE while records are still arriving."""
    horizon = 1_000
    diag = postgres_diagnostics_sandbox
    store = _store(diag, horizon_ns=horizon)
    store.ingest_batch(
        _batch(
            "active",
            (
                _record(label="start", unit="active", seq=0, observed=0),
                _record(label="middle", unit="middle", seq=1, observed=horizon + 1),
                _record(label="still", unit="active", seq=2, observed=20_000),
            ),
        )
    )
    states = _unit_states(diag)
    assert states["active"] == (0, UnitCausalState.INCOMPLETE_UNKNOWN)
    assert states["middle"] == (1, UnitCausalState.COMPLETE)


def test_issue577_opaque_uuid_boot_does_not_decide_terminal_order(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    """The live boot id sorts first. It stays open. The dead boot's dense early
    unit is COMPLETE from that boot's own watermark, and its tail is unknown
    rather than a fabricated complete claim."""
    horizon = 1_000
    live_boot, dead_boot = "11111111-live", "ffffffff-dead"
    assert live_boot < dead_boot
    diag = postgres_diagnostics_sandbox
    store = _store(diag, horizon_ns=horizon)
    store.ingest_batch(
        _batch(
            "dead",
            (
                _record(label="d0", unit="dead-0", seq=0, observed=0, boot=dead_boot),
                _record(label="d1", unit="dead-1", seq=1, observed=1, boot=dead_boot),
                _record(label="dtail", unit="dead-tail", seq=2, observed=50_000, boot=dead_boot),
            ),
            boot=dead_boot,
        )
    )
    store.ingest_batch(
        _batch(
            "live",
            (_record(label="live", unit="live", seq=0, observed=80_000, boot=live_boot),),
            boot=live_boot,
        )
    )
    states = _unit_states(diag)
    assert states["dead-0"] == (1, UnitCausalState.COMPLETE)
    assert states["dead-1"] == (1, UnitCausalState.COMPLETE)
    assert states["dead-tail"] == (1, UnitCausalState.INCOMPLETE_UNKNOWN)
    assert states["live"][0] == 0
    assert states["live"][1] != UnitCausalState.COMPLETE


def test_issue577_generation_watermark_does_not_close_another_generation(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    """Generation 0's watermark completes its own dense units and does not
    terminal a unit that only exists on generation 9."""
    horizon = 1_000
    diag = postgres_diagnostics_sandbox
    store = _store(diag, horizon_ns=horizon)
    store.ingest_batch(
        _batch(
            "gens",
            (
                _record(label="g0", unit="gen-0", seq=0, observed=0, generation=0),
                _record(label="g1", unit="gen-1", seq=1, observed=1, generation=0),
                _record(label="gtail", unit="gen-tail", seq=2, observed=50_000, generation=0),
                _record(label="gother", unit="gen-other", seq=3, observed=10, generation=9),
            ),
        )
    )
    states = _unit_states(diag)
    assert states["gen-0"] == (1, UnitCausalState.COMPLETE)
    assert states["gen-1"] == (1, UnitCausalState.COMPLETE)
    assert states["gen-tail"][0] == 0
    assert states["gen-tail"][1] != UnitCausalState.COMPLETE
    assert states["gen-other"] == (0, UnitCausalState.INCOMPLETE_UNKNOWN)


def test_issue577_epoch_watermark_does_not_fabricate_complete(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    """A newer epoch closes the previous epoch without calling it complete.
    Dense units the old epoch itself observed past the horizon are COMPLETE.
    A lone older epoch with no internal watermark proof stays unknown."""
    horizon = 1_000
    diag = postgres_diagnostics_sandbox
    store = _store(diag, horizon_ns=horizon)
    store.ingest_batch(
        _batch(
            "epochs",
            (
                _record(label="e0", unit="epoch-0", seq=0, observed=0, epoch=0),
                _record(label="e1", unit="epoch-1", seq=1, observed=1, epoch=0),
                _record(label="etail", unit="epoch-tail", seq=2, observed=50_000, epoch=0),
                _record(label="lone", unit="epoch-lone", seq=3, observed=100, epoch=4),
                _record(label="new", unit="epoch-new", seq=4, observed=80_000, epoch=8),
            ),
        )
    )
    states = _unit_states(diag)
    assert states["epoch-0"] == (1, UnitCausalState.COMPLETE)
    assert states["epoch-1"] == (1, UnitCausalState.COMPLETE)
    assert states["epoch-tail"] == (1, UnitCausalState.INCOMPLETE_UNKNOWN)
    assert states["epoch-lone"] == (1, UnitCausalState.INCOMPLETE_UNKNOWN)
    assert states["epoch-new"][0] == 0
    assert states["epoch-new"][1] != UnitCausalState.COMPLETE
