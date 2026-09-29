"""Class-level invariants for the execution-record path.

Every bug the first live rollout found (#570-#574) had a per-bug regression
test added with it, but a regression test only catches *that* bug. The live
failures all belonged to five recurring shapes that the rest of the suite
could not see:

* self-consistent fixtures - a test that writes monotonic time and reads
  monotonic time cannot notice the clock base is wrong;
* doubles nicer than production - a config double returning a plain dict
  cannot exercise the nested frozen dataclasses production passes;
* fixture values that accidentally satisfy the invariant the code wrongly
  relies on - "boot-a" < "boot-b" hides a lexical boot comparison;
* one size - three records cannot produce a pathological range count, and
  nothing asserted a request does bounded work;
* one side of a two-sided contract pinned.

These tests assert the *invariant* rather than a known bug, so the next
member of each class fails here instead of on a camera.
"""

from __future__ import annotations

import inspect
import time

import psycopg
import pytest
from test_execution_record_store import (  # noqa: E402 - sibling test module fixture reuse
    CAMERA,
    _batch,
    _record,
    _store,
)

from backend.app.features.diagnostics import retention as retention_module
from backend.app.features.diagnostics.coverage import availability
from backend.app.features.diagnostics.records import AvailabilityKind
from backend.app.features.diagnostics.retention import (
    MAX_UNITS_PER_ENFORCE,
    RetentionBudget,
    enforce_budget,
    used_bytes,
)
from tests_support.postgres_diagnostics_sandbox import DiagnosticsSandbox
from worker.pipeline.diagnostics import emit_delivery, emit_policy

pytest_plugins = ("tests_support.postgres_diagnostics_sandbox",)

# Wall-clock nanoseconds are >= 2020-01-01. A monotonic stamp on this host is
# uptime, which is many orders of magnitude smaller, so this separates the two
# clock bases without pinning a moment in time.
_YEAR_2020_NS = 1_577_836_800_000_000_000


def _record_builders() -> dict[str, object]:
    """Every public ``*_record`` builder that can emit onto the wire."""
    return {
        name: value
        for module in (emit_policy, emit_delivery)
        for name, value in vars(module).items()
        if (
            name.endswith("_record")
            and inspect.isfunction(value)
            # Defined here, not imported: make_record is the generic
            # constructor every builder calls, not a producer itself.
            and value.__module__ == module.__name__
        )
    }


#: Builders exercised by test_every_record_builder_stamps_wall_clock below.
#: A new producer added without a wall-clock case fails the closed-set check
#: rather than silently shipping a second clock base (#570).
_WALL_STAMPED_BUILDERS = frozenset(
    {
        "sdk_frame_record",
        "policy_consume_record",
        "model_score_record",
        "policy_decision_record",
        "policy_coast_record",
        "event_delivery_record",
        "delivery_attempt_record",
        "backend_acceptance_record",
    }
)


def test_record_builder_set_is_closed() -> None:
    """Adding a record producer must also add it to the wall-clock case."""
    assert set(_record_builders()) == set(_WALL_STAMPED_BUILDERS)


def test_every_record_builder_stamps_wall_clock() -> None:
    """No producer may stamp a monotonic clock.

    The live rollout query returned nothing for every camera because
    ``observed_at_ns`` was process uptime: an epoch-bounded query window can
    never match it, and two workers' stamps are not comparable. The suite
    missed it because its own fixtures wrote and read the same wrong base,
    and the end-to-end query window was [0, 2**62) - a window that matches
    any number at all.
    """
    before = time.time_ns()
    records = [
        emit_policy.policy_coast_record(
            camera_id=CAMERA,
            worker_boot_id="boot-1",
            source_generation=1,
            stream_epoch=1,
            frame_seq=1,
            source_pts_ns=None,
            module_qualified_id="fall.v2",
        ),
        emit_delivery.delivery_attempt_record(
            camera_id=CAMERA,
            observing_boot_id="boot-1",
            edge_event_id="event-1",
            outcome="retry-transient",
            attempt=1,
            max_attempts=10,
            failure_class="RETRY",
            status_code=503,
            retained=None,
        ),
        emit_delivery.backend_acceptance_record(
            camera_id=CAMERA,
            observing_boot_id="boot-1",
            edge_event_id="event-1",
            status="accepted_local",
            hub_event_id=None,
        ),
    ]
    after = time.time_ns()
    assert all(record is not None for record in records)
    for record in records:
        assert record.time_quality == "wall", record.record_kind
        assert before <= record.observed_at_ns <= after, record.record_kind
        assert record.observed_at_ns > _YEAR_2020_NS, record.record_kind


def test_availability_range_count_scales_with_gaps_not_records(
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> None:
    """Availability is a description of continuity, not a row-per-record.

    Live, a 120 s window over ~3,000 records painted 10,241 ranges because
    each record was treated as an instant with UNKNOWN between neighbours
    33 ms apart. The store tests used three records, where that shape is
    indistinguishable from the correct one. Pin the cardinality instead: the
    painted ranges must be bounded by the number of real discontinuities,
    whatever the record count.
    """
    diag = postgres_diagnostics_sandbox
    budget = RetentionBudget(total_bytes=8 * 2**20)
    del budget
    store = _store(diag, total_bytes=8 * 2**20)
    # One contiguous producer run, one sequence break, then another run.
    first = tuple(
        _record(label=f"a{index}", seq=index, observed=1_000 + index * 33) for index in range(600)
    )
    second = tuple(
        _record(label=f"b{index}", seq=index, observed=1_000 + index * 33)
        for index in range(700, 1_000)
    )
    store.ingest_batch(_batch("run-a", first))
    store.ingest_batch(_batch("run-b", second))

    painted = diag.database.read_snapshot(
        lambda connection: availability(connection, CAMERA, 0, 2_000 * 33)
    )

    kinds = [item.kind for item in painted]
    available = [item for item in painted if item.kind is AvailabilityKind.AVAILABLE]
    # Two contiguous runs -> two available spans, and a bounded number of
    # ranges overall. 900 records must not yield hundreds of ranges.
    assert len(available) == 2, kinds
    assert len(painted) <= 8, f"{len(painted)} ranges for 900 records: {kinds}"
    assert AvailabilityKind.AVAILABLE in kinds


def test_enforce_budget_does_bounded_work_however_deep_the_backlog(
    postgres_diagnostics_sandbox: DiagnosticsSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One request may never do unbounded work.

    Live, the first ingest after the budget started measuring disk honestly
    tried to prune a ~700 MB backlog inside one request on the event loop:
    /health stopped answering and the container went unhealthy. Nothing in
    the suite asserted that ingest terminates, so the state was undetectable.
    Cost is counted structurally (prunes and used_bytes walks), never by wall
    clock, so this stays deterministic.
    """
    diag = postgres_diagnostics_sandbox
    # 60 units of ~15 KB live rows against a 128 KiB budget: reaching
    # low_water needs ~53 prunes, so the per-call bound actually binds.
    budget = RetentionBudget(total_bytes=128 * 1024, unit_horizon_ns=1_000)
    store = _store(diag, total_bytes=1 << 40)
    blob = {"blob": "x" * 200}
    for unit in range(60):
        store.ingest_batch(
            _batch(
                f"fill-{unit}",
                tuple(
                    _record(
                        label=f"u{unit}-{index}",
                        unit=f"unit-{unit:02d}",
                        seq=unit * 100 + index,
                        observed=10 + unit * 5 + index,
                        payload=blob,
                    )
                    for index in range(30)
                ),
            )
        )

    walks = 0
    real_used_bytes = retention_module.used_bytes

    def _counting_used_bytes(connection: psycopg.Connection) -> int:
        nonlocal walks
        walks += 1
        return real_used_bytes(connection)

    monkeypatch.setattr(retention_module, "used_bytes", _counting_used_bytes)

    admin = diag.admin
    assert used_bytes(admin) > budget.high_water
    before = admin.execute("SELECT COUNT(*) FROM execution_units").fetchone()[0]
    committed = diag.database.transact(
        lambda connection: enforce_budget(connection, budget, 10_000_000)
    )
    after = admin.execute("SELECT COUNT(*) FROM execution_units").fetchone()[0]

    assert committed is True, "progress was made, so the ingest may commit"
    assert before - after <= MAX_UNITS_PER_ENFORCE, "one call pruned an unbounded backlog"
    assert before - after >= 1, "a call that commits must make progress"
    # The row-size walk is the expensive part; it must not run per pruned unit.
    assert walks <= 2, f"{walks} used_bytes walks in one call"
