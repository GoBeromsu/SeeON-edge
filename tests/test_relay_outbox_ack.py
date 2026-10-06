"""The relay acknowledges an alert only after its admission COMMIT.

Real PostgreSQL product sandbox, real ``/api/v1/relay/alerts`` route, real
``EdgeIngestClient`` and the contract-exact Hub fixture over loopback HTTP.
Faults are real PostgreSQL errors raised by triggers, the repo's post-COMMIT
``CommitOutcomeUnknown`` seam, or the product fence (``freeze_authority``).
Oracles read committed rows through the sandbox admin connection and Hub
requests through the fixture route ledger, never through the stores under test.

The request-path claim that follows admission (``OutboxDelivery.claim_event``)
is also checked directly as a component on the same sandbox: each delivery
state is reached through accept/claim/finish, and only lease expiry is aged by
a raw UPDATE, because no public API ages a lease without waiting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from backend.app.edge_db.authority import freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.evidence.event_outbox import EventOutbox
from backend.app.features.evidence.outbox_delivery import (
    DeliveryBudget,
    DeliveryOutcome,
    OutboxDelivery,
)
from backend.app.features.evidence.outbox_dispatch import RELAY_OUTBOX_BUDGET
from backend.app.features.evidence.relay_projection import RelayEvent
from tests_support.alert_amplification_runtime import (
    CAMERA_ID,
    FACILITY_ID,
    ServedFixture,
    hub_client,
    relay_client,
)
from tests_support.local_backend_fixture import LocalBackendFixture
from tests_support.postgres_sandbox import ProductSandbox
from tests_support.relay_postgres_runtime import (
    RELAY_HEADERS,
    clear_insert_fault,
    fail_inserts,
    outbox_rows,
    relay_postgres_app,
    row_counts,
)

pytest_plugins = ("tests_support.postgres_sandbox",)

EDGE_EVENT_ID = "00000000-0000-4000-8000-0000000000a4"


def _post_alert(client: TestClient) -> Response:
    return client.post(
        "/api/v1/relay/alerts",
        json={
            "edge_event_id": EDGE_EVENT_ID,
            "event_type": "fall",
            "probability": 0.91,
            "detected_at": "2026-08-16T00:00:00.000Z",
            "camera_id": CAMERA_ID,
            "facility_id": FACILITY_ID,
        },
        headers=RELAY_HEADERS,
    )


def _relay_audit_rows(sandbox: ProductSandbox) -> int:
    row = sandbox.admin.execute(
        "SELECT count(*) FROM audit_events WHERE action = 'relay.alert' AND target_id = %s",
        (EDGE_EVENT_ID,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _hub_sends(fixture: LocalBackendFixture) -> int:
    return sum(
        1
        for route in fixture.route_ledger
        if (route.method, route.path) == ("POST", "/api/v1/events")
    )


def _central_receipt(fixture: LocalBackendFixture) -> dict[str, str]:
    record = fixture.event_for_edge_id(EDGE_EVENT_ID)
    assert record is not None
    return {"status": "accepted", "edge_event_id": EDGE_EVENT_ID, "event_id": record.event_id}


def _restarted_audit_runtime(sandbox: ProductSandbox) -> PostgresAuditRuntime:
    # The same construction as the postgres_audit_runtime fixture: a new process.
    runtime = PostgresAuditRuntime(
        PostgresAuditStore(sandbox.database, sandbox.authority),
        maximum_snapshot_age_sec=10,
        clock=lambda: 0.0,
    )
    assert runtime.verify_once() and runtime.start_session_once()
    return runtime


def _unknown_after_admission_commit(
    transact: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    def transact_then_lose_outcome(body: Any) -> Any:
        result = transact(body)
        if body.__module__ == EventOutbox.__module__:
            raise CommitOutcomeUnknown()
        return result

    return transact_then_lose_outcome


def test_fault_before_commit_is_not_acknowledged_and_retry_commits_once(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> None:
    # Given: the outbox insert, after the incident insert in the same admission
    # transaction, fails in PostgreSQL.
    sandbox = postgres_product_sandbox
    with ServedFixture() as served:
        client = relay_client(served.origin, sandbox, postgres_audit_runtime)
        fail_inserts(sandbox, "event_outbox")

        # When: the worker posts the alert.
        refused = _post_alert(client)

        # Then: no ACK, the incident rolled back with it, no audit fact, no Hub send.
        assert (refused.status_code, refused.content) == (503, b"")
        assert row_counts(sandbox) == (0, 0)
        assert _relay_audit_rows(sandbox) == 0
        assert _hub_sends(served.fixture) == 0

        # When: PostgreSQL recovers and the worker retries the same alert.
        clear_insert_fault(sandbox, "event_outbox")
        assert postgres_audit_runtime.verify_once()
        retried = _post_alert(client)

        # Then: one incident, one delivered outbox row, one audit fact, one Hub send.
        assert retried.status_code == 202
        assert retried.json() == _central_receipt(served.fixture)
        assert row_counts(sandbox) == (1, 1)
        assert [row[:2] for row in outbox_rows(sandbox)] == [(EDGE_EVENT_ID, "SENT")]
        assert _relay_audit_rows(sandbox) == 1
        assert _hub_sends(served.fixture) == 1


def test_fault_after_commit_is_not_acknowledged_and_retry_delivers_the_committed_row(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> None:
    # Given: the admission commits, then the delivery claim that follows it
    # fails in PostgreSQL before any Hub request or response.
    sandbox = postgres_product_sandbox
    with ServedFixture() as served:
        client = relay_client(served.origin, sandbox, postgres_audit_runtime)
        fail_inserts(sandbox, "event_delivery_attempts")

        # When: the worker posts the alert.
        refused = _post_alert(client)

        # Then: no ACK, yet the incident and its pending delivery obligation stay committed.
        assert (refused.status_code, refused.content) == (503, b"")
        assert row_counts(sandbox) == (1, 1)
        assert outbox_rows(sandbox) == [(EDGE_EVENT_ID, "PENDING", CAMERA_ID, 0)]
        assert _relay_audit_rows(sandbox) == 1
        assert _hub_sends(served.fixture) == 0

        # When: PostgreSQL recovers and the worker retries the same alert.
        clear_insert_fault(sandbox, "event_delivery_attempts")
        assert postgres_audit_runtime.verify_once()
        retried = _post_alert(client)

        # Then: the retry lands on the committed row: no second incident, no
        # second audit fact, one Hub send, and the audit chain still verifies.
        assert retried.status_code == 202
        assert retried.json() == _central_receipt(served.fixture)
        assert row_counts(sandbox) == (1, 1)
        assert [row[:2] for row in outbox_rows(sandbox)] == [(EDGE_EVENT_ID, "SENT")]
        assert _relay_audit_rows(sandbox) == 1
        assert _hub_sends(served.fixture) == 1
        assert served.fixture.accepted_event_ids(EDGE_EVENT_ID) == (retried.json()["event_id"],)
        assert postgres_audit_runtime.verify_once()


def test_unknown_commit_outcome_is_not_acknowledged_and_restart_retry_delivers_once(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the admission COMMIT lands in PostgreSQL but its outcome is lost.
    sandbox = postgres_product_sandbox
    with ServedFixture() as served:
        client = relay_client(served.origin, sandbox, postgres_audit_runtime)
        with monkeypatch.context() as patch:
            patch.setattr(
                sandbox.database,
                "transact",
                _unknown_after_admission_commit(sandbox.database.transact),
            )

            # When: the worker posts the alert.
            refused = _post_alert(client)

        # Then: no ACK is invented; the committed rows stay, nothing was sent,
        # and the audit runtime latches indeterminate until the process restarts.
        assert (refused.status_code, refused.content) == (503, b"")
        assert row_counts(sandbox) == (1, 1)
        assert outbox_rows(sandbox) == [(EDGE_EVENT_ID, "PENDING", CAMERA_ID, 0)]
        assert _relay_audit_rows(sandbox) == 1
        assert _hub_sends(served.fixture) == 0
        assert postgres_audit_runtime.snapshot().indeterminate
        assert not postgres_audit_runtime.verify_once()

        # When: the relay restarts on a fresh audit runtime and the worker retries.
        restarted = _restarted_audit_runtime(sandbox)
        app = relay_postgres_app(
            sandbox, restarted, client=hub_client(served.origin), camera_id=None
        )
        retried = _post_alert(TestClient(app))

        # Then: one incident, one audit fact, one Hub send, and the chain verifies.
        assert retried.status_code == 202
        assert retried.json() == _central_receipt(served.fixture)
        assert row_counts(sandbox) == (1, 1)
        assert [row[:2] for row in outbox_rows(sandbox)] == [(EDGE_EVENT_ID, "SENT")]
        assert _relay_audit_rows(sandbox) == 1
        assert _hub_sends(served.fixture) == 1
        assert restarted.verify_once()


def test_fenced_authority_is_not_acknowledged_and_commits_nothing(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> None:
    # Given: the deployment owner fenced this edge's persistence authority.
    sandbox = postgres_product_sandbox
    with ServedFixture() as served:
        client = relay_client(served.origin, sandbox, postgres_audit_runtime)
        freeze_authority(sandbox.database, sandbox.authority)

        # When: the worker posts the alert.
        refused = _post_alert(client)

        # Then: the refusal is named, nothing is committed and nothing is sent.
        assert refused.status_code == 503
        assert refused.json() == {"detail": "edge authority is fenced"}
        assert row_counts(sandbox) == (0, 0)
        assert _relay_audit_rows(sandbox) == 0
        assert _hub_sends(served.fixture) == 0


# The request-path sender: a live lease outlasts the test and a finished retry
# is scheduled well past it, so no row turns due or stale by waiting.
_SENDER_BUDGET = DeliveryBudget(
    max_attempts=3, lease_seconds=30.0, request_timeout_seconds=10.0, retry_seconds=600.0
)

_Reach = Callable[[ProductSandbox, EventOutbox, OutboxDelivery], None]


def _admit(outbox: EventOutbox, *, forward: bool = True) -> None:
    outbox.accept(
        RelayEvent(
            edge_event_id=EDGE_EVENT_ID,
            event_type="fall",
            probability=0.91,
            detected_at="2026-08-16T00:00:00.000Z",
            camera_id=CAMERA_ID,
            facility_id=FACILITY_ID,
            resident_id=None,
            evidence=None,
            audit=None,
        ),
        backend_camera_id=CAMERA_ID if forward else None,
        forward=forward,
    )


def _claim_and_finish(
    sender: OutboxDelivery, outcome: DeliveryOutcome, reason: str, **response: Any
) -> None:
    claim = sender.claim()
    assert claim is not None
    assert sender.finish(claim, outcome, reason=reason, **response)


def _reach_sent(sandbox: ProductSandbox, outbox: EventOutbox, sender: OutboxDelivery) -> None:
    _admit(outbox)
    _claim_and_finish(sender, DeliveryOutcome.SENT, "ACCEPTED", backend_event_id="central-1")


def _reach_rejected(sandbox: ProductSandbox, outbox: EventOutbox, sender: OutboxDelivery) -> None:
    _admit(outbox)
    _claim_and_finish(sender, DeliveryOutcome.REJECTED, "VALIDATION_FAILED", http_status=422)


def _reach_exhausted(sandbox: ProductSandbox, outbox: EventOutbox, sender: OutboxDelivery) -> None:
    _admit(outbox)
    # An earlier sender with a one-attempt budget spent it. With an equal budget
    # _claim_row's attempt bound would also refuse the row; the claimer's larger
    # budget leaves only the eligibility filter between it and the terminal row.
    spent = OutboxDelivery(
        sandbox.database, sandbox.authority, replace(_SENDER_BUDGET, max_attempts=1)
    )
    _claim_and_finish(spent, DeliveryOutcome.RETRY, "CENTRAL_UNAVAILABLE", http_status=503)


def _reach_local_only(sandbox: ProductSandbox, outbox: EventOutbox, sender: OutboxDelivery) -> None:
    _admit(outbox, forward=False)


def _reach_live_lease(sandbox: ProductSandbox, outbox: EventOutbox, sender: OutboxDelivery) -> None:
    _admit(outbox)
    assert sender.claim() is not None


def _reach_expired_lease(
    sandbox: ProductSandbox, outbox: EventOutbox, sender: OutboxDelivery
) -> None:
    _reach_live_lease(sandbox, outbox, sender)
    # Raw UPDATE: no public API ages a lease without waiting, so only the
    # deadline of the lease claim() just committed moves into the past.
    sandbox.admin.execute(
        "UPDATE event_outbox SET lease_until = clock_timestamp() - interval '1 second' "
        "WHERE edge_event_id = %s",
        (EDGE_EVENT_ID,),
    )


def _reach_retry_scheduled(
    sandbox: ProductSandbox, outbox: EventOutbox, sender: OutboxDelivery
) -> None:
    _admit(outbox)
    _claim_and_finish(sender, DeliveryOutcome.RETRY, "CENTRAL_UNAVAILABLE", http_status=503)


def _delivery_history(sandbox: ProductSandbox) -> tuple[Any, list[tuple[Any, ...]]]:
    """(state, attempts, lease live, retry due), then each attempt's committed result."""
    row = sandbox.admin.execute(
        "SELECT state, attempt_count, coalesce(lease_until > clock_timestamp(), false), "
        "retry_at <= clock_timestamp() FROM event_outbox WHERE edge_event_id = %s",
        (EDGE_EVENT_ID,),
    ).fetchone()
    attempts = sandbox.admin.execute(
        "SELECT a.ordinal, r.outcome, r.reason FROM event_delivery_attempts a "
        "LEFT JOIN event_delivery_results r ON r.attempt_id = a.attempt_id "
        "WHERE a.edge_event_id = %s ORDER BY a.ordinal",
        (EDGE_EVENT_ID,),
    ).fetchall()
    return row, attempts


_SENT = (("SENT", 1, False, False), [(1, "SENT", "ACCEPTED")])
_REJECTED = (("REJECTED", 1, False, False), [(1, "REJECTED", "VALIDATION_FAILED")])
_EXHAUSTED = (("EXHAUSTED", 1, False, False), [(1, "RETRY", "CENTRAL_UNAVAILABLE")])
_LOCAL_ONLY = (("LOCAL_ONLY", 0, False, True), [])
_LIVE_LEASE = (("IN_FLIGHT", 1, True, True), [(1, None, None)])


@pytest.mark.parametrize(
    ("reach", "before", "claimed", "after"),
    [
        pytest.param(_reach_sent, _SENT, None, _SENT, id="sent"),
        pytest.param(_reach_rejected, _REJECTED, None, _REJECTED, id="rejected"),
        pytest.param(_reach_exhausted, _EXHAUSTED, None, _EXHAUSTED, id="exhausted"),
        pytest.param(_reach_local_only, _LOCAL_ONLY, None, _LOCAL_ONLY, id="local_only"),
        pytest.param(_reach_live_lease, _LIVE_LEASE, None, _LIVE_LEASE, id="live_lease"),
        pytest.param(
            _reach_expired_lease,
            (("IN_FLIGHT", 1, False, True), [(1, None, None)]),
            (EDGE_EVENT_ID, 2),
            (("IN_FLIGHT", 2, True, True), [(1, "UNKNOWN", "LEASE_EXPIRED"), (2, None, None)]),
            id="expired_lease",
        ),
        pytest.param(
            _reach_retry_scheduled,
            (("PENDING", 1, False, False), [(1, "RETRY", "CENTRAL_UNAVAILABLE")]),
            (EDGE_EVENT_ID, 2),
            (
                ("IN_FLIGHT", 2, True, False),
                [(1, "RETRY", "CENTRAL_UNAVAILABLE"), (2, None, None)],
            ),
            id="retry_scheduled",
        ),
    ],
)
def test_claim_event_claims_only_eligible_rows(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    reach: _Reach,
    before: tuple[Any, list[tuple[Any, ...]]],
    claimed: tuple[str, int] | None,
    after: tuple[Any, list[tuple[Any, ...]]],
) -> None:
    # Oracle: ADR 0009 "Only eligible pending rows are claimable by the sender",
    # and the claim_event docstring: a PENDING row is claimable before its retry
    # time, a live lease belongs to another sender. An expired lease is reclaimed
    # with the old attempt's durable UNKNOWN/LEASE_EXPIRED result.
    # Given: the one accepted event reached its delivery state through the
    # public accept/claim/finish APIs and committed it.
    sandbox = postgres_product_sandbox
    outbox = EventOutbox(
        sandbox.database,
        sandbox.authority,
        RELAY_OUTBOX_BUDGET,
        audit_runtime=postgres_audit_runtime,
    )
    sender = OutboxDelivery(sandbox.database, sandbox.authority, _SENDER_BUDGET)
    reach(sandbox, outbox, sender)
    assert _delivery_history(sandbox) == before

    # When: the request path claims that event by its ID.
    claim = sender.claim_event(EDGE_EVENT_ID)

    # Then: only an eligible row yields a claim, and an ineligible row is untouched.
    assert (None if claim is None else (claim.edge_event_id, claim.ordinal)) == claimed
    assert _delivery_history(sandbox) == after
