"""WP4 baseline measurement: real relay -> real Hub client -> fixture Hub.

This exercises the actual product delivery path (``POST /api/v1/relay/alerts``
-> ``EdgeIngestClient`` -> ``BackendEvidenceClient`` -> real loopback HTTP) and
measures exact E/A/B cardinality. No browser, no human adjudication, no live
camera, and no model/policy attribution: repeated machine-positive transitions
stay ``판정 불가`` by construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests_support.alert_amplification_harness import (
    DiagnosticOutcome,
    classify_rows,
    rows_from_relations,
    validate_route_ledger,
)
from tests_support.alert_amplification_runtime import (
    ServedFixture as _ServedFixture,
)
from tests_support.alert_amplification_runtime import (
    relay_client as _relay_client,
)

_EDGE_EVENT_ID = "00000000-0000-4000-8000-0000000000a1"
_SECOND_EDGE_EVENT_ID = "00000000-0000-4000-8000-0000000000a2"


@pytest.fixture(autouse=True)
def isolate_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(
        "backend.app.features.clips.catalog._catalog_path",
        lambda: tmp_path / "catalog.sqlite3",
    )
    yield


def _alert(edge_event_id: str) -> dict[str, object]:
    return {
        "edge_event_id": edge_event_id,
        "event_type": "fall",
        "probability": 0.91,
        "detected_at": "2026-08-16T00:00:00.000Z",
        "camera_id": "room-camera",
        "facility_id": "facility-1",
    }


def _post(client: TestClient, edge_event_id: str) -> dict[str, object]:
    response = client.post(
        "/api/v1/relay/alerts",
        json=_alert(edge_event_id),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )
    assert response.status_code == 202, response.text
    return response.json()


def test_repeated_delivery_attempts_converge_to_one_backend_identity(
    tmp_path: Path,
) -> None:
    with _ServedFixture() as served:
        client = _relay_client(served.origin, tmp_path)

        first = _post(client, _EDGE_EVENT_ID)
        retry = _post(client, _EDGE_EVENT_ID)

        assert first["edge_event_id"] == retry["edge_event_id"] == _EDGE_EVENT_ID
        assert first["event_id"] == retry["event_id"]
        accepted = served.fixture.accepted_event_ids(_EDGE_EVENT_ID)
        assert len(set(accepted)) == 1

        rows = rows_from_relations(
            transitions={_EDGE_EVENT_ID: "transition-1"},
            attempts={_EDGE_EVENT_ID: [1, 2]},
            backend_event_ids={_EDGE_EVENT_ID: list(set(accepted))},
            incidents=[],
            terminal_states={_EDGE_EVENT_ID: "ACKED"},
            clock_order_valid=True,
        )
        assert rows[0].backend_event_ids == (str(first["event_id"]),)
        # Without the API incident projection the chain is deliberately
        # incomplete, so the classifier must refuse to conclude.
        assert classify_rows(rows).outcome is DiagnosticOutcome.INCONCLUSIVE


def test_complete_chain_classifies_healthy_transport_retry(tmp_path: Path) -> None:
    from tests_support.alert_amplification_harness import IncidentProjection

    with _ServedFixture() as served:
        client = _relay_client(served.origin, tmp_path)
        first = _post(client, _EDGE_EVENT_ID)
        _post(client, _EDGE_EVENT_ID)

        rows = rows_from_relations(
            transitions={_EDGE_EVENT_ID: "transition-1"},
            attempts={_EDGE_EVENT_ID: [1, 2]},
            backend_event_ids={_EDGE_EVENT_ID: [str(first["event_id"])]},
            incidents=[
                IncidentProjection(
                    "incident-1",
                    _EDGE_EVENT_ID,
                    "2026-08-16T00:00:00Z",
                    "OPEN",
                    "ACKED",
                    "2026-08-16T00:00:01Z",
                )
            ],
            terminal_states={_EDGE_EVENT_ID: "ACKED"},
            clock_order_valid=True,
        )
        result = classify_rows(rows)

        assert result.outcome is DiagnosticOutcome.TRANSPORT_RETRY
        assert result.model_policy_cause == "판정 불가"


def test_distinct_transitions_never_attribute_model_cause(tmp_path: Path) -> None:
    from tests_support.alert_amplification_harness import IncidentProjection

    with _ServedFixture() as served:
        client = _relay_client(served.origin, tmp_path)
        first = _post(client, _EDGE_EVENT_ID)
        second = _post(client, _SECOND_EDGE_EVENT_ID)

        assert first["event_id"] != second["event_id"]
        rows = rows_from_relations(
            transitions={
                _EDGE_EVENT_ID: "transition-1",
                _SECOND_EDGE_EVENT_ID: "transition-2",
            },
            attempts={_EDGE_EVENT_ID: [1], _SECOND_EDGE_EVENT_ID: [1]},
            backend_event_ids={
                _EDGE_EVENT_ID: [str(first["event_id"])],
                _SECOND_EDGE_EVENT_ID: [str(second["event_id"])],
            },
            incidents=[
                IncidentProjection("incident-1", _EDGE_EVENT_ID, "t", "OPEN", "ACKED", "t"),
                IncidentProjection("incident-2", _SECOND_EDGE_EVENT_ID, "t", "OPEN", "ACKED", "t"),
            ],
            terminal_states={
                _EDGE_EVENT_ID: "ACKED",
                _SECOND_EDGE_EVENT_ID: "ACKED",
            },
            clock_order_valid=True,
        )
        result = classify_rows(rows)

        assert result.outcome is DiagnosticOutcome.REPEATED_MACHINE_POSITIVE
        assert result.model_policy_cause == "판정 불가"


def test_faulty_hub_identity_is_detected_through_the_real_client(
    tmp_path: Path,
) -> None:
    with _ServedFixture(faulty_event_identity=True) as served:
        client = _relay_client(served.origin, tmp_path)
        _post(client, _EDGE_EVENT_ID)
        _post(client, _EDGE_EVENT_ID)

        accepted = served.fixture.accepted_event_ids(_EDGE_EVENT_ID)
        assert len(set(accepted)) == 2

        rows = rows_from_relations(
            transitions={_EDGE_EVENT_ID: "transition-1"},
            attempts={_EDGE_EVENT_ID: [1, 2]},
            backend_event_ids={_EDGE_EVENT_ID: list(accepted)},
            incidents=[],
            terminal_states={_EDGE_EVENT_ID: "ACKED"},
            clock_order_valid=True,
        )
        assert classify_rows(rows).outcome is DiagnosticOutcome.BACKEND_IDENTITY_DUPLICATION


def test_measured_run_touches_only_allowed_hub_routes(tmp_path: Path) -> None:
    with _ServedFixture() as served:
        client = _relay_client(served.origin, tmp_path)
        _post(client, _EDGE_EVENT_ID)
        _post(client, _EDGE_EVENT_ID)

        validate_route_ledger(served.fixture.route_ledger)
        assert served.fixture.retained_media_bytes == 0
        assert {record.path for record in served.fixture.route_ledger} == {"/api/v1/events"}
