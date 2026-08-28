"""Oversized evidence must survive the complete authenticated relay path."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import shared.events.envelope_limits as limits
from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.features.evidence.record_store import CentralEvidenceQuery
from backend.app.features.relay.router import RelayAlertRequest
from backend.app.main import create_app, no_lifespan
from shared.events.delivery_queue import DeliveryQueue
from shared.events.evidence_export_contract import DeliveryFailure, EventReceipt
from tests_support.alert_amplification_runtime import (
    CAMERA_ID,
    FACILITY_ID,
    RELAY_TOKEN,
    ServedFixture,
    relay_client,
)
from worker.pipeline.output.evidence.event_payload import WorkerEventPayload
from worker.pipeline.output.evidence.evidence_sender import (
    EvidenceSender,
    SenderConfig,
    SenderStep,
)
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager

_EDGE_EVENT_ID = "00000000-0000-4000-8000-0000000000e1"
_DETECTED_AT = "2026-08-22T03:53:43Z"


@dataclass
class RelayTransport:
    """Authenticated in-process relay transport recording the actual response."""

    client: TestClient
    statuses: list[int] = field(default_factory=list)

    def send_event(
        self, payload_json: str, edge_event_id: str
    ) -> EventReceipt | DeliveryFailure:
        response = self.client.post(
            "/api/v1/relay/alerts",
            content=payload_json,
            headers={"Content-Type": "application/json", "X-Edge-Relay-Token": RELAY_TOKEN},
        )
        self.statuses.append(response.status_code)
        if response.status_code == 202:
            return EventReceipt("accepted", edge_event_id, str(response.json()["event_id"]))
        raise AssertionError(response.text)

    def send_snapshot_attachment(self, _payload: dict[str, object]) -> None:
        raise AssertionError("unexpected snapshot attachment")

    def send_snapshot_disposition(self, _payload: dict[str, object]) -> None:
        raise AssertionError("unexpected snapshot disposition")


def _event() -> WorkerEventPayload:
    return {
        "edge_event_id": _EDGE_EVENT_ID,
        "event_type": "fall",
        "probability": 0.97,
        "detected_at": _DETECTED_AT,
        "camera_id": CAMERA_ID,
        "facility_id": FACILITY_ID,
        "evidence": {"keypoints": "x" * (limits.VALUES_BYTES_MAX + 1)},
        "audit": {"model_version": "fall-v1", "detector_version": "detector-v1"},
    }


def _stager(queue_directory: Path) -> DurableEvidenceStager:
    return DurableEvidenceStager(
        queue_directory=queue_directory,
        camera_id=CAMERA_ID,
        facility_id=FACILITY_ID,
        resident_id="resident-9",
        config_version=1,
        runtime_manifest_sha256="a" * 64,
    )


def _incidents(database: Path) -> list[dict[str, object]]:
    app = create_app(lifespan=no_lifespan)
    app.state.central_evidence_query = CentralEvidenceQuery(database)
    with TestClient(app) as client:
        assert client.post(
            "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
        ).status_code == 204
        response = client.get("/api/v1/incidents")
    assert response.status_code == 200, response.text
    return list(response.json()["incidents"])


def test_required_alert_field_source_tracks_relay_model() -> None:
    assert frozenset(
        name
        for name, model_field in RelayAlertRequest.model_fields.items()
        if model_field.is_required()
    ) == limits.REQUIRED_ALERT_FIELDS


def test_oversized_event_is_shed_off_wire_and_delivered_to_incident_projection(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    queue_directory = tmp_path / "delivery-queue"
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    stager = _stager(queue_directory)
    stager.stage(_event())

    [entry] = tuple(stager.queue.entries())
    values = json.loads(base64.b64decode(str(entry["values_b64"])))
    assert len(base64.b64decode(str(entry["values_b64"]))) <= limits.VALUES_BYTES_MAX
    assert "evidence" not in values
    assert "shed_detail_keys" not in values
    assert entry["shed_detail_keys"] == ["evidence"]
    assert values["camera_id"] == CAMERA_ID
    assert values["detected_at"] == _DETECTED_AT
    assert values["event_type"] == "fall"
    assert values["facility_id"] == FACILITY_ID
    assert values["probability"] == 0.97

    with ServedFixture() as served:
        relay = relay_client(served.origin, tmp_path, database=database)
        transport = RelayTransport(relay)
        sender = EvidenceSender(
            queue_directory,
            SenderConfig("http://relay.test", RELAY_TOKEN, CAMERA_ID),
            transport=transport,
        )
        with caplog.at_level(
            logging.WARNING, logger="worker.pipeline.output.evidence.evidence_sender"
        ):
            assert sender.run_once() is SenderStep.EVENT_ACKED

    assert transport.statuses == [202]
    assert DeliveryQueue(queue_directory).accepted_count == 0
    warnings = [
        record.message
        for record in caplog.records
        if record.name == "worker.pipeline.output.evidence.evidence_sender"
    ]
    assert any(
        _EDGE_EVENT_ID in message and CAMERA_ID in message and "evidence" in message
        for message in warnings
    )

    [incident] = _incidents(database)
    assert incident["edge_event_id"] == _EDGE_EVENT_ID
