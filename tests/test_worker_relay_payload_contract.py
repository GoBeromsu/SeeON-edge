"""What the worker actually sends must satisfy what the relay actually declares.

Two production defects this session came from the same blind spot: the worker
builds a payload, the backend declares a Pydantic schema, and nothing checked
that they agree.

1. The worker emitted ``audit.runtime_manifest_sha256`` while
   ``RelayAuditEnvelope`` did not declare it and set ``extra="forbid"``. Because
   the outbox treats HTTP 422 as non-retryable, 41 live bed-exit events were
   rejected permanently rather than retried.
2. ``DurableEvidenceStager`` pops ``audit`` out of the event body into
   ``EventEntry.decision_trace`` so the decision basis is admitted atomically
   with the event, but ``EvidenceSender._payload`` transmitted only ``values``.
   The decision basis -- the thing the never-drop obligation exists to protect --
   never reached the backend, and the projection recorded ``audit=None``.

Neither was visible to unit tests on either side, because each side was
internally consistent. Only the seam was broken.

This test drives the real producer path and validates the result against the
real consumer schema, so the next mismatch of this class fails here instead of
in a nursing home.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.features.relay.router import RelayAlertRequest
from worker.pipeline.output.evidence.evidence_sender import _payload
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager

_RUNTIME_MANIFEST = "6f6f793e244baf48449e6024" + "a" * 40


def _stager(queue_directory: Path) -> DurableEvidenceStager:
    return DurableEvidenceStager(
        queue_directory=queue_directory,
        camera_id="cmsnw6rjc01vhlh01oswn99yq",
        facility_id="facility-1",
        resident_id=None,
        config_version=7,
        clock=lambda: 1.0,
        runtime_manifest_sha256=_RUNTIME_MANIFEST,
    )


def _event(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "edge_event_id": "a5e15ff2-90fd-4764-be74-a7da4f573cc9",
        "event_type": "bed-exit",
        "detected_at": "2026-08-20T17:20:58.197192Z",
        "probability": 0.87,
        "audit": {
            "model_version": "lstm-v3",
            "detector_version": "fall-2026.08",
            "operating_threshold": 0.62,
            "clock_source": "monotonic",
        },
    }
    event.update(overrides)
    return event


def _sent_payload(tmp_path: Path, event: dict[str, object]) -> dict[str, object]:
    """Drive the real staging and sending path, returning what goes on the wire."""
    queue_directory = tmp_path / "delivery-queue"
    stager = _stager(queue_directory)
    stager.stage(event)

    entries = [
        json.loads(path.read_text())
        for path in sorted(queue_directory.iterdir())
        if path.suffix == ".json"
    ]
    events = [entry for entry in entries if entry["kind"] == "EVENT"]
    assert len(events) == 1, f"expected exactly one EVENT entry, got {len(events)}"
    return json.loads(_payload(events[0]))


def test_the_wire_payload_satisfies_the_relay_schema(tmp_path: Path) -> None:
    """The whole point: producer output validates against consumer schema."""
    payload = _sent_payload(tmp_path, _event())

    request = RelayAlertRequest.model_validate(payload)

    assert request.edge_event_id == "a5e15ff2-90fd-4764-be74-a7da4f573cc9"
    assert request.event_type == "bed-exit"


def test_the_decision_envelope_survives_the_wire(tmp_path: Path) -> None:
    """Defect 2: the stager splits audit off, so the sender must rejoin it.

    Sending ``values`` alone silently drops the decision basis. This asserts the
    audit fields the worker set are present after the round trip, not merely that
    the payload parses.
    """
    payload = _sent_payload(tmp_path, _event())

    request = RelayAlertRequest.model_validate(payload)

    assert request.audit is not None, "decision envelope lost in transit"
    assert request.audit.model_version == "lstm-v3"
    assert request.audit.detector_version == "fall-2026.08"
    assert request.audit.operating_threshold == pytest.approx(0.62)
    assert request.audit.clock_source == "monotonic"


def test_the_runtime_manifest_digest_survives_the_wire(tmp_path: Path) -> None:
    """Defect 1: the worker sets this and extra=forbid rejected it for months."""
    payload = _sent_payload(tmp_path, _event())

    request = RelayAlertRequest.model_validate(payload)

    assert request.audit is not None
    assert request.audit.runtime_manifest_sha256 == _RUNTIME_MANIFEST


def test_the_config_version_the_stager_stamps_survives_the_wire(tmp_path: Path) -> None:
    """The stager injects config_version itself; it must reach the backend too."""
    payload = _sent_payload(tmp_path, _event())

    request = RelayAlertRequest.model_validate(payload)

    assert request.audit is not None
    assert request.audit.config_version == 7


def test_an_event_without_audit_still_validates(tmp_path: Path) -> None:
    """Absent audit is legal; the schema must not require what producers omit."""
    payload = _sent_payload(tmp_path, _event(audit=None))

    request = RelayAlertRequest.model_validate(payload)

    assert request.edge_event_id is not None
