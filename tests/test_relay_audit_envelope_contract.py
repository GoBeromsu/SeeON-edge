"""The audit mapping emitted by worker producers must satisfy the relay schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from backend.app.features.relay.router import RelayAlertRequest, RelayAuditEnvelope
from contracts.event import EventEvidence
from shared.events import envelope_limits as limits
from shared.events.schemas import build_audit_envelope
from worker.pipeline.output.evidence.event_payload import WorkerEventPayload
from worker.pipeline.output.evidence.evidence_sender import _payload
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.types import BusinessEvent, DecisionTraceSnapshot
from worker.types.trace import decision_trace_id

_RUNTIME_MANIFEST_SHA256 = "a" * 64
_POLICY_SHA256 = "c" * 64


def _wire_payload_from_real_producers(
    tmp_path: Path, *, extra_event_audit: dict[str, object] | None = None
) -> tuple[dict[str, object], dict[str, object]]:
    """Run the producer chain through the final relay wire payload."""
    snapshot = DecisionTraceSnapshot(
        reason="fall-onset",
        previous_state="clear",
        current_state="fall",
        triggered=True,
        track_id=5,
        bed_id=None,
    )
    event = BusinessEvent(
        domain="fall",
        event_type="fall",
        identity="a5e15ff2-90fd-4764-be74-a7da4f573cc9",
        camera_id="camera-a",
        facility_id="facility-a",
        time_sec=1.0,
        probability=0.9,
        person_id=5,
        audit={
            **(extra_event_audit or {}),
            "decision_trace_id": decision_trace_id(
                snapshot,
                module_qualified_id="fall.v1",
                effective_policy_id=_POLICY_SHA256,
            ),
        },
    )
    domain_audit = build_audit_envelope(
        model_version="lstm-v3",
        detector_version="fall-2026.08",
        operating_threshold=0.62,
    )
    attached = AlertEvidenceAttacher(
        domain_audit={"fall": domain_audit},
        runtime_manifest_sha256=_RUNTIME_MANIFEST_SHA256,
    ).attach_native(event, None)
    payload: WorkerEventPayload = {
        "edge_event_id": "a5e15ff2-90fd-4764-be74-a7da4f573cc9",
        "event_type": attached.event_type,
        "detected_at": "2026-08-20T17:20:58.197192Z",
        "probability": attached.probability,
        "audit": cast(EventEvidence, dict[str, object](attached.audit or {})),
    }
    stager = DurableEvidenceStager(
        queue_directory=tmp_path / "queue",
        camera_id="camera-a",
        facility_id="facility-a",
        config_version=7,
        clock=lambda: 1.0,
        runtime_manifest_sha256=_RUNTIME_MANIFEST_SHA256,
    )
    stager.stage(payload)
    entries = [
        json.loads(path.read_text())
        for path in sorted((tmp_path / "queue").iterdir())
        if path.suffix == ".json"
    ]
    event_entries = [entry for entry in entries if entry["kind"] == "EVENT"]
    assert len(event_entries) == 1
    wire_payload = json.loads(_payload(event_entries[0]))
    return wire_payload, event_entries[0]


def test_real_audit_producer_chain_validates_against_relay_envelope(tmp_path: Path) -> None:
    wire_payload, _ = _wire_payload_from_real_producers(tmp_path)
    audit = cast(dict[str, object], wire_payload["audit"])

    envelope = RelayAuditEnvelope.model_validate(audit)

    assert set(audit) <= set(RelayAuditEnvelope.model_fields)
    assert envelope.config_version == 7
    assert envelope.clock_source == "edge_wall_clock"
    assert envelope.model_version == "lstm-v3"
    assert envelope.detector_version == "fall-2026.08"
    assert envelope.operating_threshold == 0.62
    assert envelope.runtime_manifest_sha256 == _RUNTIME_MANIFEST_SHA256
    assert isinstance(audit["decision_trace_id"], str)


def test_relay_audit_field_source_tracks_relay_model() -> None:
    assert frozenset(RelayAuditEnvelope.model_fields) == limits.RELAY_AUDIT_FIELDS


def test_undeclared_key_from_domain_event_does_not_reach_relay(
    tmp_path: Path,
) -> None:
    """A domain mutation must not make the relay permanently discard the alert."""
    wire_payload, entry = _wire_payload_from_real_producers(
        tmp_path, extra_event_audit={"undeclared_audit_key": "mutant"}
    )

    request = RelayAlertRequest.model_validate(wire_payload)
    assert request.edge_event_id == "a5e15ff2-90fd-4764-be74-a7da4f573cc9"
    assert request.audit is not None
    assert request.audit.decision_trace_id is not None
    assert "undeclared_audit_key" not in request.audit.model_dump(exclude_none=True)
    assert entry["shed_detail_keys"] == ["audit.undeclared_audit_key"]
