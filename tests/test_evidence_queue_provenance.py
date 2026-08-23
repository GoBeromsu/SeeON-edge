from __future__ import annotations

import base64
import json
from pathlib import Path

from shared.events.delivery_queue import DeliveryQueue
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager


def test_runtime_manifest_provenance_is_immutable_in_the_admitted_event(tmp_path: Path) -> None:
    queue_directory = tmp_path / "delivery-queue"
    manifest_sha256 = "a" * 64
    stager = DurableEvidenceStager(
        queue_directory,
        camera_id="camera-a",
        facility_id="facility-a",
        config_version=7,
        runtime_manifest_sha256=manifest_sha256,
    )

    stager.stage(
        {
            "edge_event_id": "event-a",
            "event_type": "fall",
            "detected_at": "2026-08-21T00:00:00Z",
            "audit": {"model_version": "model-a"},
        }
    )

    entry = next(DeliveryQueue(queue_directory).entries())
    trace = json.loads(base64.b64decode(str(entry["decision_trace_b64"])))
    assert trace == {
        "config_version": 7,
        "runtime_manifest_sha256": manifest_sha256,
        "model_version": "model-a",
    }
