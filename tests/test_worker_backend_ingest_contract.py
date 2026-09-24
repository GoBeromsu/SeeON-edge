from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.features.relay.router import RelayAlertRequest
from contracts.relay import EventApiPayload
from shared.events.edge_ingest_client import _audit_payload_fields
from worker.runtime.config.worker_models import WorkerConfig


def _worker_config_payload() -> dict[str, object]:
    return {
        "relay": {
            "url": "http://127.0.0.1:8000",
            "token": "relay-token-1",
        },
        "cameras": [
            {
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "resident_id": "resident-1",
                "rtsp_url": "rtsp://camera-1/trackID=2",
            }
        ],
    }


def test_worker_config_uses_local_relay_and_has_no_backend_ingest_urls() -> None:
    config = WorkerConfig.model_validate(_worker_config_payload())

    assert config.relay.url == "http://127.0.0.1:8000"
    assert config.relay_alert_url == "http://127.0.0.1:8000/api/v1/relay/alerts"
    assert config.relay_heartbeat_url == "http://127.0.0.1:8000/api/v1/relay/heartbeat"
    assert not hasattr(config, "alert_api_url")
    assert not hasattr(config, "heartbeat_api_url")


@pytest.mark.parametrize("field_name", ["ingest", "alert_api_url", "heartbeat_api_url"])
def test_worker_config_rejects_backend_ingest_fields(field_name: str) -> None:
    payload = _worker_config_payload()
    payload[field_name] = (
        {"alert_api_url": "http://backend.local/ingest/alerts"}
        if field_name == "ingest"
        else "http://backend.local/ingest/alerts"
    )

    with pytest.raises(ValidationError, match=field_name):
        WorkerConfig.model_validate(payload)


@pytest.mark.parametrize("field_name", ["ingest_key_id", "ingest_secret"])
def test_worker_config_rejects_camera_backend_credentials(field_name: str) -> None:
    payload = _worker_config_payload()
    cameras = payload["cameras"]
    assert isinstance(cameras, list)
    cameras[0][field_name] = "backend-secret"

    with pytest.raises(ValidationError, match=field_name):
        WorkerConfig.model_validate(payload)


def test_relay_alert_request_accepts_optional_audit_and_snapshot() -> None:
    envelope_less = RelayAlertRequest.model_validate(
        {
            "event_type": "fall",
            "probability": 0.91,
            "detected_at": "2026-06-23T12:00:00.000Z",
            "camera_id": "camera-1",
            "facility_id": "facility-1",
        }
    )
    envelope_bearing = RelayAlertRequest.model_validate(
        {
            "event_type": "bed-exit",
            "probability": 0.87,
            "detected_at": "2026-06-23T12:01:00.000Z",
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "audit": {
                "config_version": 3,
                "model_version": "model-v1",
                "detector_version": "detector-v1",
                "operating_threshold": 0.72,
                "clock_source": "edge_wall_clock",
            },
            "snapshot_jpeg_base64": "anBlZw==",
        }
    )

    assert envelope_less.audit is None
    assert envelope_less.snapshot_jpeg_base64 is None
    assert envelope_bearing.audit is not None
    assert envelope_bearing.audit.model_dump(exclude_none=True) == {
        "config_version": 3,
        "model_version": "model-v1",
        "detector_version": "detector-v1",
        "operating_threshold": 0.72,
        "clock_source": "edge_wall_clock",
    }
    assert envelope_bearing.snapshot_jpeg_base64 == "anBlZw=="

    with pytest.raises(ValidationError, match="unexpected"):
        RelayAlertRequest.model_validate(
            {
                "event_type": "fall",
                "probability": 0.91,
                "detected_at": "2026-06-23T12:00:00.000Z",
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "unexpected": "field",
            }
        )


def test_event_api_payload_omits_audit_when_unset_and_includes_when_set() -> None:
    envelope_less = EventApiPayload(
        camera_id="camera-1",
        type="fall",
        detected_at="2026-06-23T12:00:00.000Z",
        confidence=0.91,
    ).as_dict()
    envelope_bearing = EventApiPayload(
        camera_id="camera-1",
        type="bed-exit",
        detected_at="2026-06-23T12:01:00.000Z",
        confidence=0.87,
        config_version=3,
        model_version="model-v1",
        detector_version="detector-v1",
        operating_threshold=0.72,
        clock_source="edge_wall_clock",
    ).as_dict()

    assert envelope_less == {
        "camera_id": "camera-1",
        "type": "fall",
        "detected_at": "2026-06-23T12:00:00.000Z",
        "confidence": 0.91,
    }
    assert _backend_policy_fields().isdisjoint(envelope_less)
    assert envelope_bearing == {
        "camera_id": "camera-1",
        "type": "bed-exit",
        "detected_at": "2026-06-23T12:01:00.000Z",
        "confidence": 0.87,
        "config_version": 3,
        "model_version": "model-v1",
        "detector_version": "detector-v1",
        "operating_threshold": 0.72,
        "clock_source": "edge_wall_clock",
    }


@pytest.mark.parametrize(
    ("config_version", "expect_forwarded"),
    [
        pytest.param(52_160_668_988, False, id="overflows-hub-int4"),
        pytest.param(2**31 - 1, True, id="max-hub-int4"),
    ],
)
def test_audit_payload_fields_omits_config_version_beyond_hub_int4(
    config_version: int, expect_forwarded: bool
) -> None:
    fields = _audit_payload_fields(
        {
            "config_version": config_version,
            "model_version": "model-v1",
        }
    )

    assert ("config_version" in fields) is expect_forwarded
    if expect_forwarded:
        assert fields["config_version"] == config_version
    assert fields["model_version"] == "model-v1"


def _backend_policy_fields() -> set[str]:
    return {
        "notification_recipient",
        "notification_channel",
        "recipient",
        "channel",
        "dedup_key",
        "deduplication_key",
        "outbox_id",
        "email_template",
        "email_delivery_id",
    }
