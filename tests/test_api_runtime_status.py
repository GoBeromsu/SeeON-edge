from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from pydantic import ValidationError
from starlette.applications import Starlette

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.relay.router import (
    MAX_RELAY_RUNTIME_STATUS_BODY_BYTES,
    RelayRuntimeStatusRequest,
)
from backend.app.features.status.router import _flatten_runtime_cameras
from backend.app.features.status.runtime_status_store import RuntimeStatusStore
from backend.app.main import create_app, no_lifespan
from contracts.decode_diagnostics import DecodeSelection
from shared.events.delivery_queue import (
    MAX_ACCEPTED_BYTES,
    MAX_ACCEPTED_ENTRIES,
    DeliveryQueue,
    EventEntry,
    SnapshotAttachmentEntry,
    SnapshotDispositionEntry,
)
from tests_support.compact_authority_db import prepare_compact_database
from worker.pipeline.inference_coordinator import (
    CameraInferenceTelemetry,
    InferenceTelemetrySnapshot,
)
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.runtime_status_sender import RuntimeStatusSender
from worker.runtime.telemetry.wire import RelayRuntimeStatusPayload


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "facility_id": "facility-1",
        "generation": None,
        "seq": 0,
        "cameras": [
            {
                "camera_id": "camera-1",
                "decode": {
                    "requested": "auto",
                    "selected": "opencv",
                    "fallback_count": 0,
                    "last_reason": None,
                    "updated_at_sec": 1000.0,
                },
            }
        ],
        "clip_export": {"enabled": True, "version": 4},
        "clip_recorder": {
            "available": True,
            "dropped_frames": 0,
            "dropped_events": 0,
            "failed_writes": 0,
            "finalized_clips": 2,
            "video_unavailable_clips": 0,
            "active_clips": 1,
            "encoder": "libx264",
        },
    }
    payload.update(overrides)
    return payload


def _client() -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_inventory = {
        "camera-1": {"camera_id": "camera-1", "facility_id": "facility-1"}
    }
    return TestClient(app)


def _post(client: TestClient, payload: Mapping[str, object]) -> Response:
    return client.post(
        "/api/v1/relay/runtime-status",
        json=payload,
        headers={"Authorization": "Bearer relay-token"},
    )


def _json(response: Response) -> dict[str, Any]:
    body = response.json()
    assert isinstance(body, dict)
    return body


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _payload_cameras(payload: dict[str, object]) -> list[object]:
    cameras = payload["cameras"]
    assert isinstance(cameras, list)
    return cameras


def _camera_row(**overrides: object) -> dict[str, object]:
    row = dict(_as_dict(_payload_cameras(_payload())[0]))
    row.update(overrides)
    return row


def _runtime_cameras(client: TestClient) -> dict[str, Any]:
    runtime = _json(client.get("/api/v1/status")).get("runtime")
    assert isinstance(runtime, dict)
    cameras = runtime.get("cameras")
    assert isinstance(cameras, dict)
    return cameras


def _runtime_detection(client: TestClient, camera_id: str = "camera-1") -> dict[str, object]:
    camera = _runtime_cameras(client).get(camera_id)
    assert isinstance(camera, dict)
    detection = camera.get("detection")
    assert isinstance(detection, dict)
    return detection


class _CapturingRuntimeStatusTransport:
    def __init__(self) -> None:
        self.payload: RelayRuntimeStatusPayload | None = None

    def send(self, payload: RelayRuntimeStatusPayload) -> int:
        self.payload = payload
        return 1


def _delivery_queue(tmp_path: Path) -> DeliveryQueue:
    queue = DeliveryQueue(tmp_path / "delivery-queue")
    entries = (
        EventEntry(
            edge_event_id="event-1",
            event_type="fall",
            detected_at="2026-08-21T00:00:00Z",
            camera_id="camera-1",
            facility_id="facility-1",
            decision_trace=b"trace",
            values=b"values",
        ),
        SnapshotAttachmentEntry(
            "event-1",
            "snapshot-1",
            "a" * 64,
            "snapshots/snapshot-1.jpg",
            10,
            "image/jpeg",
        ),
        SnapshotDispositionEntry("event-1", "snapshot-1", "unavailable", "camera offline"),
    )
    for entry in entries:
        assert queue.try_admit(entry).accepted
    return queue


def _post_queue_capacity(client: TestClient, queue: DeliveryQueue) -> dict[str, object]:
    transport = _CapturingRuntimeStatusTransport()
    sender = RuntimeStatusSender(
        WorkerDiagnostics(),
        "facility-1",
        transport,
        delivery_queue=queue,
    )
    assert sender.publish_once()
    assert transport.payload is not None
    response = _post(client, transport.payload)
    assert response.status_code == 200
    facility = _json(client.get("/api/v1/status"))["runtime"]["facilities"]["facility-1"]
    assert isinstance(facility, dict)
    delivery_queue = facility["delivery_queue"]
    assert isinstance(delivery_queue, dict)
    return delivery_queue


def test_status_round_trips_delivery_queue_capacity_and_kind_mix(tmp_path: Path) -> None:
    queue = _delivery_queue(tmp_path)
    expected = queue.capacity_snapshot

    projected = _post_queue_capacity(_client(), queue)

    assert projected == {
        "accepted_count": expected.accepted_count,
        "accepted_bytes": expected.accepted_bytes,
        "max_accepted_entries": MAX_ACCEPTED_ENTRIES,
        "max_accepted_bytes": MAX_ACCEPTED_BYTES,
        "by_kind": {
            "EVENT": 1,
            "SNAPSHOT_ATTACHMENT": 1,
            "SNAPSHOT_DISPOSITION": 1,
        },
        # Evidence the backend refused is retained rather than deleted, so the
        # operator needs to see it here; nothing is retained in this fixture.
        "dead_lettered_count": expected.dead_lettered_count,
        "dead_lettered_bytes": expected.dead_lettered_bytes,
    }


def test_status_reports_delivery_queue_bounds_for_headroom_calculation(tmp_path: Path) -> None:
    projected = _post_queue_capacity(_client(), _delivery_queue(tmp_path))

    assert projected["max_accepted_entries"] == MAX_ACCEPTED_ENTRIES
    assert projected["max_accepted_bytes"] == MAX_ACCEPTED_BYTES


def test_runtime_status_schema_round_trip_and_rejects_extra_fields() -> None:
    payload = _payload()

    parsed = RelayRuntimeStatusRequest.model_validate(payload)

    cameras = _payload_cameras(payload)
    camera = cameras[0]
    assert isinstance(camera, dict)
    assert parsed.model_dump() == {
        **payload,
        "gpu": None,
        "worker": None,
        "delivery_queue": None,
        "cameras": [{**camera, "measured_fps": None, "detection": None}],
    }
    with pytest.raises(ValidationError):
        RelayRuntimeStatusRequest.model_validate(_payload(unexpected=True))


def test_runtime_status_accepts_old_payload_omitting_detection() -> None:
    """Old workers remain accepted and are explicitly reported as missing."""
    payload = _payload()
    camera = _payload_cameras(payload)[0]
    assert isinstance(camera, dict)
    assert "detection" not in camera

    parsed = RelayRuntimeStatusRequest.model_validate(payload)
    client = _client()
    posted = _post(client, payload)
    accepted_camera = _runtime_cameras(client)["camera-1"]

    assert getattr(parsed.cameras[0], "detection", None) is None
    assert posted.status_code == 200
    assert accepted_camera["detection"] == _derived(state="unknown", reason="telemetry_missing")


def _detection(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "expected": True,
        "inference_admitted": 10,
        "inference_succeeded": 8,
        "inference_overwritten": 1,
        "decision_completed": 8,
    }
    payload.update(overrides)
    return payload


def _camera(**overrides: object) -> dict[str, object]:
    camera = dict(_as_dict(_payload_cameras(_payload())[0]))
    camera.update(overrides)
    return camera


def test_runtime_status_accepts_valid_zero_event_progress_detection() -> None:
    detection = _detection(
        inference_admitted=4,
        inference_succeeded=4,
        inference_overwritten=0,
        decision_completed=4,
    )
    payload = _payload(cameras=[_camera(detection=detection)])
    parsed = RelayRuntimeStatusRequest.model_validate(payload)
    client = _client()
    posted = _post(client, payload)
    stored = _runtime_cameras(client)["camera-1"]

    assert parsed.cameras[0].detection is not None
    assert parsed.cameras[0].detection.model_dump() == detection
    assert posted.status_code == 200
    stored_detection = stored["detection"]
    assert isinstance(stored_detection, dict)
    assert {field: stored_detection[field] for field in detection} == detection


def _health(
    store: RuntimeStatusStore, *, now: float = 0.0, camera_id: str = "camera-1"
) -> dict[str, object]:
    facilities = store.snapshot(now=now)["facilities"]
    facility = _as_dict(facilities)["facility-1"]
    cameras = _as_dict(facility)["cameras"]
    assert isinstance(cameras, list)
    camera = next(
        item for item in cameras if isinstance(item, dict) and item["camera_id"] == camera_id
    )
    return _as_dict(camera["detection"])


def _record_detection(
    store: RuntimeStatusStore,
    at: float,
    **values: object,
) -> None:
    camera_id = values.pop("camera_id", "camera-1")
    assert isinstance(camera_id, str)
    store.record(
        _payload(
            generation=1,
            seq=int(at * 1000),
            cameras=[_camera(camera_id=camera_id, detection=_detection(**values))],
        ),
        received_at=at,
    )


def _derived(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "state": "starting",
        "reason": None,
        "recent_success_rate": None,
        "last_completed_at_sec": None,
        "evaluation_window_sec": 10.0,
        "timeout_sec": 120.0,
    }
    result.update(overrides)
    return result


def test_detection_health_startup_disabled_missing_and_stale_states() -> None:
    startup = RuntimeStatusStore(stale_after_sec=15.0)
    _record_detection(startup, 0.0)
    disabled = RuntimeStatusStore(stale_after_sec=15.0)
    _record_detection(
        disabled,
        0.0,
        expected=False,
        inference_admitted=0,
        inference_succeeded=0,
        inference_overwritten=0,
        decision_completed=0,
    )
    missing = RuntimeStatusStore(stale_after_sec=15.0)
    missing.record(_payload(generation=1), received_at=0.0)

    assert _health(startup) == {**_detection(), **_derived()}
    assert _health(disabled) == {
        **_detection(
            expected=False,
            inference_admitted=0,
            inference_succeeded=0,
            inference_overwritten=0,
            decision_completed=0,
        ),
        **_derived(state="disabled"),
    }
    assert _health(missing) == _derived(state="unknown", reason="telemetry_missing")
    assert _health(startup, now=15.001) == {
        **_detection(),
        **_derived(state="unknown", reason="telemetry_stale"),
    }


@pytest.mark.parametrize(
    ("second_at", "expected_state", "expected_reason"),
    [
        (9.999, "starting", None),
        (10.0, "blind", "pose_not_completing"),
    ],
)
def test_detection_health_pose_failure_exact_window_boundary(
    second_at: float, expected_state: str, expected_reason: str | None
) -> None:
    store = RuntimeStatusStore(stale_after_sec=1000.0)
    _record_detection(
        store, 0.0, inference_admitted=10, inference_succeeded=8, decision_completed=8
    )
    _record_detection(
        store, second_at / 2, inference_admitted=11, inference_succeeded=8, decision_completed=8
    )
    _record_detection(
        store, second_at, inference_admitted=12, inference_succeeded=8, decision_completed=8
    )

    health = _health(store, now=second_at)
    assert health["state"] == expected_state
    assert health["reason"] == expected_reason
    assert health["recent_success_rate"] == 0.0


def test_detection_health_decision_failure_and_pose_precedence() -> None:
    store = RuntimeStatusStore(stale_after_sec=1000.0)
    _record_detection(
        store, 0.0, inference_admitted=10, inference_succeeded=8, decision_completed=8
    )
    _record_detection(
        store, 5.0, inference_admitted=11, inference_succeeded=9, decision_completed=8
    )
    _record_detection(
        store, 10.0, inference_admitted=12, inference_succeeded=10, decision_completed=8
    )

    assert _health(store, now=10.0)["reason"] == "decision_not_completing"

    pose = RuntimeStatusStore(stale_after_sec=1000.0)
    _record_detection(pose, 0.0, inference_admitted=10, inference_succeeded=8, decision_completed=8)
    _record_detection(pose, 5.0, inference_admitted=11, inference_succeeded=8, decision_completed=8)
    _record_detection(
        pose, 10.0, inference_admitted=12, inference_succeeded=8, decision_completed=8
    )
    assert _health(pose, now=10.0)["reason"] == "pose_not_completing"


@pytest.mark.parametrize(
    ("now", "expected_state", "expected_reason"),
    [
        (119.999, "starting", None),
        (120.0, "blind", "no_completed_cycles"),
    ],
)
def test_detection_health_no_completed_cycle_timeout_boundary(
    now: float, expected_state: str, expected_reason: str | None
) -> None:
    store = RuntimeStatusStore(stale_after_sec=1000.0)
    _record_detection(store, 0.0)

    health = _health(store, now=now)
    assert health["state"] == expected_state
    assert health["reason"] == expected_reason


def test_detection_health_immediate_recovery_counter_reset_and_zero_denominator() -> None:
    store = RuntimeStatusStore(stale_after_sec=1000.0)
    _record_detection(
        store, 0.0, inference_admitted=10, inference_succeeded=8, decision_completed=8
    )
    _record_detection(
        store, 5.0, inference_admitted=11, inference_succeeded=8, decision_completed=8
    )
    _record_detection(
        store, 10.0, inference_admitted=12, inference_succeeded=8, decision_completed=8
    )
    assert _health(store, now=10.0)["state"] == "blind"

    _record_detection(
        store, 15.0, inference_admitted=12, inference_succeeded=9, decision_completed=9
    )
    recovered = _health(store, now=15.0)
    assert recovered["state"] == "healthy"
    assert recovered["reason"] is None
    assert recovered["recent_success_rate"] is None
    assert recovered["last_completed_at_sec"] == 15.0

    _record_detection(
        store, 20.0, inference_admitted=1, inference_succeeded=1, decision_completed=1
    )
    reset = _health(store, now=20.0)
    assert reset["state"] == "starting"
    assert reset["reason"] == "counter_reset"
    assert reset["last_completed_at_sec"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inference_admitted", 9),
        ("inference_succeeded", 7),
        ("inference_overwritten", 1),
        ("decision_completed", 5),
    ],
)
def test_detection_health_any_counter_rollback_resets(field: str, value: int) -> None:
    store = RuntimeStatusStore(stale_after_sec=1000.0)
    baseline = _detection(
        inference_admitted=10,
        inference_succeeded=8,
        inference_overwritten=2,
        decision_completed=6,
    )
    _record_detection(store, 0.0, **baseline)
    _record_detection(store, 5.0, **{**baseline, field: value})

    health = _health(store, now=5.0)
    assert health["state"] == "starting"
    assert health["reason"] == "counter_reset"


def test_detection_health_prunes_removed_camera_and_facility_state() -> None:
    store = RuntimeStatusStore(stale_after_sec=1000.0)
    _record_detection(store, 0.0)
    _record_detection(store, 0.0, camera_id="camera-2")
    store.record(_payload(generation=1, seq=1, cameras=[]), received_at=1.0)
    store.snapshot(now=1.0)

    assert all(key[0] != "facility-1" for key in store._detection_health)

    store.record(
        _payload(
            facility_id="facility-2", generation=1, seq=0, cameras=[_camera(detection=_detection())]
        ),
        received_at=2.0,
    )
    with store._lock:
        del store._snapshots["facility-2"]
    store.snapshot(now=2.0)
    assert all(key[0] != "facility-2" for key in store._detection_health)


def test_detection_health_real_api_sequence_uses_accepted_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr("backend.app.features.status.runtime_status_store.time", lambda: clock[0])
    client = _client()

    def post_at(at: float, seq: int, detection: dict[str, object]) -> dict[str, object]:
        clock[0] = at
        assert (
            _post(
                client, _payload(generation=1, seq=seq, cameras=[_camera(detection=detection)])
            ).status_code
            == 200
        )
        return _runtime_detection(client)

    states = [
        post_at(
            0.0, 0, _detection(inference_admitted=10, inference_succeeded=8, decision_completed=8)
        )
    ]
    states.append(
        post_at(
            5.0, 1, _detection(inference_admitted=11, inference_succeeded=8, decision_completed=8)
        )
    )
    states.append(
        post_at(
            10.0, 2, _detection(inference_admitted=12, inference_succeeded=8, decision_completed=8)
        )
    )
    states.append(
        post_at(
            15.0, 3, _detection(inference_admitted=12, inference_succeeded=9, decision_completed=9)
        )
    )
    clock[0] = 31.0
    states.append(_runtime_detection(client))
    states.append(
        post_at(
            35.0, 4, _detection(inference_admitted=1, inference_succeeded=1, decision_completed=1)
        )
    )

    timeout_client = _client()
    timeout_app = timeout_client.app
    assert isinstance(timeout_app, Starlette)
    timeout_app.state.runtime_status_store = RuntimeStatusStore(stale_after_sec=1000.0)
    clock[0] = 100.0
    assert (
        _post(
            timeout_client, _payload(generation=1, seq=0, cameras=[_camera(detection=_detection())])
        ).status_code
        == 200
    )
    clock[0] = 220.0
    states.append(_runtime_detection(timeout_client))

    assert [(item["state"], item["reason"]) for item in states] == [
        ("starting", None),
        ("starting", None),
        ("blind", "pose_not_completing"),
        ("healthy", None),
        ("unknown", "telemetry_stale"),
        ("starting", "counter_reset"),
        ("blind", "no_completed_cycles"),
    ]


def test_runtime_status_rejects_impossible_counter_ordering() -> None:
    succeeded_gt_admitted = _payload(
        cameras=[_camera(detection=_detection(inference_admitted=1, inference_succeeded=2))]
    )
    completed_gt_succeeded = _payload(
        cameras=[
            _camera(
                detection=_detection(
                    inference_admitted=4,
                    inference_succeeded=2,
                    decision_completed=3,
                )
            )
        ]
    )

    with pytest.raises(ValidationError):
        RelayRuntimeStatusRequest.model_validate(succeeded_gt_admitted)
    with pytest.raises(ValidationError):
        RelayRuntimeStatusRequest.model_validate(completed_gt_succeeded)
    assert _post(_client(), completed_gt_succeeded).status_code == 422


def test_runtime_status_rejects_negative_detection_counters() -> None:
    for field in (
        "inference_admitted",
        "inference_succeeded",
        "inference_overwritten",
        "decision_completed",
    ):
        payload = _payload(cameras=[_camera(detection=_detection(**{field: -1}))])
        with pytest.raises(ValidationError):
            RelayRuntimeStatusRequest.model_validate(payload)
        assert _post(_client(), payload).status_code == 422


def test_runtime_status_rejects_unknown_detection_extras() -> None:
    payload = _payload(cameras=[_camera(detection=_detection(reason="pose_not_completing"))])

    with pytest.raises(ValidationError):
        RelayRuntimeStatusRequest.model_validate(payload)
    assert _post(_client(), payload).status_code == 422


def test_runtime_status_rejects_one_malformed_camera_without_mutating_snapshot() -> None:
    client = _client()
    first = _post(client, _payload())
    generation = _json(first)["generation"]
    rejected = _post(
        client,
        _payload(
            generation=generation,
            seq=1,
            cameras=[
                _camera(camera_id="camera-1", detection=_detection()),
                _camera(
                    camera_id="camera-2",
                    detection=_detection(decision_completed=9, inference_succeeded=1),
                ),
            ],
        ),
    )
    stored = _runtime_cameras(client)

    assert first.status_code == 200
    assert rejected.status_code == 422
    assert list(stored) == ["camera-1"]
    assert stored["camera-1"]["detection"] == _derived(state="unknown", reason="telemetry_missing")


def test_fifty_camera_detection_payload_stays_under_runtime_status_body_limit() -> None:
    cameras = [
        _camera(
            camera_id=f"camera-{index:02d}",
            measured_fps=12.5,
            detection=_detection(),
        )
        for index in range(50)
    ]
    payload = _payload(cameras=cameras, generation=1)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    parsed = RelayRuntimeStatusRequest.model_validate(payload)
    posted = _post(_client(), payload)

    assert len(body) < MAX_RELAY_RUNTIME_STATUS_BODY_BYTES
    assert len(parsed.cameras) == 50
    assert posted.status_code == 200


class _StaticInferenceSource:
    def __init__(self, cameras: dict[str, CameraInferenceTelemetry]) -> None:
        self._cameras = cameras

    def snapshot(self) -> InferenceTelemetrySnapshot:
        return InferenceTelemetrySnapshot(
            cameras=self._cameras,
            batch_sizes={},
            forward_p50_sec=0.0,
            forward_p95_sec=0.0,
        )


def test_worker_projects_detection_when_inference_telemetry_exists() -> None:
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode(
        "camera-1",
        DecodeSelection(
            requested="auto",
            selected="nvdec",
            fallback_count=0,
            last_reason=None,
            updated_at_sec=1000.0,
        ),
    )
    diagnostics.register_inference(
        _StaticInferenceSource(
            {
                "camera-1": CameraInferenceTelemetry(
                    admitted=10,
                    overwritten=2,
                    inferred=8,
                    queue_age_sec=0.1,
                )
            }
        )
    )
    for _ in range(7):
        diagnostics.record_detection_completed("camera-1")

    camera = diagnostics.to_payload("facility-1", None, 1)["cameras"][0]
    detection = camera.get("detection")
    assert detection == {
        "expected": True,
        "inference_admitted": 10,
        "inference_succeeded": 8,
        "inference_overwritten": 2,
        "decision_completed": 7,
    }


def test_worker_projects_disabled_detection_when_inference_telemetry_is_absent() -> None:
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode(
        "camera-1",
        DecodeSelection(
            requested="auto",
            selected="nvdec",
            fallback_count=0,
            last_reason=None,
            updated_at_sec=1000.0,
        ),
    )
    diagnostics.record_detection_completed("camera-1")

    camera = diagnostics.to_payload("facility-1", None, 1)["cameras"][0]
    detection = camera.get("detection")
    assert detection == {
        "expected": False,
        "inference_admitted": 0,
        "inference_succeeded": 0,
        "inference_overwritten": 0,
        "decision_completed": 0,
    }


def test_worker_fifty_camera_detection_payload_stays_under_body_limit() -> None:
    diagnostics = WorkerDiagnostics()
    diagnostics.register_inference(
        _StaticInferenceSource(
            {
                f"camera-{index:02d}": CameraInferenceTelemetry(
                    admitted=100,
                    overwritten=3,
                    inferred=90,
                    queue_age_sec=0.05,
                )
                for index in range(50)
            }
        )
    )
    for index in range(50):
        camera_id = f"camera-{index:02d}"
        diagnostics.update_decode(
            camera_id,
            DecodeSelection(
                requested="auto",
                selected="nvdec",
                fallback_count=0,
                last_reason=None,
                updated_at_sec=1000.0,
            ),
        )
        diagnostics.update_measured_fps(camera_id, 12.5)
        for _ in range(90):
            diagnostics.record_detection_completed(camera_id)
    diagnostics.set_gpu_status(
        {
            "nvml_available": True,
            "cuda_context_ok": True,
            "driver_version": "580.00",
            "device_name": "NVIDIA RTX 5070 Ti",
            "nvml_error": None,
            "captured_at_sec": 1.0,
        }
    )
    diagnostics.set_worker_status(
        {"alive": True, "pid": 1234, "started_at_sec": 1.0, "profile_boot_error": None}
    )

    payload = diagnostics.to_payload("facility-1", 12, 4)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    assert len(payload["cameras"]) == 50
    assert all("detection" in camera for camera in payload["cameras"])
    assert len(body) < MAX_RELAY_RUNTIME_STATUS_BODY_BYTES


def test_runtime_status_exposes_additive_diagnostics() -> None:
    client = _client()
    payload = _payload(
        cameras=[
            {
                **_camera_row(),
                "measured_fps": 12.5,
            }
        ],
        gpu={
            "nvml_available": False,
            "cuda_context_ok": False,
            "driver_version": None,
            "device_name": None,
            "nvml_error": None,
            "captured_at_sec": 1.0,
        },
        worker={"alive": True, "pid": 7, "started_at_sec": 1.0},
    )

    assert _post(client, payload).status_code == 200
    runtime = client.get("/api/v1/status").json()["runtime"]
    facility = runtime["facilities"]["facility-1"]
    assert facility["gpu"] == payload["gpu"]
    assert facility["worker"] == payload["worker"]
    assert facility["cameras"][0]["measured_fps"] == 12.5
    assert runtime["cameras"]["camera-1"]["measured_fps"] == 12.5
    assert runtime["worker"] == payload["worker"]
    assert runtime["device"] == {
        "backend": None,
        "available": False,
        "device_name": None,
        "captured_at_sec": 1.0,
    }
    assert runtime["clip_recorder"]["finalized_clips"] == 2
    assert runtime["clip_export_applied"] == {
        "enabled": True,
        "version": 4,
        "freshness": "fresh",
    }


def test_flatten_runtime_cameras_marks_stale_camera_without_erasing_last_fps() -> None:
    """워커가 죽어 facility가 stale이 되면, 마지막 measured_fps는 지우지 않되
    카메라별로 stale을 같이 내려야 프론트가 '멈춘 값'과 '현재 값'을 구분할 수 있다."""
    store = RuntimeStatusStore(stale_after_sec=1.0)
    store.record(
        _payload(cameras=[{**_camera_row(), "measured_fps": 5.0}]),
        received_at=0.0,
    )

    facilities = store.snapshot(now=1000.0)["facilities"]
    assert facilities["facility-1"]["stale"] is True

    cameras = _flatten_runtime_cameras(facilities)
    assert cameras["camera-1"]["measured_fps"] == 5.0
    assert cameras["camera-1"]["stale"] is True


def test_flatten_runtime_cameras_marks_fresh_camera_as_not_stale() -> None:
    store = RuntimeStatusStore(stale_after_sec=100.0)
    store.record(
        _payload(cameras=[{**_camera_row(), "measured_fps": 5.0}]),
        received_at=0.0,
    )

    facilities = store.snapshot(now=1.0)["facilities"]
    cameras = _flatten_runtime_cameras(facilities)
    assert cameras["camera-1"]["stale"] is False


def test_status_returns_unmapped_local_camera_id_unchanged(
    tmp_path: Path,
) -> None:
    """Baseline: an unmapped registry-local id is already the dashboard key.

    Characterization of the unmodified route: when the worker reports the
    local registry id and the camera has no backend_camera_id, /status must
    keep that same key. This is the identity we later normalize mapped
    aliases onto; the unmapped path must not change.
    """
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    registry_path = tmp_path / "catalog.sqlite3"
    prepare_compact_database(registry_path)
    registry = CameraRegistryStore(registry_path)
    registry.create(
        camera_id="local-unmapped-1",
        label="Lobby",
        rtsp_url="rtsp://example/unmapped",
        space_id=None,
        status="online",
        backend_camera_id=None,
    )
    app.state.camera_registry = registry
    client = TestClient(app)

    posted = _post(
        client,
        _payload(
            cameras=[
                {
                    **_camera_row(),
                    "camera_id": "local-unmapped-1",
                    "measured_fps": 8.0,
                }
            ]
        ),
    )
    runtime_cameras = _runtime_cameras(client)

    assert posted.status_code == 200
    assert list(runtime_cameras) == ["local-unmapped-1"]
    assert runtime_cameras["local-unmapped-1"]["camera_id"] == "local-unmapped-1"
    assert runtime_cameras["local-unmapped-1"]["measured_fps"] == 8.0


def _registry_status_client(
    tmp_path: Path,
    *,
    camera_id: str,
    backend_camera_id: str | None,
) -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    registry_path = tmp_path / "catalog.sqlite3"
    prepare_compact_database(registry_path)
    registry = CameraRegistryStore(registry_path)
    registry.create(
        camera_id=camera_id,
        label="Lobby",
        rtsp_url="rtsp://example/lobby",
        space_id=None,
        status="online",
        backend_camera_id=backend_camera_id,
    )
    app.state.camera_registry = registry
    return TestClient(app)


def test_status_normalizes_mapped_canonical_id_to_local_registry_id(
    tmp_path: Path,
) -> None:
    client = _registry_status_client(
        tmp_path,
        camera_id="local-uuid-1",
        backend_camera_id="backend-camera-1",
    )

    posted = _post(
        client,
        _payload(
            cameras=[
                {
                    **_camera_row(),
                    "camera_id": "backend-camera-1",
                    "measured_fps": 11.0,
                }
            ]
        ),
    )
    runtime_cameras = _runtime_cameras(client)

    assert posted.status_code == 200
    assert list(runtime_cameras) == ["local-uuid-1"]
    assert "backend-camera-1" not in runtime_cameras
    assert runtime_cameras["local-uuid-1"]["camera_id"] == "local-uuid-1"
    assert runtime_cameras["local-uuid-1"]["measured_fps"] == 11.0


def test_status_keeps_unmapped_local_id_as_dashboard_key(tmp_path: Path) -> None:
    client = _registry_status_client(
        tmp_path,
        camera_id="local-unmapped-2",
        backend_camera_id=None,
    )

    posted = _post(
        client,
        _payload(
            cameras=[
                {
                    **_camera_row(),
                    "camera_id": "local-unmapped-2",
                    "measured_fps": 6.5,
                }
            ]
        ),
    )
    runtime_cameras = _runtime_cameras(client)

    assert posted.status_code == 200
    assert list(runtime_cameras) == ["local-unmapped-2"]
    assert runtime_cameras["local-unmapped-2"]["camera_id"] == "local-unmapped-2"
    assert runtime_cameras["local-unmapped-2"]["measured_fps"] == 6.5


def test_status_retains_unknown_runtime_camera_id(tmp_path: Path) -> None:
    client = _registry_status_client(
        tmp_path,
        camera_id="local-uuid-1",
        backend_camera_id="backend-camera-1",
    )

    posted = _post(
        client,
        _payload(
            cameras=[
                {
                    **_camera_row(),
                    "camera_id": "ghost-camera",
                    "measured_fps": 3.0,
                }
            ]
        ),
    )
    runtime_cameras = _runtime_cameras(client)

    assert posted.status_code == 200
    assert list(runtime_cameras) == ["ghost-camera"]
    assert runtime_cameras["ghost-camera"]["camera_id"] == "ghost-camera"
    assert runtime_cameras["ghost-camera"]["measured_fps"] == 3.0
    assert runtime_cameras["ghost-camera"].get("unresolved") is True


def test_status_mapping_transition_emits_one_local_row_without_phantom(
    tmp_path: Path,
) -> None:
    client = _registry_status_client(
        tmp_path,
        camera_id="local-uuid-1",
        backend_camera_id="backend-camera-1",
    )
    generation = _json(
        _post(
            client,
            _payload(
                cameras=[
                    {
                        **_camera_row(),
                        "camera_id": "local-uuid-1",
                        "measured_fps": 4.0,
                    }
                ]
            ),
        )
    )["generation"]
    posted = _post(
        client,
        _payload(
            generation=generation,
            seq=1,
            cameras=[
                {
                    **_camera_row(),
                    "camera_id": "backend-camera-1",
                    "measured_fps": 9.5,
                }
            ],
        ),
    )
    runtime_cameras = _runtime_cameras(client)

    assert posted.status_code == 200
    assert list(runtime_cameras) == ["local-uuid-1"]
    assert "backend-camera-1" not in runtime_cameras
    assert runtime_cameras["local-uuid-1"]["measured_fps"] == 9.5

    both_aliases = _post(
        client,
        _payload(
            generation=generation,
            seq=2,
            cameras=[
                {
                    **_camera_row(),
                    "camera_id": "local-uuid-1",
                    "measured_fps": 1.0,
                },
                {
                    **_camera_row(),
                    "camera_id": "backend-camera-1",
                    "measured_fps": 13.0,
                },
            ],
        ),
    )
    merged = _runtime_cameras(client)

    assert both_aliases.status_code == 200
    assert list(merged) == ["local-uuid-1"]
    assert merged["local-uuid-1"]["measured_fps"] == 13.0


def test_status_keeps_unknown_camera_beside_known_mapped_camera(
    tmp_path: Path,
) -> None:
    client = _registry_status_client(
        tmp_path,
        camera_id="local-uuid-1",
        backend_camera_id="backend-camera-1",
    )

    posted = _post(
        client,
        _payload(
            cameras=[
                {
                    **_camera_row(),
                    "camera_id": "backend-camera-1",
                    "measured_fps": 10.0,
                },
                {
                    **_camera_row(),
                    "camera_id": "ghost-camera",
                    "measured_fps": 2.0,
                },
            ]
        ),
    )
    runtime_cameras = _runtime_cameras(client)

    assert posted.status_code == 200
    assert set(runtime_cameras) == {"local-uuid-1", "ghost-camera"}
    assert "backend-camera-1" not in runtime_cameras
    assert runtime_cameras["local-uuid-1"]["measured_fps"] == 10.0
    assert runtime_cameras["ghost-camera"]["measured_fps"] == 2.0
    assert runtime_cameras["ghost-camera"].get("unresolved") is True
    assert runtime_cameras["local-uuid-1"].get("unresolved") is not True


def test_latency_is_latest_memory_only_and_missing_after_restart(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    from backend.app.edge_db.migrator import migrate_database

    migrate_database(database)
    store = RuntimeStatusStore()
    store.record_latency("facility-1", "1970-01-01T00:00:00Z", received_at=10.0)
    store.record_latency("facility-1", "1970-01-01T00:00:00Z", received_at=20.0)
    store.record(_payload(), received_at=21.0)
    live = store.snapshot(now=21.0)["facilities"]["facility-1"]["latency"]
    restarted = RuntimeStatusStore()
    restarted.record(_payload(), received_at=21.0)

    assert live == {"first_attempt_samples": 2, "max_sec": 20.0, "since_sec": 10.0}
    assert restarted.snapshot(now=21.0)["facilities"]["facility-1"]["latency"] is None
    with sqlite3.connect(database) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert "runtime_latency" not in tables
    assert "control_heartbeats" not in tables


def test_runtime_status_missing_is_not_zeroed_success() -> None:
    store = RuntimeStatusStore(stale_after_sec=15.0)

    snapshot = store.snapshot(now=100.0)

    assert snapshot["facilities"] == {}
    assert snapshot["stale_after_sec"] == 15.0


def test_runtime_status_valid_empty_cameras_is_not_missing() -> None:
    store = RuntimeStatusStore(stale_after_sec=15.0)
    store.record(_payload(cameras=[]), received_at=100.0)

    snapshot = store.snapshot(now=100.0)
    facility = snapshot["facilities"]["facility-1"]

    assert facility["stale"] is False
    assert facility["cameras"] == []
    assert facility["received_at"] == 100.0


def test_runtime_status_future_received_at_is_not_accepted_as_fresh() -> None:
    store = RuntimeStatusStore(stale_after_sec=15.0, clock=lambda: 100.0)
    result = store.record(_payload(), received_at=1_000.0)

    snapshot = store.snapshot(now=100.0)

    assert result.accepted is False
    assert snapshot["facilities"] == {}


def test_runtime_status_none_generation_is_issued_and_retransmission_keeps_it() -> None:
    client = _client()

    first = _post(client, _payload())
    generation = first.json()["generation"]
    retransmission = _post(client, _payload(generation=generation, seq=1))

    assert first.status_code == 200
    assert first.json() == {"accepted": True, "generation": generation}
    assert retransmission.status_code == 200
    assert retransmission.json() == {"accepted": True, "generation": generation}
    runtime = client.get("/api/v1/status").json()["runtime"]
    assert runtime["facilities"]["facility-1"]["generation"] == generation
    assert runtime["facilities"]["facility-1"]["seq"] == 1


def test_runtime_status_none_generation_replaces_prior_worker_generation() -> None:
    client = _client()

    first_generation = _post(client, _payload()).json()["generation"]
    restarted = _post(client, _payload(generation=None, seq=0))

    assert restarted.status_code == 200
    assert restarted.json() == {"accepted": True, "generation": first_generation + 1}


def test_runtime_status_rejects_delayed_old_generation() -> None:
    client = _client()

    old_generation = _post(client, _payload()).json()["generation"]
    new_generation = _post(client, _payload(generation=None, seq=0)).json()["generation"]
    delayed = _post(client, _payload(generation=old_generation, seq=10))

    assert new_generation > old_generation
    assert delayed.status_code == 409
    assert delayed.json()["detail"] == "old_generation"


def test_runtime_status_rejects_reversed_sequence_within_generation() -> None:
    client = _client()

    generation = _post(client, _payload()).json()["generation"]
    assert _post(client, _payload(generation=generation, seq=2)).status_code == 200
    reversed_sequence = _post(client, _payload(generation=generation, seq=1))

    assert reversed_sequence.status_code == 409
    assert reversed_sequence.json()["detail"] == "old_seq"


def test_status_marks_applied_clip_export_stale_and_offline_separately() -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_inventory = {
        "camera-1": {"camera_id": "camera-1", "facility_id": "facility-1"}
    }
    app.state.runtime_status_store = RuntimeStatusStore(stale_after_sec=1.0)
    client = TestClient(app)

    app.state.runtime_status_store.record(_payload(), received_at=0.0)
    stale = client.get("/api/v1/status").json()["runtime"]["clip_export_applied"]
    app.state.runtime_status_store.record(
        _payload(generation=None, worker={"alive": False, "pid": None, "started_at_sec": None})
    )
    offline = client.get("/api/v1/status").json()["runtime"]["clip_export_applied"]

    assert stale == {"enabled": True, "version": 4, "freshness": "stale"}
    assert offline == {"enabled": True, "version": 4, "freshness": "offline"}


def test_runtime_status_snapshot_is_stale_after_ttl() -> None:
    store = RuntimeStatusStore(stale_after_sec=15.0)
    store.record(_payload(), received_at=1000.0)

    snapshot = store.snapshot(now=1016.0)

    assert snapshot["facilities"]["facility-1"]["received_at"] == 1000.0
    assert snapshot["facilities"]["facility-1"]["generation"] == 1
    assert snapshot["facilities"]["facility-1"]["stale"] is True


def test_runtime_status_rejects_missing_and_invalid_tokens() -> None:
    client = _client()

    missing = client.post("/api/v1/relay/runtime-status", json=_payload())
    wrong = client.post(
        "/api/v1/relay/runtime-status",
        json=_payload(),
        headers={"Authorization": "Bearer wrong"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 403


def test_runtime_status_accepts_despite_camera_inventory_facility_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the #183 regression: a stale/mismatched camera_inventory entry
    must not blank the dashboard for a facility that is otherwise legitimate.

    Reproduces the exact production shape -- camera_inventory's own embedded
    facility_id ("facility-prod") doesn't match the payload's facility_id
    ("facility-1") -- while leaving API_FACILITY_ID unset, matching a device
    that never got a real facility_id override. Before the fix,
    _runtime_status_facility_binding() derived a facility set from
    camera_inventory and 403'd anything not in it, unconditionally, even
    though camera_inventory has no bearing on this purely-local endpoint.
    """
    monkeypatch.delenv("API_FACILITY_ID", raising=False)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_inventory = {
        "cam-edge-01": {"camera_id": "cam-edge-01", "facility_id": "facility-prod"}
    }
    client = TestClient(app)

    response = _post(client, _payload(facility_id="facility-1"))

    assert response.status_code == 200
    assert response.json()["accepted"] is True


def test_runtime_status_accepts_despite_unresolved_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A camera absent from camera_inventory/camera_registry must not block
    the runtime-status snapshot for every other camera in the same payload:
    relay_runtime_status never forwards a canonical camera id anywhere (no
    backend egress happens here at all), so _camera_binding()'s "unknown
    camera" 403 protected nothing downstream -- it just discarded the
    dashboard's only view into camera-1's state.
    """
    monkeypatch.delenv("API_FACILITY_ID", raising=False)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_inventory = {}
    client = TestClient(app)

    response = _post(client, _payload(facility_id="facility-1"))

    assert response.status_code == 200
    assert response.json()["accepted"] is True


def test_runtime_status_accepts_any_facility_without_env_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env API_FACILITY_ID is no longer an admission gate for runtime-status."""
    monkeypatch.setenv("API_FACILITY_ID", "facility-configured")
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    client = TestClient(app)

    response = _post(client, _payload(facility_id="facility-other"))

    assert response.status_code == 200
    assert response.json()["accepted"] is True


def test_status_merges_runtime_snapshot() -> None:
    client = _client()
    assert _post(client, _payload()).status_code == 200

    response = client.get("/api/v1/status")

    assert response.status_code == 200
    runtime = response.json()["runtime"]
    facility = runtime["facilities"]["facility-1"]
    assert runtime["stale_after_sec"] == 15.0
    assert facility["stale"] is False
    assert facility["cameras"][0]["decode"]["selected"] == "opencv"
    assert facility["clip_recorder"]["finalized_clips"] == 2
    assert facility["clip_recorder"]["available"] is True
    assert runtime["cameras"]["camera-1"]["decode"]["selected"] == "opencv"
    assert runtime["clip_recorder"]["finalized_clips"] == 2
    assert runtime["clip_export_applied"] == {
        "enabled": True,
        "version": 4,
        "freshness": "fresh",
    }
    assert runtime["worker"] is None
    assert runtime["device"] is None


def test_runtime_status_rejects_non_backend_decode_values() -> None:
    invalid = _payload()
    cameras = invalid["cameras"]
    assert isinstance(cameras, list)
    cameras[0]["decode"]["requested"] = "h264"  # type: ignore[index]

    with pytest.raises(ValidationError):
        RelayRuntimeStatusRequest.model_validate(invalid)


def test_worker_runtime_payload_round_trips_through_runtime_status_route() -> None:
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode(
        "camera-1",
        DecodeSelection(
            requested="auto",
            selected="nvdec",
            fallback_count=0,
            last_reason=None,
            updated_at_sec=1000.0,
        ),
    )
    client = _client()

    response = _post(client, diagnostics.to_payload("facility-1", None, 4))

    assert response.status_code == 200
    facility = client.get("/api/v1/status").json()["runtime"]["facilities"]["facility-1"]
    assert facility["seq"] == 4
    assert facility["cameras"][0]["decode"]["selected"] == "nvdec"
    assert facility["clip_recorder"] == {
        "available": False,
        "dropped_frames": None,
        "dropped_events": None,
        "failed_writes": None,
        "finalized_clips": None,
        "video_unavailable_clips": None,
        "active_clips": None,
        "encoder": None,
    }


def test_profile_boot_failure_status_with_no_cameras_is_accepted() -> None:
    diagnostics = WorkerDiagnostics()
    diagnostics.set_gpu_status(
        {
            "nvml_available": False,
            "cuda_context_ok": False,
            "driver_version": None,
            "device_name": None,
            "nvml_error": "binding_unavailable",
            "captured_at_sec": 1.0,
        }
    )
    diagnostics.set_worker_status({"alive": False, "profile_boot_error": "no usable CUDA device"})
    client = _client()

    response = _post(client, diagnostics.to_payload("facility-1", None, 0))

    assert response.status_code == 200
    facility = client.get("/api/v1/status").json()["runtime"]["facilities"]["facility-1"]
    assert facility["cameras"] == []
    assert facility["clip_recorder"]["available"] is False
    assert facility["gpu"]["nvml_available"] is False
    assert facility["worker"]["profile_boot_error"] == "no usable CUDA device"


def test_decode_open_failure_is_visible_before_any_backend_is_selected() -> None:
    diagnostics = WorkerDiagnostics()
    diagnostics.register_decode("camera-1", "auto")
    diagnostics.record_decode_open_failure("camera-1", "spawn_failed")

    camera = diagnostics.to_payload("facility-1", None, 1)["cameras"][0]

    assert camera == {
        "camera_id": "camera-1",
        "decode": {
            "requested": "auto",
            "selected": None,
            "fallback_count": 0,
            "last_reason": "spawn_failed",
            "updated_at_sec": camera["decode"]["updated_at_sec"],
        },
        "detection": {
            "expected": False,
            "inference_admitted": 0,
            "inference_succeeded": 0,
            "inference_overwritten": 0,
            "decision_completed": 0,
        },
    }


def test_runtime_status_store_preserves_highest_sequence_under_concurrent_recording() -> None:
    store = RuntimeStatusStore()
    assert store.record(_payload(generation=1, seq=0)).accepted
    barrier = threading.Barrier(17)
    threads = [
        threading.Thread(
            target=lambda seq=seq: (
                barrier.wait(),
                store.record(_payload(generation=1, seq=seq)),
            ),
        )
        for seq in range(1, 17)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    snapshot = store.snapshot()
    assert snapshot["facilities"]["facility-1"]["seq"] == 16


def test_detection_health_is_bounded_and_consistent_under_concurrent_recording() -> None:
    store = RuntimeStatusStore(stale_after_sec=1000.0)

    def payload_for(seq: int) -> dict[str, object]:
        return _payload(
            generation=1,
            seq=seq,
            cameras=[
                _camera(
                    detection=_detection(
                        inference_admitted=seq,
                        inference_succeeded=seq,
                        inference_overwritten=0,
                        decision_completed=seq,
                    )
                )
            ],
        )

    assert store.record(payload_for(0), received_at=0.0).accepted
    barrier = threading.Barrier(17)
    threads = [
        threading.Thread(
            target=lambda seq=seq: (
                barrier.wait(),
                store.record(payload_for(seq), received_at=float(seq)),
            ),
        )
        for seq in range(1, 17)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    snapshot = store.snapshot(now=16.0)
    facility = _as_dict(_as_dict(snapshot["facilities"])["facility-1"])
    cameras = facility["cameras"]
    assert isinstance(cameras, list)
    detection = _as_dict(cameras[0])["detection"]
    assert isinstance(detection, dict)
    assert facility["seq"] == 16
    assert detection["decision_completed"] == 16
    assert detection["state"] == "healthy"
    assert len(store._detection_health) == 1
