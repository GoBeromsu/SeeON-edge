from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.features.relay.router import RelayRuntimeStatusRequest
from backend.app.features.status.router import _flatten_runtime_cameras
from backend.app.features.status.runtime_status_store import RuntimeStatusStore
from backend.app.main import create_app, no_lifespan
from contracts.decode_diagnostics import DecodeSelection
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics


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


def _post(client: TestClient, payload: dict[str, object]):
    return client.post(
        "/api/v1/relay/runtime-status",
        json=payload,
        headers={"Authorization": "Bearer relay-token"},
    )


def test_runtime_status_schema_round_trip_and_rejects_extra_fields() -> None:
    payload = _payload()

    parsed = RelayRuntimeStatusRequest.model_validate(payload)

    assert parsed.model_dump() == {
        **payload,
        "gpu": None,
        "worker": None,
        "cameras": [{**payload["cameras"][0], "measured_fps": None}],
    }
    with pytest.raises(ValidationError):
        RelayRuntimeStatusRequest.model_validate(_payload(unexpected=True))


def test_runtime_status_exposes_additive_diagnostics() -> None:
    client = _client()
    payload = _payload(
        cameras=[
            {
                **_payload()["cameras"][0],
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
        _payload(cameras=[{**_payload()["cameras"][0], "measured_fps": 5.0}]),
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
        _payload(cameras=[{**_payload()["cameras"][0], "measured_fps": 5.0}]),
        received_at=0.0,
    )

    facilities = store.snapshot(now=1.0)["facilities"]
    cameras = _flatten_runtime_cameras(facilities)
    assert cameras["camera-1"]["stale"] is False


def test_latency_max_persists_and_excludes_nonfirst_attempts(tmp_path: Path) -> None:
    state_path = tmp_path / "catalog.sqlite3"
    store = RuntimeStatusStore(latency_state_path=state_path)

    store.record_latency("facility-1", "1970-01-01T00:00:00Z", received_at=10.0)
    store.record_latency("facility-1", "1970-01-01T00:00:00Z", received_at=20.0)
    restarted = RuntimeStatusStore(latency_state_path=state_path)
    restarted.record_latency("facility-1", "1970-01-01T00:00:00Z", received_at=11.0)
    restarted.record(_payload(), received_at=21.0)

    latency = restarted.snapshot()["facilities"]["facility-1"]["latency"]
    assert latency == {"first_attempt_samples": 3, "max_sec": 20.0, "since_sec": 10.0}


def test_persist_latency_degrades_gracefully_when_the_connection_is_unusable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A broken SQLite connection at persist time must not crash the caller
    -- the store keeps functioning in-memory for the rest of the process
    lifetime, it just loses latency history across restarts. Mirrors the
    pre-SQLite-conversion OSError-only degradation contract, now broadened to
    also catch sqlite3.Error since the failure surface changed from file I/O
    to SQL."""
    state_path = tmp_path / "catalog.sqlite3"
    store = RuntimeStatusStore(latency_state_path=state_path)
    store.record_latency("facility-1", "1970-01-01T00:00:00Z", received_at=10.0)

    # Simulate the connection breaking underneath the store between a
    # successful load and a subsequent persist (e.g. the underlying file
    # became unwritable, or SQLite closed the connection on error).
    store._connection.close()

    with caplog.at_level("WARNING"):
        store.record_latency("facility-1", "1970-01-01T00:00:00Z", received_at=20.0)

    assert f"runtime latency store unavailable at {state_path}" in caplog.text
    # The store still functions in-memory: the new sample is reflected even
    # though it could not be durably persisted.
    latency = store._latency_for_facility("facility-1")
    assert latency == {"first_attempt_samples": 2, "max_sec": 20.0, "since_sec": 10.0}
    assert store.snapshot()["facilities"] == {}


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
