"""Identity-boundary regression tests for issue #308.

The defect: an edge-local camera id was substituted wherever a Hub-issued
canonical id was required, so the Hub rejected it with
FACILITY_BINDING_MISMATCH and the edge only saw an opaque relay 502 -- which
was repeatedly misdiagnosed as an authentication problem.

Two properties are pinned here, and they pull in opposite directions:

1. The edge-local id must never cross the Hub boundary.
2. An unmapped camera must still be watched locally. worker-config's camera
   list is the exact ingest set (``worker/runtime/worker.py`` feeds it to
   ``build_camera_source_registry``), so dropping a camera there stops fall
   detection for that room. On a nursing-home edge an unwatched room is worse
   than a rejected upstream submission.

Enforcement therefore belongs at the Hub egress point, not at the worker
projection. That mirrors the periodic heartbeat relay, which already refuses to
push under an unmapped id.

These tests judge edge *reaction* (which id egresses, whether the camera is
still served) rather than whether a fixture's bytes satisfy the edge's own
parser, so they do not re-enter the fixture/parser self-consistency loop.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from test_api_ingest_relay import FakeBackendIngestClient

from backend.app.features.cameras.router import _mapping_state
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.main import create_app, no_lifespan

_LOCAL_ID = "11111111-2222-3333-4444-555555555555"
_HUB_ID = "cmsnvr-abc123"
_RELAY_HEADERS = {"X-Edge-Relay-Token": "relay-token"}


def _app_with_camera(tmp_path: Path, *, backend_camera_id: str | None, pending: bool = False):
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id=_LOCAL_ID,
        label="Room 101",
        rtsp_url="rtsp://example/room-101",
        space_id=None,
        status="online",
        backend_camera_id=backend_camera_id,
        mapping_pending=pending,
    )
    app.state.camera_registry = store
    return app, store


def _alert_body(camera_id: str) -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "facility_id": "facility-1",
        "event_type": "bed-exit",
        "detected_at": "2026-08-17T10:00:00.000Z",
        "probability": 0.97,
    }


def test_relay_alert_never_egresses_local_id_for_unmapped_camera(tmp_path) -> None:
    """An unmapped camera must not have its local id pushed to the Hub.

    The old code sent ``binding.get("camera_id") or payload.camera_id``, so the
    edge-local UUID reached the Hub and came back as FACILITY_BINDING_MISMATCH.
    """
    app, _ = _app_with_camera(tmp_path, backend_camera_id=None)
    fake = FakeBackendIngestClient()
    app.state.backend_ingest_client = fake

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/alerts", json=_alert_body(_LOCAL_ID), headers=_RELAY_HEADERS
        )

    # Locally accepted: the alert is still recorded on the edge.
    assert response.status_code == 202
    # But nothing addressed the Hub under a fabricated id.
    assert _LOCAL_ID not in fake.egress_camera_ids
    assert fake.alerts == []


def test_relay_alert_egresses_hub_id_when_mapped(tmp_path) -> None:
    """A mapped camera egresses under the Hub-issued id, not the local one."""
    app, _ = _app_with_camera(tmp_path, backend_camera_id=_HUB_ID)
    fake = FakeBackendIngestClient()
    app.state.backend_ingest_client = fake

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/alerts", json=_alert_body(_LOCAL_ID), headers=_RELAY_HEADERS
        )

    assert response.status_code == 202
    assert fake.egress_camera_ids == [_HUB_ID]
    assert _LOCAL_ID not in fake.egress_camera_ids
    assert len(fake.alerts) == 1


def test_worker_config_still_serves_unmapped_camera(tmp_path) -> None:
    """Coverage must survive an unmapped camera.

    worker-config.cameras is the worker's ingest set. Omitting an unmapped
    camera here would stop fall detection for that room, which is a strictly
    worse failure than a rejected upstream submission.
    """
    app, _ = _app_with_camera(tmp_path, backend_camera_id=None)
    app.state.backend_ingest_client = FakeBackendIngestClient()

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=_RELAY_HEADERS)

    assert response.status_code == 200
    cameras = response.json()["cameras"]
    assert len(cameras) == 1, "an unmapped camera must still reach the worker"
    assert cameras[0]["rtsp_url"] == "rtsp://example/room-101"


def test_mapping_state_distinguishes_pending_from_unmapped() -> None:
    """pending (waiting on Hub sync) is not the same as unmapped.

    A technician who has just registered a camera is in ``pending``; that is
    normal, not a defect, and the operator surface must say so.
    """
    assert _mapping_state({"backend_camera_id": _HUB_ID}) == "mapped"
    assert _mapping_state({"backend_camera_id": None, "mapping_pending": True}) == "pending"
    assert _mapping_state({"backend_camera_id": None, "mapping_pending": False}) == "unmapped"
    assert _mapping_state({}) == "unmapped"
    # Blank strings are not a mapping.
    assert _mapping_state({"backend_camera_id": "   "}) == "unmapped"
