from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.main import create_app, no_lifespan
from tests_support.compact_authority_db import prepare_compact_database


def test_models_reports_gateway_metadata_only(tmp_path: Path) -> None:
    app = create_app(lifespan=no_lifespan)
    registry_path = tmp_path / "catalog.sqlite3"
    prepare_compact_database(registry_path)
    store = CameraRegistryStore(registry_path)
    store.create(
        camera_id="camera-1",
        label="c1",
        rtsp_url="rtsp://example/1",
        space_id=None,
        status="online",
    )
    app.state.camera_registry = store
    app.state.backend_ingest_client = object()

    response = TestClient(app).get("/api/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "service": "ml-api",
        "role": "gateway",
        "ml": "external-worker",
        "relay": {"backend_configured": True, "camera_count": 1},
    }
