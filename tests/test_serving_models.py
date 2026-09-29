from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


@pytest.fixture
def app(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> FastAPI:
    return postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)


def test_models_reports_gateway_metadata_only(app: FastAPI) -> None:
    app.state.camera_registry.create(
        camera_id="camera-1",
        label="c1",
        rtsp_url="rtsp://example/1",
        space_id=None,
        status="online",
    )
    app.state.backend_ingest_client = object()

    response = TestClient(app).get("/api/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "service": "ml-api",
        "role": "gateway",
        "ml": "external-worker",
        "relay": {"backend_configured": True, "camera_count": 1},
    }
