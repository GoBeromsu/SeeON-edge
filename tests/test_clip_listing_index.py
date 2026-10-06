"""``GET /clips`` answers from the PostgreSQL catalogue without a request-time listing index."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


def test_http_listing_does_not_install_a_listing_index(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    with TestClient(app) as client:
        login = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
        assert login.status_code == 204
        response = client.get("/api/v1/clips", params={"limit": 48})
    assert response.status_code == 200
    assert response.json()["pagination"]["next_cursor"] is None
