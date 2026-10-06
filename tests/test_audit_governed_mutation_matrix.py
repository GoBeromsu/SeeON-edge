from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from contracts.edge_provisioning_models import (
    EnrollmentVerificationResult,
    FacilityIdentity,
    MachinePrincipal,
)
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert response.status_code == 204


def _verified_enrollment() -> EnrollmentVerificationResult:
    return EnrollmentVerificationResult(
        principal=MachinePrincipal("d17e0eb8-cb81-4d8e-a427-dfe690518f2b", 3),
        facility=FacilityIdentity("87d79f24-b32f-49a3-b534-19f0af7d9135", "Ward A"),
        server_revision=7,
    )


def _reject_audit_inserts(sandbox: ProductSandbox) -> None:
    sandbox.admin.execute(
        "CREATE OR REPLACE FUNCTION reject_audit_test() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'injected audit failure'; END $$"
    )
    sandbox.admin.execute(
        "CREATE TRIGGER reject_audit_test BEFORE INSERT ON audit_events "
        "FOR EACH ROW EXECUTE FUNCTION reject_audit_test()"
    )


def _edge_site(sandbox: ProductSandbox) -> list[tuple[object, ...]]:
    # Every governed mutation in the matrix owns columns of the single edge_site row.
    return sandbox.admin.execute("SELECT * FROM edge_site ORDER BY 1").fetchall()


def _nothing(_root: Path, _monkeypatch: pytest.MonkeyPatch) -> None:
    return None


def _prepare_storage(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIP_STORE_DIR", str(root / "clips"))
    (root / "clips" / "archive").mkdir(parents=True)


def _prepare_connection(_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.app.features.connection.router.verify_enrollment",
        lambda *_args, **_kwargs: _verified_enrollment(),
    )


@pytest.mark.parametrize(
    ("endpoint", "payload", "prepare"),
    (
        (
            "/api/v1/runtime-settings",
            {"clip_export_enabled": True, "expected_version": 0},
            _nothing,
        ),
        (
            "/api/v1/detection-settings",
            {
                "domains": {
                    "fall": {"on": True, "mode": "always"},
                    "bed_exit": {"on": False, "mode": "always"},
                }
            },
            _nothing,
        ),
        (
            "/api/v1/clips/storage/location",
            {"path": "archive"},
            _prepare_storage,
        ),
        (
            "/api/v1/connection",
            {
                "facility_code": "NH-7H2K9M4QXP",
                "facility_token": "eft_v1.token.secret",
                "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            },
            _prepare_connection,
        ),
    ),
)
def test_real_postgres_audit_denial_rolls_back_each_governed_mutation(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    payload: dict[str, JsonValue],
    prepare: Callable[[Path, pytest.MonkeyPatch], None],
) -> None:
    # Given: a valid governed mutation and a real PostgreSQL rejection only at audit INSERT.
    sandbox = postgres_product_sandbox
    prepare(tmp_path, monkeypatch)
    with TestClient(postgres_api_app(sandbox, postgres_audit_runtime)) as client:
        _login(client)
        before = _edge_site(sandbox)
        _reject_audit_inserts(sandbox)

        # When: the route attempts its caller-owned transactional audit append.
        response = client.put(endpoint, json=payload)

        # Then: no success bytes or business-state commit escape the failed transaction.
        assert response.status_code == 503
        assert response.content == b""
        assert _edge_site(sandbox) == before


def test_each_governed_mutation_commits_exactly_one_action(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: valid payloads for every mutation omitted by the first implementation.
    sandbox = postgres_product_sandbox
    _prepare_storage(tmp_path, monkeypatch)
    _prepare_connection(tmp_path, monkeypatch)
    mutations = (
        ("/api/v1/runtime-settings", {"clip_export_enabled": True, "expected_version": 0}),
        (
            "/api/v1/detection-settings",
            {
                "domains": {
                    "fall": {"on": True, "mode": "always"},
                    "bed_exit": {"on": False, "mode": "always"},
                }
            },
        ),
        ("/api/v1/clips/storage/location", {"path": "archive"}),
        (
            "/api/v1/connection",
            {
                "facility_code": "NH-7H2K9M4QXP",
                "facility_token": "eft_v1.token.secret",
                "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            },
        ),
    )

    # When: all four routes commit successfully.
    with TestClient(postgres_api_app(sandbox, postgres_audit_runtime)) as client:
        _login(client)
        responses = tuple(client.put(endpoint, json=payload) for endpoint, payload in mutations)

    # Then: each closed action appears exactly once.
    assert [response.status_code for response in responses] == [200, 200, 200, 200]
    counts = dict(
        sandbox.admin.execute(
            "SELECT action, COUNT(*) FROM audit_events WHERE action IN "
            "('runtime-settings.update','detection-settings.update',"
            "'clip-storage.update','connection.update') GROUP BY action"
        ).fetchall()
    )
    assert counts == {
        "runtime-settings.update": 1,
        "detection-settings.update": 1,
        "clip-storage.update": 1,
        "connection.update": 1,
    }
