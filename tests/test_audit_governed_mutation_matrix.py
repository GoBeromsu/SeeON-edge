from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue

from backend.app.features.audit.store import AuditEvent, AuditRecord, AuditStore
from backend.app.features.audit.verification import SqlValue
from backend.app.main import create_app, no_lifespan
from contracts.edge_provisioning_models import (
    EnrollmentVerificationResult,
    FacilityIdentity,
    MachinePrincipal,
)


class AuthorizerAuditDenyStore(AuditStore):
    """Exercise SQLite's real authorizer at the audit INSERT boundary."""

    def _append(
        self, connection: sqlite3.Connection, event: AuditEvent
    ) -> AuditRecord:
        def authorize(
            action: int,
            arg1: str | None,
            _arg2: str | None,
            _database: str | None,
            _source: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_INSERT and arg1 == "audit_events":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return super()._append(connection, event)


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
    )
    assert response.status_code == 204


def _verified_enrollment() -> EnrollmentVerificationResult:
    return EnrollmentVerificationResult(
        principal=MachinePrincipal("d17e0eb8-cb81-4d8e-a427-dfe690518f2b", 3),
        facility=FacilityIdentity("87d79f24-b32f-49a3-b534-19f0af7d9135", "Ward A"),
        server_revision=7,
    )


def _snapshot(path: Path, table: str) -> tuple[tuple[SqlValue, ...], ...]:
    with sqlite3.connect(path) as connection:
        try:
            return tuple(connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall())
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error):
                raise
            return ()


def _nothing(_path: Path, _monkeypatch: pytest.MonkeyPatch) -> None:
    return None


def _prepare_storage(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIP_STORE_DIR", str(path.parent / "clips"))
    (path.parent / "clips" / "archive").mkdir(parents=True)


def _prepare_connection(_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.app.features.connection.router.verify_enrollment",
        lambda *_args, **_kwargs: _verified_enrollment(),
    )


@pytest.mark.parametrize(
    ("endpoint", "payload", "table", "prepare"),
    (
        (
            "/api/v1/runtime-settings",
            {"clip_export_enabled": True, "expected_version": 0},
            "runtime_settings",
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
            "detection_settings",
            _nothing,
        ),
        (
            "/api/v1/clips/storage/location",
            {"path": "archive"},
            "clip_storage_location",
            _prepare_storage,
        ),
        (
            "/api/v1/connection",
            {
                "facility_code": "NH-7H2K9M4QXP",
                "facility_token": "eft_v1.token.secret",
                "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            },
            "edge_site",
            _prepare_connection,
        ),
    ),
)
def test_real_sqlite_audit_denial_rolls_back_each_governed_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    payload: dict[str, JsonValue],
    table: str,
    prepare: Callable[[Path, pytest.MonkeyPatch], None],
) -> None:
    # Given: a valid governed mutation and a real authorizer denial only at audit INSERT.
    edge_database_path = tmp_path / ".central-fixture" / "edge.sqlite3"
    prepare(edge_database_path, monkeypatch)
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        before = _snapshot(edge_database_path, table)
        client.app.state.audit_store = AuthorizerAuditDenyStore(edge_database_path)

        # When: the route attempts its caller-owned transactional audit append.
        response = client.put(endpoint, json=payload)

        # Then: no success bytes or business-state commit escape the failed transaction.
        after = _snapshot(edge_database_path, table)
        assert response.status_code == 503
        assert response.content == b""
        assert after == before


def test_each_governed_mutation_commits_exactly_one_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: valid payloads for every mutation omitted by the first implementation.
    edge_database_path = tmp_path / ".central-fixture" / "edge.sqlite3"
    clips = edge_database_path.parent / "clips"
    (clips / "archive").mkdir(parents=True)
    monkeypatch.setenv("CLIP_STORE_DIR", str(clips))
    monkeypatch.setattr(
        "backend.app.features.connection.router.verify_enrollment",
        lambda *_args, **_kwargs: _verified_enrollment(),
    )
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
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        responses = tuple(client.put(endpoint, json=payload) for endpoint, payload in mutations)

    # Then: each closed action appears exactly once.
    assert [response.status_code for response in responses] == [200, 200, 200, 200]
    with sqlite3.connect(edge_database_path) as connection:
        counts = dict(
            connection.execute(
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
