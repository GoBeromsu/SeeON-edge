"""Hermetic HTTP tests for execution-record ingest and engineer query."""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg import sql
from pydantic import ValidationError

from backend.app.core.config import Settings, get_settings
from backend.app.edge_db.postgres import PoolBudget, PostgresDatabase
from backend.app.features.audit.catalog import AuditAction
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.diagnostics.retention import RetentionBudget
from backend.app.features.diagnostics.store import ExecutionRecordStore
from backend.app.features.relay.router import RELAY_TOKEN_HEADER
from backend.app.lifespan import lifespan
from backend.app.main import create_app
from shared.events.execution_records import (
    MAX_EXECUTION_RECORD_BODY_BYTES,
    WireBatch,
    WireGap,
    WireProvenance,
    WireRecord,
)
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_app_env import inject_sandbox_root
from tests_support.postgres_diagnostics_sandbox import DiagnosticsSandbox
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = (
    "tests_support.postgres_sandbox",
    "tests_support.postgres_app_env",
    "tests_support.postgres_diagnostics_sandbox",
)

_RELAY_TOKEN = "relay-token"
_BUILD_REVISION = "backend-rev-1"
_BUDGET_BYTES = 2**20
_PATH = "/api/v1/relay/execution-records"
_QUERY = "/api/v1/diagnostics/executions"
_PROVENANCE = WireProvenance(
    worker_build_revision="abc123",
    worker_image_digest="sha256:deadbeef",
    model_digest="m1",
    calibration_digest="c1",
    preprocessing_identity="pose-bbox56/v1",
    config_digest="cfg1",
    policy_identity="fall.policy:2",
)


def _record(seq: int, **overrides: object) -> WireRecord:
    fields: dict[str, object] = {
        "record_kind": "model.score",
        "camera_id": "cam-1",
        "worker_boot_id": "boot-1",
        "source_generation": 0,
        "stream_epoch": 3,
        "producer": "model",
        "producer_sequence": seq,
        "observed_at_ns": 1_000 + seq,
        "time_quality": "monotonic",
        "causal_unit_id": "unit-1",
        "outcome": "scored",
        "payload": {"raw_logit": -0.25, "temperature": 1.7},
    }
    fields.update(overrides)
    return WireRecord(**fields)  # type: ignore[arg-type]


def _batch(*records: WireRecord, gaps: tuple[WireGap, ...] = ()) -> WireBatch:
    return WireBatch("cam-1", "boot-1", _PROVENANCE, records, gaps)


def _oversized_chunks(total_bytes: int, *, chunk: int = 64 * 1024) -> Iterator[bytes]:
    sent = 0
    while sent < total_bytes:
        step = min(chunk, total_bytes - sent)
        sent += step
        yield b"a" * step


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


@pytest.fixture
def enabled_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_API_EXECUTION_RECORDS_ENABLED", "true")
    monkeypatch.setenv("ML_API_EXECUTION_RECORDS_BUDGET_BYTES", str(_BUDGET_BYTES))
    monkeypatch.setenv("ML_API_BUILD_REVISION", _BUILD_REVISION)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def product_app(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> FastAPI:
    """A no-lifespan app on the sandbox root, so ``_login`` reaches a real session."""
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.edge_relay_token = _RELAY_TOKEN
    return app


def _client_with_store(app: FastAPI, store: ExecutionRecordStore) -> TestClient:
    app.state.backend_build_revision = _BUILD_REVISION
    app.state.execution_record_store = store
    return TestClient(app)


@pytest.fixture
def enabled_client(
    enabled_settings: None,
    product_app: FastAPI,
    postgres_diagnostics_sandbox: DiagnosticsSandbox,
) -> TestClient:
    return _client_with_store(
        product_app,
        ExecutionRecordStore(
            postgres_diagnostics_sandbox.database, RetentionBudget(total_bytes=_BUDGET_BYTES)
        ),
    )


def test_relay_post_requires_token(enabled_client: TestClient) -> None:
    client = enabled_client
    missing = client.post(_PATH, json=_batch(_record(0)).to_json())
    assert missing.status_code == 401
    wrong = client.post(
        _PATH,
        json=_batch(_record(0)).to_json(),
        headers={RELAY_TOKEN_HEADER: "wrong"},
    )
    assert wrong.status_code == 403


def test_oversized_content_length_is_rejected(enabled_client: TestClient) -> None:
    client = enabled_client
    response = client.post(
        _PATH,
        headers={
            RELAY_TOKEN_HEADER: _RELAY_TOKEN,
            "Content-Type": "application/json",
            "Content-Length": str(MAX_EXECUTION_RECORD_BODY_BYTES + 1),
        },
        content=b"{}",
    )
    assert response.status_code == 413


def test_chunked_oversized_body_is_rejected(enabled_client: TestClient) -> None:
    client = enabled_client
    over = MAX_EXECUTION_RECORD_BODY_BYTES + 4096
    response = client.post(
        _PATH,
        headers={
            RELAY_TOKEN_HEADER: _RELAY_TOKEN,
            "Content-Type": "application/json",
        },
        content=_oversized_chunks(over),
    )
    assert response.status_code == 413


def test_contract_violation_is_422(enabled_client: TestClient) -> None:
    client = enabled_client
    body = _batch(_record(0)).to_json()
    body["records"][0]["record_id"] = "0" * 64
    bad_id = client.post(_PATH, json=body, headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert bad_id.status_code == 422
    assert "record_id" in bad_id.json()["detail"]
    body = _batch(_record(0)).to_json()
    body["records"][0]["record_kind"] = "not.a.kind"
    del body["batch_id"]
    del body["records"][0]["record_id"]
    bad_kind = client.post(_PATH, json=body, headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert bad_kind.status_code == 422
    assert "record_kind" in bad_kind.json()["detail"]


def test_committed_receipt_round_trip_and_idempotent_replay(enabled_client: TestClient) -> None:
    client = enabled_client
    payload = _batch(_record(0), _record(1)).to_json()
    first = client.post(_PATH, json=payload, headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert first.status_code == 200
    receipt = first.json()
    assert receipt["storage_state"] == "committed"
    assert receipt["accepted"] == 2
    assert receipt["duplicates"] == 0
    assert receipt["batch_id"] == payload["batch_id"]
    replay = client.post(_PATH, json=payload, headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert replay.status_code == 200
    assert replay.json() == receipt


@pytest.mark.usefixtures("enabled_settings")
def test_storage_unavailable_receipt_is_still_200(
    product_app: FastAPI, postgres_diagnostics_sandbox: DiagnosticsSandbox
) -> None:
    client = _client_with_store(
        product_app,
        ExecutionRecordStore(
            postgres_diagnostics_sandbox.database, RetentionBudget(total_bytes=256)
        ),
    )
    response = client.post(
        _PATH,
        json=_batch(_record(0)).to_json(),
        headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN},
    )
    assert response.status_code == 200
    assert response.json()["storage_state"] == "STORAGE_UNAVAILABLE"
    assert response.json()["accepted"] == 0


def test_disabled_feature_answers_503(product_app: FastAPI) -> None:
    client = TestClient(product_app)
    ingest = client.post(
        _PATH,
        json=_batch(_record(0)).to_json(),
        headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN},
    )
    assert ingest.status_code == 503
    assert ingest.json()["detail"] == "execution records disabled"
    _login(client)
    query = client.get(
        _QUERY,
        params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 10},
    )
    assert query.status_code == 503
    assert query.json()["detail"] == "execution records disabled"


@pytest.mark.usefixtures("enabled_settings")
def test_missing_diagnostics_database_answers_503(
    product_app: FastAPI, postgres_diagnostics_sandbox: DiagnosticsSandbox
) -> None:
    """A diagnostics database that refuses admission (server down at boot, so
    ``open_diagnostics_database`` could not start its pool) must not leak an
    opaque 500 -- PostgresUnavailable is mapped to a clear 503 (#579/#580, round 2)."""
    unstarted = PostgresDatabase(
        postgres_diagnostics_sandbox.dsn,
        postgres_diagnostics_sandbox.schema,
        PoolBudget(
            max_connections=1,
            max_waiting=1,
            acquire_timeout_sec=1.0,
            statement_timeout_ms=5000,
            lock_timeout_ms=3000,
            startup_timeout_sec=5.0,
        ),
    )
    client = _client_with_store(
        product_app, ExecutionRecordStore(unstarted, RetentionBudget(total_bytes=_BUDGET_BYTES))
    )
    ingest = client.post(
        _PATH,
        json=_batch(_record(0)).to_json(),
        headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN},
    )
    assert ingest.status_code == 503
    assert ingest.json()["detail"] == (
        "diagnostics store unavailable: check PostgreSQL and run migration provision"
    )
    _login(client)
    query = client.get(
        _QUERY,
        params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 10},
    )
    assert query.status_code == 503
    assert query.json()["detail"] == (
        "diagnostics store unavailable: check PostgreSQL and run migration provision"
    )


def test_query_requires_dashboard_session(enabled_client: TestClient) -> None:
    client = enabled_client
    response = client.get(
        _QUERY,
        params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 10},
    )
    assert response.status_code == 401


def test_query_returns_records_and_unknown_tails(enabled_client: TestClient) -> None:
    client = enabled_client
    payload = _batch(_record(0), _record(1), _record(2)).to_json()
    posted = client.post(_PATH, json=payload, headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert posted.status_code == 200
    _login(client)
    empty = client.get(
        _QUERY,
        params={"camera_id": "missing", "from_ns": 0, "to_ns": 50},
    )
    assert empty.status_code == 200
    body = empty.json()
    assert body["records"] == []
    assert body["availability"]
    assert all(row["kind"] == "UNKNOWN" for row in body["availability"])
    page = client.get(
        _QUERY,
        params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 5_000, "limit": 2},
    )
    assert page.status_code == 200
    first = page.json()
    assert [row["producer_sequence"] for row in first["records"]] == [0, 1]
    assert first["next_cursor"] is not None
    rest = client.get(
        _QUERY,
        params={
            "camera_id": "cam-1",
            "from_ns": 0,
            "to_ns": 5_000,
            "limit": 2,
            "cursor": first["next_cursor"],
        },
    )
    assert rest.status_code == 200
    assert [row["producer_sequence"] for row in rest.json()["records"]] == [2]
    assert rest.json()["next_cursor"] is None


def test_query_limit_bounds_are_422(enabled_client: TestClient) -> None:
    client = enabled_client
    _login(client)
    too_low = client.get(
        _QUERY,
        params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 10, "limit": 0},
    )
    assert too_low.status_code == 422
    too_high = client.get(
        _QUERY,
        params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 10, "limit": 501},
    )
    assert too_high.status_code == 422
    inverted = client.get(
        _QUERY,
        params={"camera_id": "cam-1", "from_ns": 20, "to_ns": 10},
    )
    assert inverted.status_code == 422


def test_boot_refuses_when_enabled_without_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_API_EXECUTION_RECORDS_ENABLED", "true")
    monkeypatch.delenv("ML_API_EXECUTION_RECORDS_BUDGET_BYTES", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValidationError, match="ML_API_EXECUTION_RECORDS_BUDGET_BYTES"):
        Settings()
    get_settings.cache_clear()


@pytest.mark.usefixtures("postgres_app_env", "postgres_lifespan_diagnostics_schema")
def test_lifespan_constructs_store_when_enabled(
    monkeypatch: pytest.MonkeyPatch, enabled_settings: None
) -> None:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", _RELAY_TOKEN)
    app = create_app(lifespan=lifespan)
    with TestClient(app) as client:
        assert isinstance(client.app.state.execution_record_store, ExecutionRecordStore)
        assert client.app.state.backend_build_revision == _BUILD_REVISION


@pytest.mark.usefixtures("enabled_settings", "postgres_app_env")
def test_lifespan_diagnostics_query_and_product_write_skip_a_pending_diagnostics_write(
    monkeypatch: pytest.MonkeyPatch,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    postgres_lifespan_diagnostics_schema: str,
) -> None:
    """Through the real lifespan wiring: while another session holds the
    diagnostics ingest lock (standing in for an in-flight ingest write), a
    diagnostics query and a product write both complete. The diagnostics store
    has its own schema and pool, and a snapshot query never takes the ingest
    lock (#579/#580, S4).
    """
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", _RELAY_TOKEN)
    app = create_app(lifespan=lifespan)
    inject_sandbox_root(app, postgres_product_sandbox, postgres_audit_runtime)
    admin = postgres_product_sandbox.admin
    with TestClient(app) as client:
        posted = client.post(
            _PATH,
            json=_batch(_record(0)).to_json(),
            headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN},
        )
        assert posted.status_code == 200
        _login(client)

        holder = psycopg.connect(postgres_product_sandbox.dsn, connect_timeout=5)
        try:
            # The first statement of every ingest transaction.
            holder.execute(
                sql.SQL("LOCK TABLE {}.execution_batches IN SHARE ROW EXCLUSIVE MODE").format(
                    sql.Identifier(postgres_lifespan_diagnostics_schema)
                )
            )
            query = client.get(
                _QUERY,
                params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 5_000},
            )
            assert query.status_code == 200
            assert [row["producer_sequence"] for row in query.json()["records"]] == [0]

            # A governed product write: the session read appends an audit event.
            session = client.get("/api/v1/auth/session")
            assert session.status_code == 204
            committed = admin.execute(
                "SELECT count(*) FROM audit_events WHERE action = %s",
                (AuditAction.AUTH_SESSION_READ.value,),
            ).fetchone()
            assert committed == (1,)
        finally:
            holder.rollback()
            holder.close()
