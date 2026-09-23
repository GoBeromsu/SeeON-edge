"""Hermetic HTTP tests for execution-record ingest and engineer query."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.core.config import Settings, get_settings
from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.features.diagnostics.retention import RetentionBudget
from backend.app.features.diagnostics.store import ExecutionRecordStore
from backend.app.features.relay.router import RELAY_TOKEN_HEADER
from backend.app.lifespan import lifespan
from backend.app.main import create_app, no_lifespan
from shared.events.execution_records import (
    MAX_EXECUTION_RECORD_BODY_BYTES,
    WireBatch,
    WireGap,
    WireProvenance,
    WireRecord,
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


def _enabled_client(tmp_path: Path) -> TestClient:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = _RELAY_TOKEN
    app.state.backend_build_revision = _BUILD_REVISION
    app.state.execution_record_store = ExecutionRecordStore(
        lambda: open_runtime_database(database, actor=RuntimeActor.API),
        RetentionBudget(total_bytes=_BUDGET_BYTES),
    )
    return TestClient(app)


def test_relay_post_requires_token(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    missing = client.post(_PATH, json=_batch(_record(0)).to_json())
    assert missing.status_code == 401
    wrong = client.post(
        _PATH,
        json=_batch(_record(0)).to_json(),
        headers={RELAY_TOKEN_HEADER: "wrong"},
    )
    assert wrong.status_code == 403


def test_oversized_content_length_is_rejected(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
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


def test_chunked_oversized_body_is_rejected(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
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


def test_contract_violation_is_422(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
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


def test_committed_receipt_round_trip_and_idempotent_replay(
    tmp_path: Path, enabled_settings: None
) -> None:
    client = _enabled_client(tmp_path)
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


def test_storage_unavailable_receipt_is_still_200(tmp_path: Path, enabled_settings: None) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = _RELAY_TOKEN
    app.state.backend_build_revision = _BUILD_REVISION

    app.state.execution_record_store = ExecutionRecordStore(
        lambda: open_runtime_database(database, actor=RuntimeActor.API),
        RetentionBudget(total_bytes=256),
    )
    client = TestClient(app)
    response = client.post(
        _PATH,
        json=_batch(_record(0)).to_json(),
        headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN},
    )
    assert response.status_code == 200
    assert response.json()["storage_state"] == "STORAGE_UNAVAILABLE"
    assert response.json()["accepted"] == 0


def test_disabled_feature_answers_503(tmp_path: Path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = _RELAY_TOKEN
    client = TestClient(app)
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


def test_query_requires_dashboard_session(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    response = client.get(
        _QUERY,
        params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 10},
    )
    assert response.status_code == 401


def test_query_returns_records_and_unknown_tails(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
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


def test_query_limit_bounds_are_422(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
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


def test_lifespan_constructs_store_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled_settings: None
) -> None:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", _RELAY_TOKEN)
    app = create_app(lifespan=lifespan)
    with TestClient(app) as client:
        assert isinstance(client.app.state.execution_record_store, ExecutionRecordStore)
        assert client.app.state.backend_build_revision == _BUILD_REVISION
