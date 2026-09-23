"""Adversarial ingest/query cases for execution-record HTTP and store surfaces."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from observability_stack_fixtures import serve_backend

from backend.app.core.config import get_settings
from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.features.diagnostics.records import CoverageKind, StorageState
from backend.app.features.diagnostics.retention import RetentionBudget
from backend.app.features.diagnostics.store import ExecutionRecordStore
from backend.app.features.relay.router import RELAY_TOKEN_HEADER
from backend.app.main import create_app, no_lifespan
from shared.events.execution_records import (
    MAX_EXECUTION_RECORD_BODY_BYTES,
    PROCESS_SCOPE,
    WireBatch,
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
        "payload": {"n": seq},
    }
    fields.update(overrides)
    return WireRecord(**fields)  # type: ignore[arg-type]


def _batch(*records: WireRecord) -> WireBatch:
    return WireBatch("cam-1", "boot-1", _PROVENANCE, records)


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


@pytest.fixture
def enabled_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ML_API_EXECUTION_RECORDS_ENABLED", "true")
    monkeypatch.setenv("ML_API_EXECUTION_RECORDS_BUDGET_BYTES", str(_BUDGET_BYTES))
    monkeypatch.setenv("ML_API_BUILD_REVISION", _BUILD_REVISION)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _enabled_client(tmp_path: Path, *, budget_bytes: int = _BUDGET_BYTES) -> TestClient:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = _RELAY_TOKEN
    app.state.backend_build_revision = _BUILD_REVISION
    app.state.execution_record_store = ExecutionRecordStore(
        lambda: open_runtime_database(database, actor=RuntimeActor.API),
        RetentionBudget(total_bytes=budget_bytes),
    )
    return TestClient(app)


def _count(path: Path, table: str) -> int:
    connection = open_runtime_database(path, actor=RuntimeActor.API)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def test_a1_tampered_record_id_is_422(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    body = _batch(_record(0)).to_json()
    body["records"][0]["record_id"] = "0" * 64
    response = client.post(_PATH, json=body, headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert response.status_code == 422
    assert "record_id" in response.json()["detail"]


def test_a2_tampered_batch_id_is_422(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    body = _batch(_record(0)).to_json()
    body["batch_id"] = "f" * 64
    response = client.post(_PATH, json=body, headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert response.status_code == 422
    assert "batch_id" in response.json()["detail"]


def test_a3_unknown_record_kind_is_422(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    body = _batch(_record(0)).to_json()
    body["records"][0]["record_kind"] = "not.a.kind"
    del body["batch_id"]
    del body["records"][0]["record_id"]
    response = client.post(_PATH, json=body, headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert response.status_code == 422
    assert "record_kind" in response.json()["detail"]


def test_a4_process_scoped_kind_with_stream_epoch_is_422(
    tmp_path: Path, enabled_settings: None
) -> None:
    client = _enabled_client(tmp_path)
    valid = _record(
        0,
        record_kind="backend.acceptance",
        producer="backend",
        source_generation=PROCESS_SCOPE,
        stream_epoch=PROCESS_SCOPE,
        outcome="accepted",
        payload={"state": "local"},
    )
    body = _batch(valid).to_json()
    body["records"][0]["stream_epoch"] = 3
    del body["batch_id"]
    del body["records"][0]["record_id"]
    response = client.post(_PATH, json=body, headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "process-scoped" in detail or "PROCESS_SCOPE" in detail


def test_a5_body_exactly_at_cap_is_not_413(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    payload = b"{" + (b" " * (MAX_EXECUTION_RECORD_BODY_BYTES - 2)) + b"}"
    assert len(payload) == MAX_EXECUTION_RECORD_BODY_BYTES
    response = client.post(
        _PATH,
        headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN, "Content-Type": "application/json"},
        content=payload,
    )
    assert response.status_code != 413
    assert response.status_code == 422


def test_a6_body_one_byte_over_cap_is_413(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    payload = b"{" + (b" " * (MAX_EXECUTION_RECORD_BODY_BYTES - 1)) + b"}"
    assert len(payload) == MAX_EXECUTION_RECORD_BODY_BYTES + 1
    response = client.post(
        _PATH,
        headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN, "Content-Type": "application/json"},
        content=payload,
    )
    assert response.status_code == 413


def test_a7_missing_relay_token_is_401(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    response = client.post(_PATH, json=_batch(_record(0)).to_json())
    assert response.status_code == 401


def test_a8_wrong_relay_token_is_403(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    response = client.post(
        _PATH,
        json=_batch(_record(0)).to_json(),
        headers={RELAY_TOKEN_HEADER: "wrong"},
    )
    assert response.status_code == 403


def test_a9_duplicate_batch_replay_is_byte_identical(
    tmp_path: Path, enabled_settings: None
) -> None:
    database = tmp_path / "edge.sqlite3"
    client = _enabled_client(tmp_path)
    payload = _batch(_record(0), _record(1)).to_json()
    first = client.post(_PATH, json=payload, headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert first.status_code == 200
    rows = _count(database, "execution_records")
    replay = client.post(_PATH, json=payload, headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert replay.status_code == 200
    assert replay.content == first.content
    assert replay.json()["storage_state"] == "committed"
    assert _count(database, "execution_records") == rows
    assert _count(database, "execution_batches") == 1


def test_a10_conflicting_record_keeps_original(tmp_path: Path, enabled_settings: None) -> None:
    database = tmp_path / "edge-state" / "edge.sqlite3"
    bootstrap_database(database)
    store = ExecutionRecordStore(
        lambda: open_runtime_database(database, actor=RuntimeActor.API),
        RetentionBudget(total_bytes=_BUDGET_BYTES),
    )
    from backend.app.features.diagnostics.records import (
        ExecutionRecordInput,
        IngestBatch,
        Provenance,
        RecordKind,
    )

    provenance = Provenance(
        worker_build_revision="worker-rev",
        worker_image_digest="sha256:worker",
        model_digest="sha256:model",
        calibration_digest="sha256:cal",
        preprocessing_identity="pre-v1",
        config_digest="sha256:cfg",
        policy_identity="policy-v1",
        backend_build_revision=_BUILD_REVISION,
    )
    record_id = _hex("same")

    def _input(payload: dict[str, object]) -> ExecutionRecordInput:
        return ExecutionRecordInput(
            record_id=record_id,
            record_kind=RecordKind.SDK_FRAME,
            camera_id="cam-a",
            worker_boot_id="boot-1",
            source_generation=0,
            stream_epoch=0,
            producer="sdk",
            producer_sequence=1,
            observed_at_ns=100,
            time_quality="trusted",
            causal_unit_id="unit-a",
            outcome="ok",
            payload=payload,
        )

    first = store.ingest_batch(
        IngestBatch(
            batch_id=_hex("first"),
            camera_id="cam-a",
            worker_boot_id="boot-1",
            provenance=provenance,
            records=(_input({"n": 1}),),
        )
    )
    assert first.storage_state is StorageState.COMMITTED
    conflict = store.ingest_batch(
        IngestBatch(
            batch_id=_hex("third"),
            camera_id="cam-a",
            worker_boot_id="boot-1",
            provenance=provenance,
            records=(_input({"n": 2}),),
        )
    )
    assert conflict.accepted == 0
    assert conflict.rejected == ((record_id, "conflict"),)
    connection = open_runtime_database(database, actor=RuntimeActor.API)
    try:
        payload = connection.execute(
            "SELECT payload FROM execution_records WHERE record_id = ?", (record_id,)
        ).fetchone()
    finally:
        connection.close()
    assert payload is not None and '"n":1' in str(payload[0])
    assert _count(database, "execution_records") == 1


def test_a11_payload_over_max_record_bytes_writes_rejected_oversize(
    tmp_path: Path, enabled_settings: None
) -> None:
    client = _enabled_client(tmp_path, budget_bytes=512 * 1024)
    budget = RetentionBudget(total_bytes=512 * 1024)
    record = _record(0, payload={"blob": "x" * (budget.max_record_bytes + 8)})
    response = client.post(
        _PATH, json=_batch(record).to_json(), headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] == 0
    assert body["rejected"] == [[record.record_id, "oversize"]]
    database = tmp_path / "edge.sqlite3"
    connection = open_runtime_database(database, actor=RuntimeActor.API)
    try:
        kinds = {
            str(row[0])
            for row in connection.execute("SELECT coverage_kind FROM execution_coverage")
        }
        records = connection.execute("SELECT COUNT(*) FROM execution_records").fetchone()
    finally:
        connection.close()
    assert CoverageKind.REJECTED_OVERSIZE in kinds
    assert records == (0,)


def test_a12_valid_json_that_is_not_an_object_is_422(
    tmp_path: Path, enabled_settings: None
) -> None:
    client = _enabled_client(tmp_path)
    array_body = client.post(_PATH, json=[1, 2, 3], headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN})
    assert array_body.status_code == 422
    string_body = client.post(
        _PATH,
        headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN, "Content-Type": "application/json"},
        content=b'"not-an-object"',
    )
    assert string_body.status_code == 422


def test_a13_ten_k_record_batch_is_bounded(tmp_path: Path) -> None:
    database = tmp_path / "edge-state" / "edge.sqlite3"
    bootstrap_database(database)
    store = ExecutionRecordStore(
        lambda: open_runtime_database(database, actor=RuntimeActor.API),
        RetentionBudget(total_bytes=32 * 2**20),
    )
    from backend.app.features.diagnostics.records import (
        ExecutionRecordInput,
        IngestBatch,
        Provenance,
        RecordKind,
    )

    provenance = Provenance(
        worker_build_revision="worker-rev",
        worker_image_digest="sha256:worker",
        model_digest="sha256:model",
        calibration_digest="sha256:cal",
        preprocessing_identity="pre-v1",
        config_digest="sha256:cfg",
        policy_identity="policy-v1",
        backend_build_revision=_BUILD_REVISION,
    )
    records = tuple(
        ExecutionRecordInput(
            record_id=_hex(f"r{index}"),
            record_kind=RecordKind.SDK_FRAME,
            camera_id="cam-a",
            worker_boot_id="boot-1",
            source_generation=0,
            stream_epoch=0,
            producer="sdk",
            producer_sequence=index,
            observed_at_ns=100 + index,
            time_quality="trusted",
            causal_unit_id="unit-bulk",
            outcome="ok",
            payload={},
        )
        for index in range(10_000)
    )
    started = time.perf_counter()
    receipt = store.ingest_batch(
        IngestBatch(
            batch_id=_hex("bulk-10k"),
            camera_id="cam-a",
            worker_boot_id="boot-1",
            provenance=provenance,
            records=records,
        )
    )
    elapsed_s = time.perf_counter() - started
    assert receipt.storage_state is StorageState.COMMITTED
    assert receipt.accepted == 10_000
    assert elapsed_s < 30.0
    Path(".omo/evidence/observability").mkdir(parents=True, exist_ok=True)
    Path(".omo/evidence/observability/redteam-a13-elapsed.json").write_text(
        json.dumps({"ten_k_ingest_sec": elapsed_s, "accepted": receipt.accepted}),
        encoding="utf-8",
    )


def test_b6_query_from_ns_greater_than_to_ns_is_422(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    _login(client)
    response = client.get(_QUERY, params={"camera_id": "cam-1", "from_ns": 20, "to_ns": 10})
    assert response.status_code == 422


def test_b7_query_limit_zero_is_422(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    _login(client)
    response = client.get(
        _QUERY, params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 10, "limit": 0}
    )
    assert response.status_code == 422


def test_b8_query_limit_501_is_422(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    _login(client)
    response = client.get(
        _QUERY, params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 10, "limit": 501}
    )
    assert response.status_code == 422


def test_b9_cursor_from_other_camera_never_leaks(tmp_path: Path, enabled_settings: None) -> None:
    client = _enabled_client(tmp_path)
    first = client.post(
        _PATH,
        json=_batch(_record(0), _record(1), _record(2)).to_json(),
        headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN},
    )
    assert first.status_code == 200
    other = _record(0, camera_id="cam-2", causal_unit_id="unit-2")
    other_batch = WireBatch("cam-2", "boot-1", _PROVENANCE, (other,))
    posted = client.post(
        _PATH, json=other_batch.to_json(), headers={RELAY_TOKEN_HEADER: _RELAY_TOKEN}
    )
    assert posted.status_code == 200
    _login(client)
    page = client.get(
        _QUERY,
        params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 5_000, "limit": 1},
    )
    assert page.status_code == 200
    cursor = page.json()["next_cursor"]
    assert cursor
    leaked = client.get(
        _QUERY,
        params={
            "camera_id": "cam-2",
            "from_ns": 0,
            "to_ns": 5_000,
            "limit": 10,
            "cursor": cursor,
        },
    )
    assert leaked.status_code in {200, 422}
    if leaked.status_code == 200:
        records = leaked.json()["records"]
        assert all(row.get("causal_unit_id") != "unit-1" for row in records)
        cameras = {row.get("causal_unit_id") for row in records}
        assert cameras <= {"unit-2"} or records == []


def test_live_uvicorn_relay_and_query_adversarial(tmp_path: Path) -> None:
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        url = f"{backend.base_url}{_PATH}"
        headers = {RELAY_TOKEN_HEADER: backend.relay_token}
        tampered = _batch(_record(0)).to_json()
        tampered["records"][0]["record_id"] = "0" * 64
        bad_id = httpx.post(url, json=tampered, headers=headers, timeout=5.0)
        assert bad_id.status_code == 422
        missing = httpx.post(url, json=_batch(_record(0)).to_json(), timeout=5.0)
        assert missing.status_code == 401
        payload = _batch(_record(0), _record(1)).to_json()
        first = httpx.post(url, json=payload, headers=headers, timeout=5.0)
        assert first.status_code == 200
        replay = httpx.post(url, json=payload, headers=headers, timeout=5.0)
        assert replay.status_code == 200
        assert replay.content == first.content
        session = backend.dashboard_session()
        try:
            inverted = session.get(_QUERY, params={"camera_id": "cam-1", "from_ns": 9, "to_ns": 1})
            assert inverted.status_code == 422
            over = session.get(
                _QUERY,
                params={"camera_id": "cam-1", "from_ns": 0, "to_ns": 10, "limit": 501},
            )
            assert over.status_code == 422
        finally:
            session.close()
        oversize = httpx.post(
            url,
            headers={**headers, "Content-Type": "application/json"},
            content=b"x" * (MAX_EXECUTION_RECORD_BODY_BYTES + 1),
            timeout=5.0,
        )
        assert oversize.status_code == 413
