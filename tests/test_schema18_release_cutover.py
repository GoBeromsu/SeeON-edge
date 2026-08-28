"""Schema-18 release identity, Compose cutover, and deterministic failure gates."""

from __future__ import annotations

import io
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from backend.app.edge_db.ownership import COMPACT_APPLICATION_TABLES
from backend.app.main import create_app, no_lifespan
from shared.release_identity import (
    EDGE_DATABASE_FORMAT_IDENTITY,
    EDGE_DATABASE_SCHEMA_VERSION,
    ReleaseIdentityMismatchError,
    require_peer_schema_identity,
)

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "compose.edge.yaml"
RUNBOOK = ROOT / "docs" / "runbooks" / "schema-18-ten-table-cutover.md"
IMAGE_IDENTITY_MARKER = "/opt/seeon/edge-database-schema-version"
SCHEMA18_DRAIN_SENTINEL = "EDGE_DB_DRAIN_INCOMPLETE"


def _compose() -> dict[str, Any]:
    class Loader(yaml.SafeLoader):
        pass

    def compose_tag(loader: Loader, _suffix: str, node: yaml.Node) -> object:
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return None

    Loader.add_multi_constructor("!", compose_tag)
    payload = yaml.load(COMPOSE.read_text(encoding="utf-8"), Loader=Loader)
    assert isinstance(payload, dict)
    return payload


def _in_flight(database: Path) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id, detected_at, payload_json, state, queued_at, "
            " next_attempt_at, attempt_count, lease_owner, lease_expires_at, "
            " delivery_state) "
            "VALUES ('e1', '2026-08-22T00:00:00Z', '{}', 'IN_FLIGHT', "
            "        1787000000.0, 1787000000.0, 0, 'sender-1', 1787003600.0, "
            "        'PENDING')"
        )
        connection.commit()
    finally:
        connection.close()


def test_shared_front_and_worker_images_advertise_schema_18() -> None:
    assert EDGE_DATABASE_SCHEMA_VERSION == 18
    assert EDGE_DATABASE_FORMAT_IDENTITY == "seeon-edge-v1"
    front = (ROOT / "front" / "src" / "shared" / "releaseIdentity.ts").read_text(
        encoding="utf-8"
    )
    assert re.search(r"EDGE_DATABASE_SCHEMA_VERSION\s*=\s*18", front)
    assert "seeon-edge-v1" in front
    backend_image = (ROOT / "Dockerfile.backend").read_text(encoding="utf-8")
    worker_image = (ROOT / "Dockerfile.edge").read_text(encoding="utf-8")
    for source in (backend_image, worker_image):
        assert IMAGE_IDENTITY_MARKER in source
        assert "seeon.edge.database.schema-version" in source
        assert "EDGE_DATABASE_SCHEMA_VERSION" in source


def test_mixed_image_identity_refuses() -> None:
    require_peer_schema_identity(18)
    with pytest.raises(ReleaseIdentityMismatchError, match="17"):
        require_peer_schema_identity(17)


def test_api_advertises_schema_18_release_identity() -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        response = client.get("/health/release-identity")
    assert response.status_code == 200
    assert response.json() == {
        "format": EDGE_DATABASE_FORMAT_IDENTITY,
        "edge_database_schema_version": EDGE_DATABASE_SCHEMA_VERSION,
    }


def test_worker_startup_refuses_mixed_api_identity() -> None:
    from worker.runtime.config.release_pair import require_api_release_identity

    def urlopen(request: object, timeout: float = 0) -> io.BytesIO:
        del request, timeout
        return io.BytesIO(b'{"format":"seeon-edge-v1","edge_database_schema_version":17}')

    with pytest.raises(ReleaseIdentityMismatchError, match="17"):
        require_api_release_identity("http://ml-api:8000", urlopen=urlopen)


def test_compose_orders_inventory_candidate_cutover_api_worker() -> None:
    compose = _compose()
    services = compose["services"]
    assert isinstance(services, dict)
    inventory = services["edge-filesystem-inventory"]
    migrator = services["edge-db-migrator"]
    api = services["ml-api"]
    worker = services["ml-worker"]
    assert isinstance(inventory, dict)
    assert isinstance(migrator, dict)
    assert isinstance(api, dict)
    assert isinstance(worker, dict)
    assert inventory["command"][-1] == "backend.app.edge_db.inventory"
    assert "backend.app.edge_db.compact_cutover" in migrator["command"]
    assert migrator["command"][migrator["command"].index("--live") + 1] == (
        "/var/lib/seeon-state/edge.sqlite3"
    )
    assert migrator["depends_on"] == {
        "edge-filesystem-inventory": {"condition": "service_completed_successfully"}
    }
    assert api["depends_on"] == {
        "edge-db-migrator": {"condition": "service_completed_successfully"}
    }
    # The worker also waits on the verified models volume (edge-model-fetch);
    # the database cutover ordering above is unchanged by that.
    assert worker["depends_on"] == {
        "ml-api": {"condition": "service_healthy"},
        "edge-model-fetch": {"condition": "service_completed_successfully"},
    }


def test_cli_help_matches_compose_and_runbook_flags() -> None:
    from backend.app.edge_db.compact_cutover import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    help_text = io.StringIO()
    import sys

    original = sys.stdout
    sys.stdout = help_text
    try:
        try:
            main(["--help"])
        except SystemExit:
            pass
    finally:
        sys.stdout = original
    rendered = help_text.getvalue()
    for flag in (
        "--source",
        "--live",
        "--archive",
        "--candidate",
        "--receipt",
        "--clip-store",
        "--worker-state",
        "--expected-source-sha256",
        "--rollback",
    ):
        assert flag in rendered
    compose = COMPOSE.read_text(encoding="utf-8")
    runbook = RUNBOOK.read_text(encoding="utf-8")
    assert "backend.app.edge_db.compact_cutover" in compose
    assert "python -m backend.app.edge_db.compact_cutover" in runbook
    assert "--rollback" in runbook


def test_empty_v17_candidate_cutover_installs_schema_18(
    tmp_path: Path, supported_compact_cutover_sqlite: None
) -> None:
    from compact_cutover_fixtures import cutover_request, sha256

    from backend.app.edge_db.compact_cutover import run_compact_cutover

    request = cutover_request(tmp_path)
    before = sha256(request.source)
    result = run_compact_cutover(request)
    assert result.source_rows >= 1
    assert sha256(request.archive) == before
    receipts = request.receipt.read_text(encoding="utf-8").splitlines()
    assert len(receipts) == result.source_rows
    assert all('"action":' in line for line in receipts)
    with sqlite3.connect(request.live) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (18,)
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert tables == COMPACT_APPLICATION_TABLES
    assert oct(request.archive.stat().st_mode & 0o777) == "0o400"


def test_populated_v17_emits_one_receipt_per_source_row(
    tmp_path: Path, supported_compact_cutover_sqlite: None
) -> None:
    import json

    from compact_cutover_dense_fixture import dense_cutover_request
    from compact_cutover_fixtures import sha256

    from backend.app.edge_db.compact_cutover import run_compact_cutover

    request = dense_cutover_request(tmp_path)
    source_hash = sha256(request.source)
    with sqlite3.connect(request.source) as source:
        tables = [
            str(row[0])
            for row in source.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        source_rows = sum(
            int(source.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
            for table in tables
        )
    assert len(tables) == 72
    result = run_compact_cutover(request)
    receipts = [json.loads(line) for line in request.receipt.read_text().splitlines()]
    assert result.source_rows == source_rows == len(receipts)
    assert {record["action"] for record in receipts} <= {"MAP", "REBUILD", "NONE"}
    assert {record["source_table"] for record in receipts} == set(tables)
    assert sha256(request.source) == source_hash == sha256(request.archive)
    with sqlite3.connect(request.live) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (18,)


def test_old_sqlite_refuses_before_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from compact_cutover_fixtures import cutover_request

    from backend.app.edge_db import compact_cutover
    from backend.app.edge_db.compact_cutover import run_compact_cutover
    from backend.app.edge_db.sqlite_runtime import SqliteVersionTooOldError

    request = cutover_request(tmp_path)
    before = request.live.read_bytes()
    monkeypatch.setattr(compact_cutover, "_runtime_sqlite_version", lambda: (3, 45, 1))
    with pytest.raises(SqliteVersionTooOldError, match="3.45.1"):
        run_compact_cutover(request)
    assert request.live.read_bytes() == before
    assert not request.archive.exists()
    assert not request.candidate.exists()


def test_in_flight_source_refuses_without_replacement(
    tmp_path: Path, supported_compact_cutover_sqlite: None
) -> None:
    from compact_cutover_fixtures import cutover_request

    from backend.app.edge_db.compact_cutover import run_compact_cutover
    from backend.app.edge_db.schema import SchemaV18MigrationError

    request = cutover_request(tmp_path)
    _in_flight(request.source)
    before = request.live.read_bytes()
    with pytest.raises(SchemaV18MigrationError, match=SCHEMA18_DRAIN_SENTINEL):
        run_compact_cutover(request)
    assert request.live.read_bytes() == before
    assert not request.candidate.exists()
    with sqlite3.connect(request.live) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)


def test_bad_archive_hash_refuses_without_replacement(
    tmp_path: Path, supported_compact_cutover_sqlite: None
) -> None:
    from compact_cutover_fixtures import cutover_request

    from backend.app.edge_db.compact_cutover import CompactCutoverError, run_compact_cutover

    request = cutover_request(tmp_path)
    before = request.live.read_bytes()
    request.archive.write_bytes(b"not-the-source-archive")
    with pytest.raises(CompactCutoverError, match="EDGE_DB_CUTOVER_STALE_ARCHIVE"):
        run_compact_cutover(request)
    assert request.live.read_bytes() == before
    assert not request.candidate.exists()
    assert request.archive.read_bytes() == b"not-the-source-archive"


def test_post_first_write_rollback_is_forward_only(
    tmp_path: Path, supported_compact_cutover_sqlite: None
) -> None:
    from compact_cutover_fixtures import cutover_request

    from backend.app.edge_db.compact_cutover import (
        CompactCutoverError,
        rollback_compact_cutover,
        run_compact_cutover,
    )

    request = cutover_request(tmp_path)
    run_compact_cutover(request)
    with sqlite3.connect(request.live) as connection:
        connection.execute(
            "INSERT INTO credentials "
            "(id, username, algorithm, salt, password_hash, updated_at) "
            "VALUES (1, 'ops', 'scrypt', ?, ?, '2026-08-25T00:00:00.000Z')",
            (b"\x00" * 16, b"\x01" * 64),
        )
        connection.commit()
    after_write = request.live.read_bytes()
    with pytest.raises(CompactCutoverError, match="EDGE_DB_CUTOVER_FORWARD_ONLY"):
        rollback_compact_cutover(request)
    assert request.live.read_bytes() == after_write
    with sqlite3.connect(request.live) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (18,)
        assert connection.execute("SELECT username FROM credentials").fetchone() == ("ops",)


def test_rollback_before_first_write_restores_archive(
    tmp_path: Path, supported_compact_cutover_sqlite: None
) -> None:
    from compact_cutover_fixtures import cutover_request, sha256

    from backend.app.edge_db.compact_cutover import rollback_compact_cutover, run_compact_cutover
    from backend.app.edge_db.compact_cutover_preflight import schema_version

    request = cutover_request(tmp_path)
    archive_hash = sha256(request.source)
    run_compact_cutover(request)
    rollback_compact_cutover(request)
    assert schema_version(request.live) == 17
    assert sha256(request.live) == archive_hash == sha256(request.archive)


def test_runtime_does_not_open_archive_or_legacy_tables() -> None:
    runtime_files = (
        ROOT / "backend" / "app" / "lifespan.py",
        ROOT / "backend" / "app" / "edge_db" / "configuration.py",
        ROOT / "worker" / "runtime" / "worker.py",
    )
    forbidden = (
        "edge-v17-archive",
        "v17-archive",
        "derivative_jobs",
        "clip_listing_generation",
        "qa_replay_runs",
    )
    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} still names {token}"


def test_scoped_docs_do_not_require_retired_families() -> None:
    docs = (
        ROOT / "backend" / "app" / "AGENTS.md",
        ROOT / "backend" / "app" / "features" / "AGENTS.md",
        ROOT / "worker" / "pipeline" / "output" / "evidence" / "AGENTS.md",
        ROOT / "worker" / "types" / "AGENTS.md",
        ROOT / "docs" / "architecture.md",
    )
    retired = (
        "API actor writes `control_*`",
        "maintain_clip_listing",
        "QaStore opens as",
        "writes `qa_*`",
        "derivative evidence, overlay/MJPEG",
        "derivatives/<incident>/<sha256>.mp4",
        "python -m backend.app.edge_db.importer",
    )
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for token in retired:
            assert token not in text, f"{path} still requires {token!r}"
