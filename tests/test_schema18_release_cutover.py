"""Schema-18 release identity, Compose cutover, and deterministic failure gates."""

from __future__ import annotations

import hashlib
import io
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from backend.app.edge_db.migrator import MIGRATIONS, migrate_database
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


def _v17(database: Path) -> None:
    migrate_database(database, migrations=MIGRATIONS[:17])


def _cutover_paths(tmp_path: Path) -> dict[str, Path]:
    state = tmp_path / "state"
    state.mkdir()
    return {
        "source": state / "edge.sqlite3",
        "live": state / "edge.sqlite3",
        "archive": state / "edge-v17-archive.sqlite3",
        "candidate": state / "edge-v18-candidate.sqlite3",
        "receipt": state / "schema18-cutover-receipts.jsonl",
        "clip_store": tmp_path / "clip-store",
        "worker_state": tmp_path / "worker-state",
    }


SUPPORTED_SQLITE = (3, 51, 3)


def _cutover(paths: dict[str, Path], **kwargs: object) -> object:
    from backend.app.edge_db.compact_cutover import run_compact_cutover

    return run_compact_cutover(
        _request(paths),
        sqlite_version=SUPPORTED_SQLITE,
        **kwargs,
    )


def _request(paths: dict[str, Path], **overrides: object) -> object:
    from backend.app.edge_db.compact_cutover import CompactCutoverRequest

    values = dict(paths)
    values.update(overrides)
    return CompactCutoverRequest(
        source=values["source"],
        live=values["live"],
        archive=values["archive"],
        candidate=values["candidate"],
        receipt=values["receipt"],
        clip_store=values["clip_store"],
        worker_state=values["worker_state"],
        expected_source_sha256=values.get("expected_source_sha256"),
    )


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
    assert worker["depends_on"] == {"ml-api": {"condition": "service_healthy"}}


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


def test_empty_v17_candidate_cutover_installs_schema_18(tmp_path: Path) -> None:
    paths = _cutover_paths(tmp_path)
    _v17(paths["live"])
    before = paths["live"].read_bytes()
    result = _cutover(paths)
    assert result.current_version == 18
    assert hashlib.sha256(paths["archive"].read_bytes()).hexdigest() == hashlib.sha256(
        before
    ).hexdigest()
    with sqlite3.connect(paths["live"]) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (18,)
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert tables == COMPACT_APPLICATION_TABLES
    assert oct(paths["archive"].stat().st_mode & 0o777) == "0o400"


def test_old_sqlite_refuses_before_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.app.edge_db import compact_cutover
    from backend.app.edge_db.compact_cutover import CompactCutoverError, run_compact_cutover

    paths = _cutover_paths(tmp_path)
    _v17(paths["live"])
    before = paths["live"].read_bytes()
    monkeypatch.setattr(compact_cutover.sqlite3, "sqlite_version_info", (3, 45, 1))
    with pytest.raises(CompactCutoverError, match="sqlite 3.45.1"):
        run_compact_cutover(_request(paths))
    assert paths["live"].read_bytes() == before
    assert not paths["archive"].exists()
    assert not paths["candidate"].exists()


def test_in_flight_source_refuses_without_replacement(tmp_path: Path) -> None:
    from backend.app.edge_db.compact_cutover import CompactCutoverError

    paths = _cutover_paths(tmp_path)
    _v17(paths["live"])
    _in_flight(paths["live"])
    before = paths["live"].read_bytes()
    with pytest.raises(CompactCutoverError, match=SCHEMA18_DRAIN_SENTINEL):
        _cutover(paths)
    assert paths["live"].read_bytes() == before
    assert not paths["candidate"].exists()
    with sqlite3.connect(paths["live"]) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)


def test_bad_archive_hash_refuses_without_replacement(tmp_path: Path) -> None:
    from backend.app.edge_db.compact_cutover import CompactCutoverError

    paths = _cutover_paths(tmp_path)
    _v17(paths["live"])
    before = paths["live"].read_bytes()
    paths["archive"].write_bytes(b"not-the-source-archive")
    with pytest.raises(CompactCutoverError, match="EDGE_DB_CUTOVER_STALE_ARCHIVE"):
        _cutover(paths)
    assert paths["live"].read_bytes() == before
    assert not paths["candidate"].exists()
    assert paths["archive"].read_bytes() == b"not-the-source-archive"


def test_post_first_write_rollback_is_forward_only(tmp_path: Path) -> None:
    from backend.app.edge_db.compact_cutover import CompactCutoverError

    paths = _cutover_paths(tmp_path)
    _v17(paths["live"])
    _cutover(paths)
    with sqlite3.connect(paths["live"]) as connection:
        connection.execute(
            "INSERT INTO credentials "
            "(id, username, algorithm, salt, password_hash, updated_at) "
            "VALUES (1, 'ops', 'scrypt', ?, ?, '2026-08-25T00:00:00.000Z')",
            (b"\x00" * 16, b"\x01" * 64),
        )
        connection.commit()
    after_write = paths["live"].read_bytes()
    with pytest.raises(CompactCutoverError, match="EDGE_DB_CUTOVER_FORWARD_ONLY"):
        _cutover(paths, rollback=True)
    assert paths["live"].read_bytes() == after_write
    with sqlite3.connect(paths["live"]) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (18,)
        assert connection.execute("SELECT username FROM credentials").fetchone() == ("ops",)


def test_rollback_before_first_write_restores_archive(tmp_path: Path) -> None:
    paths = _cutover_paths(tmp_path)
    _v17(paths["live"])
    archive_bytes = paths["live"].read_bytes()
    _cutover(paths)
    result = _cutover(paths, rollback=True)
    assert result.current_version == 17
    assert paths["live"].read_bytes() == archive_bytes
    assert paths["archive"].read_bytes() == archive_bytes


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
