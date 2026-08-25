"""The schema-18 runbook commands must match the executable CLI and Compose."""

from __future__ import annotations

import io
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from backend.app.edge_db.migrator import MIGRATIONS, migrate_database

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "runbooks" / "schema-18-ten-table-cutover.md"
COMPOSE = ROOT / "compose.edge.yaml"
WORKER_ROLLBACK = ROOT / "docs" / "runbooks" / "worker-migration-rollback.md"


def _shell_commands(path: Path) -> str:
    return "\n".join(
        re.findall(r"```sh\n(.*?)```", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    )


def _runbook_gate_sql() -> str:
    text = RUNBOOK.read_text(encoding="utf-8")
    match = re.search(
        r"(SELECT EXISTS\(SELECT 1 FROM evidence_events.*?)\\\"\\\"\\\"", text, re.S
    )
    assert match, "schema-18 runbook is missing an executable drain predicate"
    return match.group(1).replace('\\"', '"').strip()


def test_schema18_runbook_commands_match_cli_help_and_compose() -> None:
    from backend.app.edge_db.compact_cutover import main

    commands = _shell_commands(RUNBOOK)
    compose = COMPOSE.read_text(encoding="utf-8")
    stdout = io.StringIO()
    import sys

    original = sys.stdout
    sys.stdout = stdout
    try:
        try:
            main(["--help"])
        except SystemExit as error:
            assert error.code == 0
    finally:
        sys.stdout = original
    help_text = stdout.getvalue()
    assert "python -m backend.app.edge_db.compact_cutover" in commands
    assert "backend.app.edge_db.compact_cutover" in compose
    for flag in (
        "--source",
        "--live",
        "--archive",
        "--candidate",
        "--receipt",
        "--clip-store",
        "--worker-state",
    ):
        assert flag in commands
        assert flag in help_text
        assert flag in compose
    assert "--rollback" in commands
    assert "--rollback" in help_text
    assert "down -v" not in commands
    assert "archive delete" not in commands.lower()
    assert not re.search(r"\b(?:CREATE|ALTER|DROP)\s+(?:TABLE|INDEX)\b", commands, re.I)


def test_worker_rollback_runbook_orders_inventory_then_cutover() -> None:
    commands = _shell_commands(WORKER_ROLLBACK)
    inventory = "$DC up --pull always edge-filesystem-inventory"
    migrator = "$DC up --pull always edge-db-migrator"
    api = "$DC up -d --wait ml-api"
    worker = "$DC up -d --wait ml-worker"
    assert inventory in commands
    assert migrator in commands
    assert api in commands
    assert worker in commands
    assert commands.index(inventory) < commands.index(migrator) < commands.index(api) < (
        commands.index(worker)
    )
    assert "--no-deps edge-db-migrator" not in commands
    assert "down -v" not in commands


def _seeded(tmp_path: Path, seed: Callable[[sqlite3.Connection], None], name: str) -> Path:
    database = tmp_path / f"{name}.sqlite3"
    migrate_database(database, migrations=MIGRATIONS[:17])
    connection = sqlite3.connect(database)
    try:
        seed(connection)
        connection.commit()
    finally:
        connection.close()
    return database


def test_runbook_drain_query_matches_cutover_refusal(tmp_path: Path) -> None:
    from backend.app.edge_db.compact_cutover import (
        CompactCutoverError,
        CompactCutoverRequest,
        run_compact_cutover,
    )

    def in_flight(connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id, detected_at, payload_json, state, queued_at, "
            " next_attempt_at, attempt_count, lease_owner, lease_expires_at, "
            " delivery_state) "
            "VALUES ('e1', '2026-08-22T00:00:00Z', '{}', 'IN_FLIGHT', "
            "        1787000000.0, 1787000000.0, 0, 'sender-1', 1787003600.0, "
            "        'PENDING')"
        )

    for_query = _seeded(tmp_path, in_flight, "query")
    for_cutover = _seeded(tmp_path, in_flight, "cutover")
    connection = sqlite3.connect(f"file:{for_query}?mode=ro", uri=True)
    try:
        blocked = bool(connection.execute(_runbook_gate_sql()).fetchone()[0])
    finally:
        connection.close()
    assert blocked is True
    request = CompactCutoverRequest(
        source=for_cutover,
        live=for_cutover,
        archive=tmp_path / "archive.sqlite3",
        candidate=tmp_path / "candidate.sqlite3",
        receipt=tmp_path / "receipt.jsonl",
        clip_store=tmp_path / "clips",
        worker_state=tmp_path / "worker",
    )
    with pytest.raises(CompactCutoverError, match="EDGE_DB_DRAIN_INCOMPLETE"):
        run_compact_cutover(request, sqlite_version=(3, 51, 3))
    with sqlite3.connect(for_cutover) as live:
        assert live.execute("PRAGMA user_version").fetchone() == (17,)
