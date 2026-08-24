from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from compact_cutover_dense_fixture import dense_cutover_request
from compact_cutover_fixtures import sha256

from backend.app.edge_db.compact_cutover import run_compact_cutover

pytestmark = pytest.mark.usefixtures("supported_compact_cutover_sqlite")


def test_dense_all_72_table_fixture_reconciles_bidirectionally(tmp_path: Path) -> None:
    request = dense_cutover_request(tmp_path)
    source_hash = sha256(request.source)
    with sqlite3.connect(request.source) as source:
        tables = [
            str(row[0])
            for row in source.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        counts = {
            table: int(source.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
            for table in tables
        }
    assert len(tables) == 72
    assert min(counts.values()) >= 1
    source_rows = sum(counts.values())

    result = run_compact_cutover(request)

    receipts = [json.loads(line) for line in request.receipt.read_text().splitlines()]
    assert len(receipts) == result.source_rows == source_rows
    assert {record["source_table"] for record in receipts} == set(tables)
    assert all(record["target_pks"] for record in receipts if record["action"] == "MAP")
    assert all(record["reason"] for record in receipts if record["action"] == "NONE")
    assert sha256(request.source) == source_hash == sha256(request.archive)
    with sqlite3.connect(request.live) as target:
        values = {
            "credentials": target.execute("SELECT username FROM credentials").fetchone(),
            "site": target.execute(
                "SELECT registry_version,facility_id,fall_on,clip_export_enabled,"
                "topology_server_revision FROM edge_site"
            ).fetchone(),
            "camera": target.execute(
                "SELECT label,room_location_id,bed_image_width FROM cameras"
            ).fetchone(),
            "review": target.execute(
                "SELECT review_version,review_disposition FROM incidents"
            ).fetchone(),
            "artifact": target.execute("SELECT kind,state FROM artifacts ORDER BY kind").fetchall(),
            "audit": target.execute(
                "SELECT audit_id,actor_id,target_id FROM audit_events"
            ).fetchone(),
            "tables": target.execute(
                "SELECT count(*) FROM pragma_table_list() WHERE schema='main' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchone(),
        }
    assert values == {
        "credentials": ("admin",),
        "site": (11, "facility:fixture", 1, 1, 3),
        "camera": ("Fixture", "room-edge", 640),
        "review": (1, "FP"),
        "artifact": [("PRIMARY_CLIP", "AVAILABLE")],
        "audit": (7, "operator", "clip:fixture"),
        "tables": (10,),
    }
