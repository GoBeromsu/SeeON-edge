from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from compact_cutover_fixtures import TS, cutover_request

from backend.app.edge_db.compact_cutover import run_compact_cutover

pytestmark = pytest.mark.usefixtures("supported_compact_cutover_sqlite")


def _refresh_live(source: Path, live: Path) -> None:
    with sqlite3.connect(source, isolation_level=None) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copyfile(source, live)


def test_audit_is_mapped_and_retired_catalog_rows_have_none_reasons(tmp_path: Path) -> None:
    request = cutover_request(tmp_path)
    payload = {
        "actor_type": "user",
        "actor_id": "operator-1",
        "auth_mechanism": "session",
        "target_type": "clip",
        "target_id": "clip-1",
        "outcome": "success",
        "request_id": "request-1",
        "safe": "preserved",
        "token": "must-be-redacted",
    }
    with sqlite3.connect(request.source) as connection:
        connection.execute(
            "INSERT INTO audit (audit_id,occurred_at,action,payload_json) VALUES (7,?,?,?)",
            (TS, "clip-view", json.dumps(payload)),
        )
        connection.execute(
            "INSERT INTO events (edge_event_id,camera_id,event_type,detected_at,payload_json) "
            "VALUES ('catalog-only','camera-1','fall',?,'{}')",
            (TS,),
        )
        connection.execute(
            "INSERT INTO topology_dirty (id,registry_version,created_at) VALUES (1,1,?)",
            (TS,),
        )
        connection.commit()
    _refresh_live(request.source, request.live)

    run_compact_cutover(request)

    receipts = [json.loads(line) for line in request.receipt.read_text().splitlines()]
    by_table = {
        record["source_table"]: record
        for record in receipts
        if record["source_table"] in {"audit", "events", "topology_dirty"}
    }
    assert by_table["audit"]["action"] == "MAP"
    assert by_table["audit"]["target_pks"] == ["audit_events:audit_id=7"]
    assert by_table["events"]["action"] == "NONE"
    assert by_table["events"]["reason"] == "catalog_duplicate_retired"
    assert by_table["topology_dirty"]["action"] == "NONE"
    assert by_table["topology_dirty"]["reason"] == "derived_dirty_marker_retired"
    with sqlite3.connect(request.live) as connection:
        audit = connection.execute(
            "SELECT audit_id,actor_id,action,target_id,detail_json FROM audit_events"
        ).fetchone()
        counts = (
            connection.execute("SELECT count(*) FROM incidents").fetchone(),
            connection.execute("SELECT topology_dirty_registry_version FROM edge_site").fetchone(),
        )
    assert audit[:4] == (7, "operator-1", "clip-view", "clip-1")
    assert json.loads(audit[4]) == {key: value for key, value in payload.items() if key != "token"}
    assert counts == ((0,), None)
