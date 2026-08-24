from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from compact_cutover_dense_fixture import dense_cutover_request
from compact_cutover_fixtures import sha256
from compact_cutover_sensitive_fixture import DENSE_SECRETS

from backend.app.edge_db.compact_cutover import run_compact_cutover
from backend.app.edge_db.functions import audit_record_hash

pytestmark = pytest.mark.usefixtures("supported_compact_cutover_sqlite")

_LOCKED_MAP_TABLES = {
    "clips",
    "connection_store_migrations",
    "control_detection_policy_revisions",
    "control_detection_policy_state",
    "control_evidence_review_revisions",
    "evidence_events",
    "evidence_incident_snapshots",
    "evidence_media_objects",
    "snapshots",
}


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
    by_table = {
        table: [record for record in receipts if record["source_table"] == table]
        for table in _LOCKED_MAP_TABLES
    }
    assert all(
        any(record["action"] == "MAP" for record in records) for records in by_table.values()
    )
    assert all(
        record["action"] == "MAP"
        for table, records in by_table.items()
        if table != "control_detection_policy_revisions"
        for record in records
    )
    revision_actions = [
        record["action"] for record in by_table["control_detection_policy_revisions"]
    ]
    assert revision_actions == ["NONE", "MAP", "MAP"]
    targets = {
        (record["source_table"], tuple(record["source_pk"])): record["target_pks"]
        for record in receipts
        if record["source_table"] in _LOCKED_MAP_TABLES
    }
    assert targets[("clips", ("clip_id=legacy-clip",))] == ["clips:clip_id=legacy-clip"]
    assert targets[("connection_store_migrations", ("version=1",))] == [
        "schema_migrations:version=1"
    ]
    assert targets[("control_detection_policy_state", ("facility_id=facility:fixture",))] == [
        "policies:policy_id=1"
    ]
    assert targets[("control_detection_policy_revisions", ("revision_id=2",))] == [
        "policies:policy_id=1"
    ]
    assert targets[("control_detection_policy_revisions", ("revision_id=3",))] == [
        "policies:policy_id=1"
    ]
    assert targets[("control_evidence_review_revisions", ("review_id=review:fixture",))] == [
        "audit_events:audit_id=8"
    ]
    assert targets[("control_evidence_review_revisions", ("review_id=review:current",))] == [
        "audit_events:audit_id=9",
        "incidents:incident_id=incident:fixture",
    ]
    assert targets[("evidence_events", ("edge_event_id=event:complete",))] == [
        "incidents:incident_id=incident:fixture"
    ]
    assert targets[("evidence_incident_snapshots", ("incident_id=incident:fixture",))] == [
        "artifacts:incident_id=incident:fixture,kind=SNAPSHOT"
    ]
    assert targets[("snapshots", ("snapshot_id=snapshot:fixture",))] == [
        "artifacts:incident_id=incident:fixture,kind=SNAPSHOT"
    ]
    assert targets[("evidence_media_objects", ("media_id=media:fixture",))] == [
        "artifacts:incident_id=incident:fixture,kind=PRIMARY_CLIP",
        "clips:clip_id=clip:fixture",
    ]
    assert targets[("evidence_media_objects", ("media_id=media:snapshot",))] == [
        "artifacts:incident_id=incident:fixture,kind=SNAPSHOT"
    ]
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
            "incident": target.execute(
                "SELECT backend_event_id,review_version,review_disposition,review_actor "
                "FROM incidents"
            ).fetchone(),
            "policy": target.execute(
                "SELECT active_values_json,active_content_sha256,previous_values_json,"
                "previous_content_sha256,activation_generation,status FROM policies"
            ).fetchone(),
            "clips": target.execute(
                "SELECT clip_id,publish_state,local_state FROM clips ORDER BY clip_id"
            ).fetchall(),
            "artifacts": target.execute(
                "SELECT incident_id,kind,artifact_id,state,content_sha256,size_bytes,mime_type "
                "FROM artifacts ORDER BY kind"
            ).fetchall(),
            "audit": target.execute(
                "SELECT audit_id,actor_id,action,target_id,detail_json,previous_hash,record_hash "
                "FROM audit_events ORDER BY audit_id"
            ).fetchall(),
            "audit_hash_rows": target.execute(
                "SELECT occurred_at,recorded_at,clock_quality,actor_type,actor_id,auth_mechanism,"
                "action,target_type,target_id,outcome,reason,request_id,interaction_id,detail_json,"
                "previous_hash,record_hash,retention_class,hold_reference "
                "FROM audit_events ORDER BY audit_id"
            ).fetchall(),
            "tables": target.execute(
                "SELECT count(*) FROM pragma_table_list() WHERE schema='main' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchone(),
        }
    assert values == {
        "credentials": ("admin",),
        "site": (11, "facility:fixture", 1, 1, 3),
        "camera": ("Fixture", "room-edge", 640),
        "incident": ("backend:event:fixture", 2, "TP", "senior-operator"),
        "policy": (
            '{"threshold":0.3}',
            "3" * 64,
            '{"threshold":0.2}',
            "2" * 64,
            4,
            "applied",
        ),
        "clips": [
            ("clip:fixture", "PUBLISHED", "AVAILABLE"),
            ("legacy-clip", "WAITING", "UNAVAILABLE"),
        ],
        "artifacts": [
            (
                "incident:fixture",
                "PRIMARY_CLIP",
                "media:fixture",
                "AVAILABLE",
                "e" * 64,
                10,
                "video/mp4",
            ),
            (
                "incident:fixture",
                "SNAPSHOT",
                "snapshot:fixture",
                "AVAILABLE",
                "f" * 64,
                12,
                "image/jpeg",
            ),
        ],
        "audit": values["audit"],
        "audit_hash_rows": values["audit_hash_rows"],
        "tables": (10,),
    }
    audits = values["audit"]
    assert [(row[1], row[2], row[3]) for row in audits] == [
        ("operator", "clip-view", "clip:fixture"),
        ("operator", "incident-review-migrated", "incident:fixture"),
        ("senior-operator", "incident-review-migrated", "incident:fixture"),
    ]
    serialized_audit = json.dumps(audits, default=str)
    serialized_receipts = request.receipt.read_text()
    assert all(secret not in serialized_audit for secret in DENSE_SECRETS)
    assert all(secret not in serialized_receipts for secret in DENSE_SECRETS)
    assert json.loads(audits[0][4])["safe"] == {"preserved": [1, "yes"]}
    assert audits[0][5] == "0" * 64
    assert all(
        current[5] == previous[6] for previous, current in zip(audits, audits[1:], strict=False)
    )
    hash_keys = (
        "occurred_at",
        "recorded_at",
        "clock_quality",
        "actor_type",
        "actor_id",
        "auth_mechanism",
        "action",
        "target_type",
        "target_id",
        "outcome",
        "reason",
        "request_id",
        "interaction_id",
        "detail_json",
        "previous_hash",
        "retention_class",
        "hold_reference",
    )
    for row in values["audit_hash_rows"]:
        payload = dict(zip(hash_keys, (*row[:15], *row[16:]), strict=True))
        assert row[15] == audit_record_hash(
            row[14], json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
