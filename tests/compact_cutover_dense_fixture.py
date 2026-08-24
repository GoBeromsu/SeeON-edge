"""Dense valid schema-17 source with every one of the 72 tables populated."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from test_edge_db_schema16_fixtures import (
    DIGEST,
    MANIFEST,
    MEDIA,
    NOW,
    TRACE,
    build_schema16_fixture,
)

from backend.app.edge_db.compact_cutover import CompactCutoverRequest
from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.schema import MIGRATIONS


def dense_cutover_request(tmp_path: Path) -> CompactCutoverRequest:
    state = tmp_path / "dense-state"
    state.mkdir(mode=0o700)
    source = state / "source.sqlite3"
    build_schema16_fixture(source, drain_blocked=False)
    migrate_database(source, migrations=MIGRATIONS[:17])
    with sqlite3.connect(source) as connection:
        _populate_missing_tables(connection)
        connection.commit()
    with sqlite3.connect(source, isolation_level=None) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    live = state / "edge.sqlite3"
    shutil.copyfile(source, live)
    clip_store = state / "clip-store"
    clip_dir = clip_store / "clips" / "clip:fixture"
    clip_dir.mkdir(parents=True)
    media = clip_dir / "clip.mp4"
    media.write_bytes(b"0123456789")
    manifest = {
        "clip_id": "clip:fixture",
        "camera_id": "camera:fixture",
        "event_ref": "event:complete",
        "event_type": "fall",
        "started_at": NOW,
        "duration_s": 1.0,
        "codec": "h264",
        "path": "clip.mp4",
        "video_available": True,
        "finalized": True,
    }
    (clip_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    worker = state / "worker-state"
    worker.mkdir()
    return CompactCutoverRequest(
        source,
        live,
        state / "archive.sqlite3",
        state / "candidate.sqlite3",
        state / "receipt.jsonl",
        clip_store,
        worker,
    )


def _populate_missing_tables(connection: sqlite3.Connection) -> None:
    camera = json.dumps(
        [
            {
                "id": "camera:fixture",
                "label": "Fixture",
                "rtsp_url": "rtsp://camera.invalid/live",
                "space_id": "space:fixture",
                "backend_camera_id": "backend:fixture",
                "mapping_pending": False,
                "never_connected": False,
                "created_at": NOW,
                "edge_ref": "camera-edge",
                "room_edge_ref": "room-edge",
            }
        ],
        separators=(",", ":"),
    )
    statements = (
        ("INSERT INTO credentials VALUES (1,'admin','scrypt',?,?,?)", (b"s" * 16, b"h" * 64, NOW)),
        ("INSERT INTO camera_registry VALUES (1,11,?)", (camera,)),
        ("INSERT INTO camera_topology_floors VALUES ('floor-edge','Floor',0)", ()),
        (
            "INSERT INTO camera_topology_rooms VALUES "
            "('room-edge','floor-edge','Room','ROOM',1,'space:fixture')",
            (),
        ),
        (
            "INSERT INTO camera_topology_cameras VALUES "
            "('camera:fixture','camera-edge','room-edge')",
            (),
        ),
        (
            "INSERT INTO camera_bed_zone VALUES ('camera:fixture','[[0,0],[1,0],[1,1]]',640,480,?)",
            (NOW,),
        ),
        ("INSERT INTO detection_settings VALUES ('fall',1,'always',NULL,NULL)", ()),
        ("INSERT INTO runtime_settings VALUES (1,1,4)", ()),
        ("INSERT INTO clip_storage_location VALUES (1,'facility-a')", ()),
        ("INSERT INTO topology_dirty VALUES (1,11,?)", (NOW,)),
        ("INSERT INTO runtime_latency VALUES ('facility:fixture','{}')", ()),
        (
            "INSERT INTO connection_settings VALUES "
            "(1,NULL,NULL,'facility:fixture','secret',?,'FC','client','edge-install',1,?,?)",
            (NOW, NOW, NOW),
        ),
        ("INSERT INTO connection_store_migrations VALUES (1,'fixture',?,NULL,NULL,NULL)", (NOW,)),
        (
            "INSERT INTO edge_topology_sync_state "
            "(id,edge_installation_id,enrollment_generation,"
            "last_snapshotted_registry_version,last_client_revision,server_revision,"
            "consecutive_failures) VALUES (1,'edge-install',1,11,2,3,0)",
            (),
        ),
        (
            "INSERT INTO edge_topology_confirmation_preview VALUES "
            "(1,'confirm','dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd',?,'snapshot',2,3,11,'edge-install',1,1,1,1,1,'accepted')",
            (NOW,),
        ),
        (
            "INSERT INTO audit VALUES (7,?,'clip-view',?)",
            (
                NOW,
                json.dumps(
                    {
                        "actor_type": "user",
                        "actor_id": "operator",
                        "target_type": "clip",
                        "target_id": "clip:fixture",
                        "outcome": "success",
                    }
                ),
            ),
        ),
        ("INSERT INTO events VALUES ('catalog-only','camera:fixture','fall',?,NULL,'{}')", (NOW,)),
        ("INSERT INTO cameras VALUES ('legacy-camera','Legacy',NULL,'{}')", ()),
        (
            "INSERT INTO clips VALUES "
            "('legacy-clip','camera:fixture','fall','ready',?,'clips/legacy',?,10,'video/mp4','h264','{}')",
            (NOW, MEDIA),
        ),
        (
            "INSERT INTO snapshots VALUES "
            "('snapshot:fixture','camera:fixture','event:complete',?,'snap.jpg',?,10,'image/jpeg','{}')",
            (NOW, MEDIA),
        ),
        (
            "INSERT INTO labels VALUES ('clip:fixture','TRUE_POSITIVE','legacy-reviewer',?,'{}')",
            (NOW,),
        ),
        (
            "INSERT INTO clip_listing_rows VALUES "
            "(1,'clip:fixture','clips/clip:fixture/manifest.json',1,1,'camera:fixture','event:complete','fall','fall',?,1.0,'h264','clip.mp4',1,NULL,1,10)",
            (NOW,),
        ),
        ("INSERT INTO clip_listing_thumbnails VALUES (1,'clip:fixture',1,1,1)", ()),
        ("INSERT INTO clip_listing_summary VALUES (1,'camera:fixture','fall',1)", ()),
        (
            "INSERT INTO evidence_incident_snapshots VALUES "
            "('incident:fixture','snapshot:fixture','media:fixture',?,'camera:fixture',?)",
            (NOW, NOW),
        ),
        (
            "INSERT INTO control_evidence_review_revisions VALUES "
            "('review:fixture','incident:fixture','clip:fixture',1,'operator',?,'FALSE_POSITIVE',NULL)",
            (NOW,),
        ),
        (
            "INSERT INTO control_evidence_review_state VALUES "
            "('incident:fixture','clip:fixture',1)",
            (),
        ),
        (
            "INSERT INTO control_legacy_label_migrations VALUES "
            "('clip:fixture','MIGRATED','incident:fixture','review:fixture')",
            (),
        ),
        ("INSERT INTO runtime_settings VALUES (1,1,4) ON CONFLICT(id) DO NOTHING", ()),
    )
    for statement, values in statements:
        connection.execute(statement, values)
    render = (
        "INSERT INTO derivative_render_records VALUES "
        "('incident:fixture','ANNOTATED_CLIP',?,'media:fixture','clip:fixture',?,?,?,?,1,"
        "'opencv-cpu','cpu','host','overlay-cpu.v1',320,180,0,1000,?)"
    )
    connection.execute(render, ("1" * 64, MEDIA, TRACE, MANIFEST, DIGEST, NOW))
    artifact = list(connection.execute("SELECT * FROM derivative_render_records").fetchone())
    artifact[1] = "STILL"
    connection.execute(
        f"INSERT INTO derivative_artifacts VALUES ({','.join('?' for _ in artifact)})",
        artifact,
    )
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    empty = [
        table
        for table in tables
        if connection.execute(f'SELECT count(*) FROM "{table}"').fetchone() == (0,)
    ]
    if empty:
        raise AssertionError(f"dense fixture left empty tables: {empty}")


__all__ = ["dense_cutover_request"]
