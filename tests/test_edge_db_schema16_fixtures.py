from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.ownership import TABLE_FAMILIES, writer_for_table
from backend.app.edge_db.schema import MIGRATIONS
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipLocalState,
    ClipPublishState,
    EventDeliveryState,
)
from worker.pipeline.output.evidence.evidence_record_models import EvidenceLifecycle

NOW = "2026-08-21T12:00:00Z"
MANIFEST = "a" * 64
ANALYSIS = "b" * 64
TRACE = "c" * 64
POLICY = "d" * 64
MEDIA = "e" * 64
REQUEST = "f" * 64
RUN_ONE = "1" * 64
RUN_TWO = "2" * 64
COMPARISON = "3" * 64
DIGEST = "4" * 64


def _insert(connection: sqlite3.Connection, statement: str, values: tuple[object, ...]) -> None:
    connection.execute(statement, values)


def _populate_fixture(connection: sqlite3.Connection, *, drain_blocked: bool) -> None:
    """Insert representative schema-16 rows using the released table contracts."""
    _insert(
        connection,
        "INSERT INTO config_history VALUES (7,3,11,?,1000.0)",
        ('{"clip_export_enabled":true}',),
    )
    _insert(
        connection,
        "INSERT INTO config_current VALUES (1,3,7,11,?,1000.0)",
        ('{"clip_export_enabled":true}',),
    )
    _insert(
        connection,
        "INSERT INTO runtime_manifest_contents VALUES (?,1,?,?)",
        (MANIFEST, '{"camera_id":"camera:fixture"}', NOW),
    )
    _insert(
        connection,
        "INSERT INTO runtime_manifest_boots VALUES ('boot:fixture',?,?)",
        (MANIFEST, NOW),
    )
    _insert(
        connection,
        "INSERT INTO runtime_manifest_cameras VALUES ('boot:fixture','camera:fixture',?,?)",
        (MANIFEST, NOW),
    )
    _insert(
        connection,
        "INSERT INTO runtime_analysis_traces "
        "(trace_id,trace_schema_version,worker_boot_id,camera_id,stream_epoch,frame_seq,"
        "pts,source_time_sec,frame_width,frame_height,bed_region_provenance) "
        "VALUES (?,1,'boot:fixture','camera:fixture',0,1,1.0,1.0,640,480,'fresh')",
        (ANALYSIS,),
    )
    _insert(
        connection,
        "INSERT INTO runtime_analysis_components VALUES (?,0,'pose.fixture','observed')",
        (ANALYSIS,),
    )
    _insert(
        connection,
        "INSERT INTO runtime_analysis_beds VALUES (?,0,1,2,3,4,0.9,'fresh')",
        (ANALYSIS,),
    )
    _insert(connection, "INSERT INTO runtime_analysis_bed_points VALUES (?,0,0,1,2)", (ANALYSIS,))
    _insert(
        connection,
        "INSERT INTO runtime_analysis_persons VALUES (?,0,9,NULL,1,2,3,4,0.8)",
        (ANALYSIS,),
    )
    _insert(
        connection, "INSERT INTO runtime_analysis_keypoints VALUES (?,0,0,2,3,0.7)", (ANALYSIS,)
    )
    _insert(
        connection,
        "INSERT INTO runtime_trace_cursors (camera_id,newest_retained_seq,updated_at_source_sec) "
        "VALUES ('camera:fixture',1,1.0)",
        (),
    )

    _insert(
        connection,
        "INSERT INTO evidence_decision_traces "
        "(trace_id,trace_schema_version,analysis_trace_id,module_qualified_id,policy_qualified_id,"
        "effective_policy_id,runtime_manifest_sha256,reason,previous_state,current_state,triggered,"
        "track_id,bed_id) VALUES (?,1,?,'fall.fixture','fall.policy.fixture',?,?,'fall-onset',"
        "'clear','fall',1,9,0)",
        (TRACE, ANALYSIS, POLICY, MANIFEST),
    )
    _insert(
        connection,
        "INSERT INTO evidence_decision_values VALUES (?,'fall_probability',0.95,NULL)",
        (TRACE,),
    )

    event_states = (
        (
            ("event:staged", "STAGED", None, None),
            ("event:ready", "READY", None, None),
            ("event:flight", "IN_FLIGHT", "worker:fixture", 2000.0),
        )
        if drain_blocked
        else ()
    ) + (("event:complete", "ACKED", None, None),)
    for event_id, state, owner, expires in event_states:
        _insert(
            connection,
            "INSERT INTO evidence_events "
            "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at,lease_owner,"
            "lease_expires_at,delivery_state) VALUES (?,?,?, ?,1000.0,1000.0,?,?,?)",
            (
                event_id,
                NOW,
                '{"event_type":"fall"}',
                state,
                owner,
                expires,
                EventDeliveryState.ACKED.value
                if state == "ACKED"
                else EventDeliveryState.PENDING.value,
            ),
        )
    _insert(
        connection, "INSERT INTO evidence_event_trace_refs VALUES ('event:complete',?)", (TRACE,)
    )

    clip_state = (
        ClipPublishState.IN_FLIGHT.value if drain_blocked else ClipPublishState.PUBLISHED.value
    )
    _insert(
        connection,
        "INSERT INTO evidence_clips "
        "(clip_id,local_state,state_version,publish_state,publish_attempt_count,publish_next_attempt_at,"
        "publish_lease_owner,publish_lease_expires_at) VALUES "
        "('clip:fixture',?,2,?,1,1000.0,?,?)",
        (
            ClipLocalState.VERIFIED.value,
            clip_state,
            "worker:fixture" if drain_blocked else None,
            2000.0 if drain_blocked else None,
        ),
    )
    _insert(connection, "INSERT INTO clip_events VALUES ('clip:fixture','event:complete',0)", ())
    _insert(
        connection,
        "INSERT INTO evidence_clip_trace_refs VALUES ('clip:fixture','event:complete',?)",
        (TRACE,),
    )
    _insert(
        connection,
        "INSERT INTO evidence_media_objects "
        "(media_id,content_sha256,size_bytes,mime_type,contained_relpath,basename,created_at) "
        "VALUES ('media:fixture',?,10,'video/mp4','clips/fixture.mp4','fixture.mp4',?)",
        (MEDIA, NOW),
    )
    _insert(
        connection,
        "INSERT INTO evidence_incidents "
        "(incident_id,edge_event_id,camera_id,event_type,detected_at,runtime_manifest_sha256,"
        "decision_trace_id,module_qualified_id,policy_qualified_id,effective_policy_id,provenance_state,"
        "primary_clip_id,lifecycle_state,created_at,updated_at) VALUES "
        "('incident:fixture','event:complete','camera:fixture','fall',?,?,?,?,?,?,'QUALIFIED',"
        "'clip:fixture',?,?,?)",
        (
            NOW,
            MANIFEST,
            TRACE,
            "fall.fixture",
            "fall.policy.fixture",
            POLICY,
            EvidenceLifecycle.COMPLETE.value,
            NOW,
            NOW,
        ),
    )
    _insert(
        connection,
        "INSERT INTO evidence_primary_clips "
        "(incident_id,clip_id,manifest_relpath,manifest_sha256,manifest_size_bytes,media_id,"
        "source_packet_preserved,source_media_json,truncation_json,created_at) VALUES "
        "('incident:fixture','clip:fixture','clips/fixture.json',?,10,'media:fixture',1,'{}','[]',?)",
        (DIGEST, NOW),
    )
    _insert(
        connection,
        "INSERT INTO evidence_artifact_slots VALUES ('incident:fixture','PRIMARY_CLIP','AVAILABLE',"
        "'media:fixture',NULL,1,?,?)",
        (NOW, NOW),
    )
    _insert(
        connection,
        "INSERT INTO evidence_retention_states VALUES ('clip:fixture','PURGED',NULL,1,?,?)",
        (NOW, NOW),
    )

    job_state = (
        "PENDING" if drain_blocked else "CANCELLED"
    )
    reason = None if drain_blocked else "fixture complete"
    _insert(
        connection,
        "INSERT INTO derivative_jobs "
        "(incident_id,derivative_kind,request_id,state,reason,created_at,updated_at) "
        "VALUES ('incident:fixture','STILL',?,?,?, ?,?)",
        (REQUEST, job_state, reason, NOW, NOW),
    )
    _insert(
        connection,
        "INSERT INTO derivative_evidence_slots VALUES "
        "('incident:fixture','ANNOTATED_CLIP',?,?,?,1,?,?)",
        (
            "PENDING" if drain_blocked else "UNAVAILABLE",
            None,
            None if drain_blocked else "fixture complete",
            NOW,
            NOW,
        ),
    )

    _insert(
        connection,
        "INSERT INTO control_detection_policy_revisions VALUES "
        "(1,'facility:fixture','camera:fixture','fall',1,'fall-policy',1,'{}',?,?)",
        (POLICY, NOW),
    )
    _insert(
        connection, "INSERT INTO control_detection_policy_state VALUES ('facility:fixture',1)", ()
    )
    _insert(
        connection,
        "INSERT INTO control_detection_policy_activations VALUES "
        "(1,'facility:fixture','camera:fixture','fall',1,1,NULL,1,'applied',NULL,?,?)",
        (NOW, NOW),
    )
    _insert(
        connection,
        "INSERT INTO control_heartbeats VALUES ('camera:fixture','facility:fixture',1000.0,7)",
        (),
    )
    _insert(
        connection,
        "INSERT INTO qa_replay_runs VALUES "
        "(?,'camera:fixture','fall.fixture','fall.policy.fixture',?,1,1,"
        "'captured',NULL,'api:fixture',?,?,?)",
        (RUN_ONE, POLICY, NOW, DIGEST, "{}"),
    )
    _insert(
        connection,
        "INSERT INTO qa_replay_runs VALUES "
        "(?,'camera:fixture','fall.fixture','fall.policy.fixture',?,1,1,"
        "'replay',?,'api:fixture',?,?,?)",
        (RUN_TWO, POLICY, RUN_ONE, NOW, DIGEST, "{}"),
    )
    _insert(
        connection,
        "INSERT INTO qa_replay_comparisons VALUES (?,?,?,1,0,?,?,?)",
        (COMPARISON, RUN_ONE, RUN_TWO, NOW, DIGEST, "{}"),
    )
    _insert(
        connection,
        "INSERT INTO qa_label_revisions VALUES ('label:fixture',?,1,'api:fixture',?,'TP',NULL)",
        (COMPARISON, NOW),
    )
    _insert(connection, "INSERT INTO qa_label_state VALUES (?,1)", (COMPARISON,))

    _insert(
        connection,
        "INSERT INTO faults VALUES (1,100,'2026-08-21T00:00:00Z','fixture','worker','detect',"
        "'camera:fixture',1,1.0,'[480,640]',?,? ,1,'RuntimeError','fixture',1,'restart',?)",
        (MEDIA, MANIFEST, NOW),
    )
    _insert(
        connection,
        "INSERT INTO schema_import_sources VALUES ('legacy-worker','16',?,1,1,1,?)",
        (DIGEST, NOW),
    )
    _insert(
        connection,
        "INSERT INTO schema_import_receipts VALUES ('legacy-worker','rows','16',?,1,?)",
        (DIGEST, NOW),
    )
    connection.commit()


def _table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return {
        table: int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
        for table in tables
    }


def _content_checksum(connection: sqlite3.Connection) -> str:
    """Hash all fixture content, excluding migrator timestamps in schema_migrations."""
    tables = sorted(table for table in _table_counts(connection) if table != "schema_migrations")
    content = []
    for table in tables:
        columns = [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]
        order_by = ", ".join(f'"{column}"' for column in columns)
        rows = connection.execute(f'SELECT * FROM "{table}" ORDER BY {order_by}').fetchall()
        content.append((table, rows))
    return hashlib.sha256(
        json.dumps(content, default=repr, separators=(",", ":")).encode()
    ).hexdigest()


def build_schema16_fixture(path: Path, *, drain_blocked: bool) -> dict[str, object]:
    migrate_database(path, migrations=MIGRATIONS[:16])
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        assert connection.execute("PRAGMA user_version").fetchone() == (16,)
        _populate_fixture(connection, drain_blocked=drain_blocked)
        return {"row_counts": _table_counts(connection), "checksum": _content_checksum(connection)}


def test_schema16_fixture_covers_every_writer_family_and_drain_states(tmp_path: Path) -> None:
    allowed = build_schema16_fixture(tmp_path / "allowed.sqlite3", drain_blocked=False)
    blocked = build_schema16_fixture(tmp_path / "blocked.sqlite3", drain_blocked=True)

    with sqlite3.connect(tmp_path / "blocked.sqlite3") as connection:
        assert connection.execute(
            "SELECT state FROM evidence_events ORDER BY edge_event_id"
        ).fetchall() == [("ACKED",), ("IN_FLIGHT",), ("READY",), ("STAGED",)]
        assert connection.execute("SELECT publish_state FROM evidence_clips").fetchone() == (
            ClipPublishState.IN_FLIGHT.value,
        )
        assert connection.execute("SELECT state FROM derivative_jobs").fetchone() == (
            "PENDING",
        )
        assert connection.execute("SELECT state FROM derivative_evidence_slots").fetchone() == (
            "PENDING",
        )
        populated_families = {
            family.prefix
            for (table, count) in blocked["row_counts"].items()
            if count
            and (
                family := next(
                    (item for item in TABLE_FAMILIES if table.startswith(item.prefix)), None
                )
            )
        }
    assert populated_families == {family.prefix for family in TABLE_FAMILIES}
    assert all(
        writer_for_table(table) is not None
        for table, count in blocked["row_counts"].items()
        if count
    )
    with sqlite3.connect(tmp_path / "allowed.sqlite3") as connection:
        assert connection.execute("SELECT state FROM evidence_events").fetchall() == [("ACKED",)]
        assert connection.execute("SELECT publish_state FROM evidence_clips").fetchone() == (
            ClipPublishState.PUBLISHED.value,
        )
        assert connection.execute("SELECT state FROM derivative_jobs").fetchone() == (
            "CANCELLED",
        )
        assert connection.execute("SELECT state FROM derivative_evidence_slots").fetchone() == (
            "UNAVAILABLE",
        )
    assert allowed["checksum"] != blocked["checksum"]


def test_schema16_fixture_is_deterministic_across_rebuilds(tmp_path: Path) -> None:
    first = build_schema16_fixture(tmp_path / "first.sqlite3", drain_blocked=True)
    second = build_schema16_fixture(tmp_path / "second.sqlite3", drain_blocked=True)

    assert first["row_counts"] == second["row_counts"]
    assert first["checksum"] == second["checksum"]
