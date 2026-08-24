from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.functions import audit_record_hash, register_edge_db_functions
from backend.app.edge_db.migrator import migrate_database

TS = "2026-08-24T00:00:00Z"
HASH_A = "ab" * 32
HASH_B = "cd" * 32


def _db(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "edge.sqlite3"
    migrate_database(path)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    register_edge_db_functions(connection)
    return connection


def _seed_room(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO locations "
        "(location_id, kind, parent_location_id, parent_kind, name, order_index, "
        "created_at, updated_at) VALUES "
        "('floor-1','FLOOR',NULL,NULL,'Floor',0,?,?)",
        (TS, TS),
    )
    connection.execute(
        "INSERT INTO locations "
        "(location_id, kind, parent_location_id, parent_kind, name, order_index, "
        "created_at, updated_at) VALUES "
        "('room-1','ROOM','floor-1','FLOOR','Room',0,?,?)",
        (TS, TS),
    )


def _seed_camera(connection: sqlite3.Connection) -> None:
    _seed_room(connection)
    connection.execute(
        "INSERT INTO cameras ("
        "camera_id,label,rtsp_url,normalized_stream_identity,"
        "room_location_id,room_location_kind,edge_ref,mapping_state,"
        "never_connected,revision,created_at,updated_at"
        ") VALUES ("
        "'cam-1','Cam','rtsp://127.0.0.1/one','stream-one',"
        "'room-1','ROOM','edge-1','UNMAPPED',1,1,?,?)",
        (TS, TS),
    )


@pytest.mark.parametrize(
    ("sql", "params"),
    [
        (
            "INSERT INTO credentials "
            "(id,username,algorithm,salt,password_hash,updated_at) "
            "VALUES (1,'admin','scrypt',?,?,?)",
            (b"s" * 16, b"h" * 64, "xxxx-xx-xxTxx:xx:xxZ"),
        ),
        (
            "INSERT INTO edge_site (id, fall_on, fall_mode, fall_start_time, fall_end_time, "
            "updated_at) VALUES (1,1,'window','zz:zz','08:00',?)",
            (TS,),
        ),
        (
            "INSERT INTO edge_site (id, fall_on, fall_mode, fall_start_time, fall_end_time, "
            "updated_at) VALUES (1,1,'window','99:99','08:00',?)",
            (TS,),
        ),
        (
            "INSERT INTO policies ("
            "facility_id,module_id,module_version,schema_id,schema_version,"
            "active_values_json,active_content_sha256,previous_present,"
            "activation_generation,status,activated_at,updated_at"
            ") VALUES ('fac','fall',1,'fall-schema',1,?,?,0,1,'applied',?,?)",
            ('{"x":"' + ("é" * 9000) + '"}', HASH_A, TS, TS),
        ),
        (
            "INSERT INTO clips ("
            "clip_id,camera_id,event_facet,started_at,local_state,local_reason,"
            "publish_state,retention_state,revision,created_at,updated_at,"
            "manifest_relpath"
            ") VALUES ('clip-1','cam-1','fall',?,'UNAVAILABLE','MISSING',"
            "'WAITING','RETAINED',1,?,?,'clips/a.json')",
            (TS, TS, TS),
        ),
        (
            "INSERT INTO audit_events ("
            "occurred_at,recorded_at,clock_quality,actor_type,actor_id,auth_mechanism,"
            "action,target_type,target_id,outcome,previous_hash,record_hash,"
            "retention_class"
            ") VALUES (?,?, 'trusted','system','sys','none','boot','db','edge',"
            "'success',?,?, 'standard')",
            (TS, TS, HASH_A, HASH_B),
        ),
        (
            "INSERT INTO audit_events ("
            "occurred_at,recorded_at,clock_quality,actor_type,actor_id,auth_mechanism,"
            "action,target_type,target_id,outcome,previous_hash,record_hash,"
            "retention_class,hold_reference"
            ") VALUES (?,?, 'trusted','system','sys','none','boot','db','edge',"
            "'success',?,?, 'standard','hold-1')",
            (TS, TS, "0" * 64, HASH_B),
        ),
    ],
)
def test_schema18_rejects_previously_accepted_bad_tuples(
    tmp_path: Path,
    sql: str,
    params: tuple[object, ...],
) -> None:
    connection = _db(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(sql, params)
    finally:
        connection.close()


def test_incident_identity_and_revision_are_immutable(tmp_path: Path) -> None:
    connection = _db(tmp_path)
    try:
        connection.execute(
            "INSERT INTO incidents ("
            "incident_id,edge_event_id,facility_id,camera_id,event_type,detected_at,"
            "lifecycle_state,provenance_state,provenance_missing_reason,"
            "review_version,revision,created_at,updated_at"
            ") VALUES ('inc-1','evt-1','fac','cam-1','fall',?,'OPEN','MISSING',"
            "'NOT_RECORDED',0,1,?,?)",
            (TS, TS, TS),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE incidents SET edge_event_id = 'evt-2', revision = 2 "
                "WHERE incident_id = 'inc-1'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE incidents SET updated_at = ?, revision = 1 WHERE incident_id = 'inc-1'",
                (TS,),
            )
    finally:
        connection.close()


def test_artifact_identity_transition_and_revision_are_guarded(tmp_path: Path) -> None:
    connection = _db(tmp_path)
    try:
        connection.execute(
            "INSERT INTO incidents ("
            "incident_id,edge_event_id,facility_id,camera_id,event_type,detected_at,"
            "lifecycle_state,provenance_state,provenance_missing_reason,"
            "review_version,revision,created_at,updated_at"
            ") VALUES ('inc-1','evt-1','fac','cam-1','fall',?,'OPEN','MISSING',"
            "'NOT_RECORDED',0,1,?,?)",
            (TS, TS, TS),
        )
        connection.execute(
            "INSERT INTO artifacts ("
            "incident_id,kind,state,revision,created_at,updated_at,captured_at"
            ") VALUES ('inc-1','SNAPSHOT','PENDING',1,?,?,?)",
            (TS, TS, TS),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE artifacts SET state = 'PURGED', revision = 2 "
                "WHERE incident_id = 'inc-1'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE artifacts SET kind = 'PRIMARY_CLIP', revision = 2 "
                "WHERE incident_id = 'inc-1'"
            )
    finally:
        connection.close()


def _insert_incident(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO incidents ("
        "incident_id,edge_event_id,facility_id,camera_id,event_type,detected_at,"
        "lifecycle_state,provenance_state,provenance_missing_reason,"
        "review_version,revision,created_at,updated_at"
        ") VALUES ('inc-1','evt-1','fac','cam-1','fall',?,'OPEN','MISSING',"
        "'NOT_RECORDED',0,1,?,?)",
        (TS, TS, TS),
    )


def _insert_clip_row(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO clips ("
        "clip_id,camera_id,event_facet,started_at,local_state,local_reason,"
        "publish_state,retention_state,revision,created_at,updated_at,"
        "manifest_relpath,manifest_sha256,manifest_size_bytes,"
        "media_relpath,media_sha256,media_size_bytes"
        ") VALUES ('clip-1','cam-1','fall',?,'AVAILABLE',NULL,'WAITING','RETAINED',"
        "1,?,?, 'clips/a.json',?,?, 'clips/a.mp4',?,?)",
        (TS, TS, TS, HASH_A, 10, HASH_B, 20),
    )


def test_available_to_corrupt_preserves_retained_identity(tmp_path: Path) -> None:
    connection = _db(tmp_path)
    later = "2026-08-24T00:00:01Z"
    try:
        _insert_incident(connection)
        _insert_clip_row(connection)
        connection.execute(
            "INSERT INTO artifacts ("
            "incident_id,kind,artifact_id,clip_id,state,contained_relpath,"
            "content_sha256,size_bytes,mime_type,codec,captured_at,revision,"
            "created_at,updated_at"
            ") VALUES ('inc-1','PRIMARY_CLIP','art-1','clip-1','AVAILABLE',"
            "'clips/a.mp4',?,20,'video/mp4','hevc',NULL,1,?,?)",
            (HASH_B, TS, TS),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE artifacts SET state='CORRUPT', reason='HASH_MISMATCH', "
                "revision=2, contained_relpath='clips/other.mp4', updated_at=? "
                "WHERE incident_id='inc-1'",
                (later,),
            )
        connection.execute(
            "UPDATE artifacts SET state='CORRUPT', reason='HASH_MISMATCH', "
            "revision=2, updated_at=? WHERE incident_id='inc-1'",
            (later,),
        )
        row = connection.execute(
            "SELECT state, reason, contained_relpath, content_sha256, size_bytes, "
            "mime_type, captured_at, revision FROM artifacts WHERE incident_id='inc-1'"
        ).fetchone()
        assert row == (
            "CORRUPT",
            "HASH_MISMATCH",
            "clips/a.mp4",
            HASH_B,
            20,
            "video/mp4",
            None,
            2,
        )
    finally:
        connection.close()


def test_snapshot_available_to_corrupt_rejects_rewritten_identity(tmp_path: Path) -> None:
    connection = _db(tmp_path)
    later = "2026-08-24T00:00:01Z"
    try:
        _insert_incident(connection)
        connection.execute(
            "INSERT INTO artifacts ("
            "incident_id,kind,artifact_id,state,contained_relpath,content_sha256,"
            "size_bytes,mime_type,captured_at,revision,created_at,updated_at"
            ") VALUES ('inc-1','SNAPSHOT','snap-1','AVAILABLE','snaps/a.jpg',?,"
            "8,'image/jpeg',?,1,?,?)",
            (HASH_A, TS, TS, TS),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE artifacts SET state='CORRUPT', reason='BAD', revision=2, "
                "content_sha256=?, updated_at=? WHERE incident_id='inc-1'",
                (HASH_B, later),
            )
        connection.execute(
            "UPDATE artifacts SET state='CORRUPT', reason='BAD', revision=2, "
            "updated_at=? WHERE incident_id='inc-1'",
            (later,),
        )
    finally:
        connection.close()


def test_legal_artifact_transitions_are_pinned(tmp_path: Path) -> None:
    connection = _db(tmp_path)
    later = "2026-08-24T00:00:01Z"
    try:
        _insert_incident(connection)
        connection.execute(
            "INSERT INTO artifacts ("
            "incident_id,kind,state,revision,created_at,updated_at,captured_at"
            ") VALUES ('inc-1','SNAPSHOT','PENDING',1,?,?,?)",
            (TS, TS, TS),
        )
        connection.execute(
            "UPDATE artifacts SET state='AVAILABLE', artifact_id='snap-1', "
            "contained_relpath='snaps/a.jpg', content_sha256=?, size_bytes=8, "
            "mime_type='image/jpeg', revision=2, updated_at=? WHERE incident_id='inc-1'",
            (HASH_A, later),
        )
        assert connection.execute(
            "SELECT state, revision FROM artifacts WHERE incident_id='inc-1'"
        ).fetchone() == ("AVAILABLE", 2)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE artifacts SET state='PENDING', revision=3, updated_at=? "
                "WHERE incident_id='inc-1'",
                (later,),
            )
    finally:
        connection.close()


def test_audit_hash_is_previous_bytes_plus_sorted_compact_utf8() -> None:
    payload = {"b": 1, "a": "é"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(bytes.fromhex("0" * 64) + canonical.encode()).hexdigest()
    assert audit_record_hash("0" * 64, '{"b":1,"a":"é"}') == expected
    assert audit_record_hash("0" * 64, '{"a":"é","b":1}') == expected


def test_audit_insert_requires_derived_record_hash(tmp_path: Path) -> None:
    connection = _db(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO audit_events ("
                "occurred_at,recorded_at,clock_quality,actor_type,actor_id,auth_mechanism,"
                "action,target_type,target_id,outcome,previous_hash,record_hash,"
                "retention_class"
                ") VALUES (?,?, 'trusted','system','sys','none','boot','db','edge',"
                "'success',?,?, 'standard')",
                (TS, TS, "0" * 64, "1" * 64),
            )
    finally:
        connection.close()
