from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest
from compact_cutover_fixtures import TS, cutover_request

from backend.app.edge_db.compact_cutover import run_compact_cutover

pytestmark = pytest.mark.usefixtures("supported_compact_cutover_sqlite")


def test_evidence_incident_and_current_review_win_conflicts(tmp_path: Path) -> None:
    # Given canonical evidence disagrees with the generic catalog and has a current review
    request = cutover_request(tmp_path)
    with sqlite3.connect(request.source) as connection:
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at) "
            "VALUES ('event-1',?,'{}','ACKED',1,1)",
            (TS,),
        )
        connection.execute(
            "INSERT INTO evidence_clips (clip_id,local_state,state_version) "
            "VALUES ('clip-1','VERIFIED',1)"
        )
        connection.execute(
            "INSERT INTO evidence_incidents "
            "(incident_id,edge_event_id,camera_id,event_type,detected_at,"
            "provenance_state,provenance_missing_reason,primary_clip_id,lifecycle_state,revision,"
            "created_at,updated_at) VALUES "
            "('incident-1','event-1','canonical-camera','fall',?,'MISSING',"
            "'NOT_RECORDED','clip-1','STAGING',1,?,?)",
            (TS, TS, TS),
        )
        connection.execute(
            "INSERT INTO evidence_primary_clips "
            "(incident_id,clip_id,source_packet_preserved,source_missing_reason,"
            "truncation_json,unavailable_reason,created_at) "
            "VALUES ('incident-1','clip-1',0,'NOT_RECORDED','[]','MISSING',?)",
            (TS,),
        )
        connection.execute(
            "INSERT INTO events "
            "(edge_event_id,camera_id,event_type,detected_at,payload_json) "
            "VALUES ('event-1','catalog-camera','bed-exit',?,'{}')",
            (TS,),
        )
        connection.execute(
            "INSERT INTO control_evidence_review_revisions "
            "(review_id,incident_id,clip_id,review_version,actor_id,reviewed_at,"
            "disposition,notes) VALUES "
            "('review-1','incident-1','clip-1',1,'operator',?,'FALSE_POSITIVE',NULL)",
            (TS,),
        )
        connection.execute(
            "INSERT INTO control_evidence_review_state "
            "(incident_id,clip_id,current_version) VALUES ('incident-1','clip-1',1)"
        )
        connection.commit()
    connection.close()
    with sqlite3.connect(request.source, isolation_level=None) as checkpoint:
        checkpoint.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copyfile(request.source, request.live)

    # When the compact projection reconciles the conflicting facts
    run_compact_cutover(request)

    # Then canonical evidence wins and the closed TP/FP classifier is deterministic.
    with sqlite3.connect(request.live) as connection:
        row = connection.execute(
            "SELECT camera_id,event_type,review_version,review_disposition "
            "FROM incidents WHERE incident_id='incident-1'"
        ).fetchone()
    assert row == ("canonical-camera", "fall", 1, "FP")
