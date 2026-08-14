from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.clips import artifacts as artifact_module
from backend.app.main import create_app, no_lifespan

NOW = "2026-08-13T00:00:00Z"
PRIMARY = b"clean-source-packet-media"
ANNOTATED = b"annotated-derivative-media"
MANIFEST_ID = "a" * 64
ANALYSIS_ID = "b" * 64
TRACE_ID = "c" * 64
POLICY_ID = "d" * 64
SCENE_ID = "e" * 64


@pytest.fixture
def clip_artifact_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "clip-store"
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    monkeypatch.setenv("API_LABEL_STORE", str(tmp_path / "labels"))
    clip_dir = root / "clips" / "clip-a"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(PRIMARY)
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-a",
                "camera_id": "camera-a",
                "event_ref": "event-a",
                "event_type": "fall",
                "started_at": NOW,
                "duration_s": 1.0,
                "codec": "h264",
                "path": "clips/clip-a/clip.mp4",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    derivative_sha = hashlib.sha256(ANNOTATED).hexdigest()
    derivative_relpath = f"derivatives/incident-a/{derivative_sha}.mp4"
    derivative_path = root / derivative_relpath
    derivative_path.parent.mkdir(parents=True)
    derivative_path.write_bytes(ANNOTATED)
    _seed_central_artifacts(
        artifact_module.EDGE_DATABASE_PATH,
        derivative_relpath,
        derivative_sha,
    )
    return root


def _seed_central_artifacts(database: Path, relpath: str, derivative_sha: str) -> None:
    primary_sha = hashlib.sha256(PRIMARY).hexdigest()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO runtime_manifest_contents VALUES (?,1,?,?)",
            (MANIFEST_ID, '{"cameras":[{"camera_id":"camera-a"}]}', NOW),
        )
        connection.execute(
            "INSERT INTO runtime_manifest_boots VALUES ('boot-a',?,?)", (MANIFEST_ID, NOW)
        )
        connection.execute(
            "INSERT INTO runtime_manifest_cameras VALUES ('boot-a','camera-a',?,?)",
            (MANIFEST_ID, NOW),
        )
        connection.execute(
            "INSERT INTO runtime_analysis_traces "
            "(trace_id,trace_schema_version,worker_boot_id,camera_id,stream_epoch,"
            "frame_seq,pts,source_time_sec,frame_width,frame_height,"
            "bed_region_provenance,storage_bytes) "
            "VALUES (?,1,'boot-a','camera-a',1,1,0,0,320,180,'fresh',1)",
            (ANALYSIS_ID,),
        )
        connection.execute(
            "INSERT INTO evidence_decision_traces VALUES "
            "(?,1,?,'fall.v1','fall.policy.v1',?,?,'fall-onset','clear','fall',1,"
            "7,NULL,NULL,'not-applicable')",
            (TRACE_ID, ANALYSIS_ID, POLICY_ID, MANIFEST_ID),
        )
        connection.execute(
            "INSERT INTO evidence_decision_values VALUES (?, 'fall_probability', 0.9, NULL)",
            (TRACE_ID,),
        )
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at,"
            "delivery_state) VALUES ('event-a',?,'{}','ACKED',1,1,'ACKED')",
            (NOW,),
        )
        connection.execute(
            "INSERT INTO evidence_clips (clip_id,local_state,state_version,publish_state) "
            "VALUES ('clip-a','VERIFIED',2,'PUBLISHED')"
        )
        connection.execute("INSERT INTO clip_events VALUES ('clip-a','event-a',0)")
        connection.execute(
            "INSERT INTO evidence_event_trace_refs VALUES ('event-a',?)", (TRACE_ID,)
        )
        connection.execute(
            "INSERT INTO evidence_clip_trace_refs VALUES ('clip-a','event-a',?)", (TRACE_ID,)
        )
        connection.execute(
            "INSERT INTO evidence_media_objects "
            "(media_id,content_sha256,size_bytes,mime_type,contained_relpath,basename,"
            "created_at) VALUES "
            "('primary-media',?,?,'video/mp4','clips/clip-a/clip.mp4','clip.mp4',?)",
            (primary_sha, len(PRIMARY), NOW),
        )
        connection.execute(
            "INSERT INTO evidence_media_objects "
            "(media_id,content_sha256,size_bytes,mime_type,contained_relpath,basename,"
            "created_at) VALUES "
            "('derivative-media',?,?,'video/mp4',?,'annotated.mp4',?)",
            (derivative_sha, len(ANNOTATED), relpath, NOW),
        )
        connection.execute(
            "INSERT INTO evidence_incidents "
            "(incident_id,edge_event_id,camera_id,event_type,detected_at,"
            "runtime_manifest_sha256,decision_trace_id,module_qualified_id,"
            "policy_qualified_id,effective_policy_id,provenance_state,primary_clip_id,"
            "lifecycle_state,created_at,updated_at) VALUES "
            "('incident-a','event-a','camera-a','fall',?,?,?,?,?,?,'QUALIFIED','clip-a',"
            "'COMPLETE',?,?)",
            (NOW, MANIFEST_ID, TRACE_ID, "fall.v1", "fall.policy.v1", POLICY_ID, NOW, NOW),
        )
        connection.execute(
            "INSERT INTO evidence_artifact_slots "
            "(incident_id,slot_name,state,media_id,created_at,updated_at) VALUES "
            "('incident-a','PRIMARY_CLIP','AVAILABLE','primary-media',?,?)",
            (NOW, NOW),
        )
        connection.execute(
            "INSERT INTO evidence_primary_clips "
            "(incident_id,clip_id,manifest_relpath,manifest_sha256,manifest_size_bytes,"
            "media_id,source_packet_preserved,source_media_json,truncation_json,created_at) "
            "VALUES ('incident-a','clip-a','clips/clip-a/manifest.json',?,1,"
            "'primary-media',1,'{}','[]',?)",
            ("f" * 64, NOW),
        )
        connection.execute(
            "INSERT INTO derivative_evidence_slots "
            "(incident_id,derivative_kind,state,media_id,created_at,updated_at) "
            "VALUES ('incident-a','ANNOTATED_CLIP','AVAILABLE','derivative-media',?,?)",
            (NOW, NOW),
        )
        connection.execute(
            "INSERT INTO derivative_render_records VALUES "
            "('incident-a','ANNOTATED_CLIP',?,'derivative-media','clip-a',?,?,?,?,"
            "1,'opencv-cpu','cpu','host','overlay-cpu.v1',320,180,0,1000,?)",
            ("1" * 64, primary_sha, TRACE_ID, MANIFEST_ID, SCENE_ID, NOW),
        )
        connection.commit()


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert response.status_code == 204


def test_authenticated_artifact_views_analysis_and_annotated_range(
    clip_artifact_store: Path,
) -> None:
    del clip_artifact_store
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        assert client.get("/api/v1/clips/clip-a/artifacts").status_code == 401
        _login(client)
        views = client.get("/api/v1/clips/clip-a/artifacts")
        analysis = client.get("/api/v1/clips/clip-a/analysis")
        video = client.get(
            "/api/v1/clips/clip-a/video?view=annotated",
            headers={"Range": "bytes=2-8"},
        )

    assert views.status_code == 200
    assert views.json() == {
        "clip_id": "clip-a",
        "clean": "AVAILABLE",
        "analysis": "AVAILABLE",
        "annotated": "AVAILABLE",
        "playback_view": "annotated",
        "annotated_fallback_to_clean": False,
    }
    assert "relpath" not in views.text
    assert analysis.status_code == 200
    assert analysis.json()["decision_trace_id"] == TRACE_ID
    assert analysis.json()["values"] == [
        {"name": "fall_probability", "value": 0.9, "missing_reason": None}
    ]
    assert video.status_code == 206
    assert video.content == ANNOTATED[2:9]
    assert video.headers["x-clip-view"] == "annotated"


def test_missing_or_mutated_annotation_falls_back_to_clean(
    clip_artifact_store: Path,
) -> None:
    derivative = next((clip_artifact_store / "derivatives").glob("*/*.mp4"))
    derivative.write_bytes(b"mutated")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips/clip-a/video?view=annotated")

    assert response.status_code == 200
    assert response.content == PRIMARY
    assert response.headers["x-clip-view"] == "clean"
    assert response.headers["x-clip-view-fallback"] == "clean"
