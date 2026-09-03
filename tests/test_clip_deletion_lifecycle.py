"""Operator-initiated primary clip deletion: worker-owned retention reuse.

Covers the layers introduced for explicit clip deletion --
``ClipMaintenance.purge_clip`` / ``ClipRecorder.delete_clip`` (delegation to
the existing, already-tested ``EvidenceRetention.purge``),
``ClipDeletionControlService`` (idempotent short-circuit + per-clip
serialization), and the worker's authenticated HTTP surface
(``DELETE /clips/{clip_id}`` on the shared derivative-control listener).

Crash/restart convergence itself is already exhaustively covered by
``tests/test_evidence_reconciliation.py`` (``test_retention_tombstone_...``,
``test_restart_completes_retention_after_crash_between_delete_and_db_transition``)
against the same ``begin_clip_retention``/``complete_clip_retention`` primitives
this feature reuses unchanged; the tests here additionally prove *this*
feature's own entry points (``purge_clip``, the control service, the HTTP
route) reach those primitives and converge the same way.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path

from worker.pipeline.output.evidence.clip_maintenance import ClipMaintenance, default_disk_usage
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder, ClipRecorderConfig
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderStats
from worker.pipeline.output.evidence.evidence_retention import PurgeResult
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.mjpeg_server import MjpegServer, MjpegServerConfig
from worker.runtime.clip_deletion_control import ClipDeletionControlService

NOW = "2026-05-01T00:00:00Z"
LATER = "2026-05-01T00:00:01Z"
MANIFEST_ID = "a" * 64
POLICY_ID = "b" * 64
TRACE_ID = "c" * 64


def _write_finalized_clip(store_dir: Path, clip_id: str, *, media: bytes = b"clean-source") -> Path:
    clip_dir = store_dir / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(media)
    manifest = {
        "clip_id": clip_id,
        "finalized": True,
        "started_at": NOW,
        "video_available": True,
        "path": f"clips/{clip_id}/clip.mp4",
    }
    (clip_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return clip_dir


# --- delegation: ClipMaintenance/ClipRecorder reuse EvidenceRetention.purge ---


def test_clip_maintenance_purge_clip_deletes_verified_clip_and_calls_db_hooks(
    tmp_path: Path,
) -> None:
    store_dir = tmp_path / "clip-store"
    clip_dir = _write_finalized_clip(store_dir, "clip-a")
    begun: list[str] = []
    completed: list[str] = []
    config = ClipRecorderConfig(store_dir=store_dir)
    maintenance = ClipMaintenance(
        config,
        ClipRecorderStats(),
        is_clip_held=lambda _clip_id: False,
        disk_usage_provider=default_disk_usage,
        begin_clip_purge=lambda clip_id: begun.append(clip_id) or True,
        complete_clip_purge=lambda clip_id: completed.append(clip_id),
    )

    result = maintenance.purge_clip("clip-a")

    assert result is PurgeResult.PURGED
    assert not clip_dir.exists()
    assert begun == ["clip-a"]
    assert completed == ["clip-a"]


def test_clip_recorder_delete_clip_delegates_to_maintenance(tmp_path: Path) -> None:
    store_dir = tmp_path / "clip-store"
    clip_dir = _write_finalized_clip(store_dir, "clip-a")
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=store_dir),
        is_clip_held=lambda _clip_id: False,
        begin_clip_purge=lambda _clip_id: True,
        complete_clip_purge=lambda _clip_id: None,
    )

    result = recorder.delete_clip("clip-a")

    assert result is PurgeResult.PURGED
    assert not clip_dir.exists()


def test_clip_maintenance_refuses_symlinked_clips_root_without_deleting_outside(
    tmp_path: Path,
) -> None:
    store_dir = tmp_path / "clip-store"
    outside = tmp_path / "outside"
    clip_dir = outside / "clip-a"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"external")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-a",
                "finalized": True,
                "started_at": NOW,
                "video_available": True,
                "path": "clips/clip-a/clip.mp4",
            }
        ),
        encoding="utf-8",
    )
    store_dir.mkdir()
    (store_dir / "clips").symlink_to(outside, target_is_directory=True)
    maintenance = ClipMaintenance(
        ClipRecorderConfig(store_dir=store_dir),
        ClipRecorderStats(),
        is_clip_held=lambda _clip_id: False,
        disk_usage_provider=default_disk_usage,
    )

    assert maintenance.preflight_clip("clip-a") is PurgeResult.UNVERIFIABLE
    assert maintenance.purge_clip("clip-a") is PurgeResult.UNVERIFIABLE
    assert clip_dir.is_dir()
    assert (clip_dir / "clip.mp4").is_file()


def test_clip_maintenance_refuses_symlinked_store_path_intermediate(
    tmp_path: Path,
) -> None:
    actual_parent = tmp_path / "actual-parent"
    actual_store = actual_parent / "clip-store"
    clip_dir = _write_finalized_clip(actual_store, "clip-a")
    configured_parent = tmp_path / "configured-parent"
    configured_parent.symlink_to(actual_parent, target_is_directory=True)
    maintenance = ClipMaintenance(
        ClipRecorderConfig(store_dir=configured_parent / "clip-store"),
        ClipRecorderStats(),
        is_clip_held=lambda _clip_id: False,
        disk_usage_provider=default_disk_usage,
    )

    assert maintenance.preflight_clip("clip-a") is PurgeResult.UNVERIFIABLE
    assert maintenance.purge_clip("clip-a") is PurgeResult.UNVERIFIABLE
    assert clip_dir.is_dir()


def test_root_swap_after_preflight_refuses_destructive_delete(
    tmp_path: Path,
) -> None:
    store_dir = tmp_path / "clip-store"
    clip_dir = _write_finalized_clip(store_dir, "clip-a")
    outside = tmp_path / "outside"
    outside_clip = outside / "clip-a"
    outside_clip.mkdir(parents=True)
    (outside_clip / "clip.mp4").write_bytes(b"external")
    (outside_clip / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-a",
                "finalized": True,
                "started_at": NOW,
                "video_available": True,
                "path": "clips/clip-a/clip.mp4",
            }
        ),
        encoding="utf-8",
    )
    (outside / "sentinel").write_text("preserve", encoding="utf-8")
    maintenance = ClipMaintenance(
        ClipRecorderConfig(store_dir=store_dir),
        ClipRecorderStats(),
        is_clip_held=lambda _clip_id: False,
        disk_usage_provider=default_disk_usage,
    )
    preflight_finished = threading.Event()
    root_swapped = threading.Event()
    result: list[PurgeResult | None] = []

    def delete_after_preflight() -> None:
        assert maintenance.preflight_clip("clip-a") is None
        preflight_finished.set()
        assert root_swapped.wait(2.0)
        result.append(maintenance.purge_clip("clip-a"))

    deletion = threading.Thread(target=delete_after_preflight)
    deletion.start()
    assert preflight_finished.wait(2.0)
    original_root = store_dir / "clips"
    original_root.rename(store_dir / "clips-before-swap")
    original_root.symlink_to(outside, target_is_directory=True)
    root_swapped.set()
    deletion.join(2.0)

    assert not deletion.is_alive()
    assert result == [PurgeResult.UNVERIFIABLE]
    assert (store_dir / "clips-before-swap" / clip_dir.name).is_dir()
    assert outside_clip.is_dir()
    assert (outside_clip / "clip.mp4").is_file()
    assert (outside / "sentinel").read_text(encoding="utf-8") == "preserve"


def test_clip_recorder_delete_clip_reports_unverifiable_for_symlink_escape(
    tmp_path: Path,
) -> None:
    store_dir = tmp_path / "clip-store"
    (store_dir / "clips").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (store_dir / "clips" / "clip-a").symlink_to(outside, target_is_directory=True)
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=store_dir),
        is_clip_held=lambda _clip_id: False,
    )

    result = recorder.delete_clip("clip-a")

    assert result is PurgeResult.UNVERIFIABLE
    assert outside.exists()


def test_clip_recorder_delete_clip_reports_missing_for_unknown_clip(tmp_path: Path) -> None:
    store_dir = tmp_path / "clip-store"
    (store_dir / "clips").mkdir(parents=True)
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=store_dir),
        is_clip_held=lambda _clip_id: False,
    )

    assert recorder.delete_clip("never-existed") is PurgeResult.MISSING


def test_clip_recorder_delete_clip_rejects_path_traversal_clip_id(tmp_path: Path) -> None:
    """Unlike automatic ``rotate()`` (whose candidates only ever come from
    enumerating real ``clips/*/manifest.json`` directories), ``delete_clip``
    builds its path directly from an operator-supplied string and must
    refuse to escape the clips root itself, before any filesystem check.
    """
    store_dir = tmp_path / "clip-store"
    (store_dir / "clips").mkdir(parents=True)
    outside = store_dir / "escaped-sibling"
    outside.mkdir()
    (outside / "canary.txt").write_text("do not delete")
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=store_dir),
        is_clip_held=lambda _clip_id: False,
    )

    assert recorder.delete_clip("../escaped-sibling") is PurgeResult.UNVERIFIABLE
    assert (outside / "canary.txt").exists()


def test_operator_delete_preserves_sixty_day_floor(tmp_path: Path) -> None:
    store_dir = tmp_path / "clip-store"
    clip_dir = store_dir / "clips" / "young"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"young")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "young",
                "finalized": True,
                "started_at": "2026-08-23T00:00:00Z",
                "video_available": True,
                "path": "clips/young/clip.mp4",
            }
        ),
        encoding="utf-8",
    )
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=store_dir),
        is_clip_held=lambda _clip_id: False,
    )

    assert recorder.preflight_clip_deletion("young") is PurgeResult.HELD
    assert recorder.delete_clip("young") is PurgeResult.HELD
    assert clip_dir.is_dir()


def test_clip_recorder_delete_clip_reports_held_when_is_clip_held_true(tmp_path: Path) -> None:
    store_dir = tmp_path / "clip-store"
    clip_dir = _write_finalized_clip(store_dir, "clip-a")
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=store_dir),
        is_clip_held=lambda _clip_id: True,
    )

    assert recorder.delete_clip("clip-a") is PurgeResult.HELD
    assert clip_dir.exists()


# --- ClipDeletionControlService: non-destructive preflight --------------------


def test_control_service_preflight_reports_hold_without_invoking_delete() -> None:
    calls: list[str] = []
    service = ClipDeletionControlService(
        preflight_clip=lambda _clip_id: PurgeResult.HELD,
        delete_clip=lambda clip_id: calls.append(clip_id) or PurgeResult.PURGED,
    )

    assert service.preflight("clip-a") == {"clip_id": "clip-a", "status": "HELD"}
    assert calls == []


def test_control_service_preflight_reports_ready_without_invoking_delete() -> None:
    calls: list[str] = []
    service = ClipDeletionControlService(
        preflight_clip=lambda _clip_id: None,
        delete_clip=lambda clip_id: calls.append(clip_id) or PurgeResult.PURGED,
    )

    assert service.preflight("clip-a") == {"clip_id": "clip-a", "status": "READY"}
    assert calls == []


def test_control_service_delete_reports_missing_truthfully() -> None:
    service = ClipDeletionControlService(
        preflight_clip=lambda _clip_id: PurgeResult.MISSING,
        delete_clip=lambda _clip_id: PurgeResult.MISSING,
    )

    assert service.delete("clip-a") == {"clip_id": "clip-a", "status": "MISSING"}


def test_control_service_keeps_durable_state_out_of_worker_adapter() -> None:
    calls: list[str] = []
    service = ClipDeletionControlService(
        preflight_clip=lambda _clip_id: None,
        delete_clip=lambda clip_id: calls.append(clip_id) or PurgeResult.PURGED,
    )

    assert service.delete("clip-a") == {"clip_id": "clip-a", "status": "PURGED"}
    assert calls == ["clip-a"]


def test_control_service_serializes_same_clip_delete_at_a_barrier() -> None:
    first_started = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    calls: list[str] = []
    calls_lock = threading.Lock()
    results: list[dict[str, object]] = []

    def delete_clip(clip_id: str) -> PurgeResult:
        with calls_lock:
            calls.append(clip_id)
            call_number = len(calls)
        if call_number == 1:
            first_started.set()
            assert release_first.wait(2.0)
        return PurgeResult.PURGED

    service = ClipDeletionControlService(
        preflight_clip=lambda _clip_id: None,
        delete_clip=delete_clip,
    )

    first = threading.Thread(target=lambda: results.append(service.delete("clip-a")))
    first.start()
    assert first_started.wait(2.0)

    def second_delete() -> None:
        second_entered.set()
        results.append(service.delete("clip-a"))

    second = threading.Thread(target=second_delete)
    second.start()
    assert second_entered.wait(2.0)
    with calls_lock:
        assert calls == ["clip-a"]
    release_first.set()
    first.join(2.0)
    second.join(2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert calls == ["clip-a", "clip-a"]
    assert results == [
        {"clip_id": "clip-a", "status": "PURGED"},
        {"clip_id": "clip-a", "status": "PURGED"},
    ]


# --- worker HTTP surface: DELETE /clips/{clip_id} ------------------------------


def _seed_central_evidence(
    database: Path,
    *,
    clip_id: str = "clip-a",
    incident_id: str = "incident-a",
    lifecycle_state: str = "COMPLETE",
    publish_state: str | None = "PUBLISHED",
) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_clips (clip_id,local_state,state_version,publish_state) "
            "VALUES (?,'VERIFIED',2,?)",
            (clip_id, publish_state),
        )
        if publish_state != "PUBLISHED":
            connection.commit()
            return
        connection.execute(
            "INSERT INTO runtime_manifest_contents VALUES (?,1,'{}',?)", (MANIFEST_ID, NOW)
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
            "VALUES ('d'*64,1,'boot-a','camera-a',1,1,0,0,16,16,'fresh',1)".replace(
                "'d'*64", f"'{'d' * 64}'"
            ),
        )
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at,"
            "delivery_state) VALUES ('event-a',?,'{}','ACKED',1,1,'ACKED')",
            (NOW,),
        )
        connection.execute(
            "INSERT INTO evidence_decision_traces "
            "(trace_id,trace_schema_version,analysis_trace_id,module_qualified_id,"
            "policy_qualified_id,effective_policy_id,runtime_manifest_sha256,reason,"
            "previous_state,current_state,triggered,track_missing_reason,bed_missing_reason) "
            "VALUES (?,1,?,'fall.v1','fall.policy.v1',?,?,'fall-onset','clear','fall',1,"
            "'not-applicable','not-applicable')",
            (TRACE_ID, "d" * 64, POLICY_ID, MANIFEST_ID),
        )
        connection.execute(
            "INSERT INTO evidence_incidents "
            "(incident_id,edge_event_id,camera_id,event_type,detected_at,"
            "runtime_manifest_sha256,decision_trace_id,module_qualified_id,"
            "policy_qualified_id,effective_policy_id,provenance_state,primary_clip_id,"
            "lifecycle_state,created_at,updated_at) VALUES "
            "(?,'event-a','camera-a','fall',?,?,?,?,?,?,'QUALIFIED',?,?,?,?)",
            (
                incident_id,
                NOW,
                MANIFEST_ID,
                TRACE_ID,
                "fall.v1",
                "fall.policy.v1",
                POLICY_ID,
                clip_id,
                lifecycle_state,
                NOW,
                NOW,
            ),
        )
        connection.commit()


def _seed_derivative_job(database: Path, incident_id: str, *, state: str = "PENDING") -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO derivative_jobs "
            "(incident_id,derivative_kind,request_id,state,created_at,updated_at) "
            "VALUES (?,'STILL',?,?,?,?)",
            (incident_id, "1" * 64, state, NOW, NOW),
        )
        connection.commit()


_RETENTION: dict[tuple[Path, str], str] = {}


def _control_service(database: Path, store_dir: Path) -> ClipDeletionControlService:
    del database
    maintenance = ClipMaintenance(
        ClipRecorderConfig(store_dir=store_dir),
        ClipRecorderStats(),
        is_clip_held=lambda _clip_id: False,
        disk_usage_provider=lambda _path: shutil.disk_usage(store_dir),
    )
    return ClipDeletionControlService(
        preflight_clip=maintenance.preflight_clip,
        delete_clip=maintenance.purge_clip,
    )


def _begin(database: Path):
    def begin(clip_id: str) -> bool:
        key = (database, clip_id)
        if _RETENTION.get(key) == "PURGED":
            return False
        _RETENTION[key] = "PENDING"
        return True

    return begin


def _complete(database: Path):
    def complete(clip_id: str) -> None:
        _RETENTION[(database, clip_id)] = "PURGED"

    return complete


def _fail(database: Path):
    def fail(clip_id: str, reason: str) -> None:
        del reason
        _RETENTION[(database, clip_id)] = "FAILED"

    return fail


def _preflight_over_http(
    port: int, clip_id: str, *, token: str | None = "relay-token"
) -> tuple[int, dict[str, object]]:
    headers = {} if token is None else {"X-Edge-Relay-Token": token}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/clips/{clip_id}/deletion-preflight",
        method="GET",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, {}


def _delete_over_http(
    port: int, clip_id: str, *, token: str | None = "relay-token"
) -> tuple[int, dict[str, object]]:
    headers = {} if token is None else {"X-Edge-Relay-Token": token}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/clips/{clip_id}",
        method="DELETE",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, {}


def test_worker_http_clip_delete_unavailable_without_control_composed(tmp_path: Path) -> None:
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token="relay-token"),
    )
    server.start()
    try:
        status, _payload = _delete_over_http(server.port, "clip-a")
    finally:
        server.stop()
    assert status == 503


# --- crash/restart convergence through this feature's own entry points --------
