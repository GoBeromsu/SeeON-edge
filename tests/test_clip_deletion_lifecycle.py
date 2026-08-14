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

import hashlib
import json
import shutil
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path

from shared.edge_db.migrator import migrate_database
from worker.pipeline.output.evidence.clip_maintenance import ClipMaintenance, default_disk_usage
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder, ClipRecorderConfig
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderStats
from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox
from worker.pipeline.output.evidence.evidence_reconciliation import reconcile_event_evidence
from worker.pipeline.output.evidence.evidence_retention import PurgeResult
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.mjpeg_server import MjpegServer, MjpegServerConfig
from worker.runtime.clip_deletion_control import ClipDeletionControlService

NOW = "2026-08-13T00:00:00Z"
LATER = "2026-08-13T00:00:01Z"
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


def test_clip_recorder_delete_clip_reports_held_when_is_clip_held_true(tmp_path: Path) -> None:
    store_dir = tmp_path / "clip-store"
    clip_dir = _write_finalized_clip(store_dir, "clip-a")
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=store_dir),
        is_clip_held=lambda _clip_id: True,
    )

    assert recorder.delete_clip("clip-a") is PurgeResult.HELD
    assert clip_dir.exists()


# --- ClipDeletionControlService: idempotence and serialization -----------------


def test_control_service_short_circuits_purged_without_reinvoking_delete() -> None:
    calls: list[str] = []
    service = ClipDeletionControlService(
        delete_clip=lambda clip_id: calls.append(clip_id) or PurgeResult.PURGED,
        retention_state=lambda _clip_id: "PURGED",
    )

    payload = service.delete("clip-a")

    assert payload == {"clip_id": "clip-a", "status": "PURGED"}
    assert calls == []


def test_control_service_retries_delete_when_state_is_pending() -> None:
    calls: list[str] = []
    service = ClipDeletionControlService(
        delete_clip=lambda clip_id: calls.append(clip_id) or PurgeResult.PURGED,
        retention_state=lambda _clip_id: "PENDING",
    )

    payload = service.delete("clip-a")

    assert payload == {"clip_id": "clip-a", "status": "PURGED"}
    assert calls == ["clip-a"]


def test_control_service_first_request_reports_missing_truthfully() -> None:
    service = ClipDeletionControlService(
        delete_clip=lambda _clip_id: PurgeResult.MISSING,
        retention_state=lambda _clip_id: None,
    )

    assert service.delete("clip-a") == {"clip_id": "clip-a", "status": "MISSING"}


def test_control_service_serializes_concurrent_requests_for_the_same_clip() -> None:
    started = threading.Event()
    release = threading.Event()
    call_count = 0
    lock = threading.Lock()
    purged = threading.Event()

    def slow_delete(_clip_id: str) -> PurgeResult:
        nonlocal call_count
        with lock:
            call_count += 1
        started.set()
        assert release.wait(2.0)
        purged.set()
        return PurgeResult.PURGED

    service = ClipDeletionControlService(
        delete_clip=slow_delete,
        # Mirrors the real ``EvidenceExportRuntime.clip_retention_state``
        # contract: PURGED only becomes visible once the first delete's DB
        # transition actually lands.
        retention_state=lambda _clip_id: "PURGED" if purged.is_set() else None,
    )
    results: list[dict[str, object]] = []

    def run() -> None:
        results.append(service.delete("clip-a"))

    first = threading.Thread(target=run)
    first.start()
    assert started.wait(2.0)
    second = threading.Thread(target=run)
    second.start()
    # The second request must block behind the first's lock, not race it.
    second.join(0.1)
    assert second.is_alive()
    release.set()
    first.join(2.0)
    second.join(2.0)

    assert call_count == 1
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


def _control_service(database: Path, store_dir: Path) -> ClipDeletionControlService:
    config = ClipRecorderConfig(store_dir=store_dir)
    maintenance = ClipMaintenance(
        config,
        ClipRecorderStats(),
        is_clip_held=lambda _clip_id: False,
        disk_usage_provider=lambda _path: shutil.disk_usage(store_dir),
        begin_clip_purge=_begin(database),
        complete_clip_purge=_complete(database),
        fail_clip_purge=_fail(database),
    )

    def retention_state(clip_id: str) -> str | None:
        with EvidenceOutbox.open(database) as outbox:
            return outbox.clip_retention_state(clip_id)

    return ClipDeletionControlService(
        delete_clip=maintenance.purge_clip,
        retention_state=retention_state,
        complete_pending_purge=_complete(database),
    )


def _begin(database: Path):
    def begin(clip_id: str) -> bool:
        with EvidenceOutbox.open(database) as outbox:
            return outbox.begin_clip_retention(clip_id, updated_at=NOW)

    return begin


def _complete(database: Path):
    def complete(clip_id: str) -> None:
        with EvidenceOutbox.open(database) as outbox:
            outbox.complete_clip_retention(clip_id, updated_at=LATER)

    return complete


def _fail(database: Path):
    def fail(clip_id: str, reason: str) -> None:
        with EvidenceOutbox.open(database) as outbox:
            outbox.fail_clip_retention(clip_id, reason=reason, updated_at=LATER)

    return fail


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


def test_worker_http_deletes_clip_preserves_shared_derivative_and_is_idempotent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    store_dir = tmp_path / "clip-store"
    migrate_database(database)
    clip_dir = _write_finalized_clip(store_dir, "clip-a")
    _seed_central_evidence(database, clip_id="clip-a", lifecycle_state="COMPLETE")
    derivative_path = (
        store_dir / "derivatives" / "incident-a" / (hashlib.sha256(b"x").hexdigest() + ".mp4")
    )
    derivative_path.parent.mkdir(parents=True)
    derivative_path.write_bytes(b"shared-derivative")

    service = _control_service(database, store_dir)
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token="relay-token"),
        clip_deletion_control=service,
    )
    server.start()
    try:
        status, payload = _delete_over_http(server.port, "clip-a")
        assert status == 202
        assert payload == {"clip_id": "clip-a", "status": "PURGED"}
        assert not clip_dir.exists()
        assert derivative_path.exists(), "shared derivative blobs must never be deleted"

        # Duplicate request: idempotent, no crash, no re-delete attempt.
        status_again, payload_again = _delete_over_http(server.port, "clip-a")
        assert status_again == 202
        assert payload_again == {"clip_id": "clip-a", "status": "PURGED"}
    finally:
        server.stop()

    with EvidenceOutbox.open(database) as outbox:
        assert outbox.clip_retention_state("clip-a") == "PURGED"


def test_worker_http_clip_delete_requires_relay_auth(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    store_dir = tmp_path / "clip-store"
    migrate_database(database)
    _write_finalized_clip(store_dir, "clip-a")
    _seed_central_evidence(database, clip_id="clip-a")
    service = _control_service(database, store_dir)
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token="relay-token"),
        clip_deletion_control=service,
    )
    server.start()
    try:
        assert _delete_over_http(server.port, "clip-a", token=None)[0] == 403
        assert _delete_over_http(server.port, "clip-a", token="wrong")[0] == 403
    finally:
        server.stop()


def test_worker_http_holds_delete_for_incomplete_incident(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    store_dir = tmp_path / "clip-store"
    migrate_database(database)
    clip_dir = _write_finalized_clip(store_dir, "clip-a")
    _seed_central_evidence(database, clip_id="clip-a", lifecycle_state="STAGING")
    service = _control_service(database, store_dir)
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token="relay-token"),
        clip_deletion_control=service,
    )
    server.start()
    try:
        status, payload = _delete_over_http(server.port, "clip-a")
    finally:
        server.stop()

    assert status == 202
    assert payload == {"clip_id": "clip-a", "status": "HELD"}
    assert clip_dir.exists()


def test_worker_http_holds_delete_for_active_derivative_job(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    store_dir = tmp_path / "clip-store"
    migrate_database(database)
    clip_dir = _write_finalized_clip(store_dir, "clip-a")
    _seed_central_evidence(
        database, clip_id="clip-a", incident_id="incident-a", lifecycle_state="COMPLETE"
    )
    _seed_derivative_job(database, "incident-a", state="PENDING")
    service = _control_service(database, store_dir)
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token="relay-token"),
        clip_deletion_control=service,
    )
    server.start()
    try:
        status, payload = _delete_over_http(server.port, "clip-a")
    finally:
        server.stop()

    assert status == 202
    assert payload == {"clip_id": "clip-a", "status": "HELD"}
    assert clip_dir.exists()


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


def test_pending_retention_with_directory_still_present_converges_on_retry(
    tmp_path: Path,
) -> None:
    """Simulates a crash *before* the filesystem delete ran: the DB already
    recorded PENDING (as ``ClipDeletionControlService.delete`` would have, via
    ``begin_clip_purge``), but the clip directory is untouched. A fresh
    ``purge_clip`` call (what the next operator delete request, or this
    worker's own retry, drives) must still converge to PURGED -- retryable,
    not stuck.
    """
    database = tmp_path / "edge.sqlite3"
    store_dir = tmp_path / "clip-store"
    migrate_database(database)
    clip_dir = _write_finalized_clip(store_dir, "clip-a")
    _seed_central_evidence(database, clip_id="clip-a")
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.begin_clip_retention("clip-a", updated_at=NOW)
        assert outbox.clip_retention_state("clip-a") == "PENDING"

    service = _control_service(database, store_dir)
    payload = service.delete("clip-a")

    assert payload == {"clip_id": "clip-a", "status": "PURGED"}
    assert not clip_dir.exists()
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.clip_retention_state("clip-a") == "PURGED"


def test_pending_retention_with_directory_already_removed_converges_on_same_process_http_retry(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    store_dir = tmp_path / "clip-store"
    migrate_database(database)
    clip_dir = _write_finalized_clip(store_dir, "clip-a")
    _seed_central_evidence(database, clip_id="clip-a")
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.begin_clip_retention("clip-a", updated_at=NOW)
    shutil.rmtree(clip_dir)

    service = _control_service(database, store_dir)
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token="relay-token"),
        clip_deletion_control=service,
    )
    server.start()
    try:
        status, payload = _delete_over_http(server.port, "clip-a")
    finally:
        server.stop()

    assert status == 202
    assert payload == {"clip_id": "clip-a", "status": "PURGED"}
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.clip_retention_state("clip-a") == "PURGED"


def test_pending_retention_with_directory_already_removed_converges_on_restart_reconciliation(
    tmp_path: Path,
) -> None:
    """Simulates a crash *after* the filesystem delete ran but before the DB
    transitioned PENDING -> PURGED. This worker never re-attempts the (now
    missing) delete itself -- ``reconcile_event_evidence`` (run once at every
    worker boot, ``EvidenceExportRuntime.initialize_under_lock``) is what
    completes the tombstone.
    """
    database = tmp_path / "edge.sqlite3"
    store_dir = tmp_path / "clip-store"
    migrate_database(database)
    clip_dir = _write_finalized_clip(store_dir, "clip-a")
    _seed_central_evidence(database, clip_id="clip-a")
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.begin_clip_retention("clip-a", updated_at=NOW)
    shutil.rmtree(clip_dir)

    with EvidenceOutbox.open(database) as outbox:
        reconcile_event_evidence(store_dir, outbox)
        state = outbox.clip_retention_state("clip-a")

    assert state == "PURGED"
