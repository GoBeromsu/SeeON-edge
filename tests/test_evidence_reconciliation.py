from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.app.features.evidence.record_store import CentralEvidenceQuery
from shared.edge_db.migrator import migrate_database
from worker.pipeline.output.evidence.evidence_media import MediaFacts
from worker.pipeline.output.evidence.evidence_outbox import (
    ClaimLease,
    EvidenceOutbox,
)
from worker.pipeline.output.evidence.evidence_reconciliation import reconcile_event_evidence
from worker.pipeline.output.evidence.evidence_records import (
    ArtifactState,
    EvidenceLifecycle,
    EvidenceRecordStore,
)
from worker.pipeline.output.evidence.evidence_retention import (
    EvidenceRetention,
    PurgeCandidate,
    PurgeResult,
)
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager

EVENT_ID = "00000000-0000-4000-8000-000000000013"
TRACE_ID = "d" * 64
ANALYSIS_ID = "e" * 64
MANIFEST_ID = "a" * 64
POLICY_ID = "b" * 64
MEDIA_BYTES = b"source-packet-primary"
MEDIA_SHA256 = hashlib.sha256(MEDIA_BYTES).hexdigest()
SNAPSHOT_RELPATH = (
    f"snapshots/{hashlib.sha256(b'camera:opaque').hexdigest()[:16]}/2026-08-13/"
    f"{hashlib.sha256(b'snapshot:opaque').hexdigest()}.jpg"
)
NOW = "2026-08-13T00:00:00Z"


def _seed_provenance(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO runtime_manifest_contents VALUES (?,1,?,?)",
            (MANIFEST_ID, '{"cameras":[{"camera_id":"camera:opaque"}]}', NOW),
        )
        connection.execute(
            "INSERT INTO runtime_manifest_boots VALUES ('boot:one',?,?)",
            (MANIFEST_ID, NOW),
        )
        connection.execute(
            "INSERT INTO runtime_manifest_cameras VALUES ('boot:one','camera:opaque',?,?)",
            (MANIFEST_ID, NOW),
        )
        connection.execute(
            """
            INSERT INTO runtime_analysis_traces (
                trace_id, trace_schema_version, worker_boot_id, camera_id,
                stream_epoch, frame_seq, pts, source_time_sec, frame_width,
                frame_height, bed_region_provenance, storage_bytes
            ) VALUES (?,1,'boot:one','camera:opaque',7,11,42.25,42.25,16,16,'fresh',1)
            """,
            (ANALYSIS_ID,),
        )
        connection.execute(
            """
            INSERT INTO evidence_decision_traces (
                trace_id, trace_schema_version, analysis_trace_id,
                module_qualified_id, policy_qualified_id, effective_policy_id,
                runtime_manifest_sha256, reason, previous_state, current_state,
                triggered, track_id, track_missing_reason, bed_id, bed_missing_reason
            ) VALUES (?,1,?,'fall.v1','fall.policy.v1',?,?,'fall-onset',
                      'clear','triggered',1,NULL,'not-applicable',NULL,'not-applicable')
            """,
            (TRACE_ID, ANALYSIS_ID, POLICY_ID, MANIFEST_ID),
        )
        connection.commit()


def _event(*, snapshot: dict[str, object] | None = None) -> dict[str, object]:
    event: dict[str, object] = {
        "edge_event_id": EVENT_ID,
        "event_type": "fall",
        "probability": 0.95,
        "detected_at": NOW,
        "camera_id": "camera:opaque",
        "evidence": {"domain": "fall"},
        "audit": {
            "runtime_manifest_sha256": MANIFEST_ID,
            "decision_trace_id": TRACE_ID,
        },
    }
    if snapshot is not None:
        event["snapshot"] = snapshot
    return event


def _stage(database: Path, store_dir: Path, *, snapshot: bool = False) -> None:
    snapshot_record = None
    if snapshot:
        content = b"snapshot"
        relpath = SNAPSHOT_RELPATH
        target = store_dir / relpath
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
        snapshot_record = {
            "snapshot_id": "snapshot:opaque",
            "path": relpath,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "mime_type": "image/jpeg",
            "captured_at": NOW,
            "camera_id": "camera:opaque",
            "edge_event_id": EVENT_ID,
        }
    stager = DurableEvidenceStager(
        database,
        "camera:opaque",
        "facility:private",
        None,
        9,
        lambda: 1.0,
        MANIFEST_ID,
    )
    stager.stage(_event(snapshot=snapshot_record))
    stager.complete(EVENT_ID, "clip:opaque")


def _write_final_clip(store_dir: Path, *, media: bytes = MEDIA_BYTES) -> Path:
    clip_dir = store_dir / "clips" / "clip:opaque"
    clip_dir.mkdir(parents=True, exist_ok=True)
    media_path = clip_dir / "clip.mp4"
    media_path.write_bytes(media)
    payload = {
        "manifest_schema_version": 2,
        "state": "READY",
        "clip_id": "clip:opaque",
        "camera_id": "camera:opaque",
        "event_refs": [EVENT_ID],
        "event_ref": EVENT_ID,
        "event_type": "fall",
        "domain": "fall",
        "decision_trace_id": TRACE_ID,
        "clip_start_at": "2026-08-13T00:00:00.000Z",
        "clip_end_at": "2026-08-13T00:00:01.000Z",
        "finalized_at": "2026-08-13T00:00:02.000Z",
        "sha256": MEDIA_SHA256,
        "size_bytes": len(MEDIA_BYTES),
        "mime_type": "video/mp4",
        "codec": "h264",
        "duration_ms": 1000,
        "state_version": 2,
        "runtime_manifest_sha256": MANIFEST_ID,
        "path": "clips/clip:opaque/clip.mp4",
        "source_media": {
            "remux_method": "source-packet-stream-copy",
            "configuration_id": "configuration:opaque",
            "timestamp_translation_seconds": "0/1",
            "streams": [
                {
                    "index": 0,
                    "time_base": "1/1000",
                    "packet_count": 2,
                    "timestamp_translation_ticks": 0,
                }
            ],
        },
        "time_origin": {
            "worker_boot_id": "boot:one",
            "camera_id": "camera:opaque",
            "stream_epoch": 7,
            "generation": 3,
            "media_origin_pts_sec": 40.0,
            "event_pts_sec": 42.25,
            "requested_start_pts_sec": 12.25,
            "requested_end_pts_sec": 72.25,
            "event_media_time_ms": 2250.0,
        },
        "truncation_reasons": ["PRE_ROLL_KEYFRAME_OVERHANG"],
    }
    manifest = clip_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    return manifest


def _migrated(tmp_path: Path) -> tuple[Path, Path]:
    database = tmp_path / "edge.sqlite3"
    store_dir = tmp_path / "clip-store"
    migrate_database(database)
    _seed_provenance(database)
    return database, store_dir


def test_duplicate_event_delivery_reuses_one_incident_and_primary_relation(
    tmp_path: Path,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir)
    _stage(database, store_dir)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM evidence_incidents WHERE edge_event_id = ?",
            (EVENT_ID,),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM clip_events WHERE edge_event_id = ?",
            (EVENT_ID,),
        ).fetchone() == (1,)


def test_restart_reconciliation_publishes_one_authoritative_qualified_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir, snapshot=True)
    manifest = _write_final_clip(store_dir)
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts(MEDIA_SHA256, len(MEDIA_BYTES), 1000),
    )

    with EvidenceOutbox.open(database) as outbox:
        report = reconcile_event_evidence(store_dir, outbox)
        event_claim = outbox.claim(ClaimLease("relay:event", 1.0, 30.0))
        assert event_claim is not None and outbox.acknowledge(
            event_claim, backend_event_id="backend:event"
        )
        clip_claim = outbox.claim_clip(ClaimLease("relay:clip", 1.0, 30.0))
        assert clip_claim is not None and outbox.acknowledge_clip(
            clip_claim, acknowledged_at=2.0, remote_state="READY"
        )
    with EvidenceOutbox.open(database) as outbox:
        restarted = reconcile_event_evidence(store_dir, outbox)

    record = EvidenceRecordStore(database).get(EVENT_ID)
    assert report.verified == 1
    assert restarted.completed == 1
    assert record is not None
    assert record.lifecycle is EvidenceLifecycle.COMPLETE
    assert record.camera_id == "camera:opaque"
    assert record.module_qualified_id == "fall.v1"
    assert record.policy_qualified_id == "fall.policy.v1"
    assert record.runtime_manifest_sha256 == MANIFEST_ID
    assert record.decision_trace_id == TRACE_ID
    assert record.primary is not None
    assert record.primary.media_sha256 == MEDIA_SHA256
    assert record.primary.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert record.primary.source_packet_preserved is True
    assert record.primary.time_origin["event_media_time_ms"] == 2250.0
    assert record.primary.truncation_reasons == ("PRE_ROLL_KEYFRAME_OVERHANG",)
    assert record.snapshot_state is ArtifactState.AVAILABLE
    assert record.event_delivery_state == "ACKED"
    assert record.clip_publish_state == "PUBLISHED"
    assert "facility:private" not in repr(record)
    assert not hasattr(record, "operator_only")
    assert not hasattr(record, "validation_run_id")


@pytest.mark.parametrize(
    ("mutation", "expected_state"),
    (
        (lambda payload: payload.__setitem__("camera_id", "camera:other"), ArtifactState.CORRUPT),
        (lambda payload: payload.pop("runtime_manifest_sha256"), ArtifactState.CORRUPT),
        (
            lambda payload: payload.__setitem__("source_media", {"streams": "invalid"}),
            ArtifactState.CORRUPT,
        ),
        (
            lambda payload: payload.__setitem__("time_origin", {"camera_id": []}),
            ArtifactState.CORRUPT,
        ),
    ),
)
def test_recovery_refuses_unqualified_manifest_provenance_before_immutable_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, object]], object],
    expected_state: ArtifactState,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir)
    manifest = _write_final_clip(store_dir)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutation(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts(MEDIA_SHA256, len(MEDIA_BYTES), 1000),
    )

    with EvidenceOutbox.open(database) as outbox:
        report = reconcile_event_evidence(store_dir, outbox)
        outcome = outbox.clip_outcome("clip:opaque")

    record = EvidenceRecordStore(database).get(EVENT_ID)
    assert report.corrupt == 1
    assert outcome is not None and outcome.local_state.value == "CORRUPT"
    assert record is not None
    assert record.lifecycle is EvidenceLifecycle.FAILED
    assert record.primary_state is expected_state
    assert record.primary is not None
    assert record.primary.media_sha256 is None
    assert record.primary.manifest_sha256 is None


def test_recovery_cross_validates_event_type_domain_and_direct_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir)
    manifest = _write_final_clip(store_dir)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update(
        {
            "event_ref": "00000000-0000-4000-8000-000000000099",
            "event_type": "bed-exit",
            "domain": "bed_exit",
            "decision_trace_id": "f" * 64,
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts(MEDIA_SHA256, len(MEDIA_BYTES), 1000),
    )

    with EvidenceOutbox.open(database) as outbox:
        report = reconcile_event_evidence(store_dir, outbox)

    record = EvidenceRecordStore(database).get(EVENT_ID)
    assert report.corrupt == 1
    assert record is not None and record.lifecycle is EvidenceLifecycle.FAILED
    assert record.primary is not None and record.primary.media_sha256 is None


def test_stream_epoch_mismatch_unavailable_manifest_converges_once(
    tmp_path: Path,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir)
    clip_dir = store_dir / "clips" / "clip:opaque"
    clip_dir.mkdir(parents=True)
    payload = {
        "manifest_schema_version": 2,
        "state": "UNAVAILABLE",
        "clip_id": "clip:opaque",
        "camera_id": "camera:opaque",
        "event_refs": [EVENT_ID],
        "event_ref": EVENT_ID,
        "event_type": "fall",
        "domain": "fall",
        "runtime_manifest_sha256": MANIFEST_ID,
        "decision_trace_id": TRACE_ID,
        "clip_start_at": "2026-08-13T00:00:00Z",
        "clip_end_at": "2026-08-13T00:00:01Z",
        "finalized_at": "2026-08-13T00:00:02Z",
        "state_version": 2,
        "reason_code": "STREAM_EPOCH_MISMATCH",
    }
    (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with EvidenceOutbox.open(database) as outbox:
        first = reconcile_event_evidence(store_dir, outbox)
    with EvidenceOutbox.open(database) as outbox:
        second = reconcile_event_evidence(store_dir, outbox)
        outcome = outbox.clip_outcome("clip:opaque")

    assert first.unavailable == second.unavailable == 1
    assert outcome is not None
    assert outcome.unavailable_reason is not None
    assert outcome.unavailable_reason.value == "STREAM_EPOCH_MISMATCH"
    record = EvidenceRecordStore(database).get(EVENT_ID)
    assert record is not None and record.lifecycle is EvidenceLifecycle.FAILED
    assert record.primary is not None
    assert record.primary.unavailable_reason == "STREAM_EPOCH_MISMATCH"


def test_relay_retry_changes_only_mutable_outbox_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir)
    _write_final_clip(store_dir)
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts(MEDIA_SHA256, len(MEDIA_BYTES), 1000),
    )
    with EvidenceOutbox.open(database) as outbox:
        reconcile_event_evidence(store_dir, outbox)
    before = EvidenceRecordStore(database).get(EVENT_ID)
    assert before is not None

    with EvidenceOutbox.open(database) as outbox:
        claim = outbox.claim(ClaimLease("relay:first", 1.0, 10.0))
        assert claim is not None
        assert outbox.schedule_retry(claim, next_attempt_at=20.0)
        assert outbox.claim(ClaimLease("relay:early", 19.0, 10.0)) is None
        retry = outbox.claim(ClaimLease("relay:retry", 20.0, 10.0))
        assert retry is not None
        assert retry.attempt_count == 2

    after = EvidenceRecordStore(database).get(EVENT_ID)
    assert after is not None
    assert after.primary == before.primary
    assert after.lifecycle == before.lifecycle
    assert after.revision == before.revision
    assert after.event_attempt_count == 2


def test_mutated_published_media_faults_without_rewriting_immutable_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir)
    _write_final_clip(store_dir)
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts(MEDIA_SHA256, len(MEDIA_BYTES), 1000),
    )
    with EvidenceOutbox.open(database) as outbox:
        reconcile_event_evidence(store_dir, outbox)
    before = EvidenceRecordStore(database).get(EVENT_ID)
    assert before is not None and before.primary is not None

    (store_dir / "clips" / "clip:opaque" / "clip.mp4").write_bytes(b"mutated")
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts(hashlib.sha256(b"mutated").hexdigest(), 7, 1000),
    )
    with EvidenceOutbox.open(database) as outbox:
        report = reconcile_event_evidence(store_dir, outbox)

    after = EvidenceRecordStore(database).get(EVENT_ID)
    assert report.corrupt == 1
    assert after is not None and after.primary is not None
    assert after.lifecycle is EvidenceLifecycle.FAILED
    assert after.failure_reason == "CORRUPT"
    assert after.primary.media_sha256 == before.primary.media_sha256
    assert after.primary.manifest_sha256 == before.primary.manifest_sha256
    assert after.primary_state is ArtifactState.CORRUPT


def test_mutated_snapshot_faults_incident_without_rewriting_snapshot_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir, snapshot=True)
    _write_final_clip(store_dir)
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts(MEDIA_SHA256, len(MEDIA_BYTES), 1000),
    )
    with EvidenceOutbox.open(database) as outbox:
        reconcile_event_evidence(store_dir, outbox)
    before = EvidenceRecordStore(database).get(EVENT_ID)
    assert before is not None
    with sqlite3.connect(database) as connection:
        identity_before = connection.execute(
            "SELECT media_id FROM evidence_incident_snapshots WHERE incident_id = ?",
            (EVENT_ID,),
        ).fetchone()

    snapshot_path = store_dir / SNAPSHOT_RELPATH
    snapshot_path.write_bytes(b"mutated-snapshot")
    with EvidenceOutbox.open(database) as outbox:
        report = reconcile_event_evidence(store_dir, outbox)

    after = EvidenceRecordStore(database).get(EVENT_ID)
    with sqlite3.connect(database) as connection:
        identity_after = connection.execute(
            "SELECT media_id FROM evidence_incident_snapshots WHERE incident_id = ?",
            (EVENT_ID,),
        ).fetchone()
    assert report.snapshot_corrupt == 1
    assert after is not None
    assert after.lifecycle is EvidenceLifecycle.FAILED
    assert after.failure_reason == "CORRUPT"
    assert after.snapshot_state is ArtifactState.CORRUPT
    assert identity_after == identity_before


def test_orphan_final_and_temporary_staging_directories_are_quarantined(
    tmp_path: Path,
) -> None:
    database, store_dir = _migrated(tmp_path)
    orphan_final = store_dir / "clips" / "orphan:final"
    orphan_final.mkdir(parents=True)
    (orphan_final / "manifest.json.tmp").write_text("partial")
    orphan_staging = store_dir / "clips" / ".staging" / "orphan:staging"
    orphan_staging.mkdir(parents=True)
    (orphan_staging / "clip.mp4.tmp").write_bytes(b"partial")

    with EvidenceOutbox.open(database) as outbox:
        report = reconcile_event_evidence(store_dir, outbox)

    quarantine = store_dir / "clips" / ".quarantine"
    assert report.quarantined == 2
    assert not orphan_final.exists()
    assert not orphan_staging.exists()
    assert {path.name for path in quarantine.iterdir()} == {
        "final-orphan:final",
        "staging-orphan:staging",
    }


def test_missing_verified_directory_faults_from_database_state_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir)
    _write_final_clip(store_dir)
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts(MEDIA_SHA256, len(MEDIA_BYTES), 1000),
    )
    with EvidenceOutbox.open(database) as outbox:
        reconcile_event_evidence(store_dir, outbox)
    clip_dir = store_dir / "clips" / "clip:opaque"
    for child in clip_dir.iterdir():
        child.unlink()
    clip_dir.rmdir()

    with EvidenceOutbox.open(database) as outbox:
        report = reconcile_event_evidence(store_dir, outbox)

    record = EvidenceRecordStore(database).get(EVENT_ID)
    assert report.corrupt == 1
    assert record is not None
    assert record.lifecycle is EvidenceLifecycle.FAILED
    assert record.failure_reason == "MISSING"
    assert record.primary_state is ArtifactState.CORRUPT


def test_retention_tombstone_atomically_distinguishes_purge_from_missing_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir)
    _write_final_clip(store_dir)
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts(MEDIA_SHA256, len(MEDIA_BYTES), 1000),
    )
    with EvidenceOutbox.open(database) as outbox:
        reconcile_event_evidence(store_dir, outbox)
        event_claim = outbox.claim(ClaimLease("relay:event", 1.0, 30.0))
        assert event_claim is not None and outbox.acknowledge(event_claim)
        clip_claim = outbox.claim_clip(ClaimLease("relay:clip", 1.0, 30.0))
        assert clip_claim is not None and outbox.acknowledge_clip(
            clip_claim, acknowledged_at=2.0, remote_state="READY"
        )
        reconcile_event_evidence(store_dir, outbox)

    def begin(clip_id: str) -> bool:
        with EvidenceOutbox.open(database) as outbox:
            return outbox.begin_clip_retention(clip_id, updated_at="2026-10-13T00:00:00Z")

    def complete(clip_id: str) -> None:
        with EvidenceOutbox.open(database) as outbox:
            outbox.complete_clip_retention(clip_id, updated_at="2026-10-13T00:00:01Z")

    retention = EvidenceRetention(
        store_dir,
        is_held=lambda _clip_id: False,
        disk_usage_provider=lambda _path: shutil.disk_usage(store_dir),
        begin_purge=begin,
        complete_purge=complete,
    )
    clip_dir = store_dir / "clips" / "clip:opaque"
    result = retention.purge(
        PurgeCandidate(
            clip_id="clip:opaque",
            clip_dir=clip_dir,
            finalized_at=datetime(2026, 8, 13, tzinfo=UTC),
        )
    )
    with EvidenceOutbox.open(database) as outbox:
        restarted = reconcile_event_evidence(store_dir, outbox)
        retention_state = outbox.clip_retention_state("clip:opaque")

    record = EvidenceRecordStore(database).get(EVENT_ID)
    summary = CentralEvidenceQuery(database).get(EVENT_ID)
    assert result is PurgeResult.PURGED
    assert not clip_dir.exists()
    assert restarted.corrupt == 0
    assert retention_state == "PURGED"
    assert record is not None and record.lifecycle is EvidenceLifecycle.COMPLETE
    assert record.primary_state is ArtifactState.UNAVAILABLE
    assert record.primary is not None and record.primary.media_sha256 == MEDIA_SHA256
    assert summary is not None and summary.retention_state == "PURGED"


def test_restart_completes_retention_after_crash_between_delete_and_db_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir)
    _write_final_clip(store_dir)
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts(MEDIA_SHA256, len(MEDIA_BYTES), 1000),
    )
    with EvidenceOutbox.open(database) as outbox:
        reconcile_event_evidence(store_dir, outbox)
        event_claim = outbox.claim(ClaimLease("relay:event", 1.0, 30.0))
        assert event_claim is not None and outbox.acknowledge(event_claim)
        clip_claim = outbox.claim_clip(ClaimLease("relay:clip", 1.0, 30.0))
        assert clip_claim is not None and outbox.acknowledge_clip(
            clip_claim, acknowledged_at=2.0, remote_state="READY"
        )
        reconcile_event_evidence(store_dir, outbox)
        assert outbox.begin_clip_retention("clip:opaque", updated_at="2026-10-13T00:00:00Z")

    shutil.rmtree(store_dir / "clips" / "clip:opaque")
    with EvidenceOutbox.open(database) as outbox:
        report = reconcile_event_evidence(store_dir, outbox)
        state = outbox.clip_retention_state("clip:opaque")

    record = EvidenceRecordStore(database).get(EVENT_ID)
    assert report.corrupt == 0
    assert state == "PURGED"
    assert record is not None and record.lifecycle is EvidenceLifecycle.COMPLETE
    assert record.primary_state is ArtifactState.UNAVAILABLE


def test_missing_bound_media_has_explicit_failure_and_never_partial_published_state(
    tmp_path: Path,
) -> None:
    database, store_dir = _migrated(tmp_path)
    _stage(database, store_dir)

    with EvidenceOutbox.open(database) as outbox:
        report = reconcile_event_evidence(store_dir, outbox)

    record = EvidenceRecordStore(database).get(EVENT_ID)
    assert report.unavailable == 1
    assert record is not None
    assert record.lifecycle is EvidenceLifecycle.FAILED
    assert record.failure_reason == "MISSING"
    assert record.primary_state is ArtifactState.UNAVAILABLE
    assert record.primary is not None
    assert record.primary.media_sha256 is None
    assert record.primary.unavailable_reason == "MISSING"
