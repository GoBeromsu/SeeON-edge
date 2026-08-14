from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from contracts.frame import Frame
from shared.edge_db.migrator import migrate_database
from worker.pipeline.output.event_sink import EvidenceEventSink
from worker.pipeline.output.evidence.event_payload import WorkerEventPayload
from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence.snapshot_store import (
    SnapshotCapacityError,
    SnapshotConflictError,
    SnapshotLimits,
    SnapshotStore,
    StoredSnapshot,
)
from worker.types import BusinessEvent, FramePacket

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


@dataclass(slots=True)
class _FailingStager:
    staged: list[WorkerEventPayload] = field(default_factory=list)

    def stage(self, event: WorkerEventPayload) -> None:
        self.staged.append(event)
        raise sqlite3.OperationalError("injected DB crash")

    def attach_snapshot(self, edge_event_id: str, snapshot: dict[str, object]) -> None:
        del edge_event_id, snapshot
        raise AssertionError("attach is unreachable")

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        del edge_event_id, clip_id
        raise AssertionError("complete is unreachable")


@dataclass(slots=True)
class _Recorder:
    calls: int = 0

    def on_event(
        self,
        trigger_packet: FramePacket,
        event: BusinessEvent,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        del trigger_packet, event, allow_new_clip
        self.calls += 1
        return None


def _event(identity: str = "event-1", camera_id: str = "camera-1") -> BusinessEvent:
    return BusinessEvent(
        domain="fall",
        event_type="fall",
        identity=identity,
        camera_id=camera_id,
        facility_id="facility-private",
        time_sec=1.0,
        probability=0.9,
        snapshot_jpeg=b"jpeg-bytes",
    )


def _packet(camera_id: str = "camera-1") -> FramePacket:
    return FramePacket(
        camera_id,
        Frame(1, 1.0, np.zeros((2, 2, 3), dtype=np.uint8)),
        1.0,
        1,
        2,
        2,
        0.1,
    )


def _snapshot_payload(record: StoredSnapshot) -> dict[str, object]:
    return {
        "snapshot_id": record.snapshot_id,
        "path": record.path,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "mime_type": record.mime_type,
        "captured_at": record.captured_at,
        "camera_id": record.camera_id,
        "edge_event_id": record.edge_event_id,
    }


def test_db_crash_leaves_only_resumable_staging_and_no_orphan_final(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    recorder = _Recorder()
    sink = EvidenceEventSink(
        stager=_FailingStager(),
        recorder=recorder,
        snapshot_store=store,
        now=lambda: NOW,
    )
    packet = _packet()

    with pytest.raises(sqlite3.OperationalError, match="injected DB crash"):
        sink.emit_for_frame(_event(), packet)

    assert list((tmp_path / ".snapshot-staging").glob("*.json"))
    assert list((tmp_path / ".snapshot-staging").glob("*.jpg"))
    assert list((tmp_path / "snapshots").rglob("*.jpg")) == []
    assert recorder.calls == 0


def test_central_snapshot_relation_is_pending_until_validated_publication(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    store_root = tmp_path / "store"
    migrate_database(database)
    store = SnapshotStore(store_root)
    staged = store.stage(
        b"jpeg-bytes",
        snapshot_id="event-1",
        captured_at="2026-08-13T12:00:00Z",
        camera_id="camera-1",
        edge_event_id="event-1",
    )
    payload: WorkerEventPayload = {
        "edge_event_id": "event-1",
        "event_type": "fall",
        "probability": 0.9,
        "detected_at": staged.captured_at,
        "camera_id": "camera-1",
        "facility_id": "facility-private",
        "snapshot": _snapshot_payload(staged),
    }
    stager = DurableEvidenceStager(
        database,
        "camera-1",
        "facility-private",
        None,
        1,
        lambda: 1.0,
    )

    stager.stage(payload)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state FROM evidence_artifact_slots "
            "WHERE incident_id = 'event-1' AND slot_name = 'SNAPSHOT'"
        ).fetchone() == ("PENDING",)
        assert connection.execute(
            "SELECT count(*) FROM evidence_incident_snapshots"
        ).fetchone() == (0,)

    store.publish(staged)
    stager.attach_snapshot("event-1", _snapshot_payload(staged))
    store.commit(staged)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state FROM evidence_artifact_slots "
            "WHERE incident_id = 'event-1' AND slot_name = 'SNAPSHOT'"
        ).fetchone() == ("AVAILABLE",)
        assert connection.execute(
            "SELECT snapshot_id FROM evidence_incident_snapshots"
        ).fetchone() == ("event-1",)
    assert (store_root / staged.path).read_bytes() == b"jpeg-bytes"
    assert list((store_root / ".snapshot-staging").glob("*")) == []


def test_snapshot_stage_is_idempotent_and_rejects_identity_rebinding(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    values = {
        "snapshot_id": "event-1",
        "captured_at": "2026-08-13T12:00:00Z",
        "camera_id": "camera-1",
        "edge_event_id": "event-1",
    }

    first = store.stage(b"jpeg-bytes", **values)
    second = store.stage(b"jpeg-bytes", **values)

    assert first == second
    assert len(list((tmp_path / ".snapshot-staging").glob("*.jpg"))) == 1
    with pytest.raises(SnapshotConflictError, match="conflicts"):
        store.stage(b"other", **values)


def test_snapshot_stage_backpressure_is_global_and_per_camera_observable(
    tmp_path: Path,
) -> None:
    store = SnapshotStore(
        tmp_path,
        limits=SnapshotLimits(
            max_pending_global=2,
            max_pending_per_camera=1,
            max_files_global=10,
            max_files_per_camera=10,
            max_bytes_global=1024,
            max_bytes_per_camera=1024,
            max_age=timedelta(days=60),
            max_pending_age=timedelta(days=1),
        ),
    )
    store.stage(
        b"one",
        snapshot_id="event-1",
        captured_at="2026-08-13T12:00:00Z",
        camera_id="camera-1",
        edge_event_id="event-1",
    )

    with pytest.raises(SnapshotCapacityError, match="per-camera pending"):
        store.stage(
            b"two",
            snapshot_id="event-2",
            captured_at="2026-08-13T12:00:01Z",
            camera_id="camera-1",
            edge_event_id="event-2",
        )

    assert store.stats.dropped_capacity == 1
    assert store.stats.pending_files == 1
    assert store.stats.pending_bytes == 3


def test_restart_reconcile_attaches_staged_snapshot_after_full_hash_validation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    store_root = tmp_path / "store"
    migrate_database(database)
    store = SnapshotStore(store_root)
    staged = store.stage(
        b"snapshot",
        snapshot_id="event-1",
        captured_at="2026-08-13T12:00:00Z",
        camera_id="camera-1",
        edge_event_id="event-1",
    )
    stager = DurableEvidenceStager(database, "camera-1", "facility-private", None, 1, lambda: 1.0)
    stager.stage(
        {
            "edge_event_id": "event-1",
            "event_type": "fall",
            "detected_at": staged.captured_at,
            "camera_id": "camera-1",
            "facility_id": "facility-private",
            "snapshot": _snapshot_payload(staged),
        }
    )

    with EvidenceOutbox.open(database) as outbox:
        report = outbox.reconcile_snapshots(store, now=NOW)

    assert report.attached == 1
    assert (store_root / staged.path).read_bytes() == b"snapshot"
    assert list((store_root / ".snapshot-staging").glob("*")) == []


@pytest.mark.parametrize("replacement", [None, b"mutated"])
def test_missing_or_mutated_snapshot_updates_state_without_rewriting_identity(
    tmp_path: Path,
    replacement: bytes | None,
) -> None:
    database = tmp_path / "edge.sqlite3"
    store_root = tmp_path / "store"
    migrate_database(database)
    store = SnapshotStore(store_root)
    staged = store.stage(
        b"snapshot",
        snapshot_id="event-1",
        captured_at="2026-08-13T12:00:00Z",
        camera_id="camera-1",
        edge_event_id="event-1",
    )
    stager = DurableEvidenceStager(database, "camera-1", "facility-private", None, 1, lambda: 1.0)
    payload = {
        "edge_event_id": "event-1",
        "event_type": "fall",
        "detected_at": staged.captured_at,
        "camera_id": "camera-1",
        "facility_id": "facility-private",
        "snapshot": _snapshot_payload(staged),
    }
    stager.stage(payload)
    store.publish(staged)
    stager.attach_snapshot("event-1", _snapshot_payload(staged))
    store.commit(staged)
    snapshot_path = store_root / staged.path
    if replacement is None:
        snapshot_path.unlink()
    else:
        snapshot_path.write_bytes(replacement)

    with EvidenceOutbox.open(database) as outbox:
        report = outbox.reconcile_snapshots(store, now=NOW)
    with sqlite3.connect(database) as connection:
        slot = connection.execute(
            "SELECT state, reason FROM evidence_artifact_slots "
            "WHERE incident_id = 'event-1' AND slot_name = 'SNAPSHOT'"
        ).fetchone()
        identity = connection.execute(
            "SELECT snapshot_id, media_id FROM evidence_incident_snapshots"
        ).fetchone()

    assert report.corrupt == 1
    assert slot == ("CORRUPT", "MISSING_OR_MUTATED")
    assert identity == ("event-1", f"sha256:{hashlib.sha256(b'snapshot').hexdigest()}:8")


def _seed_committed_snapshot(
    connection: sqlite3.Connection,
    store: SnapshotStore,
    *,
    ordinal: int,
    captured_at: str,
    held: bool = False,
    reviewed: bool = False,
) -> StoredSnapshot:
    event_id = f"retention-event-{ordinal}"
    clip_id = f"retention-clip-{ordinal}"
    camera_id = f"camera-{ordinal % 2}"
    content = f"snapshot-{ordinal}".encode()
    record = store.store(
        content,
        snapshot_id=event_id,
        captured_at=captured_at,
        camera_id=camera_id,
        edge_event_id=event_id,
    )
    media_id = f"sha256:{record.sha256}:{record.size_bytes}"
    delivery_state = "PENDING" if held else "ACKED"
    connection.execute(
        """
        INSERT INTO evidence_events (
            edge_event_id, detected_at, payload_json, state, queued_at,
            next_attempt_at, delivery_state
        ) VALUES (?, ?, '{}', 'READY', 1, 1, ?)
        """,
        (event_id, captured_at, delivery_state),
    )
    connection.execute(
        """
        INSERT INTO evidence_clips (
            clip_id, local_state, state_version, publish_state
        ) VALUES (?, 'VERIFIED', 1, 'PUBLISHED')
        """,
        (clip_id,),
    )
    connection.execute(
        """
        INSERT INTO evidence_incidents (
            incident_id, edge_event_id, camera_id, event_type, detected_at,
            provenance_missing_reason, primary_clip_id, lifecycle_state,
            created_at, updated_at
        ) VALUES (?, ?, ?, 'fall', ?, 'NOT_RECORDED', ?, 'COMPLETE', ?, ?)
        """,
        (event_id, event_id, camera_id, captured_at, clip_id, captured_at, captured_at),
    )
    connection.execute(
        """
        INSERT INTO evidence_primary_clips (
            incident_id, clip_id, source_packet_preserved, source_missing_reason,
            truncation_json, unavailable_reason, created_at
        ) VALUES (?, ?, 0, 'NOT_RECORDED', '[]', 'MISSING', ?)
        """,
        (event_id, clip_id, captured_at),
    )
    connection.execute(
        """
        INSERT INTO evidence_media_objects (
            media_id, content_sha256, size_bytes, mime_type,
            contained_relpath, basename, created_at
        ) VALUES (?, ?, ?, 'image/jpeg', ?, ?, ?)
        """,
        (
            media_id,
            record.sha256,
            record.size_bytes,
            record.path,
            Path(record.path).name,
            captured_at,
        ),
    )
    connection.execute(
        "INSERT INTO evidence_incident_snapshots VALUES (?, ?, ?, ?, ?, ?)",
        (event_id, event_id, media_id, captured_at, camera_id, captured_at),
    )
    connection.execute(
        """
        INSERT INTO evidence_artifact_slots (
            incident_id, slot_name, state, media_id, created_at, updated_at
        ) VALUES (?, 'SNAPSHOT', 'AVAILABLE', ?, ?, ?)
        """,
        (event_id, media_id, captured_at, captured_at),
    )
    if reviewed:
        review_id = f"review-{ordinal}"
        connection.execute(
            """
            INSERT INTO control_evidence_review_revisions (
                review_id, incident_id, clip_id, review_version, actor_id,
                reviewed_at, disposition
            ) VALUES (?, ?, ?, 1, 'operator', ?, 'TRUE_POSITIVE')
            """,
            (review_id, event_id, clip_id, captured_at),
        )
        connection.execute(
            "INSERT INTO control_evidence_review_state VALUES (?, ?, 1)",
            (event_id, clip_id),
        )
    return record


def test_retention_purges_one_hundred_old_snapshots_but_preserves_held_and_reviewed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    store = SnapshotStore(root)
    old = (NOW - timedelta(days=61)).isoformat().replace("+00:00", "Z")
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        purged = [
            _seed_committed_snapshot(
                connection,
                store,
                ordinal=index,
                captured_at=old,
            )
            for index in range(100)
        ]
        held = _seed_committed_snapshot(
            connection,
            store,
            ordinal=100,
            captured_at=old,
            held=True,
        )
        reviewed = _seed_committed_snapshot(
            connection,
            store,
            ordinal=101,
            captured_at=old,
            reviewed=True,
        )
        connection.commit()

    with EvidenceOutbox.open(database) as outbox:
        report = outbox.reconcile_snapshots(store, now=NOW)

    assert report.purged == 100
    assert report.held == 2
    assert all(not (root / record.path).exists() for record in purged)
    assert (root / held.path).exists()
    assert (root / reviewed.path).exists()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM evidence_artifact_slots "
            "WHERE state = 'UNAVAILABLE' AND reason = 'RETENTION_PURGED'"
        ).fetchone() == (100,)
        assert connection.execute(
            "SELECT count(*) FROM evidence_incident_snapshots"
        ).fetchone() == (102,)


def test_restart_completes_snapshot_retention_after_delete_before_db_commit(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    store = SnapshotStore(root)
    old = (NOW - timedelta(days=61)).isoformat().replace("+00:00", "Z")
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        record = _seed_committed_snapshot(
            connection,
            store,
            ordinal=200,
            captured_at=old,
        )
        connection.commit()

    store.stage_retention(record)
    store.remove_committed(record)
    with EvidenceOutbox.open(database) as outbox:
        report = outbox.reconcile_snapshots(store, now=NOW)

    assert report.purged == 1
    assert report.corrupt == 0
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state, reason FROM evidence_artifact_slots "
            "WHERE incident_id = 'retention-event-200' AND slot_name = 'SNAPSHOT'"
        ).fetchone() == ("UNAVAILABLE", "RETENTION_PURGED")
    assert store.retention_records() == ()


def test_reconcile_purges_one_hundred_old_unreferenced_staged_snapshots(
    tmp_path: Path,
) -> None:
    store = SnapshotStore(tmp_path)
    old = NOW - timedelta(days=2)
    for index in range(100):
        captured_at = (old + timedelta(seconds=index)).isoformat().replace("+00:00", "Z")
        store.stage(
            f"snapshot-{index}".encode(),
            snapshot_id=f"event-{index}",
            captured_at=captured_at,
            camera_id=f"camera-{index % 2}",
            edge_event_id=f"event-{index}",
        )

    report = store.discard_unreferenced_staging(set(), now=NOW)

    assert report.discarded == 100
    assert list((tmp_path / ".snapshot-staging").glob("*")) == []
    assert store.stats.discarded_unreferenced == 100
