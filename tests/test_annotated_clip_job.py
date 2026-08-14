from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from contracts.frame import Frame
from shared.edge_db.migrator import migrate_database
from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    AnnotatedDerivativeLimits,
    BoundedDerivativeQueue,
    DerivativeArtifact,
    DerivativeCancelled,
    DerivativeUnavailableReason,
)
from worker.pipeline.output.evidence.derivative_store import (
    DerivativeCapacityError,
    DerivativeConflictError,
    DerivativeStore,
)
from worker.pipeline.output.evidence.evidence_records import EvidenceRecordStore
from worker.pipeline.output.overlay import OverlayRenderer
from worker.pipeline.output.overlay_scene import AppliedCameraProvenance, OverlaySceneBuilder
from worker.pipeline.trace.models import AnalysisTrace, OptionalNumber
from worker.types import FramePacket


def _scene(width: int, height: int):
    analysis = AnalysisTrace(
        trace_id="a" * 64,
        frame_key=("boot-a", "camera-a", 1, 1),
        pts=OptionalNumber(1.0),
        source_time=OptionalNumber(1.0),
        frame_width=width,
        frame_height=height,
        bed_region_provenance="empty",
        persons=(),
        beds=(),
        components=(),
    )
    return OverlaySceneBuilder().from_traces(
        analysis,
        (),
        provenance=AppliedCameraProvenance("b" * 64, "camera.v1"),
    )


def _packet(width: int, height: int) -> FramePacket:
    return FramePacket(
        camera_id="camera-a",
        frame=Frame(1, 1.0, np.zeros((height, width, 3), dtype=np.uint8)),
        pts=1.0,
        seq=1,
        width=width,
        height=height,
        decode_time_ms=0.0,
        worker_boot_id="boot-a",
        stream_epoch=1,
    )


def test_still_and_video_renderers_consume_the_same_scene_contract() -> None:
    scene = _scene(320, 180)
    renderer = OverlayRenderer()

    still = renderer.render_scene(_packet(320, 180), scene)
    video = renderer.render_scene(_packet(320, 180), scene)

    assert np.array_equal(still, video)
    assert hashlib.sha256(still.tobytes()).digest() == hashlib.sha256(video.tobytes()).digest()


@pytest.mark.parametrize(
    ("incident_id", "primary_sha256"),
    (("../private", "a" * 64), ("incident-a", "A" * 64)),
)
def test_job_rejects_path_like_ids_and_noncanonical_hashes(
    tmp_path: Path, incident_id: str, primary_sha256: str
) -> None:
    with pytest.raises(ValueError, match="identity|SHA-256"):
        AnnotatedDerivativeJob(
            incident_id,
            "clip-a",
            tmp_path / "source.mp4",
            primary_sha256,
            "d" * 64,
            "b" * 64,
            (_scene(16, 16),),
            1,
        )


def test_queue_is_bounded_and_cancellation_is_explicit(tmp_path: Path) -> None:
    queue = BoundedDerivativeQueue(
        AnnotatedDerivativeLimits(
            max_pending_jobs=1,
            max_pending_source_bytes=10,
            max_output_bytes=100,
            max_duration_seconds=1.0,
            max_disk_bytes=100,
        )
    )
    first = AnnotatedDerivativeJob(
        incident_id="incident-a",
        primary_clip_id="clip-a",
        primary_media_path=tmp_path / "a.mp4",
        primary_sha256="a" * 64,
        decision_trace_id="d" * 64,
        runtime_manifest_sha256="b" * 64,
        scenes=(_scene(16, 16),),
        source_size_bytes=9,
    )
    second = AnnotatedDerivativeJob(
        incident_id="incident-b",
        primary_clip_id="clip-b",
        primary_media_path=tmp_path / "b.mp4",
        primary_sha256="c" * 64,
        decision_trace_id="e" * 64,
        runtime_manifest_sha256="b" * 64,
        scenes=(_scene(16, 16),),
        source_size_bytes=2,
    )

    queue.submit(first)
    with pytest.raises(OverflowError, match="bounded"):
        queue.submit(second)
    queue.cancel("incident-a")
    with pytest.raises(DerivativeCancelled):
        queue.take()


def _seed_incident(database: Path, store_root: Path, primary: bytes) -> Path:
    clip_dir = store_root / "clips" / "clip-a"
    clip_dir.mkdir(parents=True)
    primary_path = clip_dir / "clip.mp4"
    primary_path.write_bytes(primary)
    digest = hashlib.sha256(primary).hexdigest()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id, detected_at, payload_json, state, queued_at, next_attempt_at, "
            "delivery_state) VALUES ('event-a','2026-08-13T00:00:00Z','{}','ACKED',1,1,'ACKED')"
        )
        connection.execute(
            "INSERT INTO evidence_clips (clip_id, local_state, state_version, publish_state) "
            "VALUES ('clip-a','VERIFIED',2,'PUBLISHED')"
        )
        connection.execute(
            "INSERT INTO evidence_media_objects "
            "(media_id,content_sha256,size_bytes,mime_type,contained_relpath,basename,created_at) "
            "VALUES ('primary-media', ?, ?, 'video/mp4', 'clips/clip-a/clip.mp4', "
            "'clip.mp4','2026-08-13T00:00:00Z')",
            (digest, len(primary)),
        )
        connection.execute(
            "INSERT INTO evidence_incidents "
            "(incident_id,edge_event_id,camera_id,event_type,detected_at,"
            "provenance_missing_reason,primary_clip_id,lifecycle_state,created_at,updated_at) "
            "VALUES ('incident-a','event-a','camera-a','fall','2026-08-13T00:00:00Z',"
            "'NOT_RECORDED','clip-a','PUBLISHED','2026-08-13T00:00:00Z',"
            "'2026-08-13T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO evidence_artifact_slots "
            "(incident_id,slot_name,state,media_id,created_at,updated_at) VALUES "
            "('incident-a','PRIMARY_CLIP','AVAILABLE','primary-media',"
            "'2026-08-13T00:00:00Z','2026-08-13T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO evidence_primary_clips "
            "(incident_id,clip_id,manifest_relpath,manifest_sha256,manifest_size_bytes,"
            "media_id,source_packet_preserved,source_media_json,truncation_json,created_at) "
            "VALUES ('incident-a','clip-a','clips/clip-a/manifest.json',?,1,'primary-media',"
            "1,'{}','[]','2026-08-13T00:00:00Z')",
            ("c" * 64,),
        )
        connection.commit()
    EvidenceRecordStore(database).request_annotated_derivative(
        "incident-a", expected_revision=1, updated_at="2026-08-13T00:00:01Z"
    )
    return primary_path


def test_artifact_hashing_is_streamed_without_unbounded_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "render.mp4"
    output.write_bytes(b"bounded-content" * 1024)

    def fail_unbounded_read(_path: Path) -> bytes:
        raise AssertionError("read_bytes must not be used for derivative hashing")

    monkeypatch.setattr(Path, "read_bytes", fail_unbounded_read)
    artifact = DerivativeArtifact.from_path(
        output,
        mime_type="video/mp4",
        width=16,
        height=16,
        start_time_ms=0,
        end_time_ms=1000,
        render_backend="opencv-cpu",
        render_version="overlay-cpu.v1",
        scene_id="a" * 64,
    )

    assert artifact.size_bytes == len(b"bounded-content" * 1024)


def test_derivative_publication_enforces_aggregate_disk_bound(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    primary = b"immutable-primary-source-packets"
    primary_path = _seed_incident(database, tmp_path / "clip-store", primary)
    output = tmp_path / "render.mp4"
    output.write_bytes(b"deterministic-annotated-video")
    scene = _scene(320, 180)
    artifact = DerivativeArtifact.from_path(
        output,
        mime_type="video/mp4",
        width=320,
        height=180,
        start_time_ms=0,
        end_time_ms=1000,
        render_backend="opencv-cpu",
        render_version="overlay-cpu.v1",
        scene_id=scene.scene_id,
    )
    job = AnnotatedDerivativeJob(
        "incident-a",
        "clip-a",
        primary_path,
        hashlib.sha256(primary).hexdigest(),
        "d" * 64,
        "b" * 64,
        (scene,),
        len(primary),
    )

    with pytest.raises(DerivativeCapacityError, match="capacity"):
        DerivativeStore(
            database,
            tmp_path / "clip-store",
            max_disk_bytes=artifact.size_bytes - 1,
        ).publish(job, artifact, updated_at="2026-08-13T00:00:02Z")

    assert primary_path.read_bytes() == primary
    assert not tuple((tmp_path / "clip-store" / "derivatives").glob("*/*.mp4"))


def test_derivative_publication_rejects_symlinked_store_path(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    store_root = tmp_path / "clip-store"
    primary = b"immutable-primary-source-packets"
    primary_path = _seed_incident(database, store_root, primary)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, store_root / "derivatives")
    output = tmp_path / "render.mp4"
    output.write_bytes(b"deterministic-annotated-video")
    scene = _scene(320, 180)
    artifact = DerivativeArtifact.from_path(
        output,
        mime_type="video/mp4",
        width=320,
        height=180,
        start_time_ms=0,
        end_time_ms=1000,
        render_backend="opencv-cpu",
        render_version="overlay-cpu.v1",
        scene_id=scene.scene_id,
    )
    job = AnnotatedDerivativeJob(
        "incident-a",
        "clip-a",
        primary_path,
        hashlib.sha256(primary).hexdigest(),
        "d" * 64,
        "b" * 64,
        (scene,),
        len(primary),
    )

    with pytest.raises(DerivativeConflictError, match="escapes"):
        DerivativeStore(database, store_root).publish(
            job, artifact, updated_at="2026-08-13T00:00:02Z"
        )

    assert not tuple(outside.iterdir())
    assert primary_path.read_bytes() == primary


def test_derivative_publication_is_content_addressed_idempotent_and_keeps_primary(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    primary = b"immutable-primary-source-packets"
    primary_path = _seed_incident(database, tmp_path / "clip-store", primary)
    output = tmp_path / "render.mp4"
    output.write_bytes(b"deterministic-annotated-video")
    scene = _scene(320, 180)
    artifact = DerivativeArtifact.from_path(
        output,
        mime_type="video/mp4",
        width=320,
        height=180,
        start_time_ms=0,
        end_time_ms=1000,
        render_backend="opencv-cpu",
        render_version="overlay-cpu.v1",
        scene_id=scene.scene_id,
    )
    store = DerivativeStore(database, tmp_path / "clip-store")
    job = AnnotatedDerivativeJob(
        incident_id="incident-a",
        primary_clip_id="clip-a",
        primary_media_path=primary_path,
        primary_sha256=hashlib.sha256(primary).hexdigest(),
        decision_trace_id="d" * 64,
        runtime_manifest_sha256="b" * 64,
        scenes=(scene,),
        source_size_bytes=len(primary),
    )

    first = store.publish(job, artifact, updated_at="2026-08-13T00:00:02Z")
    second = store.publish(job, artifact, updated_at="2026-08-13T00:00:03Z")

    assert first == second
    assert first.media_relpath.endswith(f"{artifact.sha256}.mp4")
    assert primary_path.read_bytes() == primary
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT state, media.content_sha256, media.size_bytes, media.mime_type, "
            "derivative.width, derivative.height, derivative.render_version, "
            "derivative.scene_id, derivative.primary_clip_id, "
            "derivative.decision_trace_id, derivative.runtime_manifest_sha256 "
            "FROM derivative_evidence_slots AS slot "
            "JOIN evidence_media_objects AS media USING(media_id) "
            "JOIN derivative_render_records AS derivative USING(incident_id, derivative_kind)"
        ).fetchone()
    assert row == (
        "AVAILABLE",
        artifact.sha256,
        artifact.size_bytes,
        "video/mp4",
        320,
        180,
        "overlay-cpu.v1",
        scene.scene_id,
        "clip-a",
        "d" * 64,
        "b" * 64,
    )


def test_restart_reconciles_available_and_corrupt_derivative_states(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    primary = b"immutable-primary-source-packets"
    primary_path = _seed_incident(database, tmp_path / "clip-store", primary)
    output = tmp_path / "render.mp4"
    output.write_bytes(b"deterministic-annotated-video")
    scene = _scene(320, 180)
    artifact = DerivativeArtifact.from_path(
        output,
        mime_type="video/mp4",
        width=320,
        height=180,
        start_time_ms=0,
        end_time_ms=1000,
        render_backend="opencv-cpu",
        render_version="overlay-cpu.v1",
        scene_id=scene.scene_id,
    )
    job = AnnotatedDerivativeJob(
        "incident-a",
        "clip-a",
        primary_path,
        hashlib.sha256(primary).hexdigest(),
        "d" * 64,
        "b" * 64,
        (scene,),
        len(primary),
    )
    store = DerivativeStore(database, tmp_path / "clip-store")
    store.publish(job, artifact, updated_at="2026-08-13T00:00:02Z")
    derivative = next((tmp_path / "clip-store" / "derivatives").glob("*/*.mp4"))

    derivative.write_bytes(b"mutated")
    available, corrupt = store.reconcile(updated_at="2026-08-13T00:00:03Z")

    assert (available, corrupt) == (0, 1)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state, reason FROM derivative_evidence_slots"
        ).fetchone() == ("CORRUPT", "MISSING_OR_MUTATED")


def test_restart_marks_incomplete_record_corrupt_and_quarantines_orphans(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    store_root = tmp_path / "clip-store"
    migrate_database(database)
    primary = b"immutable-primary-source-packets"
    _ = _seed_incident(database, store_root, primary)
    missing_digest = hashlib.sha256(b"missing").hexdigest()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_media_objects "
            "(media_id,content_sha256,size_bytes,mime_type,contained_relpath,basename,"
            "created_at) VALUES "
            "('missing-derivative',?,7,'video/mp4',?,'missing.mp4',"
            "'2026-08-13T00:00:02Z')",
            (missing_digest, f"derivatives/incident-a/{missing_digest}.mp4"),
        )
        connection.execute(
            "INSERT INTO derivative_render_records VALUES "
            "('incident-a','ANNOTATED_CLIP',?,'missing-derivative','clip-a',?,?,?,?,"
            "1,'opencv-cpu','cpu','host','overlay-cpu.v1',320,180,0,1000,"
            "'2026-08-13T00:00:02Z')",
            (
                "1" * 64,
                hashlib.sha256(primary).hexdigest(),
                "d" * 64,
                "b" * 64,
                _scene(320, 180).scene_id,
            ),
        )
        connection.commit()
    orphan = store_root / "derivatives" / "orphan" / "private-camera-name.mp4"
    pending = orphan.with_suffix(".mp4.pending")
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    pending.write_bytes(b"pending")

    available, corrupt = DerivativeStore(database, store_root).reconcile(
        updated_at="2026-08-13T00:00:03Z"
    )

    assert (available, corrupt) == (0, 1)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state, media_id, reason FROM derivative_evidence_slots"
        ).fetchone() == ("CORRUPT", "missing-derivative", "MISSING_OR_MUTATED")
        assert connection.execute("SELECT lifecycle_state FROM evidence_incidents").fetchone() == (
            "COMPLETE",
        )
    quarantined = tuple((store_root / ".derivative-quarantine").iterdir())
    assert len(quarantined) == 2
    assert all("private-camera-name" not in path.name for path in quarantined)
    assert not orphan.exists()
    assert not pending.exists()


def test_failed_derivative_isolated_from_clean_evidence(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    primary = b"immutable-primary-source-packets"
    primary_path = _seed_incident(database, tmp_path / "clip-store", primary)
    store = DerivativeStore(database, tmp_path / "clip-store")

    store.mark_unavailable(
        "incident-a",
        DerivativeUnavailableReason.SOURCE_TRACE_MISSING,
        updated_at="2026-08-13T00:00:02Z",
    )

    assert primary_path.read_bytes() == primary
    record = EvidenceRecordStore(database).get("incident-a")
    assert record is not None
    assert record.primary_state.value == "AVAILABLE"
    assert record.derivative_state is not None
    assert record.derivative_state.value == "UNAVAILABLE"
