from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.features.clips.artifacts import CentralClipArtifactQuery
from backend.app.main import create_app, no_lifespan
from shared.edge_db.migrator import migrate_database
from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    AnnotatedDerivativeLimits,
    CpuAnnotatedStillRenderer,
    DerivativeArtifact,
    DerivativeCancelled,
    DerivativeKind,
)
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.mjpeg_server import MjpegServer, MjpegServerConfig
from worker.runtime.derivative_runtime import (
    DerivativeControlService,
    DerivativeProductionRuntime,
)

NOW = "2026-08-13T00:00:00Z"
MANIFEST = "a" * 64
ANALYSIS = "b" * 64
DECISION = "c" * 64
POLICY = "d" * 64


def _seed(
    database: Path,
    root: Path,
    ordinal: int = 0,
    *,
    primary_content: bytes | None = None,
) -> tuple[str, str, Path]:
    suffix = chr(ord("a") + ordinal)
    incident_id = f"incident-{suffix}"
    clip_id = f"clip-{suffix}"
    event_id = f"event-{suffix}"
    camera_id = f"camera-{suffix}"
    analysis_id = hashlib.sha256(f"analysis-{suffix}".encode()).hexdigest()
    decision_id = hashlib.sha256(f"decision-{suffix}".encode()).hexdigest()
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    source = clip_dir / "clip.mp4"
    content = primary_content or f"primary-{suffix}".encode()
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO runtime_manifest_contents VALUES (?,1,'{}',?)",
            (MANIFEST, NOW),
        )
        connection.execute(
            "INSERT OR IGNORE INTO runtime_manifest_boots VALUES ('boot-a',?,?)",
            (MANIFEST, NOW),
        )
        connection.execute(
            "INSERT INTO runtime_manifest_cameras VALUES ('boot-a',?,?,?)",
            (camera_id, MANIFEST, NOW),
        )
        connection.execute(
            "INSERT INTO runtime_analysis_traces "
            "(trace_id,trace_schema_version,worker_boot_id,camera_id,stream_epoch,"
            "frame_seq,pts,source_time_sec,frame_width,frame_height,"
            "bed_region_provenance,storage_bytes) "
            "VALUES (?,1,'boot-a',?,1,1,0,0,16,16,'fresh',1)",
            (analysis_id, camera_id),
        )
        connection.execute(
            "INSERT INTO evidence_decision_traces "
            "(trace_id,trace_schema_version,analysis_trace_id,module_qualified_id,"
            "policy_qualified_id,effective_policy_id,runtime_manifest_sha256,reason,"
            "previous_state,current_state,triggered,track_missing_reason,bed_missing_reason) "
            "VALUES (?,1,?,'fall.v1','fall.policy.v1',?,?,'fall-onset','clear','fall',1,"
            "'not-applicable','not-applicable')",
            (decision_id, analysis_id, POLICY, MANIFEST),
        )
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at,"
            "delivery_state) VALUES (?,?,'{}','ACKED',1,1,'ACKED')",
            (event_id, NOW),
        )
        connection.execute(
            "INSERT INTO evidence_clips (clip_id,local_state,state_version,publish_state) "
            "VALUES (?,'VERIFIED',2,'PUBLISHED')",
            (clip_id,),
        )
        connection.execute(
            "INSERT INTO evidence_media_objects "
            "(media_id,content_sha256,size_bytes,mime_type,contained_relpath,basename,"
            "created_at) VALUES (?,?,?,'video/mp4',?,'clip.mp4',?)",
            (f"primary-{suffix}", digest, len(content), f"clips/{clip_id}/clip.mp4", NOW),
        )
        connection.execute(
            "INSERT INTO evidence_incidents "
            "(incident_id,edge_event_id,camera_id,event_type,detected_at,"
            "runtime_manifest_sha256,decision_trace_id,module_qualified_id,"
            "policy_qualified_id,effective_policy_id,provenance_state,primary_clip_id,"
            "lifecycle_state,created_at,updated_at) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,'QUALIFIED',?,'COMPLETE',?,?)",
            (
                incident_id,
                event_id,
                camera_id,
                "fall",
                NOW,
                MANIFEST,
                decision_id,
                "fall.v1",
                "fall.policy.v1",
                POLICY,
                clip_id,
                NOW,
                NOW,
            ),
        )
        connection.execute(
            "INSERT INTO evidence_artifact_slots "
            "(incident_id,slot_name,state,media_id,created_at,updated_at) "
            "VALUES (?,'PRIMARY_CLIP','AVAILABLE',?,?,?)",
            (incident_id, f"primary-{suffix}", NOW, NOW),
        )
        connection.execute(
            "INSERT INTO evidence_primary_clips "
            "(incident_id,clip_id,manifest_relpath,manifest_sha256,manifest_size_bytes,"
            "media_id,source_packet_preserved,source_media_json,time_origin_json,"
            "truncation_json,created_at) VALUES (?,?,?,?,1,?,1,'{}',?,'[]',?)",
            (
                incident_id,
                clip_id,
                f"clips/{clip_id}/manifest.json",
                "e" * 64,
                f"primary-{suffix}",
                '{"media_origin_pts_sec":0.0}',
                NOW,
            ),
        )
        connection.commit()
    return incident_id, clip_id, source


@dataclass
class _Renderer:
    payload: bytes
    started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event | None = None
    wait_for_cancel: bool = False
    calls: int = 0

    def render(
        self,
        job: AnnotatedDerivativeJob,
        destination: Path,
        *,
        cancelled: threading.Event | None = None,
    ) -> DerivativeArtifact:
        self.calls += 1
        self.started.set()
        if self.wait_for_cancel:
            assert cancelled is not None
            if cancelled.wait(2.0):
                raise DerivativeCancelled("cancelled", job)
            raise AssertionError("cancellation was not delivered")
        if self.release is not None:
            assert self.release.wait(2.0)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.payload)
        return DerivativeArtifact.from_path(
            destination,
            mime_type=job.derivative_kind.mime_type,
            width=16,
            height=16,
            start_time_ms=0,
            end_time_ms=0 if job.derivative_kind is DerivativeKind.STILL else 1000,
            render_backend="cpu-test",
            render_version="overlay-cpu.v1",
            scene_id=job.scenes[0].scene_id,
        )


def _runtime(
    database: Path,
    root: Path,
    still: _Renderer,
    video: _Renderer,
    *,
    max_pending_jobs: int = 8,
) -> DerivativeProductionRuntime:
    return DerivativeProductionRuntime(
        database,
        root,
        limits=AnnotatedDerivativeLimits(
            max_pending_jobs=max_pending_jobs,
            max_pending_source_bytes=1024,
            max_output_bytes=1024,
            max_disk_bytes=4096,
        ),
        still_renderer=still,
        video_renderer=video,
        clock=lambda: NOW,
    )


def test_production_request_schedules_still_and_video_and_exposes_query(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, primary = _seed(database, root)
    runtime = _runtime(database, root, _Renderer(b"jpeg"), _Renderer(b"video"))
    runtime.start()
    try:
        runtime.request(clip, DerivativeKind.STILL)
        runtime.request(clip, DerivativeKind.VIDEO)
        still = runtime.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0)
        video = runtime.wait_for_terminal(incident, DerivativeKind.VIDEO, timeout=2.0)
    finally:
        runtime.stop()

    assert still is not None and still.state.value == "AVAILABLE"
    assert video is not None and video.state.value == "AVAILABLE"
    projection = CentralClipArtifactQuery(database).get(clip)
    assert projection is not None and projection.still is not None and projection.video is not None
    assert projection.still.mime_type == "image/jpeg"
    assert projection.still.relpath is not None and projection.still.relpath.endswith(".jpg")
    assert projection.video.mime_type == "video/mp4"
    assert projection.video.relpath is not None and projection.video.relpath.endswith(".mp4")
    assert projection.still.primary_clip_id == clip
    assert projection.video.runtime_manifest_sha256 == MANIFEST
    assert primary.read_bytes() == b"primary-a"


def test_idempotent_request_renders_once(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    release = threading.Event()
    renderer = _Renderer(b"jpeg", release=release)
    runtime = _runtime(database, root, renderer, _Renderer(b"video"))
    runtime.start()
    try:
        first = runtime.request(clip, DerivativeKind.STILL)
        assert renderer.started.wait(2.0)
        second = runtime.request(clip, DerivativeKind.STILL)
        release.set()
        terminal = runtime.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0)
    finally:
        runtime.stop()

    assert first.request_id == second.request_id
    assert terminal is not None and terminal.state.value == "AVAILABLE"
    assert renderer.calls == 1


def test_running_cancel_is_durable_and_explicit(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    renderer = _Renderer(b"unused", wait_for_cancel=True)
    runtime = _runtime(database, root, renderer, _Renderer(b"video"))
    runtime.start()
    try:
        runtime.request(clip, DerivativeKind.STILL)
        assert renderer.started.wait(2.0)
        assert runtime.cancel(clip, DerivativeKind.STILL) is not None
        terminal = runtime.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0)
        duplicate = runtime.cancel(clip, DerivativeKind.STILL)
    finally:
        runtime.stop()

    assert terminal is not None
    assert (terminal.state.value, terminal.reason) == ("CANCELLED", "CANCELLED")
    assert duplicate is not None
    assert (duplicate.state.value, duplicate.reason) == ("CANCELLED", "CANCELLED")


def test_accepted_cancel_wins_over_publication_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If cancel is durable before publication commits, AVAILABLE must not land."""
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    entered = threading.Event()
    release = threading.Event()
    runtime = _runtime(database, root, _Renderer(b"race-jpeg"), _Renderer(b"video"))
    original_record = runtime.artifacts._record

    def _gated_record(
        job: AnnotatedDerivativeJob,
        artifact: DerivativeArtifact,
        relative: str,
        updated_at: str,
    ) -> str:
        entered.set()
        assert release.wait(2.0), "publication was not released"
        return original_record(job, artifact, relative, updated_at)

    monkeypatch.setattr(runtime.artifacts, "_record", _gated_record)
    runtime.start()
    try:
        runtime.request(clip, DerivativeKind.STILL)
        assert entered.wait(2.0), "publication never reached the commit gate"
        accepted = runtime.cancel(clip, DerivativeKind.STILL)
        assert accepted is not None
        assert accepted.state.value in {"RUNNING", "PENDING", "CANCELLED"}
        record = runtime.jobs.get(incident, DerivativeKind.STILL)
        assert record is not None and record.cancel_requested
        release.set()
        terminal = runtime.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0)
    finally:
        release.set()
        runtime.stop()

    assert terminal is not None
    assert (terminal.state.value, terminal.reason) == ("CANCELLED", "CANCELLED")
    final = runtime.jobs.get(incident, DerivativeKind.STILL)
    assert final is not None
    assert final.cancel_requested
    assert final.media_id is None
    with sqlite3.connect(database) as connection:
        artifact_rows = connection.execute(
            "SELECT 1 FROM derivative_artifacts WHERE incident_id=? AND derivative_kind='STILL'",
            (incident,),
        ).fetchall()
        media_rows = connection.execute(
            "SELECT contained_relpath FROM evidence_media_objects "
            "WHERE contained_relpath LIKE 'derivatives/%'"
        ).fetchall()
    assert artifact_rows == []
    assert media_rows == []
    objects = root / "derivatives" / "objects"
    live = tuple(objects.glob("*")) if objects.exists() else ()
    assert live == ()
    quarantined = tuple((root / ".derivative-quarantine").glob("*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"race-jpeg"


def test_publication_commit_rejects_later_cancel_with_terminal_status(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    runtime = _runtime(database, root, _Renderer(b"published"), _Renderer(b"video"))
    runtime.start()
    try:
        runtime.request(clip, DerivativeKind.STILL)
        terminal = runtime.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0)
        conflict = runtime.cancel(clip, DerivativeKind.STILL)
    finally:
        runtime.stop()

    assert terminal is not None and terminal.state.value == "AVAILABLE"
    assert conflict is not None
    assert conflict.state.value == "AVAILABLE"
    assert conflict.reason is None
    final = runtime.jobs.get(incident, DerivativeKind.STILL)
    assert final is not None
    assert final.state.value == "AVAILABLE"
    assert not final.cancel_requested
    assert final.media_id is not None
    object_files = tuple((root / "derivatives" / "objects").glob("*.jpg"))
    assert len(object_files) == 1
    assert object_files[0].read_bytes() == b"published"


def test_cancel_during_publish_preserves_shared_content_addressed_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    first_incident, first_clip, _ = _seed(database, root, 0)
    second_incident, second_clip, _ = _seed(database, root, 1)
    shared_payload = b"shared-cas-bytes"
    still_renderer = _Renderer(shared_payload)
    runtime = _runtime(
        database,
        root,
        still_renderer,
        _Renderer(b"video"),
        max_pending_jobs=8,
    )
    entered = threading.Event()
    release = threading.Event()
    shared_path = root / "derivatives" / "objects" / "placeholder.jpg"
    runtime.start()
    try:
        runtime.request(first_clip, DerivativeKind.STILL)
        first_terminal = runtime.wait_for_terminal(
            first_incident, DerivativeKind.STILL, timeout=2.0
        )
        assert first_terminal is not None and first_terminal.state.value == "AVAILABLE"
        shared_files = tuple((root / "derivatives" / "objects").glob("*.jpg"))
        assert len(shared_files) == 1
        shared_path = shared_files[0]
        shared_bytes = shared_path.read_bytes()

        original_record = runtime.artifacts._record

        def _gated_record(
            job: AnnotatedDerivativeJob,
            artifact: DerivativeArtifact,
            relative: str,
            updated_at: str,
        ) -> str:
            entered.set()
            assert release.wait(2.0), "second publication was not released"
            return original_record(job, artifact, relative, updated_at)

        monkeypatch.setattr(runtime.artifacts, "_record", _gated_record)
        runtime.request(second_clip, DerivativeKind.STILL)
        assert entered.wait(2.0), "second publication never reached commit gate"
        assert runtime.cancel(second_clip, DerivativeKind.STILL) is not None
        release.set()
        second_terminal = runtime.wait_for_terminal(
            second_incident, DerivativeKind.STILL, timeout=2.0
        )
    finally:
        release.set()
        runtime.stop()

    assert second_terminal is not None
    assert (second_terminal.state.value, second_terminal.reason) == ("CANCELLED", "CANCELLED")
    assert shared_path.exists()
    assert shared_path.read_bytes() == shared_bytes
    first = runtime.jobs.get(first_incident, DerivativeKind.STILL)
    assert first is not None and first.state.value == "AVAILABLE"
    with sqlite3.connect(database) as connection:
        first_media = connection.execute(
            "SELECT media_id FROM derivative_jobs WHERE incident_id=?",
            (first_incident,),
        ).fetchone()
        second_artifact = connection.execute(
            "SELECT 1 FROM derivative_artifacts WHERE incident_id=?",
            (second_incident,),
        ).fetchone()
        media_paths = {
            str(row[0])
            for row in connection.execute(
                "SELECT contained_relpath FROM evidence_media_objects "
                "WHERE contained_relpath LIKE 'derivatives/%'"
            ).fetchall()
        }
    assert first_media is not None and first_media[0] is not None
    assert second_artifact is None
    assert media_paths == {shared_path.relative_to(root).as_posix()}
    assert not (root / ".derivative-quarantine").exists() or not tuple(
        (root / ".derivative-quarantine").iterdir()
    )


def test_restart_honors_cancel_requested_and_does_not_publish(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    blocked = _Renderer(b"unused", wait_for_cancel=True)
    first = _runtime(database, root, blocked, _Renderer(b"video"))
    first.start()
    first.request(clip, DerivativeKind.STILL)
    assert blocked.started.wait(2.0)
    first.stop()
    mid = first.jobs.get(incident, DerivativeKind.STILL)
    assert mid is not None and mid.state.value == "PENDING"
    # Durable cancel accepted while the supervisor was down.
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE derivative_jobs SET cancel_requested=1,revision=revision+1,"
            "updated_at=? WHERE incident_id=? AND derivative_kind='STILL'",
            (NOW, incident),
        )
        connection.commit()

    recovered = _runtime(database, root, _Renderer(b"must-not-publish"), _Renderer(b"video"))
    recovered.start()
    try:
        terminal = recovered.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0)
    finally:
        recovered.stop()

    assert terminal is not None
    assert (terminal.state.value, terminal.reason) == ("CANCELLED", "CANCELLED")
    assert isinstance(recovered.still_renderer, _Renderer)
    assert recovered.still_renderer.calls == 0
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM derivative_artifacts WHERE incident_id=?",
                (incident,),
            ).fetchone()
            is None
        )
    objects = root / "derivatives" / "objects"
    assert not objects.exists() or tuple(objects.glob("*")) == ()


def test_stop_preserves_running_work_for_restart_recovery(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    interrupted = _Renderer(b"unused", wait_for_cancel=True)
    first = _runtime(database, root, interrupted, _Renderer(b"video"))
    first.start()
    first.request(clip, DerivativeKind.STILL)
    assert interrupted.started.wait(2.0)
    first.stop()

    recovered = _runtime(database, root, _Renderer(b"recovered"), _Renderer(b"video"))
    recovered.start()
    try:
        terminal = recovered.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0)
    finally:
        recovered.stop()
    assert terminal is not None and terminal.state.value == "AVAILABLE"
    assert terminal.attempt_count == 2


def test_cancel_before_stop_preserves_cancelled_terminal(tmp_path: Path) -> None:
    """Accepted cancel must stay CANCELLED when stop interrupts the same run."""
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    renderer = _Renderer(b"unused", wait_for_cancel=True)
    runtime = _runtime(database, root, renderer, _Renderer(b"video"))
    runtime.start()
    runtime.request(clip, DerivativeKind.STILL)
    assert renderer.started.wait(2.0)
    accepted = runtime.cancel(clip, DerivativeKind.STILL)
    assert accepted is not None
    mid = runtime.jobs.get(incident, DerivativeKind.STILL)
    assert mid is not None and mid.cancel_requested
    runtime.stop()

    final = runtime.jobs.get(incident, DerivativeKind.STILL)
    assert final is not None
    assert (final.state.value, final.reason) == ("CANCELLED", "CANCELLED")
    assert final.cancel_requested

    recovered = _runtime(database, root, _Renderer(b"must-not-run"), _Renderer(b"video"))
    recovered.start()
    try:
        terminal = recovered.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0)
    finally:
        recovered.stop()
    assert terminal is not None
    assert (terminal.state.value, terminal.reason) == ("CANCELLED", "CANCELLED")
    assert isinstance(recovered.still_renderer, _Renderer)
    assert recovered.still_renderer.calls == 0


def test_stop_before_cancel_race_prefers_durable_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If stop reaches interrupt first, a concurrent accepted cancel still wins."""
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    renderer = _Renderer(b"unused", wait_for_cancel=True)
    runtime = _runtime(database, root, renderer, _Renderer(b"video"))
    interrupt_entered = threading.Event()
    release_interrupt = threading.Event()
    original_interrupt = runtime.jobs.mark_interrupted

    def _gated_interrupt(
        job: AnnotatedDerivativeJob, *, updated_at: str
    ) -> bool:
        interrupt_entered.set()
        assert release_interrupt.wait(2.0), "interrupt gate was not released"
        return original_interrupt(job, updated_at=updated_at)

    monkeypatch.setattr(runtime.jobs, "mark_interrupted", _gated_interrupt)
    runtime.start()
    runtime.request(clip, DerivativeKind.STILL)
    assert renderer.started.wait(2.0)

    stop_error: list[BaseException] = []

    def _stop() -> None:
        try:
            runtime.stop()
        except BaseException as error:  # noqa: BLE001 - surface in parent
            stop_error.append(error)

    stopper = threading.Thread(target=_stop, name="derivative-stop", daemon=True)
    stopper.start()
    assert interrupt_entered.wait(2.0), "stop never reached mark_interrupted"
    # Stop already observed no cancel; operator cancel lands before interrupt commits.
    accepted = runtime.cancel(clip, DerivativeKind.STILL)
    assert accepted is not None
    gated = runtime.jobs.get(incident, DerivativeKind.STILL)
    assert gated is not None and gated.cancel_requested
    assert gated.state.value == "RUNNING"
    release_interrupt.set()
    stopper.join(timeout=2.0)
    assert not stopper.is_alive()
    assert stop_error == []

    final = runtime.jobs.get(incident, DerivativeKind.STILL)
    assert final is not None
    assert (final.state.value, final.reason) == ("CANCELLED", "CANCELLED")
    assert final.cancel_requested

    recovered = _runtime(database, root, _Renderer(b"must-not-run"), _Renderer(b"video"))
    recovered.start()
    try:
        terminal = recovered.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0)
    finally:
        recovered.stop()
    assert terminal is not None
    assert (terminal.state.value, terminal.reason) == ("CANCELLED", "CANCELLED")
    assert isinstance(recovered.still_renderer, _Renderer)
    assert recovered.still_renderer.calls == 0


def test_stop_without_cancel_still_interrupts_to_pending(tmp_path: Path) -> None:
    """Uncancelled stop interruption must remain restartable PENDING work."""
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    renderer = _Renderer(b"unused", wait_for_cancel=True)
    runtime = _runtime(database, root, renderer, _Renderer(b"video"))
    runtime.start()
    runtime.request(clip, DerivativeKind.STILL)
    assert renderer.started.wait(2.0)
    runtime.stop()

    mid = runtime.jobs.get(incident, DerivativeKind.STILL)
    assert mid is not None
    assert mid.state.value == "PENDING"
    assert not mid.cancel_requested


def test_duplicate_cancel_is_idempotent_after_terminal(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    renderer = _Renderer(b"unused", wait_for_cancel=True)
    runtime = _runtime(database, root, renderer, _Renderer(b"video"))
    runtime.start()
    try:
        runtime.request(clip, DerivativeKind.STILL)
        assert renderer.started.wait(2.0)
        first = runtime.cancel(clip, DerivativeKind.STILL)
        terminal = runtime.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0)
        second = runtime.cancel(clip, DerivativeKind.STILL)
        third = runtime.cancel(clip, DerivativeKind.STILL)
    finally:
        runtime.stop()

    assert first is not None
    assert terminal is not None
    assert (terminal.state.value, terminal.reason) == ("CANCELLED", "CANCELLED")
    assert second is not None and third is not None
    assert (second.state.value, second.reason) == ("CANCELLED", "CANCELLED")
    assert (third.state.value, third.reason) == ("CANCELLED", "CANCELLED")
    final = runtime.jobs.get(incident, DerivativeKind.STILL)
    assert final is not None and final.cancel_requested


def test_restart_detects_corruption_and_quarantines_orphan(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    runtime = _runtime(database, root, _Renderer(b"jpeg"), _Renderer(b"video"))
    runtime.start()
    runtime.request(clip, DerivativeKind.STILL)
    assert runtime.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0) is not None
    runtime.stop()
    artifact = next((root / "derivatives").glob("*/*.jpg"))
    artifact.write_bytes(b"mutated")
    orphan = root / "derivatives" / "orphan" / "private-name.mp4"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")

    restarted = _runtime(database, root, _Renderer(b"jpeg"), _Renderer(b"video"))
    restarted.start()
    restarted.stop()

    status = restarted.jobs.get(incident, DerivativeKind.STILL)
    assert status is not None
    assert (status.state.value, status.reason) == ("CORRUPT", "MISSING_OR_MUTATED")
    assert not orphan.exists()
    quarantined = tuple((root / ".derivative-quarantine").iterdir())
    assert len(quarantined) == 1
    assert "private-name" not in quarantined[0].name


def test_queue_backpressure_marks_excess_request_unavailable(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    seeded = tuple(_seed(database, root, index) for index in range(3))
    release = threading.Event()
    renderer = _Renderer(b"jpeg", release=release)
    runtime = _runtime(database, root, renderer, _Renderer(b"video"), max_pending_jobs=1)
    runtime.start()
    try:
        runtime.request(seeded[0][1], DerivativeKind.STILL)
        assert renderer.started.wait(2.0)
        runtime.request(seeded[1][1], DerivativeKind.STILL)
        third = runtime.request(seeded[2][1], DerivativeKind.STILL)
        release.set()
        assert (
            runtime.wait_for_terminal(seeded[0][0], DerivativeKind.STILL, timeout=2.0) is not None
        )
    finally:
        runtime.stop()

    assert (third.state.value, third.reason) == ("UNAVAILABLE", "RESOURCE_LIMIT")


def test_backend_request_reaches_real_worker_supervisor_without_direct_job_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(database, root)
    manifest = {
        "clip_id": clip,
        "camera_id": "camera-a",
        "event_ref": "event-a",
        "event_type": "fall",
        "started_at": NOW,
        "duration_s": 1.0,
        "codec": "h264",
        "path": f"clips/{clip}/clip.mp4",
        "video_available": True,
        "finalized": True,
    }
    (root / "clips" / clip / "manifest.json").write_text(json.dumps(manifest))
    runtime = _runtime(database, root, _Renderer(b"jpeg"), _Renderer(b"video"))
    runtime.start()
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token="relay-token"),
        derivative_control=DerivativeControlService(runtime),
    )
    server.start()
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    monkeypatch.setenv("API_LABEL_STORE", str(tmp_path / "labels"))
    monkeypatch.setattr(
        "backend.app.features.clips.derivative_control.get_settings",
        lambda: SimpleNamespace(
            worker_stream_origin=f"http://127.0.0.1:{server.port}",
            worker_stream_timeout_s=2.0,
        ),
    )
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.central_clip_artifact_query = CentralClipArtifactQuery(database)
    try:
        with TestClient(app) as client:
            login = client.post(
                "/api/v1/auth/session",
                json={"username": "admin", "password": "admin"},
            )
            assert login.status_code == 204
            requested = client.post(f"/api/v1/clips/{clip}/derivatives/still")
            assert requested.status_code == 202
            status_response = client.get(f"/api/v1/clips/{clip}/derivatives/still")
            assert status_response.status_code == 200
        terminal = runtime.wait_for_terminal(incident, DerivativeKind.STILL, timeout=2.0)
        with TestClient(app) as client:
            login = client.post(
                "/api/v1/auth/session",
                json={"username": "admin", "password": "admin"},
            )
            assert login.status_code == 204
            completed_response = client.get(f"/api/v1/clips/{clip}/derivatives/still")
    finally:
        server.stop()
        runtime.stop()
    assert terminal is not None and terminal.state.value == "AVAILABLE"
    requested_payload = requested.json()
    assert requested_payload["request_id"] == status_response.json()["request_id"]
    assert completed_response.status_code == 200
    completed_payload = completed_response.json()
    assert completed_payload["state"] == "AVAILABLE"
    assert completed_payload["mime_type"] == "image/jpeg"
    assert completed_payload["sha256"] == hashlib.sha256(b"jpeg").hexdigest()
    assert completed_payload["primary_clip_id"] == clip
    assert completed_payload["decision_trace_id"] == hashlib.sha256(b"decision-a").hexdigest()
    assert completed_payload["runtime_manifest_sha256"] == MANIFEST


def test_worker_derivative_http_requires_relay_auth(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    _incident, clip, _ = _seed(database, root)
    runtime = _runtime(database, root, _Renderer(b"jpeg"), _Renderer(b"video"))
    runtime.start()
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token="relay-token"),
        derivative_control=DerivativeControlService(runtime),
    )
    server.start()
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}/derivatives/{clip}/STILL",
        data=b"",
        method="POST",
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=2.0)
    finally:
        server.stop()
        runtime.stop()
    assert raised.value.code == 403


@pytest.mark.real_stack
def test_real_worker_http_requests_produce_decodable_still_and_video(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.fail("ffmpeg is required for derivative real-stack QA", pytrace=False)
    generated = tmp_path / "generated.mp4"
    subprocess.run(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=5:d=1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-y",
            str(generated),
        ),
        check=True,
    )
    database = tmp_path / "edge.sqlite3"
    root = tmp_path / "store"
    migrate_database(database)
    incident, clip, _ = _seed(
        database,
        root,
        primary_content=generated.read_bytes(),
    )
    runtime = DerivativeProductionRuntime(database, root)
    runtime.start()
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token="relay-token"),
        derivative_control=DerivativeControlService(runtime),
    )
    server.start()
    try:
        for kind in ("STILL", "VIDEO"):
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.port}/derivatives/{clip}/{kind}",
                data=b"",
                method="POST",
                headers={"X-Edge-Relay-Token": "relay-token"},
            )
            with urllib.request.urlopen(request, timeout=2.0) as response:
                assert response.status == 202
        still = runtime.wait_for_terminal(incident, DerivativeKind.STILL, timeout=10.0)
        video = runtime.wait_for_terminal(incident, DerivativeKind.VIDEO, timeout=10.0)
    finally:
        server.stop()
        runtime.stop()
    assert still is not None and still.state.value == "AVAILABLE"
    assert video is not None and video.state.value == "AVAILABLE"
    projection = CentralClipArtifactQuery(database).get(clip)
    assert projection is not None and projection.still is not None and projection.video is not None
    still_path = root / str(projection.still.relpath)
    video_path = root / str(projection.video.relpath)
    assert still_path.read_bytes().startswith(b"\xff\xd8")
    assert video_path.read_bytes()


def test_production_derivative_types_are_reachable() -> None:
    assert DerivativeKind.STILL.value == "STILL"
    assert DerivativeKind.VIDEO.value == "VIDEO"
    assert CpuAnnotatedStillRenderer is not None
    assert DerivativeProductionRuntime is not None
