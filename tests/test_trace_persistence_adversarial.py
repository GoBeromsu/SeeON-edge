from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from shared.edge_db.migrator import migrate_database
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.trace import (
    AnalysisTrace,
    BoundedTraceWriter,
    DecisionTrace,
    OptionalNumber,
    TraceContractError,
    TraceFrame,
    TracePersistenceError,
    TraceRetentionPolicy,
)
from worker.pipeline.trace.models import TraceComponent, TracePerson, content_id
from worker.runtime.provenance.models import AppliedRuntimeManifest
from worker.runtime.provenance.store import (
    AppliedRuntimeManifestStore,
    ProvenanceRetentionPolicy,
)
from worker.types import DecisionTraceSnapshot

_MANIFEST = "a" * 64
_POLICY = "b" * 64


def _manifest(*camera_ids: str) -> AppliedRuntimeManifest:
    canonical = json.dumps(
        {
            "cameras": [{"camera_id": camera_id} for camera_id in camera_ids],
            "manifest_schema_version": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return AppliedRuntimeManifest(1, canonical, _MANIFEST)


def _seed(database: Path, *bindings: tuple[str, str]) -> None:
    migrate_database(database)
    store = AppliedRuntimeManifestStore(database)
    for boot_id, camera_id in bindings:
        store.persist(
            _manifest(camera_id),
            boot_instance_id=boot_id,
            applied_at=f"2026-08-13T00:00:{len(boot_id):02d}Z",
        )


def _frame(
    seq: int,
    *,
    boot_id: str = "boot-a",
    camera_id: str = "camera-a",
    source_time: float = 1.0,
    triggered: bool = False,
) -> TraceFrame:
    analysis_id = content_id((boot_id, camera_id, 1, seq))
    analysis = AnalysisTrace(
        trace_id=analysis_id,
        frame_key=(boot_id, camera_id, 1, seq),
        pts=OptionalNumber(float(seq)),
        source_time=OptionalNumber(source_time),
        frame_width=4,
        frame_height=4,
        bed_region_provenance="fresh",
        persons=(),
        beds=(),
        components=(TraceComponent(0, f"pose.sha256.{'c' * 64}", "observed"),),
    )
    snapshot = DecisionTraceSnapshot(
        reason="fall-onset" if triggered else "below-threshold",
        previous_state="clear",
        current_state="triggered" if triggered else "clear",
        triggered=triggered,
        track_id=None,
        bed_id=None,
        values={"fall_probability": 0.9},
    )
    decision = DecisionTrace(
        trace_id=content_id((analysis_id, triggered)),
        analysis_trace_id=analysis_id,
        identity_index=0,
        module_qualified_id="fall.v1",
        policy_qualified_id="fall.policy.v1",
        effective_policy_id=_POLICY,
        runtime_manifest_sha256=_MANIFEST,
        snapshot=snapshot,
    )
    return TraceFrame(analysis, (decision,))


def _policy(**changes: object) -> TraceRetentionPolicy:
    values: dict[str, object] = {
        "max_frames_per_camera": 8,
        "max_rows_per_frame": 64,
        "max_bytes_per_frame": 8_000,
        "max_total_frames": 16,
        "max_total_rows": 128,
        "max_total_bytes": 64_000,
    }
    values.update(changes)
    return replace(TraceRetentionPolicy.testing(), **values)


def _event(decision_trace_id: str) -> dict[str, object]:
    return {
        "edge_event_id": "event-a",
        "event_type": "fall",
        "probability": 0.9,
        "detected_at": "2026-08-13T00:00:01Z",
        "audit": {
            "runtime_manifest_sha256": _MANIFEST,
            "decision_trace_id": decision_trace_id,
        },
    }


def _stager(database: Path) -> DurableEvidenceStager:
    return DurableEvidenceStager(
        database,
        "camera-a",
        "facility-a",
        None,
        1,
        lambda: 1.0,
        _MANIFEST,
    )


def test_sqlite_contention_waits_without_losing_accepted_event_trace(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _seed(database, ("boot-a", "camera-a"))
    blocker = sqlite3.connect(database, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    writer = BoundedTraceWriter(database, _policy(persistence_timeout_seconds=6.0))
    attempted = threading.Event()
    original = writer._store.persist_batch  # noqa: SLF001

    def observed_persist(*args: object, **kwargs: object) -> int:
        attempted.set()
        return original(*args, **kwargs)  # type: ignore[arg-type]

    writer._store.persist_batch = observed_persist  # type: ignore[method-assign]  # noqa: SLF001
    outcome: list[object] = []
    writer.start()
    submitter = threading.Thread(
        target=lambda: outcome.append(
            writer.submit(_frame(1, triggered=True), require_persisted=True)
        )
    )
    submitter.start()
    assert attempted.wait(1.0)
    blocker.commit()
    blocker.close()
    submitter.join(5.0)
    try:
        assert not submitter.is_alive()
        assert outcome == [True]
    finally:
        writer.stop()
    assert len(writer.recover_camera("camera-a").frames) == 1
    assert writer.stats().persistence_failed_frames == 0


def test_async_batch_retries_and_exhausted_failure_is_observable(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _seed(database, ("boot-a", "camera-a"))
    writer = BoundedTraceWriter(database, _policy(max_persistence_attempts=2))
    original = writer._store.persist_batch  # noqa: SLF001
    attempts = 0

    def transient(*args: object, **kwargs: object) -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    writer._store.persist_batch = transient  # type: ignore[method-assign]  # noqa: SLF001
    writer.start()
    assert writer.submit(_frame(1))
    writer.stop()
    assert attempts == 2
    assert writer.stats().retry_attempts == 1
    assert writer.stats().persisted_frames == 1

    failed = BoundedTraceWriter(database, _policy(max_persistence_attempts=2))
    failed._store.persist_batch = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]  # noqa: SLF001
        sqlite3.OperationalError("database is locked")
    )
    failed.start()
    assert failed.submit(_frame(2))
    with pytest.raises(TracePersistenceError, match="1 accepted trace frame"):
        failed.stop()
    assert failed.stats().failed_batches == 1
    assert failed.stats().persistence_failed_frames == 1


def test_global_retention_and_tied_order_use_full_frame_identity(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _seed(
        database,
        ("boot-a", "camera-a"),
        ("boot-b", "camera-a"),
        ("boot-c", "camera-a"),
    )
    writer = BoundedTraceWriter(database, _policy(max_total_frames=2))
    writer.start()
    try:
        for boot_id in ("boot-a", "boot-b", "boot-c"):
            assert writer.submit(_frame(1, boot_id=boot_id), require_persisted=True)
    finally:
        writer.stop()

    recovered = writer.recover_camera("camera-a")
    assert [frame.frame_key[0] for frame in recovered.frames] == ["boot-b", "boot-c"]
    assert recovered.truncation.oldest_retained_key == ("boot-b", "camera-a", 1, 1)
    assert recovered.truncation.newest_retained_key == ("boot-c", "camera-a", 1, 1)
    assert recovered.truncation.pruned_frames == 1


def test_per_frame_person_row_and_byte_bounds_reject_observably(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _seed(database, ("boot-a", "camera-a"))
    writer = BoundedTraceWriter(
        database,
        _policy(max_persons_per_frame=1, max_rows_per_frame=4, max_bytes_per_frame=200),
    )
    person = TracePerson(0, OptionalNumber(1), (0, 0, 1, 1), 0.9)
    oversized = replace(_frame(1), analysis=replace(_frame(1).analysis, persons=(person, person)))

    with pytest.raises(TraceContractError, match="max_persons_per_frame"):
        writer.submit(oversized)
    with pytest.raises(TraceContractError, match="max_bytes_per_frame"):
        writer.submit(_frame(2))
    assert writer.stats().rejected_frames == 2


def test_event_link_keeps_analysis_and_clip_has_direct_transactional_trace_ref(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    _seed(database, ("boot-a", "camera-a"))
    writer = BoundedTraceWriter(database, _policy(max_frames_per_camera=1, max_total_frames=1))
    event_frame = _frame(1, triggered=True)
    writer.start()
    try:
        assert writer.submit(event_frame, require_persisted=True)
        _stager(database).stage(_event(event_frame.decisions[0].trace_id))
        _stager(database).complete("event-a", "clip-a")
        assert writer.submit(_frame(2, source_time=2.0), require_persisted=True)
    finally:
        writer.stop()

    recovered = writer.recover_camera("camera-a")
    assert event_frame.analysis.trace_id in {frame.trace_id for frame in recovered.frames}
    assert recovered.truncation.pruned_frames >= 1
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT clip_id, edge_event_id, decision_trace_id FROM evidence_clip_trace_refs"
        ).fetchall() == [("clip-a", "event-a", event_frame.decisions[0].trace_id)]


def test_clip_trace_binding_rolls_back_without_a_resolving_event_trace(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _seed(database, ("boot-a", "camera-a"))
    no_trace = {
        "edge_event_id": "event-a",
        "event_type": "fall",
        "probability": 0.9,
        "detected_at": "2026-08-13T00:00:01Z",
    }
    _stager(database).stage(no_trace)

    with pytest.raises(ValueError, match="clip requires a decision trace"):
        _stager(database).complete("event-a", "clip-a")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM evidence_clips").fetchone() == (0,)
        assert connection.execute("SELECT state FROM evidence_events").fetchone() == ("STAGED",)


def test_trace_camera_must_match_boot_manifest_camera_set(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _seed(database, ("boot-a", "camera-a"))
    writer = BoundedTraceWriter(database, _policy())
    writer.start()
    try:
        with pytest.raises(TracePersistenceError, match="camera.*runtime manifest"):
            writer.submit(_frame(1, camera_id="camera-b"), require_persisted=True)
    finally:
        writer.stop()
    assert writer.recover_camera("camera-b").frames == ()


def test_duplicate_submission_counts_only_the_exact_insert(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _seed(database, ("boot-a", "camera-a"))
    writer = BoundedTraceWriter(database, _policy())
    frame = _frame(1)
    writer.start()
    try:
        assert writer.submit(frame, require_persisted=True)
        assert writer.submit(frame, require_persisted=True)
    finally:
        writer.stop()
    assert writer.stats().persisted_frames == 1
    assert writer.stats().duplicate_frames == 1


def test_stop_drains_accepted_work_and_rejects_every_later_submission(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _seed(database, ("boot-a", "camera-a"))
    writer = BoundedTraceWriter(database, _policy())
    writer.start()
    assert writer.submit(_frame(1))
    writer.stop()

    with pytest.raises(TracePersistenceError, match="stopped"):
        writer.submit(_frame(2))
    assert len(writer.recover_camera("camera-a").frames) == 1
    assert writer.stats().rejected_frames == 1


def test_provenance_history_is_bounded_without_deleting_trace_references(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    store = AppliedRuntimeManifestStore(
        database,
        ProvenanceRetentionPolicy(max_boots=2, max_boots_per_camera=1),
    )
    for index in range(3):
        store.persist(
            _manifest("camera-a"),
            boot_instance_id=f"boot-{index}",
            applied_at=f"2026-08-13T00:00:0{index}Z",
        )

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM runtime_manifest_boots").fetchone() == (1,)
        boot = connection.execute("SELECT boot_instance_id FROM runtime_manifest_boots").fetchone()
        assert boot == ("boot-2",)
        assert connection.execute(
            "SELECT pruned_boots, pruned_camera_bindings "
            "FROM runtime_provenance_retention WHERE id=1"
        ).fetchone() == (2, 2)
        contents = connection.execute("SELECT count(*) FROM runtime_manifest_contents").fetchone()
        assert contents == (1,)
