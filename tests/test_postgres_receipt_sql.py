"""Real borrowed-SQL contracts, not receipt-file or HTTP/lifespan qualification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from threading import Event
from time import monotonic

import psycopg
import pytest

from backend.app.edge_db.authority import require_authority
from backend.app.features.clips.manifest import ClipManifest
from backend.app.features.evidence.postgres_receipt_sql import (
    commit_clip,
    commit_primary_artifact,
    commit_unavailable_primary,
)
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptConflictError,
    ClipProjection,
    primary_artifact_id,
    verified_artifact,
)

pytest_plugins = ("tests_support.postgres_sandbox",)
_TIME = "2026-07-06T00:00:00Z"
_COMMIT = "2026-07-06T00:00:01Z"
_LATER = "2026-07-06T00:00:02Z"


@pytest.fixture
def projection(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"verified video")
    with media.open("rb") as handle:
        verified = verified_artifact(handle)
        receipt = ArtifactReceipt("clip-1", verified.sha256, verified.size_bytes)
        manifest = ClipManifest(
            clip_id="clip-1",
            camera_id="camera-1",
            event_ref="event-1",
            event_type="fall",
            started_at=_TIME,
            duration_s=1.0,
            codec="h264",
            path="clips/clip-1/clip.mp4",
            video_available=True,
            video_error=None,
            finalized=True,
        )
        yield ClipProjection(
            receipt,
            verified,
            manifest,
            "clips/clip-1/manifest.json",
            "clips/clip-1/clip.mp4",
            "a" * 64,
            1,
        )


@pytest.fixture
def sandbox(postgres_product_sandbox):
    sandbox = postgres_product_sandbox

    def seed(connection):
        require_authority(connection, sandbox.authority)
        for incident_id, event_id in (
            ("incident:event-1", "event-1"),
            ("incident:event-2", "event-2"),
            ("incident:max", "e" * 128),
        ):
            connection.execute(
                "INSERT INTO incidents (incident_id,edge_event_id,facility_id,camera_id,event_type,"
                "probability,detected_at,lifecycle_state,provenance_state,provenance_missing_reason,"
                "review_version,revision,created_at,updated_at) "
                "VALUES (%s,%s,'facility-1','camera-1','fall',0.8,%s,'OPEN','MISSING',"
                "'NOT_RECORDED',0,1,%s,%s)",
                (incident_id, event_id, _TIME, _TIME, _TIME),
            )

    sandbox.database.transact(seed)
    return sandbox


def _history(connection):
    return {
        table: connection.execute("SELECT * FROM " + table + " ORDER BY 1").fetchall()
        for table in ("clips", "incidents", "artifacts")
    }


def _with_authority(sandbox, operation):
    def write(connection):
        require_authority(connection, sandbox.authority)
        return operation(connection)

    return sandbox.database.transact(write)


def _commit(sandbox, projection, *, timestamp=_COMMIT, after_write=None, event_id="event-1"):
    def write(connection):
        commit_clip(connection, projection)
        commit_primary_artifact(
            connection, f"incident:{event_id}", event_id, projection, timestamp=timestamp
        )
        if after_write is not None:
            after_write(connection)

    return _with_authority(sandbox, write)


def _other_clip(projection, clip_id):
    return replace(
        projection,
        receipt=replace(projection.receipt, artifact_id=clip_id),
        manifest=replace(projection.manifest, clip_id=clip_id, path=f"clips/{clip_id}/clip.mp4"),
        manifest_relpath=f"clips/{clip_id}/manifest.json",
        media_relpath=f"clips/{clip_id}/clip.mp4",
    )


def test_projection_is_tentative_then_durable_and_retry_is_a_complete_noop(sandbox, projection):
    before = _history(sandbox.admin)
    seen = []

    def observe(connection):
        assert connection.execute("SELECT count(*) FROM clips").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (1,)
        assert _history(sandbox.admin) == before
        seen.append(True)

    _commit(sandbox, projection, after_write=observe)
    assert seen == [True]
    assert sandbox.admin.execute(
        "SELECT lifecycle_state,revision,updated_at FROM incidents WHERE edge_event_id='event-1'"
    ).fetchone() == ("COMPLETE", 2, _COMMIT)
    assert sandbox.admin.execute(
        "SELECT publish_state,published_at,revision,event_facet FROM clips"
    ).fetchone() == ("PUBLISHED", _TIME, 1, "fall")
    first = _history(sandbox.admin)
    _commit(sandbox, projection, timestamp=_LATER)
    assert _history(sandbox.admin) == first


@pytest.mark.parametrize("duration,expected", [(0.0005, 1), (0.0015, 2), (120.0, 120000)])
def test_duration_uses_existing_rounding_and_minimum(sandbox, projection, duration, expected):
    projection = replace(projection, manifest=replace(projection.manifest, duration_s=duration))
    _with_authority(sandbox, lambda connection: commit_clip(connection, projection))
    assert sandbox.admin.execute("SELECT duration_ms FROM clips").fetchone() == (expected,)


def test_waiting_clip_is_promoted_once_without_rewriting_media_identity(sandbox, projection):
    _with_authority(sandbox, lambda connection: commit_clip(connection, projection))
    sandbox.admin.execute(
        "UPDATE clips SET publish_state='WAITING',published_at=NULL,"
        "last_publish_error_code='RETRY',revision=2,updated_at=%s",
        (_COMMIT,),
    )
    _with_authority(sandbox, lambda connection: commit_clip(connection, projection))
    assert sandbox.admin.execute(
        "SELECT publish_state,published_at,last_publish_error_code,revision,media_sha256,"
        "media_size_bytes FROM clips"
    ).fetchone() == (
        "PUBLISHED",
        _TIME,
        None,
        3,
        projection.verified.sha256,
        projection.verified.size_bytes,
    )
    before = _history(sandbox.admin)
    _with_authority(sandbox, lambda connection: commit_clip(connection, projection))
    assert _history(sandbox.admin) == before


@pytest.mark.parametrize("field,value", [("sha256", "b" * 64), ("size_bytes", 999)])
def test_clip_hash_or_size_conflict_preserves_all_rows(sandbox, projection, field, value):
    _commit(sandbox, projection)
    before = _history(sandbox.admin)
    changed = replace(projection, verified=replace(projection.verified, **{field: value}))
    with pytest.raises(ArtifactReceiptConflictError, match="immutable artifact receipt"):
        _commit(sandbox, changed)
    assert _history(sandbox.admin) == before


def test_primary_content_conflict_rolls_back_a_new_clip(sandbox, projection):
    _commit(sandbox, projection)
    before = _history(sandbox.admin)
    with pytest.raises(ArtifactReceiptConflictError, match="primary clip artifact conflicts"):
        _commit(sandbox, _other_clip(projection, "clip-2"))
    assert _history(sandbox.admin) == before


def test_existing_clip_without_retained_media_identity_fails_closed(sandbox, projection):
    _with_authority(sandbox, lambda connection: commit_clip(connection, projection))
    sandbox.admin.execute(
        "UPDATE clips SET local_state='UNAVAILABLE',local_reason='MEDIA_GONE',"
        "manifest_relpath=NULL,manifest_sha256=NULL,manifest_size_bytes=NULL,"
        "media_relpath=NULL,media_sha256=NULL,media_size_bytes=NULL,revision=revision+1"
    )
    before = _history(sandbox.admin)
    with pytest.raises(ArtifactReceiptConflictError, match="immutable artifact receipt"):
        _commit(sandbox, projection)
    assert _history(sandbox.admin) == before


def test_long_id_digest_and_multiple_incident_completion(sandbox, projection):
    projection = _other_clip(projection, "c" * 128)

    def write(connection):
        commit_clip(connection, projection)
        # This caller uses a stable incident order, including a repeated fact.
        for incident_id, event_id in (
            ("incident:event-1", "event-1"),
            ("incident:max", "e" * 128),
            ("incident:max", "e" * 128),
        ):
            commit_primary_artifact(
                connection, incident_id, event_id, projection, timestamp=_COMMIT
            )

    _with_authority(sandbox, write)
    assert sandbox.admin.execute(
        "SELECT artifact_id FROM artifacts WHERE incident_id='incident:max'"
    ).fetchone() == ("primary:0fb0648ada8974a9693610ced3a5e6f1",)
    assert sandbox.admin.execute(
        "SELECT count(*) FROM incidents WHERE lifecycle_state='COMPLETE'"
    ).fetchone() == (2,)
    assert sandbox.admin.execute("SELECT count(*) FROM artifacts").fetchone() == (2,)


def test_existing_receipt_identity_is_not_rewritten(sandbox, projection):
    # A matching fact is immutable regardless of how its stored ID was assigned.
    artifact_id = "existing-artifact"
    _with_authority(sandbox, lambda connection: commit_clip(connection, projection))
    _seed_artifact(sandbox, projection, "incident:event-1", artifact_id)
    _commit(sandbox, projection)
    assert sandbox.admin.execute("SELECT artifact_id FROM artifacts").fetchone() == (artifact_id,)
    assert sandbox.admin.execute(
        "SELECT lifecycle_state FROM incidents WHERE edge_event_id='event-1'"
    ).fetchone() == ("COMPLETE",)


def _seed_artifact(sandbox, projection, incident_id, artifact_id):
    def seed(connection):
        connection.execute(
            "INSERT INTO artifacts (incident_id,kind,artifact_id,clip_id,state,contained_relpath,"
            "content_sha256,size_bytes,mime_type,codec,revision,created_at,updated_at) "
            "VALUES (%s,'PRIMARY_CLIP',%s,%s,'AVAILABLE',%s,%s,%s,'video/mp4','h264',1,%s,%s)",
            (
                incident_id,
                artifact_id,
                projection.receipt.artifact_id,
                projection.media_relpath,
                projection.verified.sha256,
                projection.verified.size_bytes,
                _TIME,
                _TIME,
            ),
        )

    _with_authority(sandbox, seed)


def test_digest_identity_owned_by_another_incident_is_a_typed_conflict(sandbox, projection):
    _with_authority(sandbox, lambda connection: commit_clip(connection, projection))
    _seed_artifact(
        sandbox, projection, "incident:event-2", primary_artifact_id("clip-1", "event-1")
    )
    before = _history(sandbox.admin)
    with pytest.raises(ArtifactReceiptConflictError, match="identity conflicts"):
        _commit(sandbox, projection)
    assert _history(sandbox.admin) == before


@pytest.mark.parametrize("via_manifest", [False, True])
def test_unavailable_primary_fails_once_and_rejects_changed_reason(
    sandbox, projection, via_manifest
):
    reason = "원인" * 40
    projection = replace(
        projection, manifest=replace(projection.manifest, video_available=False, video_error=reason)
    )

    def write(connection):
        if via_manifest:
            commit_primary_artifact(
                connection, "incident:event-1", "event-1", projection, timestamp=_COMMIT
            )
        else:
            commit_unavailable_primary(connection, "incident:event-1", reason, _COMMIT)

    _with_authority(sandbox, write)
    assert sandbox.admin.execute(
        "SELECT lifecycle_state,failure_reason,revision FROM incidents "
        "WHERE edge_event_id='event-1'"
    ).fetchone() == ("FAILED", reason[:64], 2)
    assert sandbox.admin.execute("SELECT clip_id,state,reason FROM artifacts").fetchone() == (
        None,
        "UNAVAILABLE",
        reason[:64],
    )
    before = _history(sandbox.admin)
    _with_authority(sandbox, write)
    assert _history(sandbox.admin) == before
    with pytest.raises(ArtifactReceiptConflictError):
        _with_authority(
            sandbox,
            lambda connection: commit_unavailable_primary(
                connection, "incident:event-1", "different", _LATER
            ),
        )
    assert _history(sandbox.admin) == before


@pytest.mark.parametrize("first", ["available", "unavailable"])
def test_terminal_primary_facts_cannot_overwrite_each_other(sandbox, projection, first):
    def unavailable():
        _with_authority(
            sandbox,
            lambda connection: commit_unavailable_primary(
                connection, "incident:event-1", "NO_FRAMES", _COMMIT
            ),
        )

    if first == "available":
        _commit(sandbox, projection)
    else:
        unavailable()
    before = _history(sandbox.admin)
    with pytest.raises(ArtifactReceiptConflictError):
        if first == "available":
            unavailable()
        else:
            _commit(sandbox, projection)
    assert _history(sandbox.admin) == before


def test_primary_availability_does_not_heal_an_already_failed_incident(sandbox, projection):
    sandbox.admin.execute(
        "UPDATE incidents SET lifecycle_state='FAILED',failure_reason='OTHER_FAILURE',"
        "revision=revision+1,updated_at=%s WHERE edge_event_id='event-1'",
        (_COMMIT,),
    )
    _commit(sandbox, projection, timestamp=_LATER)
    assert sandbox.admin.execute(
        "SELECT lifecycle_state,failure_reason,revision,updated_at FROM incidents "
        "WHERE edge_event_id='event-1'"
    ).fetchone() == ("FAILED", "OTHER_FAILURE", 2, _COMMIT)
    assert sandbox.admin.execute("SELECT state FROM artifacts").fetchone() == ("AVAILABLE",)


def test_outer_failure_and_real_deferred_commit_rejection_are_atomic(sandbox, projection):
    before = _history(sandbox.admin)
    failure = OSError("outer validation failed")

    def reject(connection):
        raise failure

    with pytest.raises(OSError) as raised:
        _commit(sandbox, projection, after_write=reject)
    assert raised.value is failure and _history(sandbox.admin) == before
    sandbox.admin.execute(
        "CREATE FUNCTION reject_clip_commit_test() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'injected clip commit rejection' USING ERRCODE='23514'; END $$"
    )
    sandbox.admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_clip_commit_test AFTER INSERT ON clips "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_clip_commit_test()"
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        _commit(sandbox, projection)
    assert _history(sandbox.admin) == before


@pytest.mark.parametrize("second_kind", ["matching", "hash", "incident", "identity"])
def test_competing_clip_and_primary_writers_serialize_before_comparison(
    sandbox, projection, monkeypatch, second_kind
):
    entered, release = Event(), Event()
    pids = []

    def hold(connection):
        pids.append(connection.info.backend_pid)
        entered.set()
        assert release.wait(2), "receipt writer was not released"

    if second_kind == "hash":
        candidate = replace(projection, verified=replace(projection.verified, sha256="b" * 64))
    elif second_kind in ("incident", "identity"):
        candidate = _other_clip(projection, "clip-2")
    else:
        candidate = projection
    if second_kind == "identity":
        # Force the identity race, not a claim about finding a SHA-256 collision.
        monkeypatch.setattr(
            "backend.app.features.evidence.postgres_receipt_sql.primary_artifact_id",
            lambda clip_id, event_id: "controlled-collision",
        )
    executor = ThreadPoolExecutor(max_workers=2)
    first = executor.submit(_commit, sandbox, projection, after_write=hold)
    futures = [first]
    try:
        assert entered.wait(1)
        second = executor.submit(
            _commit,
            sandbox,
            candidate,
            event_id="event-2" if second_kind == "identity" else "event-1",
        )
        futures.append(second)
        deadline, pacing = monotonic() + 1.5, Event()
        while not sandbox.admin.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE %s=ANY(pg_blocking_pids(pid)))",
            (pids[0],),
        ).fetchone()[0]:
            assert monotonic() < deadline, "second writer did not wait for the real receipt lock"
            pacing.wait(0.01)
        assert not first.done() and not second.done()
        release.set()
        assert first.result(timeout=2) is None
        if second_kind == "matching":
            assert second.result(timeout=2) is None
        else:
            with pytest.raises(ArtifactReceiptConflictError):
                second.result(timeout=2)
        assert sandbox.admin.execute("SELECT clip_id,media_sha256 FROM clips").fetchall() == [
            ("clip-1", projection.verified.sha256)
        ]
        assert sandbox.admin.execute("SELECT count(*) FROM artifacts").fetchone() == (1,)
        assert sandbox.admin.execute(
            "SELECT lifecycle_state FROM incidents WHERE edge_event_id='event-2'"
        ).fetchone() == ("OPEN",)
    finally:
        release.set()
        _, pending = wait(futures, timeout=2)
        executor.shutdown(wait=not pending, cancel_futures=True)
        assert not pending, "receipt operations did not terminate"
