"""Native receipt/file/transaction contracts; no HTTP or power-loss qualification."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import replace
from threading import Event
from time import monotonic
from uuid import uuid4

import psycopg
import pytest

from backend.app.edge_db.authority import AuthorityFenced, freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    empty_detail,
)
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.audit.store import AuditEvent, utc_now
from backend.app.features.evidence.event_outbox import EventOutbox, OutboxBudget
from backend.app.features.evidence.postgres_receipts import PostgresArtifactReceiptStore
from backend.app.features.evidence.receipt_files import ReceiptHooks
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptConflictError,
    ArtifactReceiptPersistenceError,
    ArtifactReceiptVerificationError,
    ReceiptMissingIncidentError,
    verified_artifact,
)
from backend.app.features.evidence.relay_projection import RelayEvent

pytest_plugins = ("tests_support.postgres_sandbox",)
_TIME = "2026-09-28T03:00:00.000Z"


@pytest.fixture
def audit_runtime(postgres_product_sandbox):
    sandbox = postgres_product_sandbox
    runtime = PostgresAuditRuntime(
        PostgresAuditStore(sandbox.database, sandbox.authority),
        maximum_snapshot_age_sec=10,
        clock=lambda: 0.0,
    )
    assert runtime.verify_once() and runtime.start_session_once()
    return runtime


@pytest.fixture
def sandbox(postgres_product_sandbox, audit_runtime):
    sandbox = postgres_product_sandbox
    outbox = EventOutbox(
        sandbox.database,
        sandbox.authority,
        OutboxBudget(10, 1_048_576),
        audit_runtime=audit_runtime,
    )
    for event_id in ("event-1", "event-2"):
        outbox.accept(
            RelayEvent(event_id, "fall", 0.8, _TIME, "camera-1", "facility-1", None, None, None),
            backend_camera_id=None,
            forward=False,
        )
    return sandbox


def _clip(root, *, clip_id="clip-1", refs=("event-1", "event-2")):
    directory = root / "clips" / clip_id
    directory.mkdir(parents=True)
    media = directory / "clip.mp4"
    media.write_bytes(b"verified video")
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "camera_id": "camera-1",
                "event_ref": refs[0],
                "event_refs": list(refs),
                "event_type": "fall",
                "started_at": _TIME,
                "duration_s": 1.0,
                "codec": "h264",
                "path": f"clips/{clip_id}/clip.mp4",
                "video_available": True,
                "finalized": True,
            }
        )
    )
    return (
        manifest,
        media,
        ArtifactReceipt(
            clip_id, hashlib.sha256(media.read_bytes()).hexdigest(), media.stat().st_size
        ),
    )


@pytest.fixture
def clip(tmp_path):
    return _clip(tmp_path)


def _store(sandbox, root, hooks=None):
    return PostgresArtifactReceiptStore(sandbox.database, sandbox.authority, root, hooks)


def _history(sandbox):
    return {
        table: sandbox.admin.execute("SELECT * FROM " + table + " ORDER BY 1").fetchall()
        for table in ("clips", "incidents", "artifacts", "audit_events", "event_outbox")
    }


def _callback(runtime, tokens):
    def append(connection):
        token = runtime.append_borrowed(
            connection,
            AuditEvent(
                occurred_at=utc_now(),
                actor_id="worker-relay",
                action=AuditAction.EVIDENCE_RECEIPT,
                target_id="clip-1",
                detail=empty_detail(AuditAction.EVIDENCE_RECEIPT),
                actor_type=AuditActorType.SERVICE,
                auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
            ),
        )
        runtime.validate_publication(token)
        tokens.append(token)

    return append


def test_receipt_returns_after_pool_exit_and_replay_calls_once(
    sandbox, audit_runtime, tmp_path, clip, monkeypatch
):
    manifest, media, receipt = clip
    store = _store(sandbox, tmp_path)
    assert store.get("clip-1") is None
    original = sandbox.database._pool.connection
    trace, tokens = [], []
    before = _history(sandbox)

    @contextmanager
    def connection(*args, **kwargs):
        with original(*args, **kwargs) as value:
            yield value
        assert sandbox.admin.execute("SELECT count(*) FROM clips").fetchone() == (1,)
        trace.append("pool-exit")

    monkeypatch.setattr(sandbox.database._pool, "connection", connection)
    append = _callback(audit_runtime, tokens)

    def observe(connection):
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (2,)
        assert _history(sandbox) == before
        append(connection)
        trace.append("callback")

    with media.open("rb") as handle:
        assert (
            store.commit_verified(receipt, verified_artifact(handle), after_write=observe)
            == receipt
        )
        assert not handle.closed and handle.tell() == 0
    trace.append("returned")
    assert trace == ["callback", "pool-exit", "returned"]
    assert len(tokens) == 1
    audit_runtime.publish_committed(tokens.pop())
    first = _history(sandbox)
    assert store.commit(receipt, after_write=append) == receipt
    assert len(tokens) == 1
    audit_runtime.publish_committed(tokens.pop())
    second = _history(sandbox)
    assert {k: v for k, v in first.items() if k != "audit_events"} == {
        k: v for k, v in second.items() if k != "audit_events"
    }
    assert len(second["audit_events"]) == len(first["audit_events"]) + 1
    assert store.get("clip-1") == receipt
    assert sandbox.admin.execute("SELECT manifest_sha256 FROM clips").fetchone() == (
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    assert sandbox.admin.execute(
        "SELECT lifecycle_state,revision FROM incidents ORDER BY incident_id"
    ).fetchall() == [("COMPLETE", 2), ("COMPLETE", 2)]


@pytest.mark.parametrize("phase", ["after_preflight", "before_final_check"])
@pytest.mark.parametrize("subject", ["media", "manifest"])
@pytest.mark.parametrize("kind", ["rewrite", "inode", "missing", "symlink"])
def test_file_races_leave_no_partial_native_rows_or_callback(
    sandbox, tmp_path, clip, phase, subject, kind
):
    manifest, media, receipt = clip
    target = media if subject == "media" else manifest
    before, callbacks, reached = _history(sandbox), [], []

    def change():
        content = target.read_bytes()
        if kind == "rewrite":
            changed = (
                b"x" * len(content)
                if subject == "media"
                else content.replace(b"camera-1", b"camera-2")
            )
            target.write_bytes(changed)
        elif kind == "inode":
            other = target.with_name("replacement")
            other.write_bytes(content)
            other.replace(target)
        else:
            target.unlink()
            if kind == "symlink":
                other = target.with_name("alternate")
                other.write_bytes(content)
                target.symlink_to(other)
        reached.append(True)

    store = _store(sandbox, tmp_path, ReceiptHooks(**{phase: change}))
    with media.open("rb") as handle:
        with pytest.raises(ArtifactReceiptVerificationError):
            store.commit_verified(
                receipt, verified_artifact(handle), after_write=lambda c: callbacks.append(True)
            )
        assert not handle.closed
    assert reached == [True] and callbacks == [] and _history(sandbox) == before


@pytest.mark.parametrize("fence", ["frozen", "generation", "writer"])
def test_authority_fence_precedes_mutation(sandbox, tmp_path, clip, fence):
    authority = sandbox.authority
    if fence == "frozen":
        freeze_authority(sandbox.database, authority)
    elif fence == "generation":
        authority = replace(authority, generation=authority.generation + 1)
    else:
        authority = replace(authority, writer_token=uuid4())
    store = PostgresArtifactReceiptStore(sandbox.database, authority, tmp_path)
    before, calls = _history(sandbox), []
    with pytest.raises(AuthorityFenced):
        store.commit(clip[2], after_write=lambda c: calls.append(True))
    assert calls == [] and _history(sandbox) == before


def test_missing_incident_is_typed_and_rolls_back(sandbox, tmp_path):
    _, _, receipt = _clip(tmp_path, refs=("event-1", "missing-event"))
    before, calls = _history(sandbox), []
    with pytest.raises(ReceiptMissingIncidentError, match="missing-event"):
        _store(sandbox, tmp_path).commit(receipt, after_write=lambda c: calls.append(True))
    assert calls == [] and _history(sandbox) == before


def test_incident_lock_queries_use_sorted_unique_refs(sandbox, tmp_path, monkeypatch):
    _, _, receipt = _clip(tmp_path, refs=("event-2", "event-1", "event-2"))
    original = psycopg.Connection.execute
    locked = []

    def execute(connection, query, params=None, *args, **kwargs):
        result = original(connection, query, params, *args, **kwargs)
        if query == (
            "SELECT incident_id,edge_event_id FROM incidents WHERE edge_event_id=%s FOR UPDATE"
        ):
            locked.append(params[0])
        return result

    monkeypatch.setattr(psycopg.Connection, "execute", execute)
    assert _store(sandbox, tmp_path).commit(receipt) == receipt
    assert locked == ["event-1", "event-2"]
    assert sandbox.admin.execute("SELECT count(*) FROM artifacts").fetchone() == (2,)


def test_get_does_not_acknowledge_unpublished_or_missing_identity(sandbox, tmp_path, clip):
    store = _store(sandbox, tmp_path)
    store.commit(clip[2])
    sandbox.admin.execute(
        "UPDATE clips SET publish_state='WAITING',published_at=NULL,"
        "last_publish_error_code='RETRY',revision=revision+1"
    )
    assert store.get("clip-1") is None
    sandbox.admin.execute(
        "UPDATE clips SET publish_state='PUBLISHED',published_at=%s,last_publish_error_code=NULL,"
        "local_state='UNAVAILABLE',local_reason='MEDIA_GONE',"
        "manifest_relpath=NULL,manifest_sha256=NULL,manifest_size_bytes=NULL,"
        "media_relpath=NULL,media_sha256=NULL,media_size_bytes=NULL,revision=revision+1",
        (_TIME,),
    )
    before = _history(sandbox)
    with pytest.raises(ArtifactReceiptPersistenceError, match="identity is unreadable"):
        store.get("clip-1")
    assert _history(sandbox) == before
    # Malformed digests and nonpositive non-null sizes are rejected by schema;
    # do not disable constraints just to fabricate otherwise unrepresentable rows.


@pytest.mark.parametrize("failure", [ValueError("callback failed"), BaseException("cancelled")])
def test_callback_failure_rolls_back_receipt_and_borrowed_audit(
    sandbox, audit_runtime, tmp_path, clip, failure
):
    before, tokens = _history(sandbox), []
    append = _callback(audit_runtime, tokens)

    def fail(connection):
        append(connection)
        raise failure

    with pytest.raises(type(failure)) as caught:
        _store(sandbox, tmp_path).commit(clip[2], after_write=fail)
    assert caught.value is failure and len(tokens) == 1 and _history(sandbox) == before
    audit_runtime.publish_failed(tokens.pop(), failure)
    assert not audit_runtime.snapshot().ready


def test_real_deferred_commit_rejection_never_returns_a_receipt(
    sandbox, audit_runtime, tmp_path, clip
):
    before, tokens = _history(sandbox), []
    sandbox.admin.execute(
        "CREATE FUNCTION reject_receipt_commit() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'receipt commit rejection' USING ERRCODE='23514'; END $$"
    )
    sandbox.admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_receipt_commit AFTER INSERT ON clips "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_receipt_commit()"
    )
    with pytest.raises(psycopg.errors.CheckViolation) as caught:
        _store(sandbox, tmp_path).commit(clip[2], after_write=_callback(audit_runtime, tokens))
    assert len(tokens) == 1 and _history(sandbox) == before
    audit_runtime.publish_failed(tokens.pop(), caught.value)


@pytest.mark.parametrize(
    "error", [CommitOutcomeUnknown(), OSError("pool exit"), BaseException("cancel")]
)
def test_post_owned_return_failure_is_not_replayed_or_translated(
    sandbox, tmp_path, clip, monkeypatch, error
):
    # Synthetic error after a real commit, not a wire-loss/power-loss claim.
    original = sandbox.database.transact
    calls = []

    def fail_after_commit(callback):
        original(callback)
        calls.append(True)
        raise error

    monkeypatch.setattr(sandbox.database, "transact", fail_after_commit)
    with pytest.raises(type(error)) as caught:
        _store(sandbox, tmp_path).commit(clip[2])
    assert caught.value is error and calls == [True]
    assert sandbox.admin.execute("SELECT count(*) FROM clips").fetchone() == (1,)
    monkeypatch.setattr(sandbox.database, "transact", original)
    before = _history(sandbox)
    assert _store(sandbox, tmp_path).commit(clip[2]) == clip[2]
    assert _history(sandbox) == before


def test_unavailable_commit_needs_no_media_and_retry_does_not_rewrite(sandbox, tmp_path, clip):
    _, media, _ = clip
    media.unlink()
    store = _store(sandbox, tmp_path)
    store.commit_unavailable("clip-1", "NO_FRAMES")
    assert sandbox.admin.execute("SELECT count(*) FROM clips").fetchone() == (0,)
    assert sandbox.admin.execute("SELECT state,reason FROM artifacts").fetchall() == [
        ("UNAVAILABLE", "NO_FRAMES"),
        ("UNAVAILABLE", "NO_FRAMES"),
    ]
    before = _history(sandbox)
    store.commit_unavailable("clip-1", "NO_FRAMES")
    assert _history(sandbox) == before
    with pytest.raises(ArtifactReceiptConflictError):
        store.commit_unavailable("clip-1", "OTHER_REASON")
    assert _history(sandbox) == before


def test_oppositely_ordered_multi_incident_receipts_serialize_and_conflict_atomically(
    sandbox, tmp_path, clip
):
    _, _, first_receipt = clip
    _, _, second_receipt = _clip(tmp_path, clip_id="clip-2", refs=("event-2", "event-1"))
    entered, release = Event(), Event()
    pids = []
    store = _store(sandbox, tmp_path)
    executor = ThreadPoolExecutor(max_workers=2)
    futures = []

    def hold(connection):
        pids.append(connection.info.backend_pid)
        entered.set()
        assert release.wait(2), "first receipt was not released"

    try:
        futures.append(executor.submit(store.commit, first_receipt, after_write=hold))
        assert entered.wait(1)
        futures.append(executor.submit(store.commit, second_receipt))
        deadline, pacing = monotonic() + 1.5, Event()
        while not sandbox.admin.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE %s=ANY(pg_blocking_pids(pid)))",
            (pids[0],),
        ).fetchone()[0]:
            assert monotonic() < deadline, "second receipt did not wait on incident locks"
            pacing.wait(0.01)
        assert not any(future.done() for future in futures)
        release.set()
        assert futures[0].result(timeout=2) == first_receipt
        with pytest.raises(ArtifactReceiptConflictError):
            futures[1].result(timeout=2)
        assert sandbox.admin.execute("SELECT clip_id FROM clips").fetchall() == [("clip-1",)]
        assert sandbox.admin.execute("SELECT clip_id FROM artifacts").fetchall() == [
            ("clip-1",),
            ("clip-1",),
        ]
    finally:
        release.set()
        _, pending = wait(futures, timeout=2)
        executor.shutdown(wait=not pending, cancel_futures=True)
        assert not pending, "receipt operations did not drain"
