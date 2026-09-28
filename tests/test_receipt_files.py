"""File proof and existing-owner rollback; not native PostgreSQL receipt qualification."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
from threading import Event, Thread

import pytest

from backend.app.features.clips.descriptor_files import open_contained_regular_file
from backend.app.features.clips.manifest import parse_manifest_bytes, read_manifest_file
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.compact_receipts import (
    CompactArtifactReceiptStore,
)
from backend.app.features.evidence.receipt_files import (
    ReceiptFiles,
    ReceiptHooks,
    ReceiptManifest,
    open_receipt_media,
)
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptVerificationError,
    verified_artifact,
)
from tests_support.compact_authority_db import prepare_compact_database

_TIME = "2026-07-06T00:00:00Z"


@pytest.fixture
def clip(tmp_path):
    root = tmp_path / "store"
    directory = root / "clips" / "clip-1"
    directory.mkdir(parents=True)
    media = directory / "clip.mp4"
    media.write_bytes(b"verified video")
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "clip-1",
                "camera_id": "camera-1",
                "event_ref": "event-1",
                "event_refs": ["event-1"],
                "event_type": "fall",
                "started_at": _TIME,
                "duration_s": 1.0,
                "codec": "h264",
                "path": "clips/clip-1/clip.mp4",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    receipt = ArtifactReceipt(
        "clip-1", hashlib.sha256(media.read_bytes()).hexdigest(), media.stat().st_size
    )
    return root, manifest, media, receipt


@pytest.fixture
def database(tmp_path):
    database = prepare_compact_database(tmp_path / "edge.sqlite3")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO incidents (incident_id,edge_event_id,facility_id,camera_id,event_type,"
            "probability,detected_at,lifecycle_state,provenance_state,provenance_missing_reason,"
            "review_version,revision,created_at,updated_at) "
            "VALUES ('incident:event-1','event-1','facility-1','camera-1','fall',0.8,?,'OPEN',"
            "'MISSING','NOT_RECORDED',0,1,?,?)",
            (_TIME, _TIME, _TIME),
        )
    return database


def _rows(database):
    with sqlite3.connect(database) as connection:
        return {
            table: connection.execute("SELECT * FROM " + table + " ORDER BY 1").fetchall()
            for table in ("clips", "incidents", "artifacts")
        }


def _mutate(path, kind):
    content = path.read_bytes()
    if kind == "rewrite":
        changed = (
            content.replace(b"camera-1", b"camera-2")
            if path.suffix == ".json"
            else b"x" * len(content)
        )
        assert changed != content and len(changed) == len(content)
        path.write_bytes(changed)
    elif kind == "whitespace":
        path.write_bytes(b" " + content)
    elif kind == "inode":
        replacement = path.with_name(path.name + ".replacement")
        replacement.write_bytes(content)
        replacement.replace(path)
    else:
        path.unlink()
        if kind == "symlink":
            alternate = path.with_name(path.name + ".alternate")
            alternate.write_bytes(content)
            path.symlink_to(alternate)
        elif kind == "fifo":
            os.mkfifo(path)
        else:
            assert kind == "missing"


def test_shared_proof_binds_actual_bytes_and_never_closes_borrowed_media(clip, monkeypatch):
    root, manifest_path, media, receipt = clip
    opened = []
    real_open = open_contained_regular_file

    def track(*args, **kwargs):
        result = real_open(*args, **kwargs)
        opened.append(result.handle)
        return result

    monkeypatch.setattr(
        "backend.app.features.evidence.receipt_files.open_contained_regular_file", track
    )
    monkeypatch.setattr("backend.app.features.clips.manifest.open_contained_regular_file", track)
    with media.open("rb") as source:
        verified = verified_artifact(source)
        proof = ReceiptFiles.capture(ClipStore(root), receipt, verified)
        for _ in range(2):
            projection = proof.verify()
            assert projection.manifest.camera_id == "camera-1"
            assert projection.manifest.event_refs == ("event-1",)
            assert (
                projection.manifest_hash == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            )
            assert projection.manifest_size == manifest_path.stat().st_size
            assert projection.verified.identity == verified.identity
            assert source.tell() == 0 and not source.closed
        assert opened and all(handle.closed for handle in opened)
    assert source.closed


def test_location_and_hashed_descriptor_cannot_describe_different_manifests(clip, monkeypatch):
    root, manifest_path, _, _ = clip
    store = ClipStore(root)
    locate = store.locate_manifest

    def replace_after_location(clip_id):
        located = locate(clip_id)
        _mutate(manifest_path, "rewrite")
        return located

    monkeypatch.setattr(store, "locate_manifest", replace_after_location)
    with pytest.raises(ArtifactReceiptVerificationError, match="after location"):
        ReceiptManifest.capture(store, "clip-1")


@pytest.mark.parametrize("phase", ["after_preflight", "before_final_check"])
@pytest.mark.parametrize("subject", ["manifest", "media"])
@pytest.mark.parametrize("kind", ["rewrite", "inode", "symlink", "missing", "fifo"])
def test_file_change_aborts_existing_owner_without_audit_or_partial_rows(
    clip, database, phase, subject, kind
):
    root, manifest_path, media, receipt = clip
    before, callbacks = _rows(database), []
    target = manifest_path if subject == "manifest" else media
    hooks = ReceiptHooks(**{phase: lambda: _mutate(target, kind)})
    store = CompactArtifactReceiptStore(database, root, hooks)
    with media.open("rb") as source:
        verified = verified_artifact(source)
        with pytest.raises(ArtifactReceiptVerificationError):
            store.commit_verified(
                receipt, verified, after_write=lambda connection: callbacks.append(True)
            )
        assert not source.closed
    assert callbacks == [] and _rows(database) == before


@pytest.mark.parametrize("phase", ["after_preflight", "before_final_check"])
def test_same_parsed_manifest_with_different_bytes_cannot_change_the_receipt(clip, database, phase):
    root, manifest_path, _, receipt = clip
    before = _rows(database)
    original = read_manifest_file(manifest_path)
    store = CompactArtifactReceiptStore(
        database, root, ReceiptHooks(**{phase: lambda: _mutate(manifest_path, "whitespace")})
    )
    with pytest.raises(ArtifactReceiptVerificationError):
        store.commit(receipt)
    assert read_manifest_file(manifest_path) == original
    assert _rows(database) == before


def test_valid_receipt_and_matching_retry_keep_callback_and_hash_contracts(clip, database):
    root, manifest_path, _, receipt = clip
    store = CompactArtifactReceiptStore(database, root)
    callbacks = []

    def after_write(connection):
        assert connection.execute("SELECT count(*) FROM clips").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (1,)
        callbacks.append(True)

    assert store.commit(receipt, after_write=after_write) == receipt
    first = _rows(database)
    assert store.commit(receipt, after_write=after_write) == receipt
    assert callbacks == [True, True] and _rows(database) == first
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT manifest_sha256,manifest_size_bytes FROM clips"
        ).fetchone() == (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            manifest_path.stat().st_size,
        )


def test_unavailable_receipt_rechecks_manifest_after_mutation(clip, database, monkeypatch):
    from backend.app.features.evidence import compact_receipts

    root, manifest_path, media, _ = clip
    media.unlink()  # Unavailable receipts must not require a media descriptor.
    before = _rows(database)
    original = compact_receipts.commit_unavailable_primary
    mutations = []

    def change_manifest(*args, **kwargs):
        original(*args, **kwargs)
        connection = args[0]
        assert connection.execute(
            "SELECT lifecycle_state,failure_reason FROM incidents WHERE edge_event_id='event-1'"
        ).fetchone() == ("FAILED", "NO_FRAMES")
        assert connection.execute("SELECT state,reason FROM artifacts").fetchone() == (
            "UNAVAILABLE",
            "NO_FRAMES",
        )
        mutations.append(True)
        _mutate(manifest_path, "rewrite")

    monkeypatch.setattr(compact_receipts, "commit_unavailable_primary", change_manifest)
    with pytest.raises(ArtifactReceiptVerificationError):
        CompactArtifactReceiptStore(database, root).commit_unavailable("clip-1", "NO_FRAMES")
    assert mutations == [True]
    assert _rows(database) == before


def test_unavailable_receipt_succeeds_and_retries_without_media(clip, database):
    root, _, media, _ = clip
    media.unlink()
    store = CompactArtifactReceiptStore(database, root)
    store.commit_unavailable("clip-1", "NO_FRAMES")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM clips").fetchone() == (0,)
        assert connection.execute(
            "SELECT clip_id,state,reason,revision FROM artifacts"
        ).fetchone() == (
            None,
            "UNAVAILABLE",
            "NO_FRAMES",
            1,
        )
        assert connection.execute(
            "SELECT lifecycle_state,failure_reason,revision FROM incidents"
        ).fetchone() == ("FAILED", "NO_FRAMES", 2)
    first = _rows(database)
    store.commit_unavailable("clip-1", "NO_FRAMES")
    assert not media.exists() and _rows(database) == first


@pytest.mark.parametrize("content", [b"not-json", b"{}", b"[]"])
def test_invalid_manifest_content_is_rejected_without_leaking_handles(clip, monkeypatch, content):
    root, manifest_path, _, _ = clip
    captured = ReceiptManifest.capture(ClipStore(root), "clip-1")
    opened = []

    def track(*args, **kwargs):
        result = open_contained_regular_file(*args, **kwargs)
        opened.append(result.handle)
        return result

    monkeypatch.setattr(
        "backend.app.features.evidence.receipt_files.open_contained_regular_file", track
    )
    manifest_path.write_bytes(content)
    assert parse_manifest_bytes(content) is None and read_manifest_file(manifest_path) is None
    with pytest.raises(ArtifactReceiptVerificationError):
        captured.verify()
    assert opened and all(handle.closed for handle in opened)


@pytest.mark.parametrize("reader", ["manifest", "media", "contained"])
def test_regular_to_fifo_race_finishes_without_waiting_for_a_writer(clip, monkeypatch, reader):
    root, manifest_path, media, _ = clip
    path = manifest_path if reader == "manifest" else media
    real_open = os.open
    created, finished = Event(), Event()
    results = []

    def swap_before_open(name, flags, *args, **kwargs):
        if name == path.name and kwargs.get("dir_fd") is not None and not created.is_set():
            path.unlink()
            os.mkfifo(path)
            created.set()
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_before_open)

    def run():
        try:
            if reader == "manifest":
                result = read_manifest_file(path)
            elif reader == "media":
                result = open_receipt_media(root, path)
            else:
                result = open_contained_regular_file(root, path)
            if hasattr(result, "handle"):
                result.handle.close()
            results.append(result)
        except (OSError, ArtifactReceiptVerificationError) as error:
            results.append(error)
        finally:
            finished.set()

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        assert created.wait(1), "race did not reach descriptor acquisition"
        assert finished.wait(1), "regular-to-FIFO race blocked waiting for a writer"
        if reader == "manifest":
            assert results == [None]
        else:
            expected = ArtifactReceiptVerificationError if reader == "media" else FileNotFoundError
            assert len(results) == 1 and isinstance(results[0], expected)
    finally:
        # Also release the old blocking implementation so a regression cannot
        # strand the test process. This happens only after the bounded assertion.
        if created.is_set() and not finished.is_set():
            try:
                descriptor = real_open(path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as error:
                if error.errno != errno.ENXIO:
                    raise
            else:
                os.close(descriptor)
        worker.join(2)
        assert not worker.is_alive()
