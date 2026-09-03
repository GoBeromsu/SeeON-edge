from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.catalog import CatalogStore
from backend.app.features.clips.compact_listing import CompactClipListing, CompactClipQuery
from backend.app.features.clips.manifest import ClipManifest
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.compact_receipt_sql import (
    ClipProjection,
    commit_clip,
    commit_primary_artifact,
)
from backend.app.features.evidence.compact_receipts import (
    CompactArtifactReceiptStore,
    CompactReceiptHooks,
    CompactReceiptMissingIncidentError,
)
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptConflictError,
    ArtifactReceiptStore,
    ArtifactReceiptVerificationError,
    CatalogArtifactReceiptStore,
    VerifiedArtifact,
    verify_artifact,
)
from backend.app.features.evidence.relay_projection import RelayEvent, RelayEvidenceProjection
from backend.app.features.runtime_settings.store import RuntimeSettingsStore
from backend.app.main import create_app, no_lifespan
from shared.events.evidence_export_contract import ClipReceipt
from tests_support.compact_authority_db import prepare_compact_database

TOKEN = "relay-token"
EVENT_ID = "00000000-0000-4000-8000-000000000001"


class SqliteReceiptStore:
    """Test-only implementation of the migrated backend receipt table."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute(
            "CREATE TABLE api_artifact_receipts ("
            "artifact_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, "
            "accepted INTEGER NOT NULL CHECK (accepted IN (0, 1))) STRICT"
        )

    def commit(self, receipt: ArtifactReceipt) -> ArtifactReceipt:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT sha256, size_bytes, accepted FROM api_artifact_receipts "
                "WHERE artifact_id=?",
                (receipt.artifact_id,),
            ).fetchone()
            if row is None:
                self.connection.execute(
                    "INSERT INTO api_artifact_receipts VALUES (?, ?, ?, ?)",
                    (
                        receipt.artifact_id,
                        receipt.sha256,
                        receipt.size_bytes,
                        int(receipt.accepted),
                    ),
                )
                self.connection.execute("COMMIT")
                return receipt
            committed = ArtifactReceipt(receipt.artifact_id, row[0], row[1], bool(row[2]))
            if committed != receipt:
                _raise_receipt_conflict()
            self.connection.execute("COMMIT")
            return committed  # noqa: TRY300
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def get(self, artifact_id: str) -> ArtifactReceipt | None:
        row = self.connection.execute(
            "SELECT sha256, size_bytes, accepted FROM api_artifact_receipts WHERE artifact_id=?",
            (artifact_id,),
        ).fetchone()
        return None if row is None else ArtifactReceipt(artifact_id, row[0], row[1], bool(row[2]))


class LocalBackend:
    def for_camera(self, _camera_id: str) -> LocalBackend:
        return self

    def probe_capabilities(self, _camera_id: str) -> object:
        raise AssertionError("not used")

    def publish_ready(self, request: object, media: object) -> ClipReceipt:
        return ClipReceipt("clip-1", "READY", 1, _receipt(b"verified video").sha256, 14)

    def report_unavailable(self, request: object) -> ClipReceipt:
        raise AssertionError("not used")


def _receipt(data: bytes, *, artifact_id: str = "clip-1") -> ArtifactReceipt:
    return ArtifactReceipt(artifact_id, hashlib.sha256(data).hexdigest(), len(data))


def _projection(receipt: ArtifactReceipt, *, video_available: bool = True) -> ClipProjection:
    manifest = ClipManifest(
        clip_id=receipt.artifact_id,
        camera_id="camera-1",
        event_ref="event-1",
        event_type="fall",
        started_at="2026-07-06T00:00:00Z",
        duration_s=1.0,
        codec="h264",
        path="clips/clip-1/clip.mp4",
        video_available=video_available,
        video_error=None if video_available else "NO_FRAMES",
        finalized=True,
    )
    return ClipProjection(
        receipt=receipt,
        verified=VerifiedArtifact(None, receipt.sha256, receipt.size_bytes, 1, 1),  # type: ignore[arg-type]
        manifest=manifest,
        manifest_relpath="clips/clip-1/manifest.json",
        media_relpath="clips/clip-1/clip.mp4",
        manifest_hash="a" * 64,
        manifest_size=1,
    )


def _raise_receipt_conflict() -> None:
    raise ArtifactReceiptConflictError("immutable artifact receipt fields conflict")


def test_first_commit_and_identical_retry_are_durable_and_idempotent(tmp_path: Path) -> None:
    store = SqliteReceiptStore(tmp_path / "receipts.sqlite3")
    receipt = _receipt(b"verified video")

    assert store.commit(receipt) == receipt
    assert store.commit(receipt) == receipt
    assert store.get(receipt.artifact_id) == receipt


def test_immutable_receipt_conflict_is_typed(tmp_path: Path) -> None:
    store = SqliteReceiptStore(tmp_path / "receipts.sqlite3")
    store.commit(_receipt(b"first"))

    with pytest.raises(ArtifactReceiptConflictError):
        store.commit(_receipt(b"changed"))


def test_catalog_receipt_store_commits_and_retries_durably(tmp_path: Path) -> None:
    catalog = CatalogStore.open(tmp_path / "catalog.sqlite3")
    store = CatalogArtifactReceiptStore(catalog)
    receipt = _receipt(b"verified video")
    try:
        assert store.commit(receipt) == receipt
        assert store.commit(receipt) == receipt
        with pytest.raises(ArtifactReceiptConflictError):
            store.commit(_receipt(b"different video"))
    finally:
        catalog.close()


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert response.status_code == 204


def _payload(data: bytes) -> dict[str, object]:
    return {
        "state": "READY",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "event_refs": [EVENT_ID],
        "state_version": 1,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "mime_type": "video/mp4",
        "codec": "h264",
        "duration_ms": 1000,
        "clip_start_at": "2026-07-16T00:00:00Z",
        "clip_end_at": "2026-07-16T00:00:01Z",
        "finalized_at": "2026-07-16T00:00:02Z",
    }


def _client(tmp_path: Path, store: ArtifactReceiptStore) -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = TOKEN
    app.state.artifact_receipt_store = store
    app.state.clip_store_root = tmp_path / "clip-store"
    app.state.clip_store = ClipStore(app.state.clip_store_root)
    database = tmp_path / "catalog.sqlite3"
    prepare_compact_database(database)
    registry = CameraRegistryStore(database)
    registry.create(
        camera_id="camera-1",
        label="Camera 1",
        rtsp_url="rtsp://camera/1",
        space_id="facility-1",
        status="online",
        backend_camera_id="hub-camera-1",
    )
    app.state.camera_registry = registry
    app.state.backend_evidence_client = LocalBackend()
    settings = RuntimeSettingsStore(database)
    settings.set_clip_export_enabled(True)
    app.state.runtime_settings_store = settings
    if isinstance(store, CompactArtifactReceiptStore):
        RelayEvidenceProjection(store._database_path).project_event(  # noqa: SLF001
            RelayEvent(
                edge_event_id="event-1",
                event_type="fall",
                probability=0.8,
                detected_at="2026-07-06T00:00:00Z",
                camera_id="camera-1",
                facility_id="facility-1",
                resident_id=None,
                evidence=None,
                audit=None,
            )
        )
    return TestClient(app)


def _media(
    tmp_path: Path, data: bytes, *, event_refs: list[str] | None = None
) -> Path:
    path = tmp_path / "clip-store" / "clips" / "clip-1" / "clip.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    (path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-1",
                "camera_id": "camera-1",
                "event_ref": "event-1",
                **({"event_refs": event_refs} if event_refs is not None else {}),
                "event_type": "fall",
                "started_at": "2026-07-06T00:00:00Z",
                "duration_s": 1.0,
                "codec": "h264",
                "path": "clips/clip-1",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_compact_receipt_commits_clip_and_primary_artifact(tmp_path: Path) -> None:
    # Given: verified local media and an incident matching its manifest event reference.
    data = b"verified video"
    _media(tmp_path, data)
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    RelayEvidenceProjection(database).project_event(
        RelayEvent(
            edge_event_id="event-1",
            event_type="fall",
            probability=0.8,
            detected_at="2026-07-06T00:00:00Z",
            camera_id="camera-1",
            facility_id="facility-1",
            resident_id=None,
            evidence=None,
            audit=None,
        )
    )
    store = CompactArtifactReceiptStore(database, tmp_path / "clip-store")
    receipt = _receipt(data)

    # When: the authenticated receipt is committed and reopened.
    assert store.commit(receipt) == receipt
    reopened = CompactArtifactReceiptStore(database, tmp_path / "clip-store")

    # Then: clips owns publication and artifacts owns PRIMARY_CLIP projection.
    assert reopened.get("clip-1") == receipt
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT publish_state FROM clips WHERE clip_id='clip-1'"
        ).fetchone() == ("PUBLISHED",)
        assert connection.execute(
            "SELECT clip_id,state FROM artifacts WHERE kind='PRIMARY_CLIP'"
        ).fetchone() == ("clip-1", "AVAILABLE")
        assert connection.execute(
            "SELECT lifecycle_state FROM incidents WHERE edge_event_id='event-1'"
        ).fetchone() == ("COMPLETE",)
        created_at, updated_at = connection.execute(
            "SELECT created_at, updated_at FROM incidents WHERE edge_event_id='event-1'"
        ).fetchone()
        assert created_at <= updated_at


def test_compact_receipt_completes_every_manifest_event_reference(tmp_path: Path) -> None:
    second_event = "event-2"
    data = b"verified video"
    _media(tmp_path, data, event_refs=["event-1", second_event])
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    projection = RelayEvidenceProjection(database)
    for event_id in ("event-1", second_event):
        projection.project_event(
            RelayEvent(
                edge_event_id=event_id,
                event_type="fall",
                probability=0.8,
                detected_at="2026-07-06T00:00:00Z",
                camera_id="camera-1",
                facility_id="facility-1",
                resident_id=None,
                evidence=None,
                audit=None,
            )
        )

    CompactArtifactReceiptStore(database, tmp_path / "clip-store").commit(_receipt(data))

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM clips").fetchone() == (1,)
        artifacts = connection.execute(
            "SELECT artifact_id, clip_id FROM artifacts WHERE kind = 'PRIMARY_CLIP' "
            "ORDER BY artifact_id"
        ).fetchall()
        assert len(artifacts) == 2
        assert len({row[0] for row in artifacts}) == 2
        assert {row[1] for row in artifacts} == {"clip-1"}
        assert connection.execute(
            "SELECT count(*) FROM incidents WHERE lifecycle_state = 'COMPLETE'"
        ).fetchone() == (2,)


def test_missing_manifest_incident_rolls_back_all_receipt_projection(tmp_path: Path) -> None:
    data = b"verified video"
    _media(tmp_path, data, event_refs=["event-1", "event-missing"])
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    RelayEvidenceProjection(database).project_event(
        RelayEvent(
            edge_event_id="event-1",
            event_type="fall",
            probability=0.8,
            detected_at="2026-07-06T00:00:00Z",
            camera_id="camera-1",
            facility_id="facility-1",
            resident_id=None,
            evidence=None,
            audit=None,
        )
    )

    with pytest.raises(CompactReceiptMissingIncidentError, match="event-missing"):
        CompactArtifactReceiptStore(database, tmp_path / "clip-store").commit(_receipt(data))
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM clips").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)
        assert connection.execute(
            "SELECT lifecycle_state FROM incidents WHERE edge_event_id = 'event-1'"
        ).fetchone() == ("OPEN",)


def test_primary_artifact_id_accepts_maximum_legal_inputs() -> None:
    clip_id = "c" * 128
    edge_event_id = "e" * 128

    artifact_id = "primary:" + hashlib.sha256(
        (clip_id + "\x1f" + edge_event_id).encode()
    ).hexdigest()[:32]

    assert artifact_id.startswith("primary:")
    assert len(artifact_id) == 40
    assert artifact_id == (
        "primary:"
        + hashlib.sha256((clip_id + "\x1f" + edge_event_id).encode()).hexdigest()[:32]
    )


def test_compact_receipt_replay_keeps_one_stable_digest_artifact(tmp_path: Path) -> None:
    data = b"verified video"
    _media(tmp_path, data)
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    RelayEvidenceProjection(database).project_event(
        RelayEvent(
            edge_event_id="event-1",
            event_type="fall",
            probability=0.8,
            detected_at="2026-07-06T00:00:00Z",
            camera_id="camera-1",
            facility_id="facility-1",
            resident_id=None,
            evidence=None,
            audit=None,
        )
    )
    store = CompactArtifactReceiptStore(database, tmp_path / "clip-store")
    receipt = _receipt(data)

    assert store.commit(receipt) == store.commit(receipt) == receipt
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT artifact_id FROM artifacts WHERE kind = 'PRIMARY_CLIP'"
        ).fetchall()
    expected_id = "primary:" + hashlib.sha256(b"clip-1\x1fevent-1").hexdigest()[:32]
    assert rows == [(expected_id,)]


def test_public_receipt_rejects_digest_identity_collision_without_partial_state(
    tmp_path: Path,
) -> None:
    data = b"verified video"
    _media(tmp_path, data)
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    relay = RelayEvidenceProjection(database)
    for event_id in ("event-1", "event-2"):
        relay.project_event(
            RelayEvent(
                edge_event_id=event_id,
                event_type="fall",
                probability=0.8,
                detected_at="2026-07-06T00:00:00Z",
                camera_id="camera-1",
                facility_id="facility-1",
                resident_id=None,
                evidence=None,
                audit=None,
            )
        )
    receipt = _receipt(data)
    expected_id = "primary:" + hashlib.sha256(b"clip-1\x1fevent-1").hexdigest()[:32]
    with sqlite3.connect(database) as connection:
        commit_clip(connection, _projection(receipt))
        connection.execute(
            """
            INSERT INTO artifacts (
                incident_id, kind, artifact_id, clip_id, state, contained_relpath,
                content_sha256, size_bytes, mime_type, codec, revision, created_at, updated_at
            ) VALUES ('incident:event-2', 'PRIMARY_CLIP', ?, 'clip-1', 'AVAILABLE',
                      'clips/clip-1/clip.mp4', ?, ?, 'video/mp4', 'h264', 1,
                      '2026-07-06T00:00:00Z', '2026-07-06T00:00:00Z')
            """,
            (expected_id, receipt.sha256, receipt.size_bytes),
        )

    with pytest.raises(ArtifactReceiptConflictError, match="identity conflicts"):
        CompactArtifactReceiptStore(database, tmp_path / "clip-store").commit(receipt)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM artifacts WHERE incident_id = 'incident:event-1'"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT lifecycle_state FROM incidents WHERE incident_id = 'incident:event-1'"
        ).fetchone() == ("OPEN",)


def test_legacy_primary_artifact_replay_is_a_clean_noop(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    RelayEvidenceProjection(database).project_event(
        RelayEvent(
            edge_event_id="event-1",
            event_type="fall",
            probability=0.8,
            detected_at="2026-07-06T00:00:00Z",
            camera_id="camera-1",
            facility_id="facility-1",
            resident_id=None,
            evidence=None,
            audit=None,
        )
    )
    data = b"verified video"
    receipt = _receipt(data)
    projection = _projection(receipt)
    with sqlite3.connect(database) as connection:
        commit_clip(connection, projection)
        connection.execute(
            """
            INSERT INTO artifacts (
                incident_id, kind, artifact_id, clip_id, state, contained_relpath,
                content_sha256, size_bytes, mime_type, codec, revision, created_at, updated_at
            ) VALUES ('incident:event-1', 'PRIMARY_CLIP', 'primary:clip-1', 'clip-1',
                      'AVAILABLE', 'clips/clip-1/clip.mp4', ?, ?, 'video/mp4', 'h264',
                      1, '2026-07-06T00:00:00Z', '2026-07-06T00:00:00Z')
            """,
            (receipt.sha256, receipt.size_bytes),
        )
        commit_primary_artifact(connection, "incident:event-1", "event-1", projection)
        assert connection.execute(
            "SELECT lifecycle_state FROM incidents WHERE incident_id = 'incident:event-1'"
        ).fetchone() == ("COMPLETE",)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT artifact_id FROM artifacts WHERE incident_id = 'incident:event-1'"
        ).fetchone() == ("primary:clip-1",)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (1,)


def test_primary_artifact_conflict_rolls_back_without_lifecycle_change(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    RelayEvidenceProjection(database).project_event(
        RelayEvent(
            edge_event_id="event-1",
            event_type="fall",
            probability=0.8,
            detected_at="2026-07-06T00:00:00Z",
            camera_id="camera-1",
            facility_id="facility-1",
            resident_id=None,
            evidence=None,
            audit=None,
        )
    )
    receipt = _receipt(b"verified video")
    projection = _projection(receipt)
    with sqlite3.connect(database) as connection:
        commit_clip(connection, projection)
        connection.execute(
            """
            INSERT INTO artifacts (
                incident_id, kind, clip_id, state, reason, revision, created_at, updated_at
            ) VALUES ('incident:event-1', 'PRIMARY_CLIP', 'clip-1', 'UNAVAILABLE',
                      'NO_FRAMES', 1, '2026-07-06T00:00:00Z', '2026-07-06T00:00:00Z')
            """,
        )
        with pytest.raises(ArtifactReceiptConflictError, match="primary clip artifact conflicts"):
            commit_primary_artifact(connection, "incident:event-1", "event-1", projection)
        assert connection.execute(
            "SELECT lifecycle_state FROM incidents WHERE incident_id = 'incident:event-1'"
        ).fetchone() == ("OPEN",)
        assert connection.execute(
            "SELECT state FROM artifacts WHERE incident_id = 'incident:event-1'"
        ).fetchone() == ("UNAVAILABLE",)


def test_unavailable_primary_marks_incident_failed_with_reason(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    RelayEvidenceProjection(database).project_event(
        RelayEvent(
            edge_event_id="event-1",
            event_type="fall",
            probability=0.8,
            detected_at="2026-07-06T00:00:00Z",
            camera_id="camera-1",
            facility_id="facility-1",
            resident_id=None,
            evidence=None,
            audit=None,
        )
    )
    projection = _projection(_receipt(b"unavailable"), video_available=False)
    with sqlite3.connect(database) as connection:
        commit_clip(connection, projection)
        commit_primary_artifact(connection, "incident:event-1", "event-1", projection)
        assert connection.execute(
            "SELECT lifecycle_state, failure_reason FROM incidents "
            "WHERE incident_id = 'incident:event-1'"
        ).fetchone() == ("FAILED", "NO_FRAMES")


def test_compact_receipt_rejects_equal_size_different_hash_without_partial_state(
    tmp_path: Path,
) -> None:
    # Given: current media bytes differ from an equal-size declared receipt.
    actual = b"actual-media"
    declared = b"bogus-media!"
    assert len(actual) == len(declared)
    _media(tmp_path, actual)
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    store = CompactArtifactReceiptStore(database, tmp_path / "clip-store")

    # When: the compact store independently binds the receipt to current bytes.
    with pytest.raises(ArtifactReceiptVerificationError):
        store.commit(_receipt(declared))

    # Then: neither compact authority contains a caller-declared partial fact.
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM clips").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)


def test_compact_receipt_failure_does_not_promote_existing_waiting_clip(
    tmp_path: Path,
) -> None:
    # Given: manifest reconciliation created a WAITING compact clip row.
    actual = b"actual-media"
    declared = b"bogus-media!"
    _media(tmp_path, actual)
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    listing = CompactClipListing(database)
    listing.rebuild_and_page(
        ClipStore(tmp_path / "clip-store"),
        CompactClipQuery(None, None, 10, None),
    )

    # When: a same-size, wrong-hash receipt is rejected.
    with pytest.raises(ArtifactReceiptVerificationError):
        CompactArtifactReceiptStore(database, tmp_path / "clip-store").commit(
            _receipt(declared)
        )

    # Then: publication remains WAITING and no PRIMARY_CLIP appears.
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT publish_state FROM clips WHERE clip_id='clip-1'"
        ).fetchone() == ("WAITING",)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)


@pytest.mark.parametrize("invalid_path", ["symlink", "missing"])
def test_compact_receipt_rejects_untrusted_media_path_without_partial_state(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    # Given: direct compact commit sees either no media or a symlinked final path.
    data = b"verified video"
    media = _media(tmp_path, data)
    if invalid_path == "symlink":
        target = media.with_name("target.mp4")
        target.write_bytes(data)
        media.unlink()
        media.symlink_to(target.name)
    else:
        media.unlink()
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)

    # When/Then: no untrusted pathname can produce publication state.
    with pytest.raises(ArtifactReceiptVerificationError):
        CompactArtifactReceiptStore(database, tmp_path / "clip-store").commit(_receipt(data))
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM clips").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)


@pytest.mark.parametrize("swap_kind", ["inode", "symlink", "missing"])
def test_real_route_rejects_media_swap_before_compact_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_kind: str,
) -> None:
    # Given: route verification has opened the declared inode.
    original = b"verified video"
    replacement = b"tampered bytes"
    assert len(original) == len(replacement)
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    store = CompactArtifactReceiptStore(database, tmp_path / "clip-store")
    client = _client(tmp_path, store)
    media = _media(tmp_path, original)
    original_inode = media.stat().st_ino
    original_commit = CompactArtifactReceiptStore.commit_verified
    observed_inode: int | None = None
    verified_handle: BinaryIO | None = None

    def swap_then_commit(
        compact_store: CompactArtifactReceiptStore,
        receipt: ArtifactReceipt,
        route_verified: VerifiedArtifact,
        *,
        after_write: Callable[[sqlite3.Connection], None] | None = None,
    ) -> ArtifactReceipt:
        nonlocal observed_inode, verified_handle
        verified_handle = route_verified.handle
        replacement_path = media.with_name("replacement.mp4")
        replacement_path.write_bytes(replacement)
        match swap_kind:
            case "inode":
                os.replace(replacement_path, media)
            case "symlink":
                media.unlink()
                media.symlink_to(replacement_path.name)
            case "missing":
                media.unlink()
                replacement_path.unlink()
            case unreachable:
                raise AssertionError(unreachable)
        if media.exists():
            observed_inode = media.stat().st_ino
        return original_commit(
            compact_store, receipt, route_verified, after_write=after_write
        )

    monkeypatch.setattr(
        CompactArtifactReceiptStore,
        "commit_verified",
        swap_then_commit,
    )

    # When: the real relay route crosses verification -> compact commit.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_payload(original),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: pathname/inode drift is rejected and no publication fact commits.
    assert response.status_code == 409
    assert verified_handle is not None and verified_handle.closed
    if observed_inode is not None:
        assert observed_inode != original_inode
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM clips").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)


def test_real_route_rejects_swap_after_preflight_before_transaction(
    tmp_path: Path,
) -> None:
    # Given: pathname identity is captured, then equal-size replacement occurs before DB open.
    original = b"verified video"
    replacement = b"tampered bytes"
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    media = _media(tmp_path, original)
    inode_proof: tuple[int, int] | None = None

    def swap_after_preflight() -> None:
        nonlocal inode_proof
        replacement_path = media.with_name("replacement.mp4")
        replacement_path.write_bytes(replacement)
        old_inode = media.stat().st_ino
        os.replace(replacement_path, media)
        inode_proof = old_inode, media.stat().st_ino

    store = CompactArtifactReceiptStore(
        database,
        tmp_path / "clip-store",
        CompactReceiptHooks(after_preflight=swap_after_preflight),
    )
    client = _client(tmp_path, store)

    # When: the real route reaches the exact post-preflight/pre-transaction hook.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_payload(original),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: the first in-transaction guard rejects replacement before any SQL write.
    assert response.status_code == 409
    assert inode_proof is not None and inode_proof[0] != inode_proof[1]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM clips").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)


@pytest.mark.parametrize("swap_kind", ["inode", "symlink", "missing"])
def test_real_route_rolls_back_swap_during_receipt_transaction(
    tmp_path: Path,
    swap_kind: str,
) -> None:
    # Given: SQL writes occur, then the current pathname is replaced before commit.
    original = b"verified video"
    replacement = b"tampered bytes"
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    media = _media(tmp_path, original)
    inode_proof: tuple[int, int] | None = None

    def swap_before_final_check() -> None:
        nonlocal inode_proof
        replacement_path = media.with_name("replacement.mp4")
        replacement_path.write_bytes(replacement)
        old_inode = media.stat().st_ino
        match swap_kind:
            case "inode":
                os.replace(replacement_path, media)
            case "symlink":
                media.unlink()
                media.symlink_to(replacement_path.name)
            case "missing":
                media.unlink()
                replacement_path.unlink()
            case unreachable:
                raise AssertionError(unreachable)
        if media.exists():
            inode_proof = old_inode, media.stat().st_ino

    store = CompactArtifactReceiptStore(
        database,
        tmp_path / "clip-store",
        CompactReceiptHooks(before_final_check=swap_before_final_check),
    )
    client = _client(tmp_path, store)

    # When: the deterministic hook swaps after SQL but before transaction commit.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_payload(original),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: the final in-transaction guard rolls back every compact write.
    assert response.status_code == 409
    if inode_proof is not None:
        assert inode_proof[0] != inode_proof[1]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM clips").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)


def test_real_route_valid_compact_receipt_commits_and_closes_descriptor(
    tmp_path: Path,
) -> None:
    # Given: valid bytes remain on the same pathname for both transaction guards.
    data = b"verified video"
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    media = _media(tmp_path, data)
    hook_order: list[str] = []
    captured_handle: BinaryIO | None = None

    class ObservedStore(CompactArtifactReceiptStore):
        def commit_verified(
            self,
            receipt: ArtifactReceipt,
            route_verified: VerifiedArtifact,
            *,
            after_write: Callable[[sqlite3.Connection], None] | None = None,
        ) -> ArtifactReceipt:
            nonlocal captured_handle
            captured_handle = route_verified.handle
            return super().commit_verified(
                receipt, route_verified, after_write=after_write
            )

    store = ObservedStore(
        database,
        tmp_path / "clip-store",
        CompactReceiptHooks(
            after_preflight=lambda: hook_order.append("after-preflight"),
            before_final_check=lambda: hook_order.append("before-final-check"),
        ),
    )
    client = _client(tmp_path, store)

    # When: the real route completes a descriptor-bound compact receipt.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_payload(data),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: both timing seams execute, computed identity commits, and the FD closes.
    assert response.status_code == 200
    assert hook_order == ["after-preflight", "before-final-check"]
    assert captured_handle is not None and captured_handle.closed
    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            "SELECT media_sha256,media_size_bytes,publish_state FROM clips "
            "WHERE clip_id='clip-1'"
        ).fetchone()
    assert stored == (_receipt(data).sha256, media.stat().st_size, "PUBLISHED")


def test_relay_verifies_before_durable_receipt_and_never_writes_media(tmp_path: Path) -> None:
    data = b"verified video"
    media = _media(tmp_path, data)
    store = SqliteReceiptStore(tmp_path / "receipts.sqlite3")
    client = _client(tmp_path, store)

    response = client.put(
        "/api/v1/relay/clips/clip-1", json=_payload(data), headers={"X-Edge-Relay-Token": TOKEN}
    )

    assert response.status_code == 200
    assert store.get("clip-1") == _receipt(data)
    assert media.read_bytes() == data


@pytest.mark.parametrize("declared", [b"wrong-size", b"wrong-hash"])
def test_relay_verification_failure_is_distinct_from_conflict(
    tmp_path: Path, declared: bytes
) -> None:
    data = b"verified video"
    _media(tmp_path, data)
    store = SqliteReceiptStore(tmp_path / "receipts.sqlite3")
    client = _client(tmp_path, store)

    response = client.put(
        "/api/v1/relay/clips/clip-1", json=_payload(declared), headers={"X-Edge-Relay-Token": TOKEN}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "clip media mismatch"
    assert store.get("clip-1") is None


def test_verification_failure_is_typed(tmp_path: Path) -> None:
    artifact = tmp_path / "clip.mp4"
    artifact.write_bytes(b"actual bytes")

    with pytest.raises(ArtifactReceiptVerificationError):
        verify_artifact(artifact, _receipt(b"different bytes"))


def test_a_receipt_that_exists_must_be_accepted_and_must_match(tmp_path: Path) -> None:
    """The receipt binds the bytes; it does not license the viewing.

    This used to also require a receipt to EXIST before an operator could play
    anything. A receipt is only committed after a successful upstream export,
    which needs clip export enabled (Hub-owned config, off by default) and a
    Hub-issued camera id, so on a real deployment none was ever written and
    every clip became permanently unplayable -- verified HEVC on disk, listed
    as available, and the browser answering "영상을 재생하지 못했습니다" forever.

    What the receipt actually guarantees -- that served bytes are the recorded
    ones, and that a refused artifact stays refused -- is unchanged below.
    """
    data = b"verified video"
    media = _media(tmp_path, data)
    store = SqliteReceiptStore(tmp_path / "receipts.sqlite3")
    client = _client(tmp_path, store)
    _login(client)

    # No receipt yet: local evidence is still reviewable.
    unrecorded = client.get("/api/v1/clips/clip-1/video")
    assert unrecorded.status_code == 200
    assert unrecorded.content == data

    store.commit(ArtifactReceipt("clip-1", _receipt(data).sha256, len(data), accepted=False))
    unaccepted = client.get("/api/v1/clips/clip-1/video")
    assert unaccepted.status_code == 404

    store.connection.execute(
        "UPDATE api_artifact_receipts SET accepted=1 WHERE artifact_id='clip-1'"
    )
    served = client.get("/api/v1/clips/clip-1/video")
    assert served.status_code == 200
    assert served.content == data

    media.write_bytes(b"drifted")
    drifted = client.get("/api/v1/clips/clip-1/video")
    assert drifted.status_code == 409
