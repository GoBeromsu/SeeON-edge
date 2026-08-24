from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.catalog import CatalogStore
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptConflictError,
    ArtifactReceiptStore,
    ArtifactReceiptVerificationError,
    CatalogArtifactReceiptStore,
    verify_artifact,
)
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
    return TestClient(app)


def _media(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "clip-store" / "clips" / "clip-1" / "clip.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    (path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-1",
                "camera_id": "camera-1",
                "event_ref": "event-1",
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
