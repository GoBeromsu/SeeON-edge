from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import psycopg
import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.event_outbox import EventOutbox, OutboxBudget
from backend.app.features.evidence.postgres_receipts import PostgresArtifactReceiptStore
from backend.app.features.evidence.receipt_files import ReceiptHooks
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptConflictError,
    ArtifactReceiptStore,
    ArtifactReceiptVerificationError,
    VerifiedArtifact,
    verify_artifact,
)
from backend.app.features.evidence.relay_projection import RelayEvent
from backend.app.features.runtime_settings.store import RuntimeSettingsStore
from backend.app.main import create_app, no_lifespan
from backend.app.shared.postgres_dashboard_credentials import PostgresDashboardCredentialsStore
from shared.events.evidence_export_contract import ClipReceipt
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

TOKEN = "relay-token"
# The wire payload carries Hub-facing UUIDv4 references while the receipt owner
# resolves incidents from the clip manifest's own event identity below. The two
# are deliberately unequal; nothing here asserts metadata equality between them.
EVENT_ID = "00000000-0000-4000-8000-000000000001"
# One accepted event per native client; an explicit test budget, not a deployment policy.
TEST_OUTBOX_BUDGET = OutboxBudget(4, 64 * 1024)


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


def _client(tmp_path: Path, store: ArtifactReceiptStore, sandbox: ProductSandbox) -> TestClient:
    app = create_app(lifespan=no_lifespan)
    runtime = PostgresAuditRuntime(
        PostgresAuditStore(sandbox.database, sandbox.authority),
        maximum_snapshot_age_sec=10.0,
        clock=lambda: 0.0,
    )
    assert runtime.verify_once()
    assert runtime.start_session_once()
    app.state.audit_runtime = runtime
    app.state.dashboard_credentials_store = PostgresDashboardCredentialsStore(
        sandbox.database, sandbox.authority
    )
    app.state.edge_relay_token = TOKEN
    app.state.artifact_receipt_store = store
    app.state.clip_store_root = tmp_path / "clip-store"
    app.state.clip_store = ClipStore(app.state.clip_store_root)
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
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
    settings = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    settings.set_clip_export_enabled(True)
    app.state.runtime_settings_store = settings
    if isinstance(store, PostgresArtifactReceiptStore):
        # The manifest incident comes from the real acceptance owner sharing this
        # runtime, database and authority, under an explicit small test budget.
        EventOutbox(
            sandbox.database, sandbox.authority, TEST_OUTBOX_BUDGET, audit_runtime=runtime
        ).accept(
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
            ),
            backend_camera_id=None,
            forward=False,
        )
    return TestClient(app)


def _native_store(
    tmp_path: Path, sandbox: ProductSandbox, hooks: ReceiptHooks | None = None
) -> PostgresArtifactReceiptStore:
    return PostgresArtifactReceiptStore(
        sandbox.database, sandbox.authority, tmp_path / "clip-store", hooks
    )


def _counts(sandbox: ProductSandbox) -> tuple[int, ...]:
    """Read published clips, artifacts and audit rows on the independent connection."""
    return tuple(
        sandbox.admin.execute("SELECT count(*) FROM " + table).fetchone()[0]
        for table in ("clips", "artifacts", "audit_events")
    )


def _media(
    tmp_path: Path,
    data: bytes,
    *,
    event_refs: list[str] | None = None,
    clip_id: str = "clip-1",
    event_id: str = "event-1",
) -> Path:
    path = tmp_path / "clip-store" / "clips" / clip_id / "clip.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    (path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "camera_id": "camera-1",
                "event_ref": event_id,
                **({"event_refs": event_refs} if event_refs is not None else {}),
                "event_type": "fall",
                "started_at": "2026-07-06T00:00:00Z",
                "duration_s": 1.0,
                "codec": "h264",
                "path": f"clips/{clip_id}",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("swap_kind", ["inode", "symlink", "missing"])
def test_real_route_rejects_media_swap_before_native_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_kind: str,
    postgres_product_sandbox: ProductSandbox,
) -> None:
    # Given: route verification has opened the declared inode.
    original = b"verified video"
    replacement = b"tampered bytes"
    assert len(original) == len(replacement)
    sandbox = postgres_product_sandbox
    store = _native_store(tmp_path, sandbox)
    client = _client(tmp_path, store, sandbox)
    media = _media(tmp_path, original)
    before = _counts(sandbox)
    original_inode = media.stat().st_ino
    original_commit = PostgresArtifactReceiptStore.commit_verified
    observed_inode: int | None = None
    verified_handle: BinaryIO | None = None

    def swap_then_commit(
        native_store: PostgresArtifactReceiptStore,
        receipt: ArtifactReceipt,
        route_verified: VerifiedArtifact,
        *,
        after_write: Callable[[psycopg.Connection], None] | None = None,
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
        return original_commit(native_store, receipt, route_verified, after_write=after_write)

    monkeypatch.setattr(
        PostgresArtifactReceiptStore,
        "commit_verified",
        swap_then_commit,
    )

    # When: the real relay route crosses verification -> native commit.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_payload(original),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: pathname/inode drift is rejected and no publication or audit fact commits.
    assert response.status_code == 409
    assert verified_handle is not None and verified_handle.closed
    if observed_inode is not None:
        assert observed_inode != original_inode
    assert _counts(sandbox) == before


def test_real_route_rejects_swap_after_preflight_before_transaction(
    tmp_path: Path,
    postgres_product_sandbox: ProductSandbox,
) -> None:
    # Given: pathname identity is captured, then equal-size replacement occurs before DB open.
    original = b"verified video"
    replacement = b"tampered bytes"
    sandbox = postgres_product_sandbox
    media = _media(tmp_path, original)
    inode_proof: tuple[int, int] | None = None

    def swap_after_preflight() -> None:
        nonlocal inode_proof
        replacement_path = media.with_name("replacement.mp4")
        replacement_path.write_bytes(replacement)
        old_inode = media.stat().st_ino
        os.replace(replacement_path, media)
        inode_proof = old_inode, media.stat().st_ino

    store = _native_store(tmp_path, sandbox, ReceiptHooks(after_preflight=swap_after_preflight))
    client = _client(tmp_path, store, sandbox)
    before = _counts(sandbox)

    # When: the real route reaches the exact post-preflight/pre-transaction hook.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_payload(original),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: the first in-transaction guard rejects replacement before any SQL write.
    assert response.status_code == 409
    assert inode_proof is not None and inode_proof[0] != inode_proof[1]
    assert _counts(sandbox) == before


@pytest.mark.parametrize("swap_kind", ["inode", "symlink", "missing"])
def test_real_route_rolls_back_swap_during_receipt_transaction(
    tmp_path: Path,
    swap_kind: str,
    postgres_product_sandbox: ProductSandbox,
) -> None:
    # Given: SQL writes occur, then the current pathname is replaced before commit.
    original = b"verified video"
    replacement = b"tampered bytes"
    sandbox = postgres_product_sandbox
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

    store = _native_store(
        tmp_path, sandbox, ReceiptHooks(before_final_check=swap_before_final_check)
    )
    client = _client(tmp_path, store, sandbox)
    before = _counts(sandbox)

    # When: the deterministic hook swaps after SQL but before transaction commit.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_payload(original),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: the final in-transaction guard rolls back every native write, audit included.
    assert response.status_code == 409
    if inode_proof is not None:
        assert inode_proof[0] != inode_proof[1]
    assert _counts(sandbox) == before


def test_real_route_valid_native_receipt_commits_and_closes_descriptor(
    tmp_path: Path,
    postgres_product_sandbox: ProductSandbox,
) -> None:
    # Given: valid bytes remain on the same pathname for both transaction guards.
    data = b"verified video"
    sandbox = postgres_product_sandbox
    media = _media(tmp_path, data)
    hook_order: list[str] = []
    captured_handle: BinaryIO | None = None

    class ObservedStore(PostgresArtifactReceiptStore):
        """Record the route descriptor; every native write still runs below."""

        def commit_verified(
            self,
            receipt: ArtifactReceipt,
            route_verified: VerifiedArtifact,
            *,
            after_write: Callable[[psycopg.Connection], None] | None = None,
        ) -> ArtifactReceipt:
            nonlocal captured_handle
            captured_handle = route_verified.handle
            return super().commit_verified(receipt, route_verified, after_write=after_write)

    store = ObservedStore(
        sandbox.database,
        sandbox.authority,
        tmp_path / "clip-store",
        ReceiptHooks(
            after_preflight=lambda: hook_order.append("after-preflight"),
            before_final_check=lambda: hook_order.append("before-final-check"),
        ),
    )
    client = _client(tmp_path, store, sandbox)
    clips, artifacts, audit_events = _counts(sandbox)

    # When: the real route completes a descriptor-bound native receipt.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_payload(data),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: both timing seams execute, computed identity commits, and the FD closes.
    assert response.status_code == 200
    assert hook_order == ["after-preflight", "before-final-check"]
    assert captured_handle is not None and captured_handle.closed
    assert sandbox.admin.execute(
        "SELECT media_sha256,media_size_bytes,publish_state FROM clips WHERE clip_id='clip-1'"
    ).fetchone() == (_receipt(data).sha256, media.stat().st_size, "PUBLISHED")
    assert sandbox.admin.execute(
        "SELECT lifecycle_state FROM incidents WHERE edge_event_id='event-1'"
    ).fetchone() == ("COMPLETE",)
    # One required publication for a new receipt, on the independent connection.
    assert _counts(sandbox) == (clips + 1, artifacts + 1, audit_events + 1)


def test_relay_verifies_before_durable_receipt_and_never_writes_media(
    tmp_path: Path, postgres_product_sandbox: ProductSandbox
) -> None:
    data = b"verified video"
    media = _media(tmp_path, data)
    sandbox = postgres_product_sandbox
    store = _native_store(tmp_path, sandbox)
    client = _client(tmp_path, store, sandbox)

    response = client.put(
        "/api/v1/relay/clips/clip-1", json=_payload(data), headers={"X-Edge-Relay-Token": TOKEN}
    )

    assert response.status_code == 200
    assert store.get("clip-1") == _receipt(data)
    assert media.read_bytes() == data


@pytest.mark.parametrize("declared", [b"wrong-size", b"wrong-hash"])
def test_relay_verification_failure_is_distinct_from_conflict(
    tmp_path: Path, declared: bytes, postgres_product_sandbox: ProductSandbox
) -> None:
    data = b"verified video"
    _media(tmp_path, data)
    sandbox = postgres_product_sandbox
    store = _native_store(tmp_path, sandbox)
    client = _client(tmp_path, store, sandbox)
    before = _counts(sandbox)

    response = client.put(
        "/api/v1/relay/clips/clip-1", json=_payload(declared), headers={"X-Edge-Relay-Token": TOKEN}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "clip media mismatch"
    assert store.get("clip-1") is None
    assert _counts(sandbox) == before


def test_verification_failure_is_typed(tmp_path: Path) -> None:
    artifact = tmp_path / "clip.mp4"
    artifact.write_bytes(b"actual bytes")

    with pytest.raises(ArtifactReceiptVerificationError):
        verify_artifact(artifact, _receipt(b"different bytes"))


def test_a_receipt_that_exists_must_be_accepted_and_must_match(
    tmp_path: Path, postgres_product_sandbox: ProductSandbox
) -> None:
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
    client = _client(tmp_path, store, postgres_product_sandbox)
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
