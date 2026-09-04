from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path
from typing import BinaryIO

import pytest
from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.evidence.compact_receipts import CompactArtifactReceiptStore
from backend.app.features.evidence.relay_projection import RelayEvent, RelayEvidenceProjection
from backend.app.features.runtime_settings.store import RuntimeSettingsStore
from backend.app.main import create_app, no_lifespan
from shared.events.evidence_export_client import ReadyClipRequest, UnavailableClipRequest
from shared.events.evidence_export_contract import BackendCapabilities, ClipReceipt, DeliveryFailure
from tests_support.compact_authority_db import prepare_compact_database

TOKEN = "relay-token"
EVENT_ID = "00000000-0000-4000-8000-000000000001"
MEDIA_SHA256 = hashlib.sha256(b"mp4x").hexdigest()


@dataclass
class FakeBackendEvidenceClient:
    capability_result: BackendCapabilities | DeliveryFailure = BackendCapabilities(1, 1)
    clip_result: ClipReceipt | DeliveryFailure = ClipReceipt("clip-1", "READY", 2, MEDIA_SHA256, 4)
    ready_calls: int = 0
    before_read: Callable[[], None] | None = None
    opened_media: BinaryIO | None = field(default=None, init=False)
    uploaded_bytes: bytes | None = field(default=None, init=False)
    ready_request: ReadyClipRequest | None = field(default=None, init=False)
    unavailable_request: UnavailableClipRequest | None = field(default=None, init=False)

    def for_camera(self, _camera_id: str) -> FakeBackendEvidenceClient:
        return self

    def probe_capabilities(self, _camera_id: str) -> BackendCapabilities | DeliveryFailure:
        return self.capability_result

    def publish_ready(
        self, request: ReadyClipRequest, media: BinaryIO
    ) -> ClipReceipt | DeliveryFailure:
        self.ready_calls += 1
        assert request.clip_id == "clip-1"
        self.ready_request = request
        self.opened_media = media
        if self.before_read is not None:
            self.before_read()
        self.uploaded_bytes = media.read()
        return self.clip_result

    def report_unavailable(self, request: UnavailableClipRequest) -> ClipReceipt | DeliveryFailure:
        self.unavailable_request = request
        return self.clip_result


def _client(tmp_path: Path, backend: FakeBackendEvidenceClient, *, enabled: bool) -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = TOKEN
    # Camera binding is registry-only now (no camera_inventory fallback --
    # see _camera_binding_from_registry in relay/router.py), so the fixture
    # must register "camera-1" in a CameraRegistryStore for _camera_binding
    # to resolve it instead of 403ing every export.
    database = tmp_path / "catalog.sqlite3"
    prepare_compact_database(database)
    registry = CameraRegistryStore(database)
    registry.create(
        camera_id="camera-1",
        label="Camera 1",
        rtsp_url="rtsp://camera/1",
        space_id="facility-1",
        status="online",
        # Hub-mapped, which is what every test in this module assumes: they are
        # about media resolution, descriptor verification, and receipt typing, not
        # about identity mapping. Clip export addresses the Hub, so an unmapped
        # camera is refused up front (issue #308) -- that path is pinned separately
        # by test_export_refused_when_camera_has_no_hub_mapping below.
        backend_camera_id="cmsnvr-camera-1",
    )
    app.state.camera_registry = registry
    app.state.backend_evidence_client = backend
    app.state.artifact_receipt_store = CompactArtifactReceiptStore(
        database, tmp_path / "clip-store"
    )
    RelayEvidenceProjection(database).project_event(
        RelayEvent(
            EVENT_ID,
            "fall",
            0.9,
            "2026-07-16T00:00:00Z",
            "camera-1",
            "facility-1",
            None,
            None,
            None,
        )
    )
    runtime_settings = RuntimeSettingsStore(database)
    if enabled:
        runtime_settings.set_clip_export_enabled(True)
    app.state.runtime_settings_store = runtime_settings
    app.state.clip_store_root = tmp_path / "clip-store"
    return TestClient(app)


def _write_ready_media(tmp_path: Path) -> Path:
    media = tmp_path / "clip-store" / "clips" / "clip-1" / "clip.mp4"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"mp4x")
    media.with_name("manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-1",
                "camera_id": "camera-1",
                "event_ref": EVENT_ID,
                "event_type": "fall",
                "started_at": "2026-07-16T00:00:00Z",
                "duration_s": 1.0,
                "codec": "h264",
                "path": "clips/clip-1/clip.mp4",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    return media


def _write_unavailable_manifest(tmp_path: Path, *, event_refs: list[str] | None = None) -> None:
    path = tmp_path / "clip-store" / "clips" / "clip-1" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "clip_id": "clip-1",
                "camera_id": "camera-1",
                "event_ref": EVENT_ID,
                **({"event_refs": event_refs} if event_refs is not None else {}),
                "event_type": "fall",
                "started_at": "2026-07-16T00:00:00Z",
                "duration_s": 1.0,
                "codec": "",
                "path": None,
                "video_available": False,
                "video_error": "CAPTURE_FAILED",
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )


def _ready_payload() -> dict[str, object]:
    return {
        "state": "READY",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "event_refs": [EVENT_ID],
        "state_version": 2,
        "sha256": MEDIA_SHA256,
        "size_bytes": 4,
        "mime_type": "video/mp4",
        "codec": "h264",
        "duration_ms": 1000,
        "clip_start_at": "2026-07-16T00:00:00Z",
        "clip_end_at": "2026-07-16T00:00:01Z",
        "finalized_at": "2026-07-16T00:00:02Z",
    }


def _unavailable_payload() -> dict[str, object]:
    return {
        "state": "UNAVAILABLE",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "event_refs": [EVENT_ID],
        "state_version": 3,
        "reason": "CAPTURE_FAILED",
    }


def _set_attribute(target: object, name: str, value: object) -> None:
    setattr(target, name, value)


def test_capability_requires_auth_local_enablement_and_backend_proof(tmp_path: Path) -> None:
    # Given: local support is enabled and a backend probe proves both capabilities.
    backend = FakeBackendEvidenceClient()
    client = _client(tmp_path, backend, enabled=True)

    # When: unauthenticated and authenticated callers probe the relay.
    denied = client.get("/api/v1/relay/capabilities", params={"camera_id": "camera-1"})
    accepted = client.get(
        "/api/v1/relay/capabilities",
        params={"camera_id": "camera-1"},
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: only the authenticated backend-proven result is advertised.
    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json() == {"event_idempotency": 1, "clip_export": 1}


def test_capability_stays_zero_when_feature_is_disabled(tmp_path: Path) -> None:
    # Given: the compatibility image ships with export disabled.
    backend = FakeBackendEvidenceClient()
    client = _client(tmp_path, backend, enabled=False)

    # When: the worker probes the local relay.
    response = client.get(
        "/api/v1/relay/capabilities",
        params={"camera_id": "camera-1"},
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: environment support alone never advertises backend readiness.
    assert response.status_code == 200
    assert response.json() == {"event_idempotency": 1, "clip_export": 0}


def test_capability_reads_live_persisted_setting_without_app_rebuild(tmp_path: Path) -> None:
    backend = FakeBackendEvidenceClient()
    client = _client(tmp_path, backend, enabled=False)
    headers = {"X-Edge-Relay-Token": TOKEN}

    before = client.get(
        "/api/v1/relay/capabilities",
        params={"camera_id": "camera-1"},
        headers=headers,
    )
    RuntimeSettingsStore(tmp_path / "catalog.sqlite3").set_clip_export_enabled(True)
    after = client.get(
        "/api/v1/relay/capabilities",
        params={"camera_id": "camera-1"},
        headers=headers,
    )

    assert before.json()["clip_export"] == 0
    assert after.json()["clip_export"] == 1


def test_ready_relay_resolves_owned_media_by_clip_id_and_returns_typed_receipt(
    tmp_path: Path,
) -> None:
    # Given: strict shared-store bytes exist under the route clip ID.
    _write_ready_media(tmp_path)
    backend = FakeBackendEvidenceClient()
    client = _client(tmp_path, backend, enabled=True)

    # Mutation proof: a receipt that does not match the opened bytes remains a conflict.
    bad_payload = _ready_payload() | {"sha256": "0" * 64}
    bad_receipt = client.put(
        "/api/v1/relay/clips/clip-1",
        json=bad_payload,
        headers={"X-Edge-Relay-Token": TOKEN},
    )
    assert bad_receipt.status_code == 409
    assert backend.ready_calls == 0

    # When: the worker relays matching metadata without supplying a path.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_ready_payload(),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: ml-api alone opens the owned file and returns the backend receipt.
    assert response.status_code == 200
    assert response.json() == {
        "clip_id": "clip-1",
        "state": "READY",
        "state_version": 2,
        "sha256": MEDIA_SHA256,
        "size_bytes": 4,
    }
    assert backend.ready_calls == 1
    assert backend.uploaded_bytes == b"mp4x"
    assert backend.opened_media is not None and backend.opened_media.closed
    ready_request = backend.ready_request
    assert ready_request is not None
    assert (
        ready_request.clip_id,
        ready_request.camera_id,
        ready_request.event_refs,
        ready_request.state_version,
        ready_request.sha256,
        ready_request.size_bytes,
        ready_request.mime_type,
        ready_request.codec,
        ready_request.duration_ms,
        ready_request.clip_start_at,
        ready_request.clip_end_at,
        ready_request.finalized_at,
    ) == (
        "clip-1",
        # Hub-issued id, not the edge-local registry id. The outbound clip request
        # addresses the Hub, and an id the Hub never issued is rejected with
        # FACILITY_BINDING_MISMATCH (issue #308).
        "cmsnvr-camera-1",
        (EVENT_ID,),
        2,
        MEDIA_SHA256,
        4,
        "video/mp4",
        "h264",
        1000,
        "2026-07-16T00:00:00Z",
        "2026-07-16T00:00:01Z",
        "2026-07-16T00:00:02Z",
    )
    with pytest.raises(FrozenInstanceError):
        _set_attribute(ready_request, "camera_id", "camera-other")


def test_evidence_receipt_route_commits_canonical_action_and_detail(tmp_path: Path) -> None:
    _write_ready_media(tmp_path)
    client = _client(tmp_path, FakeBackendEvidenceClient(), enabled=True)

    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_ready_payload(),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    assert response.status_code == 200
    with sqlite3.connect(tmp_path / "catalog.sqlite3") as connection:
        rows = connection.execute(
            "SELECT action,target_id,actor_type,auth_mechanism,detail_json "
            "FROM audit_events WHERE action NOT LIKE 'audit.%'"
        ).fetchall()
    assert rows == [("evidence.receipt", "clip-1", "service", "relay_token", '{"version":1}')]


def test_unavailable_relay_passes_complete_immutable_state_request(tmp_path: Path) -> None:
    # Given: capture failed before media publication, so no READY-only metadata exists.
    backend = FakeBackendEvidenceClient(
        clip_result=ClipReceipt("clip-1", "UNAVAILABLE", 3, None, None)
    )
    _write_unavailable_manifest(tmp_path)
    client = _client(tmp_path, backend, enabled=True)

    # When: the worker reports the terminal unavailable state.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_unavailable_payload(),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: the backend receives exactly the required unavailable fields as a frozen request.
    assert response.status_code == 200
    assert response.json() == {
        "clip_id": "clip-1",
        "state": "UNAVAILABLE",
        "state_version": 3,
        "sha256": None,
        "size_bytes": None,
    }
    unavailable_request = backend.unavailable_request
    assert unavailable_request is not None
    assert (
        unavailable_request.clip_id,
        unavailable_request.camera_id,
        unavailable_request.event_refs,
        unavailable_request.state_version,
        unavailable_request.reason,
    ) == ("clip-1", "cmsnvr-camera-1", (EVENT_ID,), 3, "CAPTURE_FAILED")
    with pytest.raises(FrozenInstanceError):
        _set_attribute(unavailable_request, "reason", "CORRUPT")
    with sqlite3.connect(tmp_path / "catalog.sqlite3") as connection:
        assert connection.execute(
            "SELECT lifecycle_state, failure_reason FROM incidents WHERE edge_event_id = ?",
            (EVENT_ID,),
        ).fetchone() == ("FAILED", "CAPTURE_FAILED")


def test_unavailable_relay_replay_is_noop_and_conflict_rolls_back(tmp_path: Path) -> None:
    backend = FakeBackendEvidenceClient(
        clip_result=ClipReceipt("clip-1", "UNAVAILABLE", 3, None, None)
    )
    _write_unavailable_manifest(tmp_path)
    client = _client(tmp_path, backend, enabled=True)
    headers = {"X-Edge-Relay-Token": TOKEN}

    first = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_unavailable_payload(),
        headers=headers,
    )
    assert first.status_code == 200
    replay = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_unavailable_payload(),
        headers=headers,
    )
    assert replay.status_code == 200
    conflicting = {**_unavailable_payload(), "reason": "CORRUPT"}
    conflict = client.put(
        "/api/v1/relay/clips/clip-1",
        json=conflicting,
        headers=headers,
    )
    assert conflict.status_code == 409
    with sqlite3.connect(tmp_path / "catalog.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (1,)
        assert connection.execute(
            "SELECT lifecycle_state, failure_reason FROM incidents WHERE edge_event_id = ?",
            (EVENT_ID,),
        ).fetchone() == ("FAILED", "CAPTURE_FAILED")


def test_ready_relay_uploads_verified_descriptor_when_path_is_swapped(
    tmp_path: Path,
) -> None:
    # Given: an attacker swaps the pathname only after ml-api verifies and opens it.
    media = _write_ready_media(tmp_path)
    backend = FakeBackendEvidenceClient()

    def swap_path() -> None:
        media.unlink()
        media.write_bytes(b"evil")

    backend.before_read = swap_path
    client = _client(tmp_path, backend, enabled=True)

    # When: backend upload begins after the pathname swap.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_ready_payload(),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: upload uses the already verified inode and closes that descriptor.
    assert response.status_code == 200
    assert backend.uploaded_bytes == b"mp4x"
    assert media.read_bytes() == b"evil"
    assert backend.opened_media is not None and backend.opened_media.closed


def test_clip_relay_rejects_duplicate_or_non_uuid4_event_refs(tmp_path: Path) -> None:
    backend = FakeBackendEvidenceClient()
    client = _client(tmp_path, backend, enabled=True)
    for refs in ([EVENT_ID, EVENT_ID], ["not-a-uuid"]):
        payload = _ready_payload()
        payload["event_refs"] = refs
        response = client.put(
            "/api/v1/relay/clips/clip-1",
            json=payload,
            headers={"X-Edge-Relay-Token": TOKEN},
        )
        assert response.status_code == 422
    assert backend.ready_calls == 0


def test_ready_relay_rejects_missing_media_without_backend_call(
    tmp_path: Path,
) -> None:
    # Given: no owned media exists for this clip id, and the payload claims a
    # mismatched facility. That mismatch is not what drives the 404 here,
    # though -- an Edge is single-facility by construction (one Edge install
    # serves one facility), so `facility_id` on the wire is informational,
    # not an admission key: `_camera_binding` (relay/router.py) resolves
    # ownership from the local camera registry alone and never compares it
    # to `facility_id`, and clip storage on disk isn't partitioned by
    # facility either -- `_verified_media` below builds the path from
    # `clip_id` alone (`root/clips/<clip_id>/clip.mp4`). So the only real
    # rejection reason left is what's actually true -- no media was ever
    # written for this clip id -- and 404 (not found), not 403 (forbidden),
    # is the honest status for that.
    backend = FakeBackendEvidenceClient()
    client = _client(tmp_path, backend, enabled=True)
    payload = _ready_payload()
    payload["facility_id"] = "facility-other"

    # When: the worker relays metadata for a clip whose media was never written.
    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=payload,
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    # Then: the route fails before backend egress and leaks no local path.
    assert response.status_code == 404
    assert "clip-store" not in response.text
    assert backend.ready_calls == 0


def test_export_refused_when_camera_has_no_hub_mapping(tmp_path: Path) -> None:
    """A clip export for an unmapped camera is refused before any backend call.

    Clip export exists to reach the Hub, so unlike the alert and heartbeat relays
    there is no local-accept path to fall back to. Sending the edge-local id would
    be rejected by the Hub with FACILITY_BINDING_MISMATCH, which reaches the edge
    as an opaque 502 and reads like an authentication failure (issue #308). The
    edge names the real reason instead, and never contacts the backend.
    """
    backend = FakeBackendEvidenceClient()
    client = _client(tmp_path, backend, enabled=True)
    # Same app the other tests use, but with the camera's Hub mapping removed, so
    # only the mapping state differs from the passing cases above.
    unmapped_path = tmp_path / "unmapped" / "edge.sqlite3"
    prepare_compact_database(unmapped_path)
    unmapped = CameraRegistryStore(unmapped_path)
    unmapped.create(
        camera_id="camera-1",
        label="Camera 1",
        rtsp_url="rtsp://camera/1",
        space_id="facility-1",
        status="online",
        backend_camera_id=None,
    )
    client.app.state.camera_registry = unmapped

    response = client.put(
        "/api/v1/relay/clips/clip-1",
        json=_unavailable_payload(),
        headers={"X-Edge-Relay-Token": TOKEN},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "CAMERA_MAPPING_MISSING"
    # The decisive property: the backend was never addressed at all.
    assert backend.ready_calls == 0
    assert backend.unavailable_request is None
