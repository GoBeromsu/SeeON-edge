"""Native evidence-receipt HTTP boundary: admission, publication ordering, drain.

Exercises the real relay export route against a PostgreSQL owner and native audit
runtime. The backend evidence client is a recording stub for the REMOTE boundary
only; persistence and audit are genuinely native. No lifespan/root activation is
claimed and no gate is run here.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.postgres_runtime import (
    AuditRuntimeUnavailable,
    PostgresAuditRuntime,
)
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.event_outbox import EventOutbox, OutboxBudget
from backend.app.features.evidence.postgres_receipts import PostgresArtifactReceiptStore
from backend.app.features.evidence.relay_projection import RelayEvent
from backend.app.features.runtime_settings.store import RuntimeSettingsStore
from backend.app.main import create_app, no_lifespan
from backend.app.shared.postgres_dashboard_credentials import PostgresDashboardCredentialsStore
from shared.events.evidence_export_contract import (
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
)
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

TOKEN = "relay-token"
EVENT_ID = "00000000-0000-4000-8000-000000000001"
_DATA = b"verified video"
_SHA = hashlib.sha256(_DATA).hexdigest()
TEST_OUTBOX_BUDGET = OutboxBudget(4, 64 * 1024)


class _Backend:
    """Records the remote boundary; never touches the database or audit runtime."""

    def __init__(self) -> None:
        self.ready_calls: list[BinaryIO] = []
        self.unavailable_calls: list[object] = []
        self.ready_result: object = ClipReceipt("clip-1", "READY", 1, _SHA, len(_DATA))
        self.unavailable_result: object = ClipReceipt("clip-1", "UNAVAILABLE", 1, None, None)

    def for_camera(self, _camera_id: str) -> "_Backend":
        return self

    def publish_ready(self, _request: object, media: BinaryIO) -> object:
        self.ready_calls.append(media)
        return self.ready_result

    def report_unavailable(self, request: object) -> object:
        self.unavailable_calls.append(request)
        return self.unavailable_result


class _ObservedStore(PostgresArtifactReceiptStore):
    """Capture the route-owned descriptor without replacing native writes."""

    handle: BinaryIO | None = None

    def commit_verified(self, receipt, route_verified, *, after_write=None):  # type: ignore[no-untyped-def]
        self.handle = route_verified.handle
        return super().commit_verified(receipt, route_verified, after_write=after_write)


def _media(tmp_path: Path, *, event_id: str = "event-1", data: bytes = _DATA) -> Path:
    path = tmp_path / "clip-store" / "clips" / "clip-1" / "clip.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    (path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-1",
                "camera_id": "camera-1",
                "event_ref": event_id,
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


def _ready_payload(data: bytes = _DATA) -> dict[str, object]:
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


def _unavailable_payload() -> dict[str, object]:
    return {
        "state": "UNAVAILABLE",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "event_refs": [EVENT_ID],
        "state_version": 1,
        "reason": "CAPTURE_FAILED",
    }


def _setup(
    tmp_path: Path,
    sandbox: ProductSandbox,
    *,
    store_authority=None,
    seed_event: str = "event-1",
    observed: bool = False,
) -> SimpleNamespace:
    app = create_app(lifespan=no_lifespan)
    runtime = PostgresAuditRuntime(
        PostgresAuditStore(sandbox.database, sandbox.authority),
        maximum_snapshot_age_sec=10.0,
        clock=lambda: 0.0,
    )
    assert runtime.verify_once()
    assert runtime.start_session_once()
    store_cls = _ObservedStore if observed else PostgresArtifactReceiptStore
    store = store_cls(
        sandbox.database,
        store_authority if store_authority is not None else sandbox.authority,
        tmp_path / "clip-store",
    )
    backend = _Backend()
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
    app.state.backend_evidence_client = backend
    settings = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    settings.set_clip_export_enabled(True)
    app.state.runtime_settings_store = settings
    EventOutbox(
        sandbox.database, sandbox.authority, TEST_OUTBOX_BUDGET, audit_runtime=runtime
    ).accept(
        RelayEvent(
            seed_event,
            "fall",
            0.8,
            "2026-07-06T00:00:00Z",
            "camera-1",
            "facility-1",
            None,
            None,
            None,
        ),
        backend_camera_id=None,
        forward=False,
    )
    return SimpleNamespace(
        app=app,
        client=TestClient(app),
        runtime=runtime,
        sandbox=sandbox,
        store=store,
        backend=backend,
    )


def _counts(sandbox: ProductSandbox) -> tuple[int, int, int]:
    return tuple(  # type: ignore[return-value]
        sandbox.admin.execute("SELECT count(*) FROM " + table).fetchone()[0]
        for table in ("clips", "artifacts", "audit_events")
    )


def _incident_state(sandbox: ProductSandbox) -> str:
    return sandbox.admin.execute(
        "SELECT lifecycle_state FROM incidents WHERE edge_event_id='event-1'"
    ).fetchone()[0]


def _put(setup: SimpleNamespace, payload: dict[str, object]):
    return setup.client.put(
        "/api/v1/relay/clips/clip-1", json=payload, headers={"X-Edge-Relay-Token": TOKEN}
    )


def test_unauthenticated_export_never_reaches_store_runtime_or_backend(
    tmp_path, postgres_product_sandbox
):
    setup = _setup(tmp_path, postgres_product_sandbox)
    _media(tmp_path)
    before = _counts(setup.sandbox)
    response = setup.client.put("/api/v1/relay/clips/clip-1", json=_ready_payload())
    assert response.status_code == 401
    assert setup.backend.ready_calls == [] and _counts(setup.sandbox) == before


@pytest.mark.parametrize("mode", ["missing", "wrong_type"])
def test_missing_native_runtime_refuses_before_any_egress(tmp_path, postgres_product_sandbox, mode):
    setup = _setup(tmp_path, postgres_product_sandbox)
    _media(tmp_path)
    before = _counts(setup.sandbox)
    if mode == "missing":
        del setup.app.state.audit_runtime
    else:
        setup.app.state.audit_runtime = object()
    response = _put(setup, _ready_payload())
    assert (response.status_code, response.content) == (503, b"")
    assert setup.backend.ready_calls == [] and _counts(setup.sandbox) == before


def test_owner_authority_mismatch_rejected_before_egress(tmp_path, postgres_product_sandbox):
    sandbox = postgres_product_sandbox
    foreign = replace(sandbox.authority, generation=sandbox.authority.generation + 1)
    setup = _setup(tmp_path, sandbox, store_authority=foreign)
    _media(tmp_path)
    before = _counts(sandbox)
    with pytest.raises(ValueError, match="share database and authority"):
        _put(setup, _ready_payload())
    assert setup.backend.ready_calls == [] and _counts(sandbox) == before


def test_ready_publishes_after_commit_before_remote_send(tmp_path, postgres_product_sandbox):
    setup = _setup(tmp_path, postgres_product_sandbox)
    _media(tmp_path)
    sandbox = setup.sandbox
    clips, artifacts, audit = _counts(sandbox)
    trace: list[str] = []
    publish = setup.runtime.publish_committed

    def observed_publish(token):
        # The COMMIT is done: the durable row is visible on the independent
        # connection, and the remote send has not yet been attempted.
        assert sandbox.admin.execute(
            "SELECT publish_state FROM clips WHERE clip_id='clip-1'"
        ).fetchone() == ("PUBLISHED",)
        assert _counts(sandbox) == (clips + 1, artifacts + 1, audit + 1)
        assert setup.backend.ready_calls == []
        trace.append("publication")
        return publish(token)

    original_ready = setup.backend.publish_ready

    def observed_ready(request, media):
        trace.append("remote")
        return original_ready(request, media)

    setup.runtime.publish_committed = observed_publish  # type: ignore[assignment]
    setup.backend.publish_ready = observed_ready  # type: ignore[assignment]
    response = _put(setup, _ready_payload())
    assert response.status_code == 200
    assert trace == ["publication", "remote"]
    assert _incident_state(sandbox) == "COMPLETE"
    assert not setup.runtime._pending


def test_matching_ready_retry_appends_exactly_one_more_audit(tmp_path, postgres_product_sandbox):
    setup = _setup(tmp_path, postgres_product_sandbox)
    _media(tmp_path)
    sandbox = setup.sandbox
    clips, artifacts, audit = _counts(sandbox)
    assert _put(setup, _ready_payload()).status_code == 200
    assert _counts(sandbox) == (clips + 1, artifacts + 1, audit + 1)
    # An identical receipt keeps one clip/artifact but records another required audit.
    assert _put(setup, _ready_payload()).status_code == 200
    assert _counts(sandbox) == (clips + 1, artifacts + 1, audit + 2)
    assert len(setup.backend.ready_calls) == 2


def test_missing_manifest_incident_rolls_back_without_remote_send(
    tmp_path, postgres_product_sandbox
):
    setup = _setup(tmp_path, postgres_product_sandbox)
    _media(tmp_path, event_id="event-unseeded")
    before = _counts(setup.sandbox)
    response = _put(setup, _ready_payload())
    # A missing incident is a persistence error (retryable 503 with a static
    # detail), not the empty-503 of an unavailable audit runtime.
    assert response.status_code == 503
    assert response.json()["detail"] == "artifact receipt persistence unavailable"
    assert setup.backend.ready_calls == [] and _counts(setup.sandbox) == before


def test_deferred_commit_failure_rolls_back_and_closes_descriptor(
    tmp_path, postgres_product_sandbox
):
    setup = _setup(tmp_path, postgres_product_sandbox, observed=True)
    _media(tmp_path)
    sandbox = setup.sandbox
    sandbox.admin.execute(
        "CREATE FUNCTION reject_audit() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'private detail' USING ERRCODE='23514'; END $$"
    )
    sandbox.admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_audit AFTER INSERT ON audit_events "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_audit()"
    )
    before = _counts(sandbox)
    try:
        response = _put(setup, _ready_payload())
        assert (response.status_code, response.content) == (503, b"")
    finally:
        sandbox.admin.execute("DROP TRIGGER reject_audit ON audit_events")
    assert setup.backend.ready_calls == []
    assert _counts(sandbox) == before
    assert setup.store.handle is not None and setup.store.handle.closed


def test_post_commit_unknown_retains_committed_state_without_remote_send(
    tmp_path, postgres_product_sandbox, monkeypatch
):
    setup = _setup(tmp_path, postgres_product_sandbox, observed=True)
    _media(tmp_path)
    sandbox = setup.sandbox
    clips, artifacts, audit = _counts(sandbox)
    transact = sandbox.database.transact

    def unknown_after_commit(body):
        transact(body)
        raise CommitOutcomeUnknown()

    monkeypatch.setattr(sandbox.database, "transact", unknown_after_commit)
    response = _put(setup, _ready_payload())
    assert (response.status_code, response.content) == (503, b"")
    # The real COMMIT persisted the receipt/audit; the send never ran and the
    # runtime latches indeterminate without replaying the transaction.
    assert _counts(sandbox) == (clips + 1, artifacts + 1, audit + 1)
    assert setup.backend.ready_calls == []
    assert setup.store.handle is not None and setup.store.handle.closed
    assert setup.runtime.snapshot().indeterminate
    assert not setup.runtime._pending


def test_admission_lost_after_commit_refuses_send_without_false_rollback(
    tmp_path, postgres_product_sandbox, monkeypatch
):
    setup = _setup(tmp_path, postgres_product_sandbox, observed=True)
    _media(tmp_path)
    sandbox = setup.sandbox
    clips, artifacts, audit = _counts(sandbox)
    admit = setup.runtime.require_mutation_admission
    publish = setup.runtime.publish_committed
    committed = {"done": False}

    def note_publish(token):
        result = publish(token)
        committed["done"] = True
        return result

    def admit_until_committed(owner=None):
        # Admission is lost only once the receipt is durably committed and
        # published; the post-commit recheck then refuses the remote send.
        if committed["done"]:
            raise AuditRuntimeUnavailable("mutation admission withdrawn")
        return admit(owner)

    monkeypatch.setattr(setup.runtime, "publish_committed", note_publish)
    monkeypatch.setattr(setup.runtime, "require_mutation_admission", admit_until_committed)
    response = _put(setup, _ready_payload())
    assert (response.status_code, response.content) == (503, b"")
    assert setup.backend.ready_calls == []
    assert setup.store.handle is not None and setup.store.handle.closed
    # The committed receipt/audit are real and must not be reported as rolled back.
    assert _counts(sandbox) == (clips + 1, artifacts + 1, audit + 1)
    assert _incident_state(sandbox) == "COMPLETE"


def test_unavailable_remote_failure_writes_no_local_receipt(tmp_path, postgres_product_sandbox):
    setup = _setup(tmp_path, postgres_product_sandbox)
    _media(tmp_path)
    setup.backend.unavailable_result = DeliveryFailure(DeliveryDisposition.RETRY, "NETWORK")
    before = _counts(setup.sandbox)
    response = _put(setup, _unavailable_payload())
    assert response.status_code == 503
    assert len(setup.backend.unavailable_calls) == 1
    assert _counts(setup.sandbox) == before and _incident_state(setup.sandbox) == "OPEN"


def test_unavailable_remote_success_records_zero_audit(tmp_path, postgres_product_sandbox):
    setup = _setup(tmp_path, postgres_product_sandbox)
    _media(tmp_path)
    sandbox = setup.sandbox
    clips, artifacts, audit = _counts(sandbox)
    response = _put(setup, _unavailable_payload())
    assert response.status_code == 200
    assert len(setup.backend.unavailable_calls) == 1
    # The approved unavailable-receipt contract records no audit row.
    assert _counts(sandbox) == (clips, artifacts + 1, audit)
    assert _incident_state(sandbox) == "FAILED"
    assert sandbox.admin.execute(
        "SELECT state FROM artifacts WHERE incident_id='incident:event-1' AND kind='PRIMARY_CLIP'"
    ).fetchone() == ("UNAVAILABLE",)


def test_unavailable_local_failure_after_remote_report_is_not_false_rollback(
    tmp_path, postgres_product_sandbox
):
    setup = _setup(tmp_path, postgres_product_sandbox)
    _media(tmp_path)
    sandbox = setup.sandbox
    sandbox.admin.execute(
        "CREATE FUNCTION reject_artifact() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'private detail' USING ERRCODE='23514'; END $$"
    )
    sandbox.admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_artifact AFTER INSERT ON artifacts "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_artifact()"
    )
    before = _counts(sandbox)
    try:
        response = _put(setup, _unavailable_payload())
        assert (response.status_code, response.content) == (503, b"")
    finally:
        sandbox.admin.execute("DROP TRIGGER reject_artifact ON artifacts")
    # The remote report was genuinely issued; only the local write rolled back.
    assert len(setup.backend.unavailable_calls) == 1
    assert _counts(sandbox) == before and _incident_state(sandbox) == "OPEN"
