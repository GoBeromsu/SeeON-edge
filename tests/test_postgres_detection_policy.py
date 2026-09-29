"""Detection policies against the explicit isolated PostgreSQL product sandbox."""

from __future__ import annotations

import hashlib
import json
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from queue import Queue
from threading import Barrier, Event
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4

import psycopg
import pytest
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row, tuple_row

from backend.app.edge_db.authority import AuthorityFenced, AuthorityToken, freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown, PoolBudget, PostgresDatabase
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_store import append_postgres_audit
from backend.app.features.audit.store import AuditEvent
from backend.app.features.cameras.camera_repository import CameraRegistryWriteError
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.detection_settings.policy_store import (
    DetectionPolicyNotInitialized,
    DetectionPolicyStore,
    PolicyActivation,
    PolicyActivationRefused,
    PolicyCameraIdentity,
    PolicyRevisionConflict,
    PolicyRollbackUnavailable,
)
from shared.detection_policies import (
    PolicyDocumentError,
    default_policy_bundle,
    parse_policy_bundle,
    policy_values_dict,
)

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_FACILITY = "facility/non-uuid:seoul"
_LOCAL = "local/camera:room-1"
_CAMERA = "hub-camera|opaque|A-17"
_REMAPPED = "hub-camera|opaque|B-18"
_TIME = "2026-09-27T04:00:00.123Z"


def _store(sandbox: ProductSandbox) -> DetectionPolicyStore:
    return DetectionPolicyStore(sandbox.database, sandbox.authority)


def _camera(
    sandbox: ProductSandbox, camera_id: str = _LOCAL, backend_id: str | None = _CAMERA
) -> CameraRegistryStore:
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
    registry.create(
        camera_id=camera_id,
        backend_camera_id=backend_id,
        label="Room",
        rtsp_url=f"rtsp://camera.invalid/{camera_id}",
        space_id=None,
        status="offline",
    )
    return registry


def _fall(
    threshold: float | None = 0.62,
    *,
    camera_id: str | None = None,
    facility_id: str = _FACILITY,
):
    return {
        "facility_id": facility_id,
        "module_id": "fall",
        "module_version": 2,
        "schema_id": "fall.policy",
        "schema_version": 2,
        "camera_id": camera_id,
        "values": None if threshold is None else {"transition_threshold": threshold},
    }


def _apply(
    store: DetectionPolicyStore,
    threshold: float | None = 0.62,
    *,
    camera_id: str | None = None,
    expected: int = 0,
    after_write: Callable[[psycopg.Connection], None] | None = None,
) -> PolicyActivation:
    return store.apply(
        **_fall(threshold, camera_id=camera_id),
        expected_revision_id=expected,
        after_write=after_write,
    )


def _rollback(
    store: DetectionPolicyStore,
    expected: int,
    *,
    camera_id: str | None = None,
    after_write: Callable[[psycopg.Connection], None] | None = None,
) -> PolicyActivation:
    return store.rollback(
        facility_id=_FACILITY,
        module_id="fall",
        module_version=2,
        camera_id=camera_id,
        expected_revision_id=expected,
        after_write=after_write,
    )


def _rows(sandbox: ProductSandbox):
    with sandbox.admin.cursor(row_factory=dict_row) as cursor:
        return cursor.execute("SELECT * FROM policies ORDER BY policy_id").fetchall()


def _audit(connection: psycopg.Connection, action: AuditAction = AuditAction.POLICY_APPLY) -> None:
    append_postgres_audit(
        connection,
        AuditEvent(
            occurred_at=_TIME,
            actor_id="test-operator",
            action=action,
            target_id="fall",
            detail=empty_detail(action),
        ),
    )


@contextmanager
def _other_database(sandbox: ProductSandbox) -> Iterator[PostgresDatabase]:
    database = PostgresDatabase(
        sandbox.dsn,
        sandbox.schema,
        PoolBudget(
            max_connections=2,
            max_waiting=4,
            acquire_timeout_sec=1.0,
            statement_timeout_ms=5000,
            lock_timeout_ms=3000,
            startup_timeout_sec=5.0,
        ),
    )
    database.start()
    try:
        yield database
    finally:
        database.close(timeout_sec=3.0)


def test_diff_apply_precedence_identity_encoding_and_ordering(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = _store(sandbox)
    cameras = (PolicyCameraIdentity(_CAMERA),)
    image = default_policy_bundle((_CAMERA,))
    assert store.resolve_bundle(_FACILITY, cameras) == image
    proposal = store.diff(**_fall())
    assert proposal.changed and proposal.concurrency_token == 0
    assert proposal.current == image.defaults["fall"]
    assert proposal.proposed.source == "facility-default"
    assert proposal.compared_payload == {
        key: value for key, value in _fall().items() if key != "facility_id"
    }
    assert _rows(sandbox) == []

    first = _apply(store, expected=proposal.concurrency_token)
    override = _apply(store, 0.81, camera_id=_CAMERA)
    assert first.activation_generation == first.active_revision_id == 1
    assert override.activation_generation == override.active_revision_id == 2
    assert override.previous_revision_id == 0
    assert first.previous_revision_id is None
    assert first.status == override.status == "pending"
    rows = _rows(sandbox)
    assert [row["camera_id"] for row in rows] == [None, _LOCAL]
    for row, encoded in zip(
        rows, ['{"transition_threshold":0.62}', '{"transition_threshold":0.81}'], strict=True
    ):
        assert row["active_values_json"] == encoded
        assert row["active_content_sha256"] == hashlib.sha256(encoded.encode()).hexdigest()
        assert row["applied_at"] is None and row["refusal_reason"] is None
        assert row["activated_at"] == row["updated_at"]
        assert row["activated_at"].endswith("Z")

    bundle = store.resolve_bundle(_FACILITY, cameras)
    default = bundle.defaults["fall"]
    effective = bundle.resolve(_CAMERA, "fall", 2)
    assert policy_values_dict(default.values) == {"transition_threshold": 0.62}
    assert policy_values_dict(effective.values) == {"transition_threshold": 0.81}
    assert default.source == "facility-default" and default.facility_revision_id == 1
    assert effective.source == "camera-override"
    assert effective.facility_revision_id == 1 and effective.camera_revision_id == 2
    assert parse_policy_bundle(bundle.as_dict()) == bundle
    encoded_bundle = json.dumps(
        bundle.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert bundle.content_sha256 == hashlib.sha256(encoded_bundle.encode()).hexdigest()
    assert _store(sandbox).resolve_bundle(_FACILITY, cameras) == bundle
    assert store.activations(_FACILITY) == (first, override)
    assert store.generation(_FACILITY) == 2
    # No facility remains the source-defined image default, even with stored rows.
    assert store.generation(None) == 0 and store.resolve_bundle(None, cameras) == image
    assert store.resolve_bundle("other-facility", cameras) == image


def test_equal_numbers_with_new_scope_are_changes_but_inheriting_absent_override_is_not(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = _store(sandbox)
    inherited = store.diff(**_fall(None, camera_id=_CAMERA))
    assert not inherited.changed and inherited.concurrency_token == 0
    with pytest.raises(PolicyRevisionConflict, match="already inherits"):
        _apply(store, None, camera_id=_CAMERA)
    facility = store.diff(**_fall(0.5))
    assert facility.changed and facility.current.source == "image-default"
    _apply(store, 0.5)
    camera = store.diff(**_fall(0.5, camera_id=_CAMERA))
    assert camera.changed and camera.concurrency_token == 0
    assert camera.current.source == "facility-default"
    assert camera.proposed.source == "camera-override"
    assert len(_rows(sandbox)) == 1
    applied = _apply(store, 0.5, camera_id=_CAMERA)
    unchanged = store.diff(**_fall(0.5, camera_id=_CAMERA))
    assert not unchanged.changed and unchanged.concurrency_token == applied.activation_generation


@pytest.mark.parametrize("acknowledged", [False, True])
def test_identical_apply_keeps_history_generation_status_and_timestamps_but_runs_hook(
    postgres_product_sandbox: ProductSandbox, acknowledged: bool
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    first = _apply(store)
    if acknowledged:
        store.acknowledge_applied(_FACILITY)
    before = _rows(sandbox)
    current = store.activations(_FACILITY)[0]
    hooks = []

    def hook(connection: psycopg.Connection) -> None:
        hooks.append(connection.info.transaction_status)
        assert connection.row_factory is tuple_row
        _audit(connection)

    saved = _apply(store, expected=first.activation_generation, after_write=hook)
    assert saved == current and _rows(sandbox) == before
    assert hooks == [TransactionStatus.INTRANS]
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (1,)


def test_optimistic_tokens_reject_stale_apply_and_rollback_without_hooks(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    first = _apply(store, 0.61)
    second = _apply(store, 0.72, expected=first.activation_generation)
    before = _rows(sandbox)
    hooks = []
    # Even equal values must not bypass the optimistic token check.
    for threshold in (0.72, 0.81):
        with pytest.raises(PolicyRevisionConflict):
            _apply(store, threshold, expected=first.activation_generation, after_write=hooks.append)
    with pytest.raises(PolicyRevisionConflict):
        _rollback(store, first.activation_generation, after_write=hooks.append)
    assert not hooks and _rows(sandbox) == before
    rolled = _rollback(store, second.activation_generation)
    with pytest.raises(PolicyRevisionConflict):
        _rollback(store, second.activation_generation)
    assert store.activations(_FACILITY) == (rolled,)


def test_one_previous_value_only_and_rollback_exhaustion(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    with pytest.raises(PolicyRollbackUnavailable):
        _rollback(store, 0)
    token = 0
    for threshold in (0.61, 0.62, 0.63):
        applied = _apply(store, threshold, expected=token)
        token = applied.activation_generation
    row = _rows(sandbox)[0]
    assert row["previous_values_json"] == '{"transition_threshold":0.62}'
    store.acknowledge_applied(_FACILITY)
    rolled = _rollback(store, token)
    assert rolled.activation_id == applied.activation_id
    assert rolled.activation_generation == rolled.active_revision_id == token + 1
    assert rolled.status == "pending" and rolled.previous_revision_id is None
    row = _rows(sandbox)[0]
    assert row["active_values_json"] == '{"transition_threshold":0.62}'
    assert row["previous_present"] == 0 and row["previous_values_json"] is None
    assert row["previous_content_sha256"] is None and row["applied_at"] is None
    before = _rows(sandbox)
    with pytest.raises(PolicyRollbackUnavailable):
        _rollback(store, rolled.activation_generation)
    assert _rows(sandbox) == before


@pytest.mark.parametrize("facility_default", [False, True])
def test_first_camera_override_rolls_back_to_inheritance_not_a_copied_default(
    postgres_product_sandbox: ProductSandbox, facility_default: bool
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = _store(sandbox)
    if facility_default:
        _apply(store, 0.61)
    override = _apply(store, 0.81, camera_id=_CAMERA)
    assert override.previous_revision_id == 0
    rolled = _rollback(store, override.activation_generation, camera_id=_CAMERA)
    assert rolled.active_revision_id is None and rolled.previous_revision_id is None
    assert rolled.activation_generation == override.activation_generation + 1
    assert (
        store.diff(**_fall(None, camera_id=_CAMERA)).concurrency_token
        == rolled.activation_generation
    )
    if facility_default:
        _apply(store, 0.73, expected=1)
    effective = store.resolve_bundle(_FACILITY, (PolicyCameraIdentity(_CAMERA),)).resolve(
        _CAMERA, "fall", 2
    )
    assert effective.source == ("facility-default" if facility_default else "image-default")
    assert effective.camera_revision_id is None
    assert policy_values_dict(effective.values) == {
        "transition_threshold": 0.73 if facility_default else 0.5
    }
    with pytest.raises(PolicyRollbackUnavailable):
        _rollback(store, rolled.activation_generation, camera_id=_CAMERA)


def test_nullable_bed_override_noop_and_rollback_keep_canonical_values(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = _store(sandbox)
    default = {"min_containment": 0.4, "hold_frames": 3, "grace_frames": 5}
    override = {"min_containment": 0.6, "hold_frames": 4, "grace_frames": 6}
    proposal = {
        "facility_id": _FACILITY,
        "module_id": "bed_exit",
        "module_version": 1,
        "schema_id": "bed_exit.policy",
        "schema_version": 1,
        "camera_id": None,
        "values": default,
    }
    store.apply(**proposal, expected_revision_id=0)
    camera = proposal | {"camera_id": _CAMERA, "values": override}
    applied = store.apply(**camera, expected_revision_id=0)
    inherited = camera | {"values": None}
    diff = store.diff(**inherited)
    assert diff.changed and diff.proposed.source == "facility-default"
    assert diff.compared_payload["values"] is None
    cleared = store.apply(**inherited, expected_revision_id=diff.concurrency_token)
    assert cleared.active_revision_id is None
    before = _rows(sandbox)
    assert store.apply(**inherited, expected_revision_id=cleared.activation_generation) == cleared
    assert _rows(sandbox) == before
    effective = store.resolve_bundle(_FACILITY, (PolicyCameraIdentity(_CAMERA),)).resolve(
        _CAMERA, "bed_exit", 1
    )
    assert policy_values_dict(effective.values) == default and effective.camera_revision_id is None
    rolled = store.rollback(
        facility_id=_FACILITY,
        module_id="bed_exit",
        module_version=1,
        camera_id=_CAMERA,
        expected_revision_id=cleared.activation_generation,
    )
    assert (
        rolled.activation_generation > cleared.activation_generation > applied.activation_generation
    )
    row = _rows(sandbox)[1]
    encoded = '{"grace_frames":6,"hold_frames":4,"min_containment":0.6}'
    assert row["active_values_json"] == encoded
    assert row["active_content_sha256"] == hashlib.sha256(encoded.encode()).hexdigest()
    assert row["previous_present"] == 0


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"values": None}, "facility default policy values cannot be null"),
        ({"values": {"transition_threshold": True}}, "must be numeric"),
        ({"values": {"transition_threshold": float("nan")}}, "must be finite"),
        ({"values": {"transition_threshold": 1.01}}, "must be in"),
        ({"values": {"transition_threshold": 0.5, "unexpected": 1}}, "unknown field"),
        ({"schema_version": 99}, "schema drift"),
        ({"module_version": 1}, "unsupported policy module version"),
    ],
)
def test_validation_outcomes_stay_distinct_from_driver_and_fencing_refusals(
    postgres_product_sandbox: ProductSandbox, changes: dict[str, object], reason: str
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    hooks = []
    for action in (
        lambda: store.diff(**(_fall() | changes)),
        lambda: store.apply(
            **(_fall() | changes), expected_revision_id=0, after_write=hooks.append
        ),
    ):
        with pytest.raises(PolicyDocumentError, match=reason):
            action()
    with pytest.raises(PolicyDocumentError, match="expected_revision_id"):
        _apply(store, expected=-1)
    with pytest.raises(PolicyDocumentError, match="expected_revision_id"):
        _rollback(store, -1)
    assert not hooks and _rows(sandbox) == []


@pytest.mark.parametrize("projection", ["bundle", "diff"])
@pytest.mark.parametrize("corruption", ["hash", "values", "schema", "previous"])
def test_malformed_records_are_refused_and_fenced_failed_status_is_durable(
    postgres_product_sandbox: ProductSandbox, projection: str, corruption: str
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    first = _apply(store, 0.61)
    latest = _apply(store, 0.72, expected=first.activation_generation)
    store.acknowledge_applied(_FACILITY)
    if corruption == "values":
        encoded = '{"transition_threshold":2.0}'
        sandbox.admin.execute(
            "UPDATE policies SET active_values_json=%s,active_content_sha256=%s",
            (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
        )
        reason = "transition_threshold must be in [0, 1]"
    elif corruption == "schema":
        sandbox.admin.execute("UPDATE policies SET schema_version=99")
        reason = "policy schema drift"
    else:
        column = "previous_content_sha256" if corruption == "previous" else "active_content_sha256"
        sandbox.admin.execute(f"UPDATE policies SET {column}=%s", ("0" * 64,))
        reason = "policy content hash mismatch"
    sandbox.admin.execute("UPDATE policies SET updated_at=%s", (_TIME,))
    before = _rows(sandbox)[0]
    with pytest.raises(PolicyDocumentError):
        store.activations(_FACILITY)
    assert _rows(sandbox)[0] == before
    with pytest.raises(PolicyActivationRefused) as refused:
        if projection == "bundle":
            store.resolve_bundle(_FACILITY, ())
        else:
            store.diff(**_fall())
    assert refused.value.activation_id == latest.activation_id
    assert reason in refused.value.reason
    after = _rows(sandbox)[0]
    assert after == before | {
        "status": "failed",
        "refusal_reason": refused.value.reason,
        "applied_at": None,
        "updated_at": after["updated_at"],
    }
    assert after["updated_at"] != _TIME
    failed = store.activations(_FACILITY)[0]
    assert failed.status == "failed" and failed.active_revision_id is None
    assert failed.activation_generation == latest.activation_generation
    store.acknowledge_applied(_FACILITY)
    assert _rows(sandbox)[0] == after
    freeze_authority(sandbox.database, sandbox.authority)
    # Already-recorded failures are read-only refusals, not implicit new writes.
    with pytest.raises(PolicyActivationRefused, match="refused"):
        store.resolve_bundle(_FACILITY, ())
    assert _rows(sandbox)[0] == after


@pytest.mark.parametrize("camera_id", [_CAMERA, None], ids=["camera-override", "facility-default"])
def test_failed_rollback_restores_valid_previous_numeric_policy(
    postgres_product_sandbox: ProductSandbox, camera_id: str | None
) -> None:
    sandbox = postgres_product_sandbox
    if camera_id is not None:
        _camera(sandbox)
    store = _store(sandbox)
    cameras = () if camera_id is None else (PolicyCameraIdentity(camera_id),)
    first = _apply(store, 0.61, camera_id=camera_id)
    latest = _apply(store, 0.72, camera_id=camera_id, expected=first.activation_generation)
    store.acknowledge_applied(_FACILITY)
    sandbox.admin.execute(
        "UPDATE policies SET active_content_sha256=%s WHERE policy_id=%s",
        ("0" * 64, latest.activation_id),
    )
    with pytest.raises(PolicyActivationRefused, match="content hash mismatch") as refused:
        store.resolve_bundle(_FACILITY, cameras)
    assert refused.value.activation_id == latest.activation_id
    failed = store.activations(_FACILITY)[0]
    assert failed.status == "failed" and failed.active_revision_id is None
    assert failed.activation_generation == latest.activation_generation
    before = _rows(sandbox)
    encoded = '{"transition_threshold":0.61}'
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    assert before[0]["previous_present"] == 1
    assert before[0]["previous_values_json"] == encoded
    assert before[0]["previous_content_sha256"] == digest
    hooks = []

    def hook(connection: psycopg.Connection) -> None:
        assert connection.row_factory is tuple_row
        assert connection.info.transaction_status is TransactionStatus.INTRANS
        with connection.cursor(row_factory=dict_row) as cursor:
            hooks.append(
                cursor.execute(
                    "SELECT * FROM policies WHERE policy_id=%s", (latest.activation_id,)
                ).fetchone()
            )
        _audit(connection, AuditAction.POLICY_ROLLBACK)
        assert _rows(sandbox) == before
        assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)

    with pytest.raises(PolicyRevisionConflict):
        _rollback(store, first.activation_generation, camera_id=camera_id, after_write=hook)
    assert not hooks and _rows(sandbox) == before
    rolled = _rollback(store, failed.activation_generation, camera_id=camera_id, after_write=hook)
    assert rolled.activation_id == latest.activation_id and rolled.camera_id == camera_id
    assert rolled.activation_generation == latest.activation_generation + 1
    assert rolled.status == "pending" and rolled.refusal_reason is None
    assert rolled.previous_revision_id is None
    assert store.generation(_FACILITY) == rolled.activation_generation
    row = _rows(sandbox)[0]
    assert row["activation_generation"] == rolled.activation_generation == rolled.active_revision_id
    assert row["active_values_json"] == encoded and row["active_content_sha256"] == digest
    assert row["previous_present"] == 0 and row["previous_values_json"] is None
    assert row["previous_content_sha256"] is None and row["applied_at"] is None
    assert hooks == [row]
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (1,)
    bundle = store.resolve_bundle(_FACILITY, cameras)
    effective = (
        bundle.defaults["fall"] if camera_id is None else bundle.resolve(camera_id, "fall", 2)
    )
    assert effective.source == ("facility-default" if camera_id is None else "camera-override")
    assert policy_values_dict(effective.values) == {"transition_threshold": 0.61}
    assert store.activations(_FACILITY) == (rolled,)
    with pytest.raises(PolicyRollbackUnavailable):
        _rollback(store, rolled.activation_generation, camera_id=camera_id, after_write=hook)
    assert _rows(sandbox) == [row] and hooks == [row]
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (1,)


@pytest.mark.parametrize("corruption", ["hash", "values"])
def test_failed_rollback_refuses_corrupt_previous_atomically_without_hook(
    postgres_product_sandbox: ProductSandbox, corruption: str
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = _store(sandbox)
    first = _apply(store, 0.61, camera_id=_CAMERA)
    latest = _apply(store, 0.72, camera_id=_CAMERA, expected=first.activation_generation)
    if corruption == "hash":
        sandbox.admin.execute(
            "UPDATE policies SET previous_content_sha256=%s WHERE policy_id=%s",
            ("0" * 64, latest.activation_id),
        )
        reason = "policy content hash mismatch"
    else:
        encoded = '{"transition_threshold":2.0}'
        sandbox.admin.execute(
            "UPDATE policies SET previous_values_json=%s,previous_content_sha256=%s "
            "WHERE policy_id=%s",
            (encoded, hashlib.sha256(encoded.encode()).hexdigest(), latest.activation_id),
        )
        reason = "transition_threshold must be in [0, 1]"
    with pytest.raises(PolicyActivationRefused) as refused:
        store.resolve_bundle(_FACILITY, (PolicyCameraIdentity(_CAMERA),))
    assert refused.value.activation_id == latest.activation_id
    assert refused.value.reason == reason
    failed = store.activations(_FACILITY)[0]
    assert failed.status == "failed" and failed.refusal_reason == reason
    assert failed.activation_generation == latest.activation_generation
    before = _rows(sandbox)
    site = sandbox.admin.execute("SELECT * FROM edge_site").fetchall()
    hooks = []
    with pytest.raises(PolicyDocumentError) as invalid:
        _rollback(store, failed.activation_generation, camera_id=_CAMERA, after_write=hooks.append)
    assert str(invalid.value) == reason
    assert not hooks and _rows(sandbox) == before
    assert sandbox.admin.execute("SELECT * FROM edge_site").fetchall() == site
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
    assert store.generation(_FACILITY) == latest.activation_generation
    assert store.activations(_FACILITY) == (failed,)


@pytest.mark.parametrize("facility_default", [False, True])
def test_failed_camera_rollback_restores_genuine_previous_inheritance(
    postgres_product_sandbox: ProductSandbox, facility_default: bool
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = _store(sandbox)
    if facility_default:
        _apply(store, 0.61)
    cameras = (PolicyCameraIdentity(_CAMERA),)
    inherited = store.resolve_bundle(_FACILITY, cameras).resolve(_CAMERA, "fall", 2)
    override = _apply(store, 0.81, camera_id=_CAMERA)
    previous = next(row for row in _rows(sandbox) if row["policy_id"] == override.activation_id)
    assert previous["previous_present"] == 1
    assert previous["previous_values_json"] is None and previous["previous_content_sha256"] is None
    sandbox.admin.execute(
        "UPDATE policies SET active_content_sha256=%s WHERE policy_id=%s",
        ("0" * 64, override.activation_id),
    )
    with pytest.raises(PolicyActivationRefused, match="content hash mismatch"):
        store.resolve_bundle(_FACILITY, cameras)
    failed = next(item for item in store.activations(_FACILITY) if item.camera_id == _CAMERA)
    assert failed.status == "failed" and failed.refusal_reason == "policy content hash mismatch"
    assert failed.activation_generation == override.activation_generation
    rolled = _rollback(store, failed.activation_generation, camera_id=_CAMERA)
    assert rolled.activation_id == override.activation_id
    assert rolled.activation_generation == override.activation_generation + 1
    assert rolled.active_revision_id is None and rolled.previous_revision_id is None
    assert rolled.status == "pending" and rolled.refusal_reason is None
    assert store.generation(_FACILITY) == rolled.activation_generation
    row = next(row for row in _rows(sandbox) if row["policy_id"] == override.activation_id)
    assert row["active_values_json"] is None and row["active_content_sha256"] is None
    assert row["previous_present"] == 0 and row["previous_values_json"] is None
    assert row["previous_content_sha256"] is None and row["applied_at"] is None
    effective = store.resolve_bundle(_FACILITY, cameras).resolve(_CAMERA, "fall", 2)
    assert effective == inherited
    assert effective.source == ("facility-default" if facility_default else "image-default")
    assert effective.camera_revision_id is None
    before = _rows(sandbox)
    with pytest.raises(PolicyRollbackUnavailable):
        _rollback(store, rolled.activation_generation, camera_id=_CAMERA)
    assert _rows(sandbox) == before


@pytest.mark.parametrize("read_before_repair", [False, True])
def test_apply_recovers_corrupt_active_without_copying_it_into_history(
    postgres_product_sandbox: ProductSandbox, read_before_repair: bool
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    first = _apply(store)
    sandbox.admin.execute("UPDATE policies SET active_content_sha256=%s", ("0" * 64,))
    if read_before_repair:
        with pytest.raises(PolicyActivationRefused, match="content hash mismatch"):
            store.resolve_bundle(_FACILITY, ())
    repaired = _apply(store, 0.75, expected=first.activation_generation)
    assert repaired.activation_generation == first.activation_generation + 1
    assert repaired.previous_revision_id is None and repaired.status == "pending"
    assert policy_values_dict(store.resolve_bundle(_FACILITY, ()).defaults["fall"].values) == {
        "transition_threshold": 0.75
    }


@pytest.mark.parametrize("corrupt_previous", [False, True])
def test_repair_retains_only_valid_previous_state_and_refuses_invalid_history(
    postgres_product_sandbox: ProductSandbox, corrupt_previous: bool
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    _apply(store, 0.61)
    latest = _apply(store, 0.72, expected=1)
    column = "previous_content_sha256" if corrupt_previous else "active_content_sha256"
    sandbox.admin.execute(f"UPDATE policies SET {column}=%s", ("0" * 64,))
    with pytest.raises(PolicyActivationRefused, match="content hash mismatch"):
        store.resolve_bundle(_FACILITY, ())
    before = _rows(sandbox)
    hooks = []
    if corrupt_previous:
        with pytest.raises(PolicyDocumentError, match="content hash mismatch"):
            _apply(store, 0.81, expected=latest.activation_generation, after_write=hooks.append)
        assert not hooks and _rows(sandbox) == before
    else:
        repaired = _apply(store, 0.81, expected=latest.activation_generation)
        assert repaired.previous_revision_id == max(1, repaired.activation_generation - 1)
        assert _rows(sandbox)[0]["previous_values_json"] == '{"transition_threshold":0.61}'
        _rollback(store, repaired.activation_generation)
        assert _rows(sandbox)[0]["active_values_json"] == '{"transition_threshold":0.61}'


@pytest.mark.parametrize("unknown_field", ["x" * 500, "unknown\x00field"])
def test_unstorable_validation_reason_does_not_turn_into_a_driver_refusal(
    postgres_product_sandbox: ProductSandbox, unknown_field: str
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    first = _apply(store)
    encoded = json.dumps({"transition_threshold": 0.62, unknown_field: 1})
    sandbox.admin.execute(
        "UPDATE policies SET active_values_json=%s,active_content_sha256=%s",
        (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
    )
    reason = f"fall policy contains unknown field(s): {unknown_field}"
    with pytest.raises(PolicyActivationRefused) as refused:
        store.resolve_bundle(_FACILITY, ())
    assert refused.value.activation_id == first.activation_id
    assert refused.value.reason == reason
    row = _rows(sandbox)[0]
    assert row["status"] == "failed"
    assert row["refusal_reason"] == reason.replace("\x00", r"\u0000")[:256]


def test_stale_corruption_snapshot_cannot_mark_a_concurrent_repair_failed(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    first = _apply(store)
    sandbox.admin.execute("UPDATE policies SET active_content_sha256=%s", ("0" * 64,))
    transact = sandbox.database.transact
    repairs = []
    with _other_database(sandbox) as database:
        other = DetectionPolicyStore(database, sandbox.authority)

        def interleave(callback):
            repairs.append(_apply(other, 0.78, expected=first.activation_generation))
            return transact(callback)

        with monkeypatch.context() as patch:
            patch.setattr(sandbox.database, "transact", interleave)
            with pytest.raises(PolicyActivationRefused, match="content hash mismatch"):
                store.resolve_bundle(_FACILITY, ())
    assert len(repairs) == 1
    assert store.activations(_FACILITY) == tuple(repairs)
    assert _rows(sandbox)[0]["refusal_reason"] is None


def test_registry_alias_remapping_and_namespace_collision_keep_native_policy_reference(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    registry = _camera(sandbox)
    _camera(sandbox, camera_id=_CAMERA, backend_id="second-worker-camera")
    store = _store(sandbox)
    applied = _apply(store, 0.81, camera_id=_CAMERA)
    assert _rows(sandbox)[0]["camera_id"] == _LOCAL
    assert (
        store.diff(**_fall(0.81, camera_id=_LOCAL)).concurrency_token
        == applied.activation_generation
    )
    cameras = (PolicyCameraIdentity(_CAMERA), PolicyCameraIdentity("second-worker-camera"))
    bundle = store.resolve_bundle(_FACILITY, cameras)
    assert bundle.resolve(_CAMERA, "fall", 2).source == "camera-override"
    assert bundle.resolve("second-worker-camera", "fall", 2).source == "image-default"
    registry.update(_LOCAL, {"backend_camera_id": _REMAPPED})
    renamed = store.activations(_FACILITY)[0]
    assert renamed.camera_id == _REMAPPED and renamed.activation_id == applied.activation_id
    assert (
        _apply(store, 0.81, camera_id=_REMAPPED, expected=applied.activation_generation) == renamed
    )
    registry.update(_LOCAL, {"backend_camera_id": None})
    assert store.activations(_FACILITY)[0].camera_id == _LOCAL
    assert _rows(sandbox)[0]["camera_id"] == _LOCAL
    # The restrictive native reference prevents deleting a camera with policies.
    with pytest.raises(CameraRegistryWriteError):
        registry.delete(_LOCAL)
    assert registry.get(_LOCAL) is not None and len(_rows(sandbox)) == 1


def test_native_driver_refusals_hide_row_values_and_never_supply_a_default_bundle(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    private_id = "private-camera-fragment-not-for-diagnostics"
    hooks = []
    with pytest.raises(PolicyActivationRefused) as refused:
        _apply(store, camera_id=private_id, after_write=hooks.append)
    assert refused.value.reason == "policy database operation failed"
    assert private_id not in "".join(traceback.format_exception(refused.value))
    assert refused.value.__suppress_context__ and refused.value.__cause__ is None
    assert not hooks and _rows(sandbox) == []
    # A real server error on the read path must not masquerade as image defaults.
    sandbox.admin.execute("ALTER TABLE policies RENAME TO unavailable_policies")
    with pytest.raises(PolicyActivationRefused, match="policy database operation failed"):
        store.resolve_bundle(_FACILITY, ())


@pytest.mark.parametrize("operation", ["apply", "noop", "rollback", "ack", "mark-failed"])
@pytest.mark.parametrize("fence", ["stale-generation", "stale-token", "disabled"])
def test_authority_precedes_policy_lookup_and_all_noop_paths(
    postgres_product_sandbox: ProductSandbox, operation: str, fence: str
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    first = _apply(store, 0.61)
    latest = _apply(store, 0.72, expected=first.activation_generation)
    store.acknowledge_applied(_FACILITY)
    if operation == "mark-failed":
        sandbox.admin.execute("UPDATE policies SET active_content_sha256=%s", ("0" * 64,))
    if fence == "disabled":
        freeze_authority(sandbox.database, sandbox.authority)
        authority = sandbox.authority
    else:
        authority = AuthorityToken(
            sandbox.authority.generation + (fence == "stale-generation"),
            uuid4() if fence == "stale-token" else sandbox.authority.writer_token,
        )
    fenced = DetectionPolicyStore(sandbox.database, authority)
    before = _rows(sandbox)
    hooks = []

    def invoke() -> None:
        if operation == "ack":
            fenced.acknowledge_applied(_FACILITY)
        elif operation == "mark-failed":
            fenced.resolve_bundle(_FACILITY, ())
        elif operation == "rollback":
            _rollback(fenced, latest.activation_generation, after_write=hooks.append)
        else:
            _apply(
                fenced,
                0.72 if operation == "noop" else 0.81,
                expected=latest.activation_generation,
                after_write=hooks.append,
            )

    assert fenced.generation(_FACILITY) == latest.activation_generation
    with pytest.raises(AuthorityFenced):
        invoke()
    assert not hooks and _rows(sandbox) == before
    if operation != "mark-failed":
        sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
        with pytest.raises(AuthorityFenced):
            invoke()
        assert not hooks and _rows(sandbox) == before


def test_missing_bootstrap_refuses_reads_writes_and_noops_without_creating_state(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    first = _apply(_store(sandbox))
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    store = _store(sandbox)
    before = _rows(sandbox)
    hooks = []
    for action in (
        lambda: store.generation(_FACILITY),
        lambda: store.diff(**_fall()),
        lambda: store.resolve_bundle(_FACILITY, ()),
        lambda: store.activations(_FACILITY),
        lambda: _apply(store, expected=first.activation_generation, after_write=hooks.append),
        lambda: _apply(store, 0.81, expected=first.activation_generation),
        lambda: _rollback(store, first.activation_generation),
        lambda: store.acknowledge_applied(_FACILITY),
    ):
        with pytest.raises(DetectionPolicyNotInitialized, match="bootstrap row is missing"):
            action()
    assert not hooks and _rows(sandbox) == before
    assert sandbox.admin.execute("SELECT count(*) FROM edge_site").fetchone() == (0,)


@pytest.mark.parametrize("operation", ["apply", "noop", "rollback"])
@pytest.mark.parametrize("fail", [False, True], ids=["commit", "callback-rollback"])
def test_after_write_shares_transaction_and_policy_audit_business_atomicity(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    fail: bool,
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    _apply(store, 0.61)
    latest = _apply(store, 0.72, expected=1)
    before = _rows(sandbox)
    site = sandbox.admin.execute("SELECT * FROM edge_site").fetchall()
    transact = sandbox.database.transact
    connections = []
    hooks = []

    def observe_transaction(callback):
        def observe(connection):
            connections.append(id(connection))
            return callback(connection)

        return transact(observe)

    def hook(connection: psycopg.Connection) -> None:
        hooks.append(id(connection))
        assert hooks == connections and connection.row_factory is tuple_row
        assert connection.info.transaction_status is TransactionStatus.INTRANS
        assert connection.execute("SHOW transaction_isolation").fetchone() == ("read committed",)
        assert connection.execute("SHOW transaction_read_only").fetchone() == ("off",)
        expected = {"apply": 0.81, "noop": 0.72, "rollback": 0.61}[operation]
        encoded = connection.execute("SELECT active_values_json FROM policies").fetchone()[0]
        assert json.loads(encoded) == {"transition_threshold": expected}
        connection.execute("UPDATE edge_site SET clip_export_enabled=1 WHERE id=1")
        _audit(
            connection,
            AuditAction.POLICY_ROLLBACK if operation == "rollback" else AuditAction.POLICY_APPLY,
        )
        assert _rows(sandbox) == before
        assert sandbox.admin.execute("SELECT * FROM edge_site").fetchall() == site
        assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
        if fail:
            raise RuntimeError("injected policy callback failure")

    monkeypatch.setattr(sandbox.database, "transact", observe_transaction)

    def invoke():
        if operation == "rollback":
            return _rollback(store, latest.activation_generation, after_write=hook)
        return _apply(
            store,
            0.72 if operation == "noop" else 0.81,
            expected=latest.activation_generation,
            after_write=hook,
        )

    if fail:
        with pytest.raises(RuntimeError, match="injected policy callback failure"):
            invoke()
        assert _rows(sandbox) == before
        assert sandbox.admin.execute("SELECT * FROM edge_site").fetchall() == site
    else:
        saved = invoke()
        assert saved.activation_generation == latest.activation_generation + (operation != "noop")
        assert sandbox.admin.execute("SELECT clip_export_enabled FROM edge_site").fetchone() == (1,)
    assert len(connections) == len(hooks) == 1
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (int(not fail),)
    assert sandbox.database.read(lambda connection: connection.row_factory is tuple_row)


def test_real_deferred_commit_rejection_rolls_back_policy_history_and_callback_writes(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    first = _apply(store)
    before = _rows(sandbox)
    sandbox.admin.execute(
        "CREATE TABLE policy_commit_guard (site_id bigint REFERENCES edge_site(id) "
        "DEFERRABLE INITIALLY DEFERRED)"
    )
    hooks = []

    def reject(connection: psycopg.Connection) -> None:
        hooks.append(True)
        _audit(connection)
        connection.execute("UPDATE edge_site SET clip_export_enabled=1 WHERE id=1")
        connection.execute("INSERT INTO policy_commit_guard VALUES (2)")
        assert connection.info.transaction_status is TransactionStatus.INTRANS

    with pytest.raises(PolicyActivationRefused, match="policy database operation failed"):
        _apply(store, 0.81, expected=first.activation_generation, after_write=reject)
    assert hooks == [True] and _rows(sandbox) == before
    assert sandbox.admin.execute("SELECT clip_export_enabled FROM edge_site").fetchone() == (0,)
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
    assert sandbox.admin.execute("SELECT count(*) FROM policy_commit_guard").fetchone() == (0,)


@pytest.mark.parametrize("operation", ["apply", "noop", "rollback", "ack", "mark-failed"])
@pytest.mark.parametrize(
    "committed", [False, True], ids=["lost-before-commit", "lost-after-commit"]
)
def test_real_unknown_commit_never_replays_reloads_or_publishes_activation(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    committed: bool,
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    _apply(store, 0.61)
    latest = _apply(store, 0.72, expected=1)
    if operation == "mark-failed":
        sandbox.admin.execute("UPDATE policies SET active_content_sha256=%s", ("0" * 64,))
    before = _rows(sandbox)
    transact = sandbox.database.transact
    read_snapshot = sandbox.database.read_snapshot
    commit = psycopg.Connection.commit
    attempts = []
    commits = []
    reads = []
    published = []

    def observe_transaction(callback):
        def observe(connection):
            attempts.append(connection.info.backend_pid)
            return callback(connection)

        return transact(observe)

    def observe_read(callback):
        reads.append(True)
        return read_snapshot(callback)

    def lose_receipt(connection: psycopg.Connection) -> None:
        pid = connection.info.backend_pid
        if pid in attempts:
            commits.append(pid)
            if committed:
                commit(connection)  # Actual server COMMIT, independently observed below.
            raise psycopg.OperationalError("injected COMMIT receipt loss")
        commit(connection)

    def invoke():
        if operation == "ack":
            return store.acknowledge_applied(_FACILITY)
        if operation == "mark-failed":
            return store.resolve_bundle(_FACILITY, ())
        if operation == "rollback":
            return _rollback(store, latest.activation_generation, after_write=_audit)
        return _apply(
            store,
            0.72 if operation == "noop" else 0.81,
            expected=latest.activation_generation,
            after_write=_audit,
        )

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.database, "transact", observe_transaction)
        patch.setattr(sandbox.database, "read_snapshot", observe_read)
        patch.setattr(psycopg.Connection, "commit", lose_receipt)
        with pytest.raises(CommitOutcomeUnknown):
            published.append(invoke())
    assert not published and len(attempts) == len(commits) == 1
    assert len(reads) == int(operation == "mark-failed")
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (
        int(committed and operation in {"apply", "noop", "rollback"}),
    )
    after = _rows(sandbox)
    if not committed or operation == "noop":
        assert after == before
    elif operation == "ack":
        assert after[0]["status"] == "applied" and after[0]["applied_at"] is not None
        assert after[0]["activation_generation"] == latest.activation_generation
    elif operation == "mark-failed":
        assert after[0]["status"] == "failed"
        assert after[0]["refusal_reason"] == "policy content hash mismatch"
        assert after[0]["activation_generation"] == latest.activation_generation
    else:
        assert after[0]["activation_generation"] == latest.activation_generation + 1
        assert json.loads(after[0]["active_values_json"]) == {
            "transition_threshold": 0.61 if operation == "rollback" else 0.81
        }
        assert after[0]["previous_present"] == int(operation == "apply")


def test_pool_context_failure_after_real_commit_does_not_return_candidate(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    connection_context = sandbox.database._pool.connection
    published = []
    exits = []

    @contextmanager
    def fail_release(*args, **kwargs):
        with connection_context(*args, **kwargs) as connection:
            yield connection
        exits.append(True)
        raise RuntimeError("injected pool context exit failure")

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.database._pool, "connection", fail_release)
        with pytest.raises(RuntimeError, match="pool context exit failure"):
            published.append(_apply(store, after_write=_audit))
    assert not published and exits == [True]
    assert _rows(sandbox)[0]["activation_generation"] == 1
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (1,)


@pytest.mark.parametrize("operation", ["apply", "noop", "rollback"])
def test_returned_activation_is_own_commit_not_a_later_writers_projection(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    sandbox = postgres_product_sandbox
    store = _store(sandbox)
    _apply(store, 0.61)
    latest = _apply(store, 0.72, expected=1)
    transact = sandbox.database.transact
    candidates = []
    later = []
    with _other_database(sandbox) as database:
        other = DetectionPolicyStore(database, sandbox.authority)

        def interleave(callback):
            candidate = transact(callback)
            candidates.append(candidate)
            later.append(_apply(other, 0.93, expected=candidate.activation_generation))
            return candidate

        with monkeypatch.context() as patch:
            patch.setattr(sandbox.database, "transact", interleave)
            if operation == "rollback":
                saved = _rollback(store, latest.activation_generation)
            else:
                saved = _apply(
                    store,
                    0.72 if operation == "noop" else 0.81,
                    expected=latest.activation_generation,
                )
    assert candidates == [saved]
    assert saved.activation_generation == latest.activation_generation + (operation != "noop")
    assert later[0].activation_generation == saved.activation_generation + 1
    assert store.activations(_FACILITY) == tuple(later)


def test_acknowledgement_marks_latest_pending_only_in_requested_facility_and_is_idempotent(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = _store(sandbox)
    store.acknowledge_applied(_FACILITY)
    first = _apply(store)
    second = _apply(store, 0.81, camera_id=_CAMERA)
    elsewhere = store.apply(**_fall(facility_id="other-facility"), expected_revision_id=0)
    assert elsewhere.activation_generation == 1
    store.acknowledge_applied(_FACILITY)
    rows = _rows(sandbox)
    assert [row["status"] for row in rows] == ["applied", "applied", "pending"]
    assert rows[0]["applied_at"] == rows[1]["applied_at"]
    assert rows[0]["applied_at"] == rows[0]["updated_at"]
    store.acknowledge_applied(_FACILITY)
    assert _rows(sandbox) == rows
    newer = _apply(store, 0.72, expected=first.activation_generation)
    assert newer.activation_generation == second.activation_generation + 1
    assert [activation.status for activation in store.activations(_FACILITY)] == [
        "pending",
        "applied",
    ]
    # Source semantics intentionally bind to latest inside ACK, not a wire generation.
    store.acknowledge_applied(_FACILITY)
    assert {activation.status for activation in store.activations(_FACILITY)} == {"applied"}
    assert store.activations("other-facility")[0].status == "pending"


@pytest.mark.parametrize("projection", ["bundle", "diff"])
def test_multiquery_projection_keeps_one_readonly_snapshot_across_policy_and_registry_writes(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch, projection: str
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = _store(sandbox)
    _apply(store, 0.61)
    _apply(store, 0.81, camera_id=_CAMERA)
    read_snapshot = sandbox.database.read_snapshot
    execute = psycopg.Cursor.execute
    reads = []
    interleaved = []
    with _other_database(sandbox) as database:
        other = DetectionPolicyStore(database, sandbox.authority)
        registry = CameraRegistryStore(database, sandbox.authority)

        def observe_snapshot(callback):
            def observe(connection):
                reads.append(connection.info.backend_pid)
                assert connection.row_factory is tuple_row
                assert connection.execute("SHOW transaction_read_only").fetchone() == ("on",)
                assert connection.execute("SHOW transaction_isolation").fetchone() == (
                    "repeatable read",
                )
                return callback(connection)

            return read_snapshot(observe)

        def interleave(cursor, query, *args, **kwargs):
            result = execute(cursor, query, *args, **kwargs)
            if (
                cursor.connection.info.backend_pid in reads
                and isinstance(query, str)
                and query.startswith("SELECT p.policy_id")
                and not interleaved
            ):
                interleaved.append(True)
                _apply(other, 0.72, expected=1)
                registry.update(_LOCAL, {"backend_camera_id": _REMAPPED})
            return result

        with monkeypatch.context() as patch:
            patch.setattr(sandbox.database, "read_snapshot", observe_snapshot)
            patch.setattr(psycopg.Cursor, "execute", interleave)
            if projection == "bundle":
                bundle = store.resolve_bundle(_FACILITY, (PolicyCameraIdentity(_CAMERA),))
                current = bundle.resolve(_CAMERA, "fall", 2)
                inherited = bundle.defaults["fall"]
            else:
                diff = store.diff(**_fall(None, camera_id=_CAMERA))
                assert diff.changed and diff.concurrency_token == 2
                current, inherited = diff.current, diff.proposed
    assert len(reads) == 1 and interleaved == [True]
    assert current.source == "camera-override" and current.camera_revision_id == 2
    assert current.facility_revision_id == inherited.facility_revision_id == 1
    assert policy_values_dict(inherited.values) == {"transition_threshold": 0.61}
    fresh = store.resolve_bundle(_FACILITY, (PolicyCameraIdentity(_REMAPPED),))
    assert policy_values_dict(fresh.defaults["fall"].values) == {"transition_threshold": 0.72}
    assert fresh.resolve(_REMAPPED, "fall", 2).camera_revision_id == 2
    assert fresh.resolve(_REMAPPED, "fall", 2).facility_revision_id == 3


def test_unknown_snapshot_commit_cannot_publish_or_replace_overrides_with_defaults(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = _store(sandbox)
    _apply(store, 0.61)
    _apply(store, 0.81, camera_id=_CAMERA)
    cameras = (PolicyCameraIdentity(_CAMERA),)
    expected = store.resolve_bundle(_FACILITY, cameras)
    read_snapshot = sandbox.database.read_snapshot
    commit = psycopg.Connection.commit
    reads = []
    commits = []
    published = []

    def observe_snapshot(callback):
        def observe(connection):
            reads.append(connection.info.backend_pid)
            return callback(connection)

        return read_snapshot(observe)

    def lose_receipt(connection: psycopg.Connection) -> None:
        pid = connection.info.backend_pid
        commit(connection)
        if pid in reads:
            commits.append(pid)
            raise psycopg.OperationalError("injected snapshot COMMIT receipt loss")

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.database, "read_snapshot", observe_snapshot)
        patch.setattr(psycopg.Connection, "commit", lose_receipt)
        with pytest.raises(CommitOutcomeUnknown):
            published.append(store.resolve_bundle(_FACILITY, cameras))
    assert not published and len(reads) == len(commits) == 1
    assert store.resolve_bundle(_FACILITY, cameras) == expected


def _wait_blocked(sandbox: ProductSandbox, pids: list[int]) -> None:
    deadline = monotonic() + 2
    pause = Event()
    while monotonic() < deadline:
        waiting = sandbox.admin.execute(
            "SELECT count(DISTINCT pid) FROM pg_locks WHERE pid=ANY(%s) AND NOT granted",
            (pids,),
        ).fetchone()
        if waiting == (len(pids),):
            return
        pause.wait(0.01)
    pytest.fail("policy/config writers did not wait on the singleton row lock")


@pytest.mark.parametrize("same_scope", [False, True])
def test_independent_owners_serialize_facility_generations_and_optimistic_decisions(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch, same_scope: bool
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    ready = Barrier(3)
    pids: Queue[int] = Queue()

    def synchronize(transact):
        def transaction(callback):
            def before(connection):
                pids.put(connection.info.backend_pid)
                ready.wait(timeout=2)
                return callback(connection)

            return transact(before)

        return transaction

    def attempt(store, threshold, camera_id):
        try:
            return _apply(store, threshold, camera_id=camera_id)
        except PolicyRevisionConflict as error:
            return error

    with _other_database(sandbox) as database:
        first, second = _store(sandbox), DetectionPolicyStore(database, sandbox.authority)
        with monkeypatch.context() as patch:
            patch.setattr(sandbox.database, "transact", synchronize(sandbox.database.transact))
            patch.setattr(database, "transact", synchronize(database.transact))
            with ThreadPoolExecutor(max_workers=2) as pool:
                with sandbox.admin.transaction():
                    sandbox.admin.execute("SELECT id FROM edge_site WHERE id=1 FOR UPDATE")
                    one = pool.submit(attempt, first, 0.61, None)
                    two = pool.submit(attempt, second, 0.72, None if same_scope else _CAMERA)
                    worker_pids = [pids.get(timeout=2), pids.get(timeout=2)]
                    assert len(set(worker_pids)) == 2
                    ready.wait(timeout=2)
                    _wait_blocked(sandbox, worker_pids)
                results = [one.result(timeout=5), two.result(timeout=5)]
        successes = [result for result in results if isinstance(result, PolicyActivation)]
        conflicts = [result for result in results if isinstance(result, PolicyRevisionConflict)]
        assert len(conflicts) == int(same_scope)
        assert sorted(result.activation_generation for result in successes) == (
            [1] if same_scope else [1, 2]
        )
        assert first.generation(_FACILITY) == second.generation(_FACILITY) == len(successes)
        facility = next(result for result in successes if result.camera_id is None)
        later = _apply(second, 0.93, expected=facility.activation_generation)
        assert later.activation_generation == len(successes) + 1
        assert len(_rows(sandbox)) == len(successes)


@pytest.mark.parametrize("operation", ["delete", "remap"])
def test_registry_write_lock_precedes_policy_camera_lookup_across_independent_owners(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    sandbox = postgres_product_sandbox
    registry = _camera(sandbox)
    entered, release = Event(), Event()
    policy_pids: Queue[int] = Queue()

    def hold(connection: psycopg.Connection) -> None:
        assert connection.info.transaction_status is TransactionStatus.INTRANS
        entered.set()
        assert release.wait(timeout=2), "registry write was not released"

    with _other_database(sandbox) as database:
        store = DetectionPolicyStore(database, sandbox.authority)
        transact = database.transact

        def observe(callback):
            def before(connection):
                policy_pids.put(connection.info.backend_pid)
                return callback(connection)

            return transact(before)

        monkeypatch.setattr(database, "transact", observe)
        with ThreadPoolExecutor(max_workers=2) as pool:
            if operation == "delete":
                change = pool.submit(registry.delete, _LOCAL, after_write=hold)
            else:
                change = pool.submit(
                    registry.update, _LOCAL, {"backend_camera_id": _REMAPPED}, after_write=hold
                )
            try:
                assert entered.wait(timeout=2)
                policy = pool.submit(
                    _apply, store, 0.81, camera_id=_CAMERA if operation == "delete" else _REMAPPED
                )
                _wait_blocked(sandbox, [policy_pids.get(timeout=2)])
            finally:
                release.set()
            assert change.result(timeout=5)
            if operation == "delete":
                with pytest.raises(
                    PolicyActivationRefused, match="policy database operation failed"
                ):
                    policy.result(timeout=5)
                assert _rows(sandbox) == []
            else:
                saved = policy.result(timeout=5)
                assert saved.camera_id == _REMAPPED and saved.activation_generation == 1
                assert _rows(sandbox)[0]["camera_id"] == _LOCAL
