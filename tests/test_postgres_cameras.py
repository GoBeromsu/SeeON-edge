"""Camera and topology persistence against the shared isolated PostgreSQL sandbox."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Barrier, Event
from time import monotonic
from typing import TYPE_CHECKING
from uuid import UUID

import psycopg
import pytest
from psycopg.pq import TransactionStatus
from psycopg.rows import tuple_row

from backend.app.edge_db.authority import AuthorityFenced, freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_store import append_postgres_audit
from backend.app.features.audit.store import AuditEvent
from backend.app.features.cameras.camera_repository import (
    CameraRegistryNotInitialized,
    CameraRegistryWriteError,
)
from backend.app.features.cameras.store import (
    CameraRegistryStore,
    DuplicateCameraError,
    public_camera,
    registry_expected_cameras,
)
from backend.app.features.cameras.topology import TopologyConflictError, TopologyErrorCode
from backend.app.features.cameras.update_command import CameraUpdate
from contracts.edge_provisioning_models import EdgeErrorCode

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_TIME = "2026-09-27T04:00:00.123Z"
_SPACE = "a2222222-2222-4222-8222-222222222222"
_URL = "rtsp://operator:synthetic-private@camera.invalid/live"
_ACTIONS = {
    "create": AuditAction.CAMERA_CREATE,
    "update": AuditAction.CAMERA_UPDATE,
    "delete": AuditAction.CAMERA_DELETE,
    "create_floor": AuditAction.LOCATION_CREATE,
    "update_floor": AuditAction.LOCATION_UPDATE,
    "delete_floor": AuditAction.LOCATION_DELETE,
    "create_room": AuditAction.LOCATION_CREATE,
    "update_room": AuditAction.LOCATION_UPDATE,
    "delete_room": AuditAction.LOCATION_DELETE,
}


def _create(store: CameraRegistryStore, camera_id: str = "camera-a", **overrides):
    return store.create(
        **{
            "camera_id": camera_id,
            "label": camera_id,
            "rtsp_url": f"rtsp://{camera_id}.invalid/live",
            "space_id": None,
            "status": "online",
            **overrides,
        }
    )


def _seed_topology(store: CameraRegistryStore) -> None:
    store.create_floor(edge_ref="floor-a", name="First", order_index=1)
    store.create_room(
        edge_ref="room-a",
        floor_edge_ref="floor-a",
        name="101",
        legacy_canonical_space_id=_SPACE,
    )
    _create(store, edge_ref="edge-a", room_edge_ref="room-a", rtsp_url=_URL)


def _rows(sandbox: ProductSandbox):
    return (
        sandbox.admin.execute("SELECT * FROM edge_site ORDER BY id").fetchall(),
        sandbox.admin.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall(),
        sandbox.admin.execute("SELECT * FROM locations ORDER BY location_id,kind").fetchall(),
    )


def _audit(action: AuditAction = AuditAction.CAMERA_UPDATE) -> AuditEvent:
    return AuditEvent(
        occurred_at=_TIME,
        actor_id="test-operator",
        action=action,
        target_id="camera-a",
        detail=empty_detail(action),
    )


def _mutation(store: CameraRegistryStore, name: str, after_write=None):
    operations = {
        "create": lambda: _create(store, "camera-b", status="offline", after_write=after_write),
        "update": lambda: store.update(
            "camera-a",
            CameraUpdate.model_validate({"label": "Changed", "status": "offline"}),
            after_write=after_write,
        ),
        "delete": lambda: store.delete("camera-a", after_write=after_write),
        "create_floor": lambda: store.create_floor(
            edge_ref="floor-new", name="New", order_index=3, after_write=after_write
        ),
        "update_floor": lambda: store.update_floor(
            "floor-a", name="Changed", order_index=4, after_write=after_write
        ),
        "delete_floor": lambda: store.delete_floor("floor-empty", after_write=after_write),
        "create_room": lambda: store.create_room(
            edge_ref="room-new", floor_edge_ref="floor-a", name="New", after_write=after_write
        ),
        "update_room": lambda: store.update_room("room-a", name="Changed", after_write=after_write),
        "delete_room": lambda: store.delete_room("room-empty", after_write=after_write),
        "migrate": store.migrate_legacy_string_floors,
    }
    return operations[name]()


def test_crud_complete_registry_public_output_and_revisions(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    created = _create(
        store,
        label="Bed camera",
        rtsp_url=_URL,
        space_id=_SPACE,
        mapping_pending=True,
        decode_backend="hardware",
        floor=-1,
        last_probed_at=_TIME,
    )
    incarnation = sandbox.admin.execute("SELECT incarnation FROM cameras").fetchone()[0]
    assert isinstance(incarnation, UUID)
    assert store._statuses == {"camera-a": (incarnation, "online")}
    expected = {
        "id": "camera-a",
        "label": "Bed camera",
        "rtsp_url": _URL,
        "space_id": _SPACE,
        "backend_camera_id": None,
        "mapping_pending": True,
        "status": "online",
        "decode_backend": "hardware",
        "floor": -1,
        "created_at": created["created_at"],
        "last_probed_at": _TIME,
        "last_ok_at": None,
        "never_connected": True,
        "edge_ref": None,
        "room_edge_ref": None,
    }
    assert created == expected
    assert store.get("camera-a") == expected
    assert store.snapshot() == {"registry_version": 1, "cameras": [expected]}
    assert public_camera(created) == {
        "id": "camera-a",
        "label": "Bed camera",
        "rtsp_url_masked": "rtsp://***:***@redacted-camera/live",
        "space_id": _SPACE,
        "backend_camera_id": None,
        "mapping_pending": True,
        "status": "online",
        "decode_backend": "hardware",
        "floor": -1,
        "created_at": created["created_at"],
        "never_connected": True,
        "last_ok_at": None,
        "last_probed_at": _TIME,
    }
    updated = store.update(
        "camera-a",
        CameraUpdate.model_validate(
            {
                "label": "Renamed",
                "backend_camera_id": "hub-a",
                "mapping_pending": True,
                "floor": 10,
                "status": "offline",
                "never_connected": False,
                "last_ok_at": _TIME,
            }
        ),
    )
    assert updated == expected | {
        "label": "Renamed",
        "backend_camera_id": "hub-a",
        "mapping_pending": False,
        "floor": 10,
        "status": "offline",
        "never_connected": False,
        "last_ok_at": _TIME,
    }
    assert store.get("camera-a") == updated
    assert store.snapshot() == {"registry_version": 2, "cameras": [updated]}
    assert store._statuses == {"camera-a": (incarnation, "offline")}
    assert sandbox.admin.execute(
        "SELECT incarnation,mapping_state,never_connected,floor_override,revision FROM cameras"
    ).fetchone() == (incarnation, "MAPPED", 0, "10", 2)
    assert sandbox.admin.execute(
        "SELECT registry_version,topology_dirty_registry_version FROM edge_site"
    ).fetchone() == (2, 2)
    restarted = CameraRegistryStore(sandbox.database, sandbox.authority)
    assert restarted.get("camera-a") == updated | {"status": "unknown"}
    assert registry_expected_cameras(restarted) == {
        "hub-a": {"camera_id": "hub-a", "facility_id": None, "resident_id": None},
        "camera-a": {"camera_id": "hub-a", "facility_id": None, "resident_id": None},
    }
    pending = store.update(
        "camera-a",
        CameraUpdate.model_validate({"backend_camera_id": None, "mapping_pending": True}),
    )
    assert pending["mapping_pending"] is True and pending["backend_camera_id"] is None
    unmapped = store.update(
        "camera-a", CameraUpdate.model_validate({"mapping_pending": False, "floor": None})
    )
    assert unmapped["floor"] is None and unmapped["mapping_pending"] is False
    assert pending["status"] == unmapped["status"] == "offline"
    assert sandbox.admin.execute(
        "SELECT incarnation,mapping_state,revision FROM cameras"
    ).fetchone() == (incarnation, "UNMAPPED", 4)
    before = _rows(sandbox)
    hooks = []
    assert (
        store.update(
            "absent", CameraUpdate.model_validate({"label": "No row"}), after_write=hooks.append
        )
        is None
    )
    assert store.delete("absent", after_write=hooks.append) is False
    assert _rows(sandbox) == before and not hooks
    assert store.delete("camera-a") is True
    assert store.get("camera-a") is None
    assert store.snapshot() == {"registry_version": 5, "cameras": []}
    assert store._statuses == {}
    assert sandbox.database.read(lambda connection: connection.execute("SELECT 1").fetchone()) == (
        1,
    )


@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize(
    "url",
    [
        "RTSP://different:credentials@CAMERA-A.invalid:554/live/?b=2&a=1",
        "rtsp://camera-a.invalid/live?a=1&b=2",
        " rtsp://camera-a.invalid:554/live?b=2&a=1 ",
    ],
)
def test_normalized_duplicate_identity_rolls_back_without_status_or_revision_changes(
    postgres_product_sandbox: ProductSandbox,
    operation: str,
    url: str,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    _create(store, rtsp_url="rtsp://original:secret@camera-a.invalid/live/?a=1&b=2")
    _create(store, "camera-b")
    before = _rows(sandbox)
    statuses = dict(store._statuses)
    hooks = []
    with pytest.raises(DuplicateCameraError) as error:
        if operation == "create":
            _create(store, "camera-c", rtsp_url=url, after_write=hooks.append)
        else:
            store.update(
                "camera-b",
                CameraUpdate.model_validate({"rtsp_url": url, "status": "offline"}),
                after_write=hooks.append,
            )
    assert error.value.existing_record["id"] == "camera-a"
    assert error.value.existing_record == store.get("camera-a") | {"status": "unknown"}
    assert "secret" not in str(error.value) and "credentials" not in repr(error.value)
    assert _rows(sandbox) == before and store._statuses == statuses and not hooks
    assert (
        store.update("camera-a", CameraUpdate.model_validate({"rtsp_url": url}))["rtsp_url"] == url
    )


def test_distinct_stream_paths_ports_queries_and_safe_parameters_remain_distinct(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    urls = [
        "rtsp://camera.invalid/live",
        "rtsp://camera.invalid/Live",
        "rtsp://camera.invalid:8554/live",
        "rtsp://camera.invalid/live?a=1",
        "rtsp://camera.invalid/live?a=2",
    ]
    for index, url in enumerate(urls):
        _create(store, f"camera-{index}", rtsp_url=url)
    identifier = "camera'; DELETE FROM cameras; --"
    label = "Bed's %s camera; DROP TABLE locations; --"
    saved = _create(store, identifier, label=label, rtsp_url="rtsp://camera.invalid/quoted'path")
    assert store.get(identifier) == saved and saved["label"] == label
    assert len(store.snapshot()["cameras"]) == len(urls) + 1
    assert store.delete(identifier)
    assert len(store.snapshot()["cameras"]) == len(urls)


def test_registry_and_topology_identity_order_does_not_follow_database_locale(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    refs = ("ref-a", "ref_A", "ref:A", "ref-A", "ref.a")
    for index, ref in enumerate(refs):
        _create(store, ref, rtsp_url=f"rtsp://camera.invalid/{ref}")
        store.create_floor(edge_ref=ref, name=ref, order_index=index)
        store.create_room(edge_ref=ref, floor_edge_ref=refs[0], name=ref)
    assert [camera["id"] for camera in store.snapshot()["cameras"]] == sorted(refs)
    snapshot = store.topology_snapshot()
    assert snapshot.unmapped_camera_ids == tuple(sorted(refs))
    assert [floor.edge_ref for floor in snapshot.floors] == sorted(refs)
    floor = next(floor for floor in snapshot.floors if floor.edge_ref == refs[0])
    assert [room.edge_ref for room in floor.rooms] == sorted(refs)


def test_topology_complete_projection_rename_rebind_unbind_and_deletion(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    _seed_topology(store)
    store.create_floor(edge_ref="floor-b", name="Second", order_index=0)
    store.create_room(edge_ref="room-b", floor_edge_ref="floor-a", name="102")
    store.update_floor("floor-a", name="Renamed floor", order_index=2)
    store.update_room("room-a", name="Renamed room")
    snapshot = store.topology_snapshot()
    assert snapshot.registry_version == 7
    assert snapshot.dirty is not None and snapshot.dirty.registry_version == 7
    assert (
        snapshot.dirty.created_at
        == sandbox.admin.execute("SELECT topology_dirty_created_at FROM edge_site").fetchone()[0]
    )
    assert snapshot.readiness_error is None and snapshot.unmapped_camera_ids == ()
    assert snapshot.cloud_topology() == {
        "floors": [
            {
                "edgeRef": "floor-a",
                "name": "Renamed floor",
                "orderIndex": 2,
                "rooms": [
                    {
                        "edgeRef": "room-a",
                        "name": "Renamed room",
                        "type": "ROOM",
                        "capacity": 1,
                        "cameras": [{"edgeRef": "edge-a", "label": "camera-a"}],
                        "legacyCanonicalSpaceId": _SPACE,
                    },
                    {
                        "edgeRef": "room-b",
                        "name": "102",
                        "type": "ROOM",
                        "capacity": 1,
                        "cameras": [],
                    },
                ],
            },
            {"edgeRef": "floor-b", "name": "Second", "orderIndex": 0, "rooms": []},
        ]
    }
    cloud = json.dumps(snapshot.cloud_topology())
    assert all(
        secret not in cloud
        for secret in ("rtsp", "operator", "synthetic-private", "camera.invalid")
    )
    assert public_camera(store.get("camera-a"))["edge_ref"] == "edge-a"
    assert public_camera(store.get("camera-a"))["room_edge_ref"] == "room-a"
    moved = store.update("camera-a", CameraUpdate.model_validate({"room_edge_ref": "room-b"}))
    assert moved["edge_ref"] == "edge-a" and moved["room_edge_ref"] == "room-b"
    assert store.delete_room("room-a") is True
    detached = store.update(
        "camera-a", CameraUpdate.model_validate({"edge_ref": None, "room_edge_ref": None})
    )
    assert detached["edge_ref"] is None and detached["room_edge_ref"] is None
    assert "edge_ref" not in public_camera(detached)
    unbound = store.topology_snapshot()
    assert unbound.readiness_error is EdgeErrorCode.LEGACY_MAPPING_REQUIRED
    assert unbound.unmapped_camera_ids == ("camera-a",)
    assert store.delete_room("room-b") and store.delete_floor("floor-a")
    assert store.delete_floor("floor-b") and store.delete("camera-a")
    empty = store.topology_snapshot()
    assert empty.registry_version == 14 and empty.floors == ()
    assert empty.readiness_error is None and empty.unmapped_camera_ids == ()
    before = _rows(sandbox)
    hooks = []
    assert not store.update_floor("absent", name="X", order_index=0, after_write=hooks.append)
    assert not store.update_room("absent", name="X", after_write=hooks.append)
    assert not store.delete_floor("absent", after_write=hooks.append)
    assert not store.delete_room("absent", after_write=hooks.append)
    assert not hooks and _rows(sandbox) == before


@pytest.mark.parametrize(
    ("operation", "code"),
    [
        ("duplicate_floor", TopologyErrorCode.DUPLICATE_REF),
        ("duplicate_room", TopologyErrorCode.DUPLICATE_REF),
        ("duplicate_legacy", TopologyErrorCode.DUPLICATE_REF),
        ("missing_floor", TopologyErrorCode.MISSING_PARENT),
        ("wrong_parent_kind", TopologyErrorCode.MISSING_PARENT),
        ("missing_room", TopologyErrorCode.MISSING_PARENT),
        ("invalid_ref", TopologyErrorCode.INVALID_BINDING),
        ("invalid_legacy", TopologyErrorCode.INVALID_LEGACY_SPACE_ID),
        ("occupied_room", TopologyErrorCode.ROOM_OCCUPIED),
        ("delete_floor", TopologyErrorCode.ROOM_OCCUPIED),
        ("delete_room", TopologyErrorCode.ROOM_OCCUPIED),
    ],
)
def test_reference_conflicts_restore_all_prior_rows_and_status(
    postgres_product_sandbox: ProductSandbox,
    operation: str,
    code: TopologyErrorCode,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    _seed_topology(store)
    before = _rows(sandbox)
    statuses = dict(store._statuses)
    hooks = []
    operations = {
        "duplicate_floor": lambda: store.create_floor(edge_ref="floor-a", name="X", order_index=1),
        "duplicate_room": lambda: store.create_room(
            edge_ref="room-a", floor_edge_ref="floor-a", name="X"
        ),
        "duplicate_legacy": lambda: store.create_room(
            edge_ref="room-b", floor_edge_ref="floor-a", name="X", legacy_canonical_space_id=_SPACE
        ),
        "missing_floor": lambda: store.create_room(
            edge_ref="room-b", floor_edge_ref="absent", name="X"
        ),
        "wrong_parent_kind": lambda: store.create_room(
            edge_ref="room-b", floor_edge_ref="room-a", name="X"
        ),
        "missing_room": lambda: _create(
            store, "camera-b", edge_ref="edge-b", room_edge_ref="floor-a"
        ),
        "invalid_ref": lambda: store.create_floor(edge_ref="invalid ref", name="X", order_index=1),
        "invalid_legacy": lambda: store.create_room(
            edge_ref="room-b",
            floor_edge_ref="floor-a",
            name="X",
            legacy_canonical_space_id="invalid id",
        ),
        "occupied_room": lambda: _create(
            store, "camera-b", edge_ref="edge-b", room_edge_ref="room-a", after_write=hooks.append
        ),
        "delete_floor": lambda: store.delete_floor("floor-a", after_write=hooks.append),
        "delete_room": lambda: store.delete_room("room-a", after_write=hooks.append),
    }
    with pytest.raises(TopologyConflictError) as error:
        operations[operation]()
    assert error.value.code is code
    assert "synthetic-private" not in str(error.value)
    assert _rows(sandbox) == before and store._statuses == statuses and not hooks


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"edge_ref": None}, TopologyErrorCode.INVALID_BINDING),
        ({"room_edge_ref": None}, TopologyErrorCode.INVALID_BINDING),
        ({"room_edge_ref": "absent"}, TopologyErrorCode.MISSING_PARENT),
        ({"room_edge_ref": "room-b"}, TopologyErrorCode.ROOM_OCCUPIED),
    ],
)
def test_failed_rebinding_does_not_remove_the_previous_binding(
    postgres_product_sandbox: ProductSandbox,
    updates: dict,
    code: TopologyErrorCode,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    _seed_topology(store)
    store.create_room(edge_ref="room-b", floor_edge_ref="floor-a", name="102")
    _create(store, "camera-b", edge_ref="edge-b", room_edge_ref="room-b")
    before = _rows(sandbox)
    with pytest.raises(TopologyConflictError) as error:
        store.update(
            "camera-a",
            CameraUpdate.model_validate(updates | {"status": "offline", "label": "Must roll back"}),
        )
    assert error.value.code is code and _rows(sandbox) == before
    assert store.get("camera-a")["status"] == "online"
    assert store.get("camera-a")["room_edge_ref"] == "room-a"


def test_policy_reference_blocks_camera_deletion_without_leaking_constraint_detail(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    _create(store, rtsp_url=_URL)
    sandbox.admin.execute(
        "INSERT INTO policies(facility_id,camera_id,module_id,module_version,schema_id,"
        "schema_version,previous_present,activation_generation,status,activated_at,updated_at) "
        "VALUES ('facility-a','camera-a','fall',1,'fall-policy',1,0,1,'pending',%s,%s)",
        (_TIME, _TIME),
    )
    before = _rows(sandbox)
    hooks = []
    with pytest.raises(CameraRegistryWriteError) as error:
        store.delete("camera-a", after_write=hooks.append)
    assert str(error.value) == "camera registry write rejected by a database constraint"
    assert error.value.__suppress_context__ and not hooks
    assert _rows(sandbox) == before and store.get("camera-a")["status"] == "online"


def test_constraint_errors_are_privacy_safe_and_do_not_publish_failed_status(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    _create(store, rtsp_url=_URL)
    before = _rows(sandbox)
    with pytest.raises(CameraRegistryWriteError) as error:
        store.update("camera-a", CameraUpdate.model_validate({"label": "", "status": "offline"}))
    assert "synthetic-private" not in repr(error.value)
    assert error.value.__suppress_context__ and error.value.__cause__ is None
    assert _rows(sandbox) == before and store.get("camera-a")["status"] == "online"
    with pytest.raises(CameraRegistryWriteError):
        _create(store, rtsp_url="rtsp://different.invalid/live", status="offline")
    assert _rows(sandbox) == before and store.get("camera-a")["status"] == "online"


@pytest.mark.parametrize("operation", tuple(_ACTIONS))
@pytest.mark.parametrize("fail", [False, True], ids=["commit", "rollback"])
def test_audit_hook_borrows_same_transaction_and_rolls_back_atomically(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    fail: bool,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    _seed_topology(store)
    store.create_floor(edge_ref="floor-empty", name="Empty", order_index=2)
    store.create_room(edge_ref="room-empty", floor_edge_ref="floor-a", name="Empty")
    before = _rows(sandbox)
    statuses = dict(store._statuses)
    version = store.snapshot()["registry_version"]
    transact = sandbox.database.transact
    connections = []
    hook_connections = []

    def observed_transaction(callback):
        def observe(connection):
            connections.append(id(connection))
            return callback(connection)

        return transact(observe)

    def hook(connection: psycopg.Connection) -> None:
        hook_connections.append(id(connection))
        assert hook_connections == connections
        assert connection.row_factory is tuple_row
        assert connection.info.transaction_status is TransactionStatus.INTRANS
        assert connection.execute("SHOW transaction_isolation").fetchone() == ("read committed",)
        assert connection.execute("SELECT registry_version FROM edge_site").fetchone() == (
            version + 1,
        )
        assert _rows(sandbox) == before and store._statuses == statuses
        append_postgres_audit(connection, _audit(_ACTIONS[operation]))
        assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
        if fail:
            raise RuntimeError("injected audit hook failure")

    monkeypatch.setattr(sandbox.database, "transact", observed_transaction)
    if fail:
        with pytest.raises(RuntimeError, match="injected audit hook failure"):
            _mutation(store, operation, hook)
        assert _rows(sandbox) == before and store._statuses == statuses
    else:
        _mutation(store, operation, hook)
        assert store.snapshot()["registry_version"] == version + 1
        assert _rows(sandbox) != before
    assert len(connections) == len(hook_connections) == 1
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (int(not fail),)
    assert sandbox.database.read(lambda connection: connection.row_factory is tuple_row)


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_rejected_commit_does_not_publish_status_or_partial_rows(
    postgres_product_sandbox: ProductSandbox,
    operation: str,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    _seed_topology(store)
    # Test-only deferred FK rejects the real COMMIT, not the mutation statement.
    sandbox.admin.execute(
        "CREATE TABLE camera_commit_guard (camera_id text REFERENCES cameras(camera_id) "
        "DEFERRABLE INITIALLY DEFERRED)"
    )
    before = _rows(sandbox)
    statuses = dict(store._statuses)
    registry = store.snapshot()
    hooks = []

    def reject_commit(connection: psycopg.Connection) -> None:
        hooks.append(True)
        append_postgres_audit(connection, _audit(_ACTIONS[operation]))
        connection.execute("INSERT INTO camera_commit_guard VALUES ('absent')")
        assert connection.info.transaction_status is TransactionStatus.INTRANS

    with pytest.raises(CameraRegistryWriteError):
        _mutation(store, operation, reject_commit)
    assert hooks == [True]
    assert _rows(sandbox) == before and store._statuses == statuses
    assert store.get("camera-a") == registry["cameras"][0]
    assert store.snapshot() == registry
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
    assert sandbox.admin.execute("SELECT count(*) FROM camera_commit_guard").fetchone() == (0,)


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
@pytest.mark.parametrize(
    "committed", [False, True], ids=["lost-before-commit", "lost-after-commit"]
)
def test_unknown_commit_never_replays_or_publishes_status(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    committed: bool,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    _seed_topology(store)
    before = _rows(sandbox)
    statuses = dict(store._statuses)
    registry = store.snapshot()
    incarnation = sandbox.admin.execute("SELECT incarnation FROM cameras").fetchone()[0]
    commit = psycopg.Connection.commit
    hook_pids = []
    commit_pids = []

    def hook(connection: psycopg.Connection) -> None:
        hook_pids.append(connection.info.backend_pid)
        append_postgres_audit(connection, _audit(_ACTIONS[operation]))

    def lose_receipt(connection: psycopg.Connection) -> None:
        pid = connection.info.backend_pid
        if pid in hook_pids:
            if committed:
                commit(connection)
            commit_pids.append(pid)
            raise psycopg.OperationalError("injected loss of COMMIT receipt")
        commit(connection)

    monkeypatch.setattr(psycopg.Connection, "commit", lose_receipt)
    with pytest.raises(CommitOutcomeUnknown):
        _mutation(store, operation, hook)
    monkeypatch.setattr(psycopg.Connection, "commit", commit)
    assert len(hook_pids) == len(commit_pids) == 1
    assert store._statuses == statuses
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (
        int(committed),
    )
    assert store.snapshot()["registry_version"] == 3 + int(committed)
    if not committed or operation != "delete":
        assert sandbox.admin.execute(
            "SELECT incarnation FROM cameras WHERE camera_id='camera-a'"
        ).fetchone() == (incarnation,)
    if not committed:
        assert _rows(sandbox) == before
        assert store.get("camera-a") == registry["cameras"][0]
        assert store.snapshot() == registry
    elif operation == "create":
        assert store.get("camera-b")["status"] == "unknown"
    elif operation == "update":
        assert store.get("camera-a")["label"] == "Changed"
        assert store.get("camera-a")["status"] == "online"
        assert store.snapshot()["cameras"] == [registry["cameras"][0] | {"label": "Changed"}]
    else:
        assert store.get("camera-a") is None


@pytest.mark.parametrize(
    "replacement_url",
    [_URL, "rtsp://replacement.invalid/live"],
    ids=["same-stream", "different-stream"],
)
def test_unknown_committed_delete_does_not_reuse_status_for_recreated_camera(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    replacement_url: str,
) -> None:
    sandbox = postgres_product_sandbox
    first = CameraRegistryStore(sandbox.database, sandbox.authority)
    second = CameraRegistryStore(sandbox.database, sandbox.authority)
    monkeypatch.setattr("backend.app.features.cameras.store.utc_now", lambda: _TIME)
    original = _create(first, rtsp_url=_URL)
    incarnation, revision = sandbox.admin.execute(
        "SELECT incarnation,revision FROM cameras"
    ).fetchone()
    assert isinstance(incarnation, UUID)
    statuses = dict(first._statuses)
    assert statuses == {"camera-a": (incarnation, "online")}
    commit = psycopg.Connection.commit
    hook_pids = []
    commit_pids = []

    def hook(connection: psycopg.Connection) -> None:
        hook_pids.append(connection.info.backend_pid)
        append_postgres_audit(connection, _audit(AuditAction.CAMERA_DELETE))

    def lose_receipt(connection: psycopg.Connection) -> None:
        pid = connection.info.backend_pid
        if pid in hook_pids:
            commit(connection)
            commit_pids.append(pid)
            raise psycopg.OperationalError("injected loss of COMMIT receipt")
        commit(connection)

    with monkeypatch.context() as patch:
        patch.setattr(psycopg.Connection, "commit", lose_receipt)
        with pytest.raises(CommitOutcomeUnknown):
            first.delete("camera-a", after_write=hook)
    assert len(hook_pids) == len(commit_pids) == 1
    assert sandbox.admin.execute("SELECT count(*) FROM cameras").fetchone() == (0,)
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (1,)
    assert first.get("camera-a") is None
    assert first._statuses == statuses

    recreated = _create(second, rtsp_url=replacement_url, status="offline")
    replacement_incarnation, replacement_revision = sandbox.admin.execute(
        "SELECT incarnation,revision FROM cameras"
    ).fetchone()
    assert isinstance(replacement_incarnation, UUID)
    assert replacement_incarnation != incarnation
    assert replacement_revision == revision
    assert recreated["created_at"] == original["created_at"] == _TIME
    assert recreated == original | {"rtsp_url": replacement_url, "status": "offline"}
    assert first.get("camera-a") == recreated | {"status": "unknown"}
    assert first.snapshot() == {
        "registry_version": 3,
        "cameras": [recreated | {"status": "unknown"}],
    }
    assert second.get("camera-a") == recreated
    assert second.snapshot() == {"registry_version": 3, "cameras": [recreated]}
    assert first._statuses == statuses
    assert second._statuses == {"camera-a": (replacement_incarnation, "offline")}


@pytest.mark.parametrize("operation", (*_ACTIONS, "migrate"))
def test_frozen_authority_is_checked_before_bootstrap_and_every_mutation(
    postgres_product_sandbox: ProductSandbox,
    operation: str,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    _seed_topology(store)
    freeze_authority(sandbox.database, sandbox.authority)
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    before = _rows(sandbox)
    statuses = dict(store._statuses)
    hooks = []
    with pytest.raises(AuthorityFenced):
        _mutation(store, operation, hooks.append)
    assert not hooks and _rows(sandbox) == before and store._statuses == statuses


def test_frozen_authority_precedes_invalid_input_and_missing_camera_checks(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    freeze_authority(sandbox.database, sandbox.authority)
    with pytest.raises(AuthorityFenced):
        _create(store, rtsp_url="rtsp://camera.invalid:not-a-port/live")
    with pytest.raises(AuthorityFenced):
        store.update("absent", CameraUpdate.model_validate({"edge_ref": "invalid ref"}))
    with pytest.raises(AuthorityFenced):
        store.delete("absent")
    with pytest.raises(AuthorityFenced):
        store.create_floor(edge_ref="invalid ref", name="X", order_index=-1)
    assert sandbox.admin.execute("SELECT count(*) FROM cameras").fetchone() == (0,)


def test_missing_bootstrap_fails_closed_on_reads_and_writes_without_seeding(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    _seed_topology(store)
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    before = _rows(sandbox)
    statuses = dict(store._statuses)
    for read in (store.snapshot, store.topology_snapshot, lambda: store.get("camera-a")):
        with pytest.raises(CameraRegistryNotInitialized, match="bootstrap row is missing"):
            read()
    hooks = []
    for name in (*_ACTIONS, "migrate"):
        with pytest.raises(CameraRegistryNotInitialized):
            _mutation(store, name, hooks.append)
    assert _rows(sandbox) == before and not hooks and store._statuses == statuses
    assert sandbox.admin.execute("SELECT count(*) FROM edge_site").fetchone() == (0,)


def test_explicit_floor_normalization_is_atomic_and_never_runs_on_reads(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    for camera_id, raw in (("camera-a", "B1"), ("camera-b", "2층"), ("camera-c", "5")):
        _create(store, camera_id, floor=5)
        sandbox.admin.execute(
            "UPDATE cameras SET floor_override=%s WHERE camera_id=%s", (raw, camera_id)
        )
    _create(store, "camera-d")
    before = _rows(sandbox)
    assert [row["floor"] for row in store.snapshot()["cameras"]] == [-1, 2, 5, None]
    assert _rows(sandbox) == before
    changes = store.migrate_legacy_string_floors()
    assert sorted(changes, key=lambda item: item["camera_id"]) == [
        {"camera_id": "camera-a", "old": "B1", "new": -1},
        {"camera_id": "camera-b", "old": "2층", "new": 2},
    ]
    assert sandbox.admin.execute(
        "SELECT floor_override,revision FROM cameras ORDER BY camera_id"
    ).fetchall() == [("-1", 2), ("2", 2), ("5", 1), (None, 1)]
    assert store.snapshot()["registry_version"] == 5
    before = _rows(sandbox)
    assert store.migrate_legacy_string_floors() == [] and _rows(sandbox) == before


def _compete(
    sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    first: Callable[[], object],
    second: Callable[[], object],
) -> tuple[object, object]:
    transact = sandbox.database.transact
    ready = Barrier(3)
    pids: Queue[int] = Queue()

    def synchronized_transaction(callback):
        def synchronize(connection):
            pids.put(connection.info.backend_pid)
            ready.wait(timeout=2)
            return callback(connection)

        return transact(synchronize)

    def capture(operation):
        try:
            return operation()
        except (DuplicateCameraError, TopologyConflictError) as error:
            return error

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.database, "transact", synchronized_transaction)
        with ThreadPoolExecutor(max_workers=2) as pool:
            with sandbox.admin.transaction():
                sandbox.admin.execute("SELECT id FROM edge_site WHERE id=1 FOR UPDATE")
                one = pool.submit(capture, first)
                two = pool.submit(capture, second)
                worker_pids = [pids.get(timeout=2), pids.get(timeout=2)]
                ready.wait(timeout=2)
                deadline = monotonic() + 2
                pause = Event()
                while monotonic() < deadline:
                    waiting = sandbox.admin.execute(
                        "SELECT count(DISTINCT pid) FROM pg_locks "
                        "WHERE pid=ANY(%s) AND NOT granted",
                        (worker_pids,),
                    ).fetchone()
                    if waiting == (2,):
                        break
                    pause.wait(0.01)
                else:
                    pytest.fail("both registry writers must wait on the singleton row lock")
            return one.result(timeout=5), two.result(timeout=5)


def test_competing_partial_camera_updates_merge_after_lock_without_lost_fields(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = postgres_product_sandbox
    first = CameraRegistryStore(sandbox.database, sandbox.authority)
    second = CameraRegistryStore(sandbox.database, sandbox.authority)
    _create(first)
    one, two = _compete(
        sandbox,
        monkeypatch,
        lambda: first.update(
            "camera-a", CameraUpdate.model_validate({"label": "Concurrent label"})
        ),
        lambda: second.update(
            "camera-a", CameraUpdate.model_validate({"backend_camera_id": "hub-a", "floor": -1})
        ),
    )
    assert one["label"] == "Concurrent label" and two["backend_camera_id"] == "hub-a"
    record = first.get("camera-a")
    assert (record["label"], record["backend_camera_id"], record["floor"]) == (
        "Concurrent label",
        "hub-a",
        -1,
    )
    assert first.snapshot()["registry_version"] == 3
    assert sandbox.admin.execute("SELECT revision FROM cameras").fetchone() == (3,)


def test_competing_duplicate_creates_have_exactly_one_committed_winner(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = postgres_product_sandbox
    first = CameraRegistryStore(sandbox.database, sandbox.authority)
    second = CameraRegistryStore(sandbox.database, sandbox.authority)
    results = _compete(
        sandbox,
        monkeypatch,
        lambda: _create(first, rtsp_url="rtsp://camera.invalid/live"),
        lambda: _create(
            second, "camera-b", rtsp_url="RTSP://other:secret@CAMERA.invalid:554/live/"
        ),
    )
    winners = [result for result in results if isinstance(result, dict)]
    losers = [result for result in results if isinstance(result, DuplicateCameraError)]
    assert len(winners) == len(losers) == 1
    assert losers[0].existing_record["id"] == winners[0]["id"]
    assert first.snapshot()["registry_version"] == 1
    assert len(first.snapshot()["cameras"]) == 1
    assert len(first._statuses) + len(second._statuses) == 1


def test_competing_room_delete_and_camera_binding_are_serialized(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = postgres_product_sandbox
    first = CameraRegistryStore(sandbox.database, sandbox.authority)
    second = CameraRegistryStore(sandbox.database, sandbox.authority)
    first.create_floor(edge_ref="floor-a", name="First", order_index=1)
    first.create_room(edge_ref="room-a", floor_edge_ref="floor-a", name="101")
    camera, deletion = _compete(
        sandbox,
        monkeypatch,
        lambda: _create(first, edge_ref="edge-a", room_edge_ref="room-a"),
        lambda: second.delete_room("room-a"),
    )
    assert first.snapshot()["registry_version"] == 3
    if isinstance(camera, dict):
        assert isinstance(deletion, TopologyConflictError)
        assert deletion.code is TopologyErrorCode.ROOM_OCCUPIED
        assert first.topology_snapshot().floors[0].rooms[0].cameras[0].edge_ref == "edge-a"
    else:
        assert isinstance(camera, TopologyConflictError)
        assert camera.code is TopologyErrorCode.MISSING_PARENT and deletion is True
        assert first.get("camera-a") is None and first.topology_snapshot().floors[0].rooms == ()


@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize("intervening_write", ["update", "recreate"])
def test_mutation_returns_its_own_committed_record_not_a_racing_reload(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    intervening_write: str,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    other = CameraRegistryStore(sandbox.database, sandbox.authority)
    monkeypatch.setattr("backend.app.features.cameras.store.utc_now", lambda: _TIME)
    if operation == "update":
        _create(store, status="starting")
    statuses = dict(store._statuses)
    transact = sandbox.database.transact
    interleaved = []

    def interleaved_transaction(callback):
        candidate = transact(callback)
        if not interleaved:
            incarnation, record = candidate
            interleaved.append((incarnation, record))
            assert store._statuses == statuses
            assert sandbox.admin.execute("SELECT incarnation FROM cameras").fetchone() == (
                incarnation,
            )
            if intervening_write == "recreate":
                assert other.delete("camera-a")
                _create(other, label="Later committed writer", status="offline")
            else:
                other.update(
                    "camera-a",
                    CameraUpdate.model_validate(
                        {"label": "Later committed writer", "status": "offline"}
                    ),
                )
        return candidate

    monkeypatch.setattr(sandbox.database, "transact", interleaved_transaction)
    if operation == "create":
        saved = _create(store, label="Own committed label")
    else:
        saved = store.update(
            "camera-a",
            CameraUpdate.model_validate({"label": "Own committed label", "status": "online"}),
        )
    assert saved["label"] == "Own committed label" and saved["status"] == "online"
    assert len(interleaved) == 1 and interleaved[0][1]["label"] == "Own committed label"
    incarnation = interleaved[0][0]
    assert store._statuses == {"camera-a": (incarnation, "online")}
    current_incarnation = sandbox.admin.execute("SELECT incarnation FROM cameras").fetchone()[0]
    if intervening_write == "recreate":
        assert current_incarnation != incarnation
        expected_status = "unknown"
    else:
        assert current_incarnation == incarnation
        expected_status = "online"
    visible = saved | {"label": "Later committed writer", "status": expected_status}
    assert store.get("camera-a") == visible
    assert store.snapshot()["cameras"] == [visible]
    assert other.get("camera-a") == visible | {"status": "offline"}
    assert other.snapshot()["cameras"] == [visible | {"status": "offline"}]


@pytest.mark.parametrize("projection", ["snapshot", "topology_snapshot"])
def test_snapshots_use_one_readonly_statement_snapshot_across_committed_writers(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    other = CameraRegistryStore(sandbox.database, sandbox.authority)
    _seed_topology(store)
    read_snapshot = getattr(store, projection)
    before = read_snapshot()
    execute = psycopg.Cursor.execute
    read = sandbox.database.read
    statements = []
    reads = []
    armed = True

    def interleave(cursor, query, *args, **kwargs):
        nonlocal armed
        result = execute(cursor, query, *args, **kwargs)
        if armed and isinstance(query, str) and "FROM edge_site AS s" in query:
            armed = False
            statements.append(query)
            # The read's result exists, but has not been fetched or projected.
            other.update("camera-a", CameraUpdate.model_validate({"label": "Concurrent label"}))
            other.create_floor(edge_ref="floor-b", name="New floor", order_index=2)
            other.create_room(edge_ref="room-b", floor_edge_ref="floor-b", name="201")
            _create(other, "camera-b", edge_ref="edge-b", room_edge_ref="room-b")
        return result

    def observed_read(callback):
        def observe(connection):
            reads.append(connection.info.backend_pid)
            assert connection.info.transaction_status is TransactionStatus.INTRANS
            assert connection.row_factory is tuple_row
            assert connection.execute("SHOW transaction_read_only").fetchone() == ("on",)
            assert connection.execute("SHOW transaction_isolation").fetchone() == (
                "read committed",
            )
            return callback(connection)

        return read(observe)

    monkeypatch.setattr(psycopg.Cursor, "execute", interleave)
    monkeypatch.setattr(sandbox.database, "read", observed_read)
    assert read_snapshot() == before
    assert len(statements) == len(reads) == 1 and not armed
    assert read_snapshot() != before
    assert store.snapshot()["registry_version"] == 7
    assert store.topology_snapshot().floors[1].rooms[0].cameras[0].edge_ref == "edge-b"
