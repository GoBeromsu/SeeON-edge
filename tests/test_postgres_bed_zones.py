"""Bed-zone behavior on the isolated, actual PostgreSQL product schema."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from queue import Queue
from threading import Barrier, Event
from time import monotonic
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from psycopg.pq import TransactionStatus
from psycopg.rows import tuple_row

from backend.app.edge_db.authority import AuthorityFenced, freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_store import append_postgres_audit
from backend.app.features.audit.store import AuditEvent
from backend.app.features.cameras.bed_zone_store import BedZone, BedZoneRegion, BedZoneStore
from backend.app.features.cameras.camera_repository import CameraRegistryNotInitialized
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.update_command import CameraUpdate

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_TIME = "2026-09-27T04:00:00.123Z"
_POLYGON = ((1, 2), (9, 2), (9, 8), (1, 8))
_REGION = BedZoneRegion(id="bed-a", polygon=_POLYGON, origin="manual")


def _camera(sandbox: ProductSandbox, camera_id: str = "camera-a") -> CameraRegistryStore:
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
    registry.create(
        camera_id=camera_id,
        label=camera_id,
        rtsp_url=f"rtsp://camera.invalid/{camera_id}",
        space_id=None,
        status="online",
    )
    return registry


def _put(store: BedZoneStore, camera_id: str = "camera-a", **overrides) -> BedZone:
    return store.put(
        camera_id,
        **{
            "regions": (_REGION,),
            "image_width": 640,
            "image_height": 480,
            "recognized_at": _TIME,
            **overrides,
        },
    )


def _rows(sandbox: ProductSandbox):
    return (
        sandbox.admin.execute("SELECT * FROM edge_site ORDER BY id").fetchall(),
        sandbox.admin.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall(),
    )


def _audit(connection: psycopg.Connection) -> None:
    append_postgres_audit(
        connection,
        AuditEvent(
            occurred_at=_TIME,
            actor_id="test-operator",
            action=AuditAction.BED_ZONE_UPDATE,
            target_id="camera-a",
            detail=empty_detail(AuditAction.BED_ZONE_UPDATE),
        ),
    )


def test_round_trip_preserves_regions_origins_canonical_bytes_and_revisions(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    registry = _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    regions = (_REGION, replace(_REGION, id="침대-모델", origin="model"))
    saved = _put(store, regions=regions)
    assert saved == BedZone(regions, 640, 480, _TIME)
    assert saved.as_dict() == {
        "regions": [
            {"id": "bed-a", "polygon": [[1, 2], [9, 2], [9, 8], [1, 8]], "origin": "manual"},
            {"id": "침대-모델", "polygon": [[1, 2], [9, 2], [9, 8], [1, 8]], "origin": "model"},
        ],
        "image_width": 640,
        "image_height": 480,
        "recognized_at": _TIME,
    }
    encoded = (
        '[{"id":"bed-a","polygon":[[1,2],[9,2],[9,8],[1,8]],"origin":"manual"},'
        '{"id":"침대-모델","polygon":[[1,2],[9,2],[9,8],[1,8]],"origin":"model"}]'
    )
    assert sandbox.admin.execute(
        "SELECT bed_polygon_json,bed_image_width,bed_image_height,bed_recognized_at,revision "
        "FROM cameras WHERE camera_id='camera-a'"
    ).fetchone() == (encoded, 640, 480, _TIME, 2)
    reopened = BedZoneStore(sandbox.database, sandbox.authority)
    assert reopened.camera_exists("camera-a") is True
    assert reopened.get("camera-a") == saved
    assert reopened.get_all() == {"camera-a": saved}
    # Repeating a put is still a mutation, not a content-equality no-op.
    assert _put(reopened, regions=regions) == saved
    assert registry.snapshot()["registry_version"] == 3
    assert sandbox.admin.execute(
        "SELECT registry_version,topology_dirty_registry_version FROM edge_site"
    ).fetchone() == (3, 3)
    assert sandbox.admin.execute("SELECT revision FROM cameras").fetchone() == (3,)


def test_clear_and_missing_camera_semantics_do_not_seed_or_audit_noops(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    registry = _camera(sandbox)
    _camera(sandbox, "without-zone")
    store = BedZoneStore(sandbox.database, sandbox.authority)
    _put(store)
    before = _rows(sandbox)
    hooks = []
    assert store.camera_exists("missing") is False
    assert store.camera_exists("without-zone") is True
    assert store.get("missing") is None
    assert store.get("without-zone") is None
    assert store.delete("missing", after_write=hooks.append) is False
    assert store.delete("without-zone", after_write=hooks.append) is False
    with pytest.raises(psycopg.IntegrityError, match="bed-zone camera does not exist"):
        _put(store, "missing", after_write=hooks.append)
    assert _rows(sandbox) == before and not hooks
    assert store.delete("camera-a", after_write=_audit) is True
    assert sandbox.admin.execute(
        "SELECT bed_polygon_json,bed_image_width,bed_image_height,bed_recognized_at,revision "
        "FROM cameras WHERE camera_id='camera-a'"
    ).fetchone() == (None, None, None, None, 3)
    assert registry.snapshot()["registry_version"] == 4
    assert sandbox.admin.execute("SELECT action FROM audit_events").fetchall() == [
        (AuditAction.BED_ZONE_UPDATE.value,)
    ]
    cleared = _rows(sandbox)
    assert store.delete("camera-a", after_write=hooks.append) is False
    assert _rows(sandbox) == cleared and not hooks
    assert store.get("camera-a") is None
    assert store.get_all() == {}


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"recognized_at": "candidate"}, psycopg.IntegrityError),
        ({"recognized_at": "2026-02-30T00:00:00Z"}, psycopg.IntegrityError),
        ({"recognized_at": "2026-09-27T13:00:00+09:00"}, psycopg.IntegrityError),
        ({"image_width": 2**63}, psycopg.DataError),
    ],
)
def test_native_timestamp_and_bigint_constraints_preserve_existing_zone(
    postgres_product_sandbox: ProductSandbox,
    overrides: dict[str, Any],
    error: type[Exception],
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    original = _put(store)
    before = _rows(sandbox)
    hooks = []
    with pytest.raises(error):
        _put(store, after_write=hooks.append, **overrides)
    assert not hooks and _rows(sandbox) == before
    assert store.get("camera-a") == original


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"image_width": True}, ValueError, "positive integers"),
        ({"image_height": False}, ValueError, "positive integers"),
        ({"image_width": 0}, ValueError, "positive integers"),
        ({"image_height": -1}, ValueError, "positive integers"),
        ({"image_width": 640.0}, ValueError, "positive integers"),
        ({"recognized_at": ""}, ValueError, "must not be empty"),
        ({"recognized_at": None}, ValueError, "must not be empty"),
        ({"regions": ({"id": "not-a-region"},)}, TypeError, "invalid shape"),
        ({"regions": (_REGION, _REGION)}, ValueError, "ids must be distinct"),
        (
            {"regions": tuple(replace(_REGION, id=str(i)) for i in range(9))},
            ValueError,
            "at most 8",
        ),
        ({"regions": (replace(_REGION, id=""),)}, ValueError, "1 to 64"),
        ({"regions": (replace(_REGION, id="a" * 65),)}, ValueError, "1 to 64"),
        ({"regions": (replace(_REGION, id=42),)}, ValueError, "1 to 64"),
        ({"regions": (replace(_REGION, origin="other"),)}, ValueError, "origin is invalid"),
    ],
)
def test_validation_failure_preserves_nonempty_zone_and_all_revisions(
    postgres_product_sandbox: ProductSandbox,
    overrides: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    original = _put(store)
    before = _rows(sandbox)
    hooks = []
    with pytest.raises(error, match=message):
        _put(store, after_write=hooks.append, **overrides)
    assert not hooks and _rows(sandbox) == before
    assert store.get("camera-a") == original


@pytest.mark.parametrize(
    ("polygon", "message"),
    [
        ((), "3 to 16"),
        (((1, 1), (2, 2)), "3 to 16"),
        (tuple((i, 1) for i in range(17)), "3 to 16"),
        (((True, 1), (9, 1), (1, 8)), "integers"),
        (((1, False), (9, 1), (1, 8)), "integers"),
        (((1.0, 1), (9, 1), (1, 8)), "integers"),
        (((1, "1"), (9, 1), (1, 8)), "integers"),
        (((1,), (9, 1), (1, 8)), "integers"),
        ((None, (9, 1), (1, 8)), "integers"),
        (((-1, 1), (9, 1), (1, 8)), "outside the image"),
        (((1, 1), (640, 1), (1, 8)), "outside the image"),
        (((1, 1), (9, 1), (1, 480)), "outside the image"),
        (((1, 1), (9, 1), (1, 1)), "vertices must be distinct"),
        (((1, 1), (2, 2), (3, 3)), "nondegenerate"),
        (((1, 1), (9, 8), (1, 8), (7, 1)), "self-intersect"),
        (((0, 0), (9, 0), (9, 9), (4, 0), (0, 9)), "self-intersect"),
    ],
)
def test_polygon_shape_integer_bounds_area_and_intersection_validation(
    postgres_product_sandbox: ProductSandbox, polygon: Any, message: str
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    original = _put(store)
    before = _rows(sandbox)
    with pytest.raises(ValueError, match=message):
        _put(store, regions=(replace(_REGION, polygon=polygon),))
    assert _rows(sandbox) == before and store.get("camera-a") == original


@pytest.mark.parametrize(
    "polygon",
    [
        ((0, 0), (639, 0), (0, 479)),
        tuple(reversed(_POLYGON)),
        ((0, 0), (639, 0), (320, 200), (639, 479), (0, 479)),
    ],
    ids=["triangle-image-boundary", "clockwise", "concave"],
)
def test_valid_polygon_boundaries_and_orientations_round_trip(
    postgres_product_sandbox: ProductSandbox, polygon: tuple[tuple[int, int], ...]
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    region = replace(_REGION, polygon=polygon, id="a" * 64)
    saved = _put(store, regions=(region,))
    assert store.get_all() == {"camera-a": BedZone((region,), 640, 480, _TIME)}
    assert saved.regions == (region,)


def _regions_at_byte_limit(size: int) -> tuple[BedZoneRegion, ...]:
    scale = 10**12
    quarter = scale // 4
    polygon = (
        (1, 1),
        (quarter, 1),
        (2 * quarter, 1),
        (3 * quarter, 1),
        (scale - 1, 1),
        (scale - 1, quarter),
        (scale - 1, 2 * quarter),
        (scale - 1, 3 * quarter),
        (scale - 1, scale - 1),
        (3 * quarter, scale - 1),
        (2 * quarter, scale - 1),
        (quarter, scale - 1),
        (1, scale - 1),
        (1, 3 * quarter),
        (1, 2 * quarter),
        (1, quarter),
    )
    regions = [BedZoneRegion(str(i), polygon, "manual") for i in range(8)]
    encoded = json.dumps([r.as_dict() for r in regions], ensure_ascii=False, separators=(",", ":"))
    remaining = size - len(encoded.encode("utf-8"))
    assert 0 < remaining <= 8 * 186
    for index, region in enumerate(regions):
        padding = min(186, remaining)
        regions[index] = replace(region, id=region.id + "가" * (padding // 3) + "a" * (padding % 3))
        remaining -= padding
    assert remaining == 0
    return tuple(regions)


def test_utf8_byte_limit_and_max_region_vertex_counts_are_exact(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    regions = _regions_at_byte_limit(4096)
    saved = _put(store, regions=regions, image_width=10**12, image_height=10**12)
    assert sandbox.admin.execute(
        "SELECT octet_length(bed_polygon_json) FROM cameras"
    ).fetchone() == (4096,)
    assert store.get("camera-a") == saved and len(saved.regions) == 8
    assert all(len(region.polygon) == 16 for region in saved.regions)
    before = _rows(sandbox)
    with pytest.raises(ValueError, match="exceed 4096 bytes"):
        _put(store, regions=_regions_at_byte_limit(4097), image_width=10**12, image_height=10**12)
    assert _rows(sandbox) == before and store.get("camera-a") == saved


def test_empty_region_array_is_a_stored_zone_until_explicitly_cleared(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    registry = _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    _put(store)
    saved = _put(store, regions=())
    assert saved == BedZone((), 640, 480, _TIME)
    assert store.get_all() == {"camera-a": saved}
    assert sandbox.admin.execute("SELECT bed_polygon_json FROM cameras").fetchone() == ("[]",)
    assert store.delete("camera-a") is True
    assert registry.snapshot()["registry_version"] == 4


@pytest.mark.parametrize(
    "raw",
    [
        [[1, 2], [9, 2], [9, 8], [1, 8]],
        [{"id": "bad", "polygon": [[1, 1], [9, 1], [1, 8]]}],
        [_REGION.as_dict() | {"extra": True}],
        [_REGION.as_dict() | {"origin": "retired"}],
        [_REGION.as_dict() | {"id": 1}],
        [_REGION.as_dict() | {"polygon": "not-a-polygon"}],
        [_REGION.as_dict() | {"polygon": [[1], [9, 1], [1, 8]]}],
        [_REGION.as_dict() | {"polygon": [[True, 1], [9, 1], [1, 8]]}],
        [_REGION.as_dict() | {"polygon": [[1.5, 1], [9, 1], [1, 8]]}],
        [_REGION.as_dict() | {"polygon": [[1, 1], [2, 2], [3, 3]]}],
        [_REGION.as_dict(), _REGION.as_dict()],
    ],
)
def test_reads_skip_noncanonical_rows_without_rewriting_valid_or_invalid_data(
    postgres_product_sandbox: ProductSandbox, raw: list
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    _camera(sandbox, "camera-b")
    store = BedZoneStore(sandbox.database, sandbox.authority)
    saved = _put(store)
    sandbox.admin.execute(
        "UPDATE cameras SET bed_polygon_json=%s,bed_image_width=640,bed_image_height=480,"
        "bed_recognized_at=%s WHERE camera_id='camera-b'",
        (json.dumps(raw), _TIME),
    )
    before = _rows(sandbox)
    assert store.camera_exists("camera-b") is True
    assert store.get("camera-b") is None
    assert store.get_all() == {"camera-a": saved}
    assert _rows(sandbox) == before


@pytest.mark.parametrize("operation", ["put", "delete"])
@pytest.mark.parametrize("fail", [False, True], ids=["commit", "rollback"])
def test_audit_hook_uses_same_active_connection_and_rolls_back_zone_and_revision(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    fail: bool,
) -> None:
    sandbox = postgres_product_sandbox
    registry = _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    _put(store)
    before = _rows(sandbox)
    version = registry.snapshot()["registry_version"]
    transact = sandbox.database.transact
    connections = []
    hooks = []

    def observed_transaction(callback):
        def observe(connection):
            connections.append(id(connection))
            return callback(connection)

        return transact(observe)

    def hook(connection: psycopg.Connection) -> None:
        hooks.append(id(connection))
        assert hooks == connections
        assert connection.info.transaction_status is TransactionStatus.INTRANS
        assert connection.row_factory is tuple_row
        assert connection.execute("SHOW transaction_isolation").fetchone() == ("read committed",)
        assert connection.execute(
            "SELECT registry_version,topology_dirty_registry_version FROM edge_site"
        ).fetchone() == (version + 1, version + 1)
        assert connection.execute("SELECT revision FROM cameras").fetchone() == (3,)
        assert _rows(sandbox) == before
        _audit(connection)
        assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
        if fail:
            raise RuntimeError("injected audit failure")

    monkeypatch.setattr(sandbox.database, "transact", observed_transaction)
    mutate = (
        (lambda: _put(store, regions=(replace(_REGION, id="changed"),), after_write=hook))
        if operation == "put"
        else lambda: store.delete("camera-a", after_write=hook)
    )
    if fail:
        with pytest.raises(RuntimeError, match="injected audit failure"):
            mutate()
        assert _rows(sandbox) == before
    else:
        mutate()
        assert _rows(sandbox) != before
        assert registry.snapshot()["registry_version"] == version + 1
    assert len(connections) == len(hooks) == 1
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (int(not fail),)
    assert sandbox.database.read(lambda connection: connection.row_factory is tuple_row)


@pytest.mark.parametrize("operation", ["put", "delete"])
def test_real_deferred_commit_rejection_rolls_back_zone_revision_and_audit(
    postgres_product_sandbox: ProductSandbox, operation: str
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    original = _put(store)
    sandbox.admin.execute(
        "CREATE TABLE bed_commit_guard (camera_id text REFERENCES cameras(camera_id) "
        "DEFERRABLE INITIALLY DEFERRED)"
    )
    before = _rows(sandbox)
    hooks = []

    def reject_commit(connection: psycopg.Connection) -> None:
        hooks.append(True)
        _audit(connection)
        connection.execute("INSERT INTO bed_commit_guard VALUES ('missing')")
        assert connection.info.transaction_status is TransactionStatus.INTRANS

    with pytest.raises(psycopg.IntegrityError):
        if operation == "put":
            _put(store, after_write=reject_commit)
        else:
            store.delete("camera-a", after_write=reject_commit)
    assert hooks == [True]
    assert _rows(sandbox) == before and store.get("camera-a") == original
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
    assert sandbox.admin.execute("SELECT count(*) FROM bed_commit_guard").fetchone() == (0,)


@pytest.mark.parametrize("operation", ["put", "delete"])
@pytest.mark.parametrize(
    "committed", [False, True], ids=["lost-before-commit", "lost-after-commit"]
)
def test_unknown_commit_propagates_without_replay_or_a_returned_zone(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    committed: bool,
) -> None:
    sandbox = postgres_product_sandbox
    registry = _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    original = _put(store)
    changed = replace(_REGION, id="changed")
    before = _rows(sandbox)
    commit = psycopg.Connection.commit
    hook_pids = []
    commit_pids = []
    published = []

    def hook(connection: psycopg.Connection) -> None:
        hook_pids.append(connection.info.backend_pid)
        _audit(connection)

    def lose_receipt(connection: psycopg.Connection) -> None:
        pid = connection.info.backend_pid
        if pid in hook_pids:
            if committed:
                commit(connection)
            commit_pids.append(pid)
            raise psycopg.OperationalError("injected COMMIT receipt loss")
        commit(connection)

    with monkeypatch.context() as patch:
        patch.setattr(psycopg.Connection, "commit", lose_receipt)
        with pytest.raises(CommitOutcomeUnknown):
            published.append(
                _put(store, regions=(changed,), after_write=hook)
                if operation == "put"
                else store.delete("camera-a", after_write=hook)
            )
    assert not published and len(hook_pids) == len(commit_pids) == 1
    assert registry.snapshot()["registry_version"] == 2 + int(committed)
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (
        int(committed),
    )
    if not committed:
        assert _rows(sandbox) == before and store.get("camera-a") == original
    else:
        expected = BedZone((changed,), 640, 480, _TIME) if operation == "put" else None
        assert store.get("camera-a") == expected
        assert BedZoneStore(sandbox.database, sandbox.authority).get("camera-a") == expected


@pytest.mark.parametrize("operation", ["put", "delete", "delete-absent", "invalid-put"])
def test_frozen_authority_precedes_bootstrap_and_every_mutation_including_noops(
    postgres_product_sandbox: ProductSandbox, operation: str
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    original = _put(store)
    freeze_authority(sandbox.database, sandbox.authority)
    assert store.get("camera-a") == original
    hooks = []
    for missing_bootstrap in (False, True):
        if missing_bootstrap:
            sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
        before = _rows(sandbox)
        with pytest.raises(AuthorityFenced):
            if operation in {"put", "invalid-put"}:
                _put(
                    store,
                    image_width=0 if operation == "invalid-put" else 640,
                    after_write=hooks.append,
                )
            else:
                store.delete(
                    "missing" if operation == "delete-absent" else "camera-a",
                    after_write=hooks.append,
                )
        assert not hooks and _rows(sandbox) == before


def test_missing_bootstrap_fails_closed_on_all_reads_and_writes_without_seeding(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    _put(store)
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    before = _rows(sandbox)
    hooks = []
    for operation in (
        lambda: store.camera_exists("camera-a"),
        lambda: store.camera_exists("absent"),
        lambda: store.get("camera-a"),
        lambda: store.get("absent"),
        store.get_all,
        lambda: _put(store, after_write=hooks.append),
        lambda: store.delete("camera-a", after_write=hooks.append),
        lambda: store.delete("absent", after_write=hooks.append),
    ):
        with pytest.raises(CameraRegistryNotInitialized, match="bootstrap row is missing"):
            operation()
    assert _rows(sandbox) == before and not hooks
    assert sandbox.admin.execute("SELECT count(*) FROM edge_site").fetchone() == (0,)


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

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.database, "transact", synchronized_transaction)
        with ThreadPoolExecutor(max_workers=2) as pool:
            with sandbox.admin.transaction():
                sandbox.admin.execute("SELECT id FROM edge_site WHERE id=1 FOR UPDATE")
                one = pool.submit(first)
                two = pool.submit(second)
                worker_pids = [pids.get(timeout=2), pids.get(timeout=2)]
                assert len(set(worker_pids)) == 2
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
                    pytest.fail("both writers must wait on the singleton row lock")
                # Neither writer may lock a camera before the singleton.
                sandbox.admin.execute("SELECT camera_id FROM cameras FOR UPDATE NOWAIT")
            return one.result(timeout=5), two.result(timeout=5)


@pytest.mark.parametrize("clear", [False, True], ids=["save", "clear"])
def test_competing_camera_and_bed_writes_preserve_fields_and_all_revisions(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch, clear: bool
) -> None:
    sandbox = postgres_product_sandbox
    registry = _camera(sandbox)
    zones = BedZoneStore(sandbox.database, sandbox.authority)
    original = _put(zones)
    zone, camera = _compete(
        sandbox,
        monkeypatch,
        (lambda: zones.delete("camera-a")) if clear else (lambda: _put(zones)),
        lambda: registry.update(
            "camera-a", CameraUpdate.model_validate({"label": "Concurrent label", "floor": -1})
        ),
    )
    assert (zone is True) if clear else (zone == original)
    assert camera["label"] == "Concurrent label" and camera["floor"] == -1
    assert zones.get("camera-a") == (None if clear else original)
    assert sandbox.admin.execute(
        "SELECT label,floor_override,revision FROM cameras"
    ).fetchone() == ("Concurrent label", "-1", 4)
    assert sandbox.admin.execute(
        "SELECT registry_version,topology_dirty_registry_version FROM edge_site"
    ).fetchone() == (4, 4)


def test_competing_zone_writers_each_return_their_own_committed_value(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    first = BedZoneStore(sandbox.database, sandbox.authority)
    second = BedZoneStore(sandbox.database, sandbox.authority)
    other = replace(_REGION, id="other", origin="model")
    one, two = _compete(
        sandbox, monkeypatch, lambda: _put(first), lambda: _put(second, regions=(other,))
    )
    assert one == BedZone((_REGION,), 640, 480, _TIME)
    assert two == BedZone((other,), 640, 480, _TIME)
    assert first.get("camera-a") in (one, two)
    assert sandbox.admin.execute("SELECT revision FROM cameras").fetchone() == (3,)
    assert sandbox.admin.execute("SELECT registry_version FROM edge_site").fetchone() == (3,)


def test_put_returns_own_commit_not_a_racing_reread(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    other = BedZoneStore(sandbox.database, sandbox.authority)
    transact = sandbox.database.transact
    interleaved = []

    def interleaved_transaction(callback):
        candidate = transact(callback)
        if not interleaved:
            interleaved.append(candidate)
            _put(other, regions=(replace(_REGION, id="later"),))
        return candidate

    monkeypatch.setattr(sandbox.database, "transact", interleaved_transaction)
    own = _put(store)
    assert own == BedZone((_REGION,), 640, 480, _TIME)
    assert interleaved == [own]
    assert store.get("camera-a").regions[0].id == "later"


@pytest.mark.parametrize("projection", ["camera_exists", "get", "get_all"])
def test_reads_use_one_readonly_snapshot_and_return_only_after_commit(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
) -> None:
    sandbox = postgres_product_sandbox
    registry = _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    saved = _put(store)
    _camera(sandbox, "camera-b")
    _put(store, "camera-b")
    before = True if projection == "camera_exists" else saved
    if projection == "get_all":
        before = {"camera-a": saved, "camera-b": saved}
    read = sandbox.database.read
    execute = psycopg.Cursor.execute
    reads = []
    statements = []
    armed = True

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

    def interleave(cursor, query, *args, **kwargs):
        nonlocal armed
        result = execute(cursor, query, *args, **kwargs)
        if armed and isinstance(query, str) and "FROM edge_site AS s" in query:
            armed = False
            statements.append(query)
            registry.delete("camera-a")
            registry.delete("camera-b")
        return result

    monkeypatch.setattr(sandbox.database, "read", observed_read)
    monkeypatch.setattr(psycopg.Cursor, "execute", interleave)
    result = store.get_all() if projection == "get_all" else getattr(store, projection)("camera-a")
    assert result == before and len(reads) == len(statements) == 1 and not armed
    assert store.camera_exists("camera-a") is False
    assert sandbox.admin.execute("SELECT registry_version FROM edge_site").fetchone() == (6,)


@pytest.mark.parametrize("projection", ["camera_exists", "get", "get_all"])
def test_unknown_read_commit_does_not_publish_a_snapshot_or_retry(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
) -> None:
    sandbox = postgres_product_sandbox
    _camera(sandbox)
    store = BedZoneStore(sandbox.database, sandbox.authority)
    saved = _put(store)
    read = sandbox.database.read
    commit = psycopg.Connection.commit
    reads = []
    commits = []

    def observed_read(callback):
        def observe(connection):
            reads.append(connection.info.backend_pid)
            return callback(connection)

        return read(observe)

    def lose_receipt(connection: psycopg.Connection) -> None:
        if connection.info.backend_pid in reads:
            commits.append(connection.info.backend_pid)
            commit(connection)
            raise psycopg.OperationalError("injected read COMMIT receipt loss")
        commit(connection)

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.database, "read", observed_read)
        patch.setattr(psycopg.Connection, "commit", lose_receipt)
        with pytest.raises(CommitOutcomeUnknown):
            if projection == "get_all":
                store.get_all()
            else:
                getattr(store, projection)("camera-a")
    assert len(reads) == len(commits) == 1
    assert store.get_all() == {"camera-a": saved}
