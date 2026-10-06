"""Local detection overrides against the real PostgreSQL product sandbox."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Barrier, Event
from time import monotonic
from typing import TYPE_CHECKING

import psycopg
import pytest
from fastapi import FastAPI
from psycopg.pq import TransactionStatus
from psycopg.rows import tuple_row

from backend.app.edge_db.authority import AuthorityFenced, freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_store import append_postgres_audit
from backend.app.features.audit.store import AuditEvent
from backend.app.features.cameras.bed_zone_store import BedZoneRegion, BedZoneStore
from backend.app.features.cameras.router import _apply_local_detection_overrides
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.detection_settings.store import (
    DetectionSettingsNotInitialized,
    DetectionSettingsStore,
    DomainDetectionSetting,
)
from contracts.worker_config import PulledNightWindow, PulledWorkerConfig

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_TIME = "2026-09-27T04:00:00.123Z"
_ALWAYS = DomainDetectionSetting(on=True, mode="always", start=None, end=None)
_OFF = DomainDetectionSetting(on=False, mode="always", start=None, end=None)
_NIGHT = DomainDetectionSetting(on=True, mode="window", start="22:00", end="06:00")
_DAY = DomainDetectionSetting(on=False, mode="window", start="09:00", end="18:00")


def _rows(sandbox: ProductSandbox):
    return sandbox.admin.execute("SELECT * FROM edge_site ORDER BY id").fetchall()


def _audit(connection: psycopg.Connection) -> None:
    append_postgres_audit(
        connection,
        AuditEvent(
            occurred_at=_TIME,
            actor_id="test-operator",
            action=AuditAction.DETECTION_SETTINGS_UPDATE,
            target_id="detection-settings",
            detail=empty_detail(AuditAction.DETECTION_SETTINGS_UPDATE),
        ),
    )


def _effective(
    store: DetectionSettingsStore, pulled: PulledWorkerConfig | None = None
) -> dict[str, object]:
    # Exercise the existing pure response projection, not API composition.
    app = FastAPI()
    app.state.detection_settings_store = store
    response: dict[str, object] = {"config_version": 0 if pulled is None else pulled.config_version}
    if pulled is not None:
        response["detection_windows"] = {
            domain: {"start": window.start, "end": window.end, "tz": window.tz}
            for domain, window in pulled.detection_windows.items()
        }
        if pulled.night_window is not None:
            window = pulled.night_window
            response["night_window"] = {"start": window.start, "end": window.end, "tz": window.tz}
    _apply_local_detection_overrides(app, response, pulled)
    return response


def test_partial_replacements_round_trip_persist_and_keep_canonical_public_shapes(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    assert store.replace_all({"bed_exit": _NIGHT}) is None
    assert store.get_all() == {"bed_exit": _NIGHT}
    assert sandbox.admin.execute(
        "SELECT fall_on,fall_mode,fall_start_time,fall_end_time FROM edge_site"
    ).fetchone() == (None, None, None, None)
    assert store.replace_all({"fall": _OFF}) is None
    assert store.get_all() == {"fall": _OFF, "bed_exit": _NIGHT}
    assert sandbox.admin.execute(
        "SELECT fall_on,fall_mode,fall_start_time,fall_end_time,"
        "bed_exit_on,bed_exit_mode,bed_exit_start_time,bed_exit_end_time FROM edge_site"
    ).fetchone() == (0, "always", None, None, 1, "window", "22:00", "06:00")
    reopened = DetectionSettingsStore(sandbox.database, sandbox.authority)
    assert reopened.get_all() == {"fall": _OFF, "bed_exit": _NIGHT}
    wire = {domain: setting.as_dict() for domain, setting in reopened.get_all().items()}
    assert json.dumps(wire, separators=(",", ":")) == (
        '{"fall":{"on":false,"mode":"always","start":null,"end":null},'
        '"bed_exit":{"on":true,"mode":"window","start":"22:00","end":"06:00"}}'
    )
    reopened.replace_all({"fall": _ALWAYS})
    assert store.get_all() == {"fall": _ALWAYS, "bed_exit": _NIGHT}
    reopened.replace_all({"bed_exit": _OFF, "fall": _DAY})
    assert store.get_all() == {"fall": _DAY, "bed_exit": _OFF}
    # These overrides have content-derived effective versions, not registry or
    # numeric-policy activation generations.
    assert sandbox.admin.execute(
        "SELECT registry_version,runtime_settings_version,topology_dirty_registry_version "
        "FROM edge_site"
    ).fetchone() == (0, 0, None)
    assert sandbox.admin.execute("SELECT count(*) FROM policies").fetchone() == (0,)


def test_empty_input_leaves_nonempty_settings_and_timestamp_untouched_but_calls_hook(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    store.replace_all({"fall": _ALWAYS, "bed_exit": _NIGHT})
    before = _rows(sandbox)
    hooks = []

    def hook(connection: psycopg.Connection) -> None:
        hooks.append(connection.info.transaction_status)
        _audit(connection)

    assert store.replace_all({}, after_write=hook) is None
    assert hooks == [TransactionStatus.INTRANS] and _rows(sandbox) == before
    assert store.get_all() == {"fall": _ALWAYS, "bed_exit": _NIGHT}
    assert sandbox.admin.execute("SELECT action FROM audit_events").fetchall() == [
        (AuditAction.DETECTION_SETTINGS_UPDATE.value,)
    ]


def test_identical_replacement_still_writes_and_audits_without_changing_effective_version(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    settings = {"fall": _OFF, "bed_exit": _NIGHT}
    store.replace_all(settings)
    sandbox.admin.execute("UPDATE edge_site SET updated_at=%s WHERE id=1", (_TIME,))
    before = _effective(store)
    store.replace_all(settings, after_write=_audit)
    assert store.get_all() == settings and _effective(store) == before
    assert sandbox.admin.execute("SELECT updated_at FROM edge_site").fetchone()[0] != _TIME
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (1,)


@pytest.mark.parametrize("domain", ["unknown", "fall_on=0; DELETE FROM cameras; --", "Fall"])
def test_unknown_domains_are_rejected_without_partial_writes_or_sql_interpolation(
    postgres_product_sandbox: ProductSandbox, domain: str
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    store.replace_all({"fall": _ALWAYS, "bed_exit": _NIGHT})
    before = _rows(sandbox)
    hooks = []
    with pytest.raises(KeyError) as error:
        store.replace_all({"fall": _OFF, domain: _DAY}, after_write=hooks.append)
    assert error.value.args == (domain,)
    assert not hooks and _rows(sandbox) == before
    assert store.get_all() == {"fall": _ALWAYS, "bed_exit": _NIGHT}
    with pytest.raises(KeyError) as multiple:
        store.replace_all({"zzz": _ALWAYS, "aaa": _OFF})
    assert multiple.value.args == ("aaa",)


@pytest.mark.parametrize(
    "invalid",
    [
        DomainDetectionSetting(True, "invalid", None, None),
        DomainDetectionSetting(True, "always", "22:00", "06:00"),
        DomainDetectionSetting(True, "window", None, "06:00"),
        DomainDetectionSetting(True, "window", "22:00", None),
        DomainDetectionSetting(True, "window", "24:00", "06:00"),
        DomainDetectionSetting(True, "window", "22:00", "12:60"),
        DomainDetectionSetting(True, "window", "9:00", "18:00"),
        DomainDetectionSetting(True, "window", "09:00:00", "18:00"),
        DomainDetectionSetting(True, "window", "09:00+09:00", "18:00"),
        DomainDetectionSetting(True, "window", "", "18:00"),
        DomainDetectionSetting(2, "always", None, None),
    ],
)
def test_native_schedule_constraints_roll_back_preceding_domain_update(
    postgres_product_sandbox: ProductSandbox, invalid: DomainDetectionSetting
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    original = {"fall": _ALWAYS, "bed_exit": _NIGHT}
    store.replace_all(original)
    before = _rows(sandbox)
    effective = _effective(store)
    hooks = []
    with pytest.raises(psycopg.IntegrityError):
        store.replace_all({"fall": _OFF, "bed_exit": invalid}, after_write=hooks.append)
    assert not hooks and _rows(sandbox) == before
    assert store.get_all() == original and _effective(store) == effective


@pytest.mark.parametrize(
    ("start", "end"), [("00:00", "23:59"), ("23:59", "00:00"), ("12:00", "12:00")]
)
def test_store_preserves_clock_text_without_adding_route_schedule_policy(
    postgres_product_sandbox: ProductSandbox, start: str, end: str
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    window = DomainDetectionSetting(False, "window", start, end)
    store.replace_all({"bed_exit": window})
    assert store.get_all() == {"bed_exit": window}
    assert store.get_all()["bed_exit"].as_dict() == {
        "on": False,
        "mode": "window",
        "start": start,
        "end": end,
    }


@pytest.mark.parametrize("source", ["domain", "legacy-night", "absent"])
def test_committed_override_reuses_existing_timezone_and_keeps_content_version_stable(
    postgres_product_sandbox: ProductSandbox, source: str
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    pulled_window = PulledNightWindow(start="20:00", end="05:00", tz="Asia/Seoul")
    pulled = PulledWorkerConfig(
        config_version=7,
        restart_epoch=1,
        cameras=(),
        night_window=pulled_window if source == "legacy-night" else None,
        detection_windows={"bed_exit": pulled_window} if source == "domain" else {},
    )
    store.replace_all({"bed_exit": _NIGHT})
    saved = _effective(store, pulled)
    expected_window = {
        "start": "22:00",
        "end": "06:00",
        "tz": "UTC" if source == "absent" else "Asia/Seoul",
    }
    assert saved["domains"] == {"bed_exit": {"enabled": True}}
    assert saved["detection_windows"] == {"bed_exit": expected_window}
    assert saved["night_window"] == expected_window
    assert saved["config_version"] > 7
    assert _effective(store, pulled) == saved
    assert pulled_window.start == "20:00" and pulled_window.end == "05:00"
    store.replace_all({"bed_exit": _OFF})
    disabled = _effective(store, pulled)
    assert disabled["domains"] == {"bed_exit": {"enabled": False}}
    assert "detection_windows" not in disabled and "night_window" not in disabled
    assert disabled["config_version"] != saved["config_version"]
    store.replace_all({"bed_exit": _ALWAYS})
    always = _effective(store, pulled)
    assert always["domains"] == {"bed_exit": {"enabled": True}}
    assert "detection_windows" not in always and "night_window" not in always
    assert always["config_version"] != disabled["config_version"]


@pytest.mark.parametrize("fail", [False, True], ids=["commit", "rollback"])
def test_audit_uses_same_active_connection_with_both_overrides_and_atomic_effective_policy(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    original = {"fall": _ALWAYS, "bed_exit": _NIGHT}
    changed = {"fall": _OFF, "bed_exit": _DAY}
    store.replace_all(original)
    before = _rows(sandbox)
    effective = _effective(store)
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
        assert connection.row_factory is tuple_row
        assert connection.info.transaction_status is TransactionStatus.INTRANS
        assert connection.execute("SHOW transaction_isolation").fetchone() == ("read committed",)
        assert connection.execute(
            "SELECT fall_on,fall_mode,bed_exit_on,bed_exit_mode,"
            "bed_exit_start_time,bed_exit_end_time "
            "FROM edge_site"
        ).fetchone() == (0, "always", 0, "window", "09:00", "18:00")
        assert _rows(sandbox) == before and _effective(store) == effective
        _audit(connection)
        assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
        if fail:
            raise RuntimeError("injected audit failure")

    monkeypatch.setattr(sandbox.database, "transact", observed_transaction)
    if fail:
        with pytest.raises(RuntimeError, match="injected audit failure"):
            store.replace_all(changed, after_write=hook)
        assert _rows(sandbox) == before and store.get_all() == original
        assert _effective(store) == effective
    else:
        assert store.replace_all(changed, after_write=hook) is None
        assert store.get_all() == changed and _rows(sandbox) != before
        assert _effective(store)["config_version"] != effective["config_version"]
    assert len(connections) == len(hooks) == 1
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (int(not fail),)
    assert sandbox.database.read(lambda connection: connection.row_factory is tuple_row)


def test_real_deferred_commit_failure_restores_settings_and_audit(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    original = {"fall": _ALWAYS, "bed_exit": _NIGHT}
    store.replace_all(original)
    sandbox.admin.execute(
        "CREATE TABLE detection_commit_guard (site_id bigint REFERENCES edge_site(id) "
        "DEFERRABLE INITIALLY DEFERRED)"
    )
    before = _rows(sandbox)
    effective = _effective(store)
    hooks = []

    def reject_commit(connection: psycopg.Connection) -> None:
        hooks.append(True)
        _audit(connection)
        connection.execute("INSERT INTO detection_commit_guard VALUES (2)")
        assert connection.info.transaction_status is TransactionStatus.INTRANS

    with pytest.raises(psycopg.IntegrityError):
        store.replace_all({"fall": _OFF, "bed_exit": _DAY}, after_write=reject_commit)
    assert hooks == [True]
    assert _rows(sandbox) == before and store.get_all() == original
    assert _effective(store) == effective
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
    assert sandbox.admin.execute("SELECT count(*) FROM detection_commit_guard").fetchone() == (0,)


@pytest.mark.parametrize(
    "committed", [False, True], ids=["lost-before-commit", "lost-after-commit"]
)
def test_unknown_commit_never_retries_or_publishes_success_or_cached_overrides(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch, committed: bool
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    original = {"fall": _ALWAYS, "bed_exit": _NIGHT}
    changed = {"fall": _OFF, "bed_exit": _DAY}
    store.replace_all(original)
    before = _rows(sandbox)
    effective = _effective(store)
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
            published.append(store.replace_all(changed, after_write=hook))
    assert not published and len(hook_pids) == len(commit_pids) == 1
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (
        int(committed),
    )
    expected = changed if committed else original
    assert store.get_all() == expected
    assert DetectionSettingsStore(sandbox.database, sandbox.authority).get_all() == expected
    if committed:
        assert _effective(store)["config_version"] != effective["config_version"]
    else:
        assert _rows(sandbox) == before and _effective(store) == effective


@pytest.mark.parametrize("settings", [{"fall": _OFF}, {}, {"invalid": _ALWAYS}])
def test_frozen_authority_is_checked_before_bootstrap_validation_and_empty_mutations(
    postgres_product_sandbox: ProductSandbox, settings: dict[str, DomainDetectionSetting]
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    store.replace_all({"fall": _ALWAYS, "bed_exit": _NIGHT})
    freeze_authority(sandbox.database, sandbox.authority)
    # Reads remain permitted under a frozen writer authority.
    assert store.get_all() == {"fall": _ALWAYS, "bed_exit": _NIGHT}
    before = _rows(sandbox)
    hooks = []
    with pytest.raises(AuthorityFenced):
        store.replace_all(settings, after_write=hooks.append)
    assert not hooks and _rows(sandbox) == before
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    with pytest.raises(AuthorityFenced):
        store.replace_all(settings, after_write=hooks.append)
    assert not hooks
    assert sandbox.admin.execute("SELECT count(*) FROM edge_site").fetchone() == (0,)


def test_missing_bootstrap_fails_closed_without_recreating_or_serving_defaults(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    store.replace_all({"fall": _ALWAYS, "bed_exit": _NIGHT})
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    with pytest.raises(DetectionSettingsNotInitialized, match="bootstrap row is missing"):
        store.get_all()
    hooks = []
    for settings in ({"fall": _OFF}, {}):
        with pytest.raises(DetectionSettingsNotInitialized):
            store.replace_all(settings, after_write=hooks.append)
    assert not hooks
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
                    pytest.fail("both config writers must wait on the singleton row lock")
            return one.result(timeout=5), two.result(timeout=5)


def test_competing_partial_domain_updates_preserve_both_fields_and_audit_chain(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    first = DetectionSettingsStore(sandbox.database, sandbox.authority)
    second = DetectionSettingsStore(sandbox.database, sandbox.authority)
    first.replace_all({"fall": _ALWAYS, "bed_exit": _NIGHT})
    one, two = _compete(
        sandbox,
        monkeypatch,
        lambda: first.replace_all({"fall": _OFF}, after_write=_audit),
        lambda: second.replace_all({"bed_exit": _DAY}, after_write=_audit),
    )
    assert one is None and two is None
    assert first.get_all() == {"fall": _OFF, "bed_exit": _DAY}
    assert sandbox.admin.execute(
        "SELECT fall_on,bed_exit_on,bed_exit_start_time,bed_exit_end_time FROM edge_site"
    ).fetchone() == (0, 0, "09:00", "18:00")
    audits = sandbox.admin.execute(
        "SELECT previous_hash,record_hash FROM audit_events ORDER BY audit_id"
    ).fetchall()
    assert len(audits) == 2 and audits[1][0] == audits[0][1]


def test_competing_bed_zone_and_override_writes_preserve_config_and_registry_revision(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
    registry.create(
        camera_id="camera-a",
        label="Bed",
        rtsp_url="rtsp://camera.invalid/live",
        space_id=None,
        status="online",
    )
    zones = BedZoneStore(sandbox.database, sandbox.authority)
    settings = DetectionSettingsStore(sandbox.database, sandbox.authority)
    settings.replace_all({"fall": _ALWAYS, "bed_exit": _NIGHT})
    region = BedZoneRegion("bed-a", ((0, 0), (9, 0), (0, 9)), "manual")
    zone, result = _compete(
        sandbox,
        monkeypatch,
        lambda: zones.put(
            "camera-a", regions=(region,), image_width=10, image_height=10, recognized_at=_TIME
        ),
        lambda: settings.replace_all({"fall": _OFF}),
    )
    assert result is None and zone.regions == (region,)
    assert zones.get("camera-a") == zone
    assert settings.get_all() == {"fall": _OFF, "bed_exit": _NIGHT}
    assert sandbox.admin.execute("SELECT revision FROM cameras").fetchone() == (2,)
    assert sandbox.admin.execute(
        "SELECT registry_version,topology_dirty_registry_version FROM edge_site"
    ).fetchone() == (2, 2)


def test_snapshot_reads_both_domains_in_one_readonly_statement_across_a_committed_writer(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    other = DetectionSettingsStore(sandbox.database, sandbox.authority)
    original = {"fall": _ALWAYS, "bed_exit": _NIGHT}
    changed = {"fall": _OFF, "bed_exit": _DAY}
    store.replace_all(original)
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
        if armed and isinstance(query, str) and query.startswith("SELECT fall_on,"):
            armed = False
            statements.append(query)
            other.replace_all(changed)
        return result

    monkeypatch.setattr(sandbox.database, "read", observed_read)
    monkeypatch.setattr(psycopg.Cursor, "execute", interleave)
    assert store.get_all() == original
    assert len(reads) == len(statements) == 1 and not armed
    assert store.get_all() == changed
    assert DetectionSettingsStore(sandbox.database, sandbox.authority).get_all() == changed


def test_unknown_read_commit_propagates_without_snapshot_publication_or_replay(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    original = {"fall": _ALWAYS, "bed_exit": _NIGHT}
    store.replace_all(original)
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
            store.get_all()
    assert len(reads) == len(commits) == 1
    assert store.get_all() == original
