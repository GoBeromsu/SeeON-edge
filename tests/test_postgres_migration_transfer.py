"""Fence, one-time authority transfer and the rollback predicate on a real PG18 target."""

from __future__ import annotations

import fcntl
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from backend.app.edge_db.authority import AuthorityFenced, AuthorityToken, freeze_authority
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.load import import_snapshot
from backend.app.edge_db.migration.mapping import diagnostics_schema_name
from backend.app.edge_db.migration.reconcile import reconcile
from backend.app.edge_db.migration.rollback import rollback_check
from backend.app.edge_db.migration.snapshot import export_snapshot
from backend.app.edge_db.migration.sqlite_fence import fence_sqlite
from backend.app.edge_db.migration.transfer import freeze, pending_authority_path, transfer
from backend.app.edge_db.migration.worker_state import queue_digest
from backend.app.edge_db.postgres import PoolBudget
from backend.app.features.diagnostics.postgres_database import DIAGNOSTICS_POOL_BUDGET
from backend.app.postgres_root import (
    API_POSTGRES_AUTHORITY_FILE_ENV,
    API_POSTGRES_DSN_FILE_ENV,
    API_POSTGRES_SCHEMA_ENV,
    PostgresRootError,
    close_postgres_database,
    open_postgres_root,
)
from tests_support.postgres_migration import (
    NOW,
    MigrationTarget,
    authority_file_token,
    authority_row,
    ingest_live_record,
    insert_in_flight_outbox,
    ledger,
    make_worker_state,
    runtime_database,
    runtime_role_database,
    source_and_destination,
    table_counts,
)

pytest_plugins = ("tests_support.postgres_migration",)

_Fault = Callable[[psycopg.Connection, Callable[[], None]], None]
_ROOT_BUDGET = PoolBudget(
    max_connections=1,
    max_waiting=1,
    acquire_timeout_sec=5.0,
    statement_timeout_ms=5000,
    lock_timeout_ms=5000,
    startup_timeout_sec=5.0,
)


class _ProcessDeath(BaseException):
    """The migration process dies at the COMMIT boundary."""


@dataclass(frozen=True)
class _Legacy:
    """The stopped old database, its snapshot and where its fence receipt goes."""

    source: Path
    snapshot: Path
    receipt: Path


@pytest.fixture
def legacy(tmp_path: Path) -> _Legacy:
    source, destination = source_and_destination(tmp_path)
    receipts = tmp_path / "receipts"
    receipts.mkdir(mode=0o700)
    return _Legacy(source, export_snapshot(source, destination).path, receipts / "fence.json")


@pytest.fixture
def snapshot(legacy: _Legacy) -> Path:
    return legacy.snapshot


@pytest.fixture
def imported(migration_target: MigrationTarget, snapshot: Path) -> MigrationTarget:
    target = migration_target
    import_snapshot(target.database, schema=target.schema, snapshot_path=snapshot)
    return target


def _fault_at_transfer_commit(patch: pytest.MonkeyPatch, schema: str, fault: _Fault) -> None:
    """Break the first COMMIT that would publish generation 2."""
    original = psycopg.Connection.commit
    armed = [True]
    query = sql.SQL("SELECT generation FROM {}").format(
        sql.Identifier(schema, "deployment_authority")
    )

    def commit(self: psycopg.Connection) -> None:
        if armed[0] and self.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS:
            (generation,) = self.execute(query).fetchone()
            if generation == 2:
                armed[0] = False
                fault(self, lambda: original(self))
                return
        original(self)

    patch.setattr(psycopg.Connection, "commit", commit)


def _ack_lost_after_commit(connection: psycopg.Connection, commit: Callable[[], None]) -> None:
    commit()
    raise psycopg.OperationalError("injected COMMIT receipt loss")


def _ack_lost_before_commit(connection: psycopg.Connection, commit: Callable[[], None]) -> None:
    connection.rollback()
    raise psycopg.OperationalError("injected COMMIT receipt loss")


def _died_after_commit(connection: psycopg.Connection, commit: Callable[[], None]) -> None:
    commit()
    connection.close()
    raise _ProcessDeath


def _died_before_commit(connection: psycopg.Connection, commit: Callable[[], None]) -> None:
    connection.close()
    raise _ProcessDeath


def _transfer(target: MigrationTarget, **options: object) -> AuthorityToken:
    return transfer(target.database, target.authority_path, schema=target.schema, **options)


def _fresh_source(root: Path) -> Path:
    """The operator-named legacy database path; its directory exists and the file does not."""
    directory = root / "fresh-state"
    directory.mkdir(exist_ok=True)
    return directory / "edge.sqlite3"


def _fresh_install(target: MigrationTarget, root: Path) -> AuthorityToken:
    return _transfer(target, fresh_install_source=_fresh_source(root))


def _fence(target: MigrationTarget, legacy: _Legacy) -> None:
    """Stamp the old database at the current generation, as the operator does after export.

    An absent old database is fenced without a snapshot, as on a fresh install.
    """
    generation, _ = authority_file_token(target.authority_path)
    snapshot = legacy.snapshot if legacy.source.exists() else None
    fence_sqlite(legacy.source, snapshot=snapshot, generation=generation, receipt=legacy.receipt)


def _rollback(target: MigrationTarget, legacy: _Legacy) -> dict[str, object]:
    return rollback_check(
        target.database,
        schema=target.schema,
        snapshot_path=legacy.snapshot,
        source=legacy.source,
        fence_receipt=legacy.receipt,
    )


def _file_token(path: Path) -> AuthorityToken:
    generation, writer_token = authority_file_token(path)
    return AuthorityToken(generation=generation, writer_token=writer_token)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_transfer_opens_a_new_generation_and_fences_the_old_token(
    imported: MigrationTarget,
) -> None:
    target = imported
    _, old_token = authority_file_token(target.authority_path)

    successor = _transfer(target)

    assert successor == _file_token(target.authority_path)
    assert successor.generation == 2
    assert successor.writer_token != old_token
    assert authority_row(target.admin, target.schema) == (2, successor.writer_token, True, True)
    assert stat.S_IMODE(target.authority_path.stat().st_mode) == 0o600
    assert not pending_authority_path(target.authority_path).exists()
    with pytest.raises(AuthorityFenced, match="^cannot fence a different persistence authority$"):
        freeze_authority(target.database, AuthorityToken(generation=1, writer_token=old_token))
    assert authority_row(target.admin, target.schema) == (2, successor.writer_token, True, True)


def test_transfer_before_import_changes_nothing(migration_target: MigrationTarget) -> None:
    target = migration_target
    authority_bytes = target.authority_path.read_bytes()
    generation, token = authority_file_token(target.authority_path)

    with pytest.raises(MigrationError, match="^target holds no imported snapshot$"):
        _transfer(target)

    assert authority_row(target.admin, target.schema) == (generation, token, False, False)
    assert target.authority_path.read_bytes() == authority_bytes
    assert not pending_authority_path(target.authority_path).exists()


def test_second_transfer_with_a_stale_or_spent_token_is_refused(
    imported: MigrationTarget, tmp_path: Path
) -> None:
    target = imported
    stale_directory = tmp_path / "stale"
    stale_directory.mkdir(mode=0o700)
    stale = stale_directory / "authority.json"
    stale.write_bytes(target.authority_path.read_bytes())
    stale.chmod(0o600)
    successor = _transfer(target)

    with pytest.raises(
        AuthorityFenced, match="^cannot transfer a different persistence authority$"
    ):
        transfer(target.database, stale, schema=target.schema)
    with pytest.raises(MigrationError, match="^authority was already transferred$"):
        _transfer(target)

    assert authority_row(target.admin, target.schema) == (2, successor.writer_token, True, True)
    assert _file_token(target.authority_path) == successor
    assert not pending_authority_path(target.authority_path).exists()
    assert not pending_authority_path(stale).exists()


def test_lost_commit_receipt_after_commit_publishes_the_committed_token(
    imported: MigrationTarget, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = imported

    with monkeypatch.context() as patch:
        _fault_at_transfer_commit(patch, target.schema, _ack_lost_after_commit)
        successor = _transfer(target)

    assert successor.generation == 2
    assert _file_token(target.authority_path) == successor
    assert authority_row(target.admin, target.schema) == (2, successor.writer_token, True, True)
    assert not pending_authority_path(target.authority_path).exists()


def test_lost_commit_receipt_without_commit_keeps_the_old_authority(
    imported: MigrationTarget, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = imported
    authority_bytes = target.authority_path.read_bytes()
    generation, token = authority_file_token(target.authority_path)

    with monkeypatch.context() as patch:
        _fault_at_transfer_commit(patch, target.schema, _ack_lost_before_commit)
        with pytest.raises(
            MigrationError, match="^transfer did not commit; the authority is unchanged$"
        ):
            _transfer(target)

    assert target.authority_path.read_bytes() == authority_bytes
    assert authority_row(target.admin, target.schema) == (generation, token, False, False)
    assert not pending_authority_path(target.authority_path).exists()
    assert _transfer(target).generation == 2


def test_death_after_commit_is_completed_by_the_rerun(
    imported: MigrationTarget, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = imported
    pending = pending_authority_path(target.authority_path)
    authority_bytes = target.authority_path.read_bytes()

    with monkeypatch.context() as patch:
        _fault_at_transfer_commit(patch, target.schema, _died_after_commit)
        with pytest.raises(_ProcessDeath):
            _transfer(target)

    staged = _file_token(pending)
    assert target.authority_path.read_bytes() == authority_bytes
    assert authority_row(target.admin, target.schema) == (2, staged.writer_token, True, True)

    with runtime_database(target.dsn, target.schema) as restarted:
        resumed = transfer(restarted, target.authority_path, schema=target.schema)

    assert resumed == staged
    assert _file_token(target.authority_path) == staged
    assert not pending.exists()


def test_death_before_commit_is_discarded_by_the_rerun(
    imported: MigrationTarget, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = imported
    pending = pending_authority_path(target.authority_path)
    generation, token = authority_file_token(target.authority_path)

    with monkeypatch.context() as patch:
        _fault_at_transfer_commit(patch, target.schema, _died_before_commit)
        with pytest.raises(_ProcessDeath):
            _transfer(target)

    staged = _file_token(pending)
    assert authority_row(target.admin, target.schema) == (generation, token, False, False)

    with runtime_database(target.dsn, target.schema) as restarted:
        resumed = transfer(restarted, target.authority_path, schema=target.schema)
        with pytest.raises(
            AuthorityFenced, match="^cannot fence a different persistence authority$"
        ):
            freeze_authority(restarted, staged)

    assert resumed.generation == 2
    assert resumed.writer_token != staged.writer_token
    assert _file_token(target.authority_path) == resumed
    assert authority_row(target.admin, target.schema) == (2, resumed.writer_token, True, True)
    assert not pending.exists()


@pytest.mark.parametrize(
    ("lock_file", "message"),
    [
        (".gpu.lease", "^the old worker holds its runtime lease; stop it first$"),
        (
            "delivery-queue/.delivery-queue.lock",
            "^the delivery queue is locked by another process$",
        ),
    ],
    ids=["sender-lease", "queue-lock"],
)
def test_a_live_old_worker_blocks_the_transfer(
    imported: MigrationTarget, tmp_path: Path, lock_file: str, message: str
) -> None:
    target = imported
    state = make_worker_state(tmp_path)
    generation, token = authority_file_token(target.authority_path)

    descriptor = os.open(state / lock_file, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with pytest.raises(MigrationError, match=message):
            _transfer(target, worker_state_dir=state)
        with pytest.raises(MigrationError, match=message):
            queue_digest(state)
    finally:
        os.close(descriptor)

    assert authority_row(target.admin, target.schema) == (generation, token, False, False)
    assert not pending_authority_path(target.authority_path).exists()


def test_queued_alerts_survive_the_cutover_bound_by_digest(
    imported: MigrationTarget, snapshot: Path, tmp_path: Path
) -> None:
    target = imported
    state = make_worker_state(tmp_path)
    before = queue_digest(state)
    queued_bytes = _tree_bytes(state)
    alert = state / "delivery-queue" / "0001-event-1.json"

    bound = reconcile(
        target.database,
        schema=target.schema,
        snapshot_path=snapshot,
        worker_state_dir=state,
        expected_queue_sha256=before.sha256,
    )
    alert.write_bytes(b'{"synthetic":9}\n')
    altered = reconcile(
        target.database,
        schema=target.schema,
        snapshot_path=snapshot,
        worker_state_dir=state,
        expected_queue_sha256=before.sha256,
    )
    alert.write_bytes(queued_bytes["delivery-queue/0001-event-1.json"])
    successor = _transfer(target, worker_state_dir=state)

    queue = bound["delivery_queue"]
    assert (bound["result"], bound["failures"]) == ("PASS", [])
    assert queue["result"] == "PASS"
    assert (queue["queued"], queue["temporary"], queue["dead_lettered"]) == (2, 1, 1)
    assert (altered["result"], altered["failures"]) == ("FAIL", ["delivery_queue:sha256"])
    assert successor.generation == 2
    assert queue_digest(state) == before
    assert _tree_bytes(state) == queued_bytes


def test_in_flight_delivery_blocks_transfer_and_rollback(
    imported: MigrationTarget, legacy: _Legacy
) -> None:
    target = imported
    _fence(target, legacy)
    insert_in_flight_outbox(target.admin, target.schema)
    generation, token = authority_file_token(target.authority_path)

    report = reconcile(target.database, schema=target.schema, snapshot_path=legacy.snapshot)
    with pytest.raises(
        MigrationError,
        match="^delivery tables are not empty: event_outbox, event_delivery_attempts$",
    ):
        _transfer(target)
    verdict = _rollback(target, legacy)

    assert report["result"] == "FAIL"
    assert report["failures"] == ["pending:event_outbox", "pending:event_delivery_attempts"]
    assert report["pending"]["active_leases"] == 1
    assert report["pending"]["outbox_states"] == {"IN_FLIGHT": 1}
    assert (verdict["result"], verdict["reasons"]) == (
        "DENY",
        ["delivery_history:event_outbox", "delivery_history:event_delivery_attempts"],
    )
    assert authority_row(target.admin, target.schema) == (generation, token, False, False)
    assert not pending_authority_path(target.authority_path).exists()


def test_rollback_is_denied_for_a_live_authority_or_postgres_only_history(
    imported: MigrationTarget, legacy: _Legacy
) -> None:
    target = imported

    def verdict() -> tuple[object, object]:
        decision = _rollback(target, legacy)
        return decision["result"], decision["reasons"]

    _fence(target, legacy)
    after_import = verdict()
    _transfer(target)
    after_transfer = verdict()
    freeze(target.database, target.authority_path)
    after_freeze = verdict()
    target.admin.execute(
        sql.SQL("UPDATE {} SET status = 'applied', applied_at = %s").format(
            sql.Identifier(target.schema, "policies")
        ),
        (NOW,),
    )
    after_status_flip = verdict()

    assert after_import == ("ALLOW", [])
    assert after_transfer == ("DENY", ["authority_not_fenced"])
    assert after_freeze == ("ALLOW", [])
    assert after_status_flip == ("DENY", ["target_history:policies"])


def test_live_diagnostics_and_product_writes_each_deny(
    imported: MigrationTarget, legacy: _Legacy
) -> None:
    target = imported
    diagnostics = diagnostics_schema_name(target.schema)

    def verdict() -> tuple[object, object]:
        decision = _rollback(target, legacy)
        return decision["result"], decision["reasons"]

    _fence(target, legacy)
    _transfer(target)
    with runtime_role_database(
        target.dsn, diagnostics, target.runtime_role, DIAGNOSTICS_POOL_BUDGET
    ) as live:
        ingest_live_record(live, "live-1")
    while_live = verdict()
    freeze(target.database, target.authority_path)
    after_freeze = verdict()
    written = sorted(
        table
        for table, count in table_counts(target.admin, diagnostics).items()
        if count and table != "schema_migrations"
    )
    # Diagnostics are never imported, so each written table is history SQLite never saw.
    history = [f"diagnostics_history:{table}" for table in written]
    # The same live write aimed at the product schema adds product history on top.
    with runtime_role_database(target.dsn, target.schema, target.runtime_role) as product:
        ingest_live_record(product, "live-1")
    after_product_write = verdict()

    assert "execution_records" in written
    assert while_live == ("DENY", ["authority_not_fenced", *history])
    assert after_freeze == ("DENY", history)
    assert after_product_write[0] == "DENY"
    assert sorted(after_product_write[1]) == sorted(
        [*history, *(f"target_history:{table}" for table in written)]
    )


def _root_environ(target: MigrationTarget, root: Path) -> dict[str, str]:
    """What the API process reads at startup: DSN file, authority file and schema."""
    dsn_path = root / "api-postgres.dsn"
    dsn_path.write_text(target.dsn, encoding="utf-8")
    dsn_path.chmod(0o600)
    return {
        API_POSTGRES_DSN_FILE_ENV: str(dsn_path),
        API_POSTGRES_AUTHORITY_FILE_ENV: str(target.authority_path),
        API_POSTGRES_SCHEMA_ENV: target.schema,
    }


def _seed_site(target: MigrationTarget) -> None:
    target.admin.execute(
        sql.SQL("INSERT INTO {} (id, updated_at) VALUES (1, %s)").format(
            sql.Identifier(target.schema, "edge_site")
        ),
        (NOW,),
    )


def _seed_delivery(target: MigrationTarget) -> None:
    # Every delivery table chains to an incident, so the incident is recorded too.
    target.admin.execute(
        sql.SQL(
            "INSERT INTO {} (incident_id, edge_event_id, facility_id, camera_id, event_type, "
            "detected_at, lifecycle_state, provenance_state, provenance_missing_reason, "
            "review_version, revision, created_at, updated_at) VALUES ('incident-1', 'event-1', "
            "'facility-1', 'camera-1', 'fall', %s, 'OPEN', 'MISSING', 'LEGACY', 0, 1, %s, %s)"
        ).format(sql.Identifier(target.schema, "incidents")),
        (NOW, NOW, NOW),
    )
    insert_in_flight_outbox(target.admin, target.schema)


def test_fresh_install_transfer_opens_an_accepting_generation(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    environ = _root_environ(target, tmp_path)
    ledger_before = ledger(target.admin, target.schema)
    with pytest.raises(PostgresRootError, match="not provisioned for this deployment"):
        open_postgres_root(environ, budget=_ROOT_BUDGET)

    successor = _fresh_install(target, tmp_path)
    root = open_postgres_root(environ, budget=_ROOT_BUDGET)
    try:
        opened = root.authority
    finally:
        close_postgres_database(root.database)

    assert successor.generation == 2
    assert _file_token(target.authority_path) == successor
    assert authority_row(target.admin, target.schema) == (2, successor.writer_token, True, True)
    assert not pending_authority_path(target.authority_path).exists()
    assert ledger(target.admin, target.schema) == ledger_before
    assert opened == successor


@pytest.mark.parametrize(
    "artifact",
    ["edge.sqlite3", "edge.sqlite3-wal", "edge.sqlite3-journal", "dangling-symlink"],
    ids=["database", "wal-only", "journal-only", "dangling-symlink"],
)
def test_fresh_install_refuses_when_legacy_source_exists(
    migration_target: MigrationTarget, tmp_path: Path, artifact: str
) -> None:
    target = migration_target
    source = _fresh_source(tmp_path)
    if artifact == "dangling-symlink":
        planted = source
        planted.symlink_to(source.with_name("moved-away.sqlite3"))
    else:
        planted = source.with_name(artifact)
        planted.write_bytes(b"")
    authority_bytes = target.authority_path.read_bytes()
    generation, token = authority_file_token(target.authority_path)

    with pytest.raises(
        MigrationError, match="^legacy SQLite source exists; export and import it instead$"
    ):
        _transfer(target, fresh_install_source=source)

    assert authority_row(target.admin, target.schema) == (generation, token, False, False)
    assert target.authority_path.read_bytes() == authority_bytes
    assert not pending_authority_path(target.authority_path).exists()
    # The counterfactual: the same call once the legacy artifact is gone.
    planted.unlink()
    assert _transfer(target, fresh_install_source=source).generation == 2


@pytest.mark.parametrize(
    ("seed", "occupied"),
    [
        (_seed_site, "edge_site"),
        (_seed_delivery, "event_delivery_attempts, event_outbox, incidents"),
    ],
    ids=["product", "delivery"],
)
def test_fresh_install_refuses_any_product_row(
    migration_target: MigrationTarget,
    tmp_path: Path,
    seed: Callable[[MigrationTarget], None],
    occupied: str,
) -> None:
    target = migration_target
    seed(target)
    authority_bytes = target.authority_path.read_bytes()
    generation, token = authority_file_token(target.authority_path)

    with pytest.raises(MigrationError, match=f"^target tables are not empty: {occupied}$"):
        _fresh_install(target, tmp_path)

    assert authority_row(target.admin, target.schema) == (generation, token, False, False)
    assert target.authority_path.read_bytes() == authority_bytes
    assert not pending_authority_path(target.authority_path).exists()


def test_fresh_install_refuses_after_import(imported: MigrationTarget, tmp_path: Path) -> None:
    target = imported
    authority_bytes = target.authority_path.read_bytes()
    generation, token = authority_file_token(target.authority_path)

    with pytest.raises(MigrationError, match="^target already holds an imported snapshot$"):
        _fresh_install(target, tmp_path)

    assert authority_row(target.admin, target.schema) == (generation, token, False, False)
    assert target.authority_path.read_bytes() == authority_bytes
    assert not pending_authority_path(target.authority_path).exists()
    # The counterfactual: the plain transfer accepts the same imported state.
    assert _transfer(target).generation == 2


def test_fresh_install_refuses_stale_token_and_second_transfer(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    stale_directory = tmp_path / "stale"
    stale_directory.mkdir(mode=0o700)
    stale = stale_directory / "authority.json"
    stale.write_bytes(target.authority_path.read_bytes())
    stale.chmod(0o600)
    successor = _fresh_install(target, tmp_path)

    with pytest.raises(
        AuthorityFenced, match="^cannot transfer a different persistence authority$"
    ):
        transfer(
            target.database,
            stale,
            schema=target.schema,
            fresh_install_source=_fresh_source(tmp_path),
        )
    with pytest.raises(MigrationError, match="^authority was already transferred$"):
        _fresh_install(target, tmp_path)
    with pytest.raises(MigrationError, match="^authority was already transferred$"):
        _transfer(target)

    assert authority_row(target.admin, target.schema) == (2, successor.writer_token, True, True)
    assert _file_token(target.authority_path) == successor
    assert not pending_authority_path(target.authority_path).exists()
    assert not pending_authority_path(stale).exists()


def test_fresh_install_recovers_a_commit_whose_receipt_was_lost(
    migration_target: MigrationTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = migration_target

    with monkeypatch.context() as patch:
        _fault_at_transfer_commit(patch, target.schema, _ack_lost_after_commit)
        successor = _fresh_install(target, tmp_path)

    assert successor.generation == 2
    assert _file_token(target.authority_path) == successor
    assert authority_row(target.admin, target.schema) == (2, successor.writer_token, True, True)
    assert not pending_authority_path(target.authority_path).exists()


def test_fresh_install_lost_receipt_without_commit_keeps_the_old_authority(
    migration_target: MigrationTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = migration_target
    authority_bytes = target.authority_path.read_bytes()
    generation, token = authority_file_token(target.authority_path)

    with monkeypatch.context() as patch:
        _fault_at_transfer_commit(patch, target.schema, _ack_lost_before_commit)
        with pytest.raises(
            MigrationError, match="^transfer did not commit; the authority is unchanged$"
        ):
            _fresh_install(target, tmp_path)

    assert target.authority_path.read_bytes() == authority_bytes
    assert authority_row(target.admin, target.schema) == (generation, token, False, False)
    assert not pending_authority_path(target.authority_path).exists()
    assert _fresh_install(target, tmp_path).generation == 2


def test_import_refuses_after_fresh_install(
    migration_target: MigrationTarget, snapshot: Path, tmp_path: Path
) -> None:
    target = migration_target
    successor = _fresh_install(target, tmp_path)
    counts = table_counts(target.admin, target.schema)

    with pytest.raises(
        MigrationError, match="^target authority must be the fenced provisioning generation$"
    ):
        import_snapshot(target.database, schema=target.schema, snapshot_path=snapshot)

    assert table_counts(target.admin, target.schema) == counts
    assert authority_row(target.admin, target.schema) == (2, successor.writer_token, True, True)


def test_rollback_check_denies_after_fresh_install(
    migration_target: MigrationTarget, legacy: _Legacy, tmp_path: Path
) -> None:
    target = migration_target
    fresh = replace(legacy, source=_fresh_source(tmp_path))

    def verdict() -> tuple[object, object]:
        decision = _rollback(target, fresh)
        return decision["result"], decision["reasons"]

    _fresh_install(target, tmp_path)
    # A fresh install refuses a path that already holds a file, so the fence follows it.
    _fence(target, fresh)
    activated = verdict()
    freeze(target.database, target.authority_path)
    frozen = verdict()

    assert activated == ("DENY", ["authority_not_fenced", "sqlite:source_absent"])
    assert frozen == ("DENY", ["sqlite:source_absent"])
