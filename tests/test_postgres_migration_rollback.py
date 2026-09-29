"""The rollback check names one reason per cause; unfence trusts its ALLOW only up to the fence.

Each expected reason follows from one thing the test did, to the fenced file or to
PostgreSQL. The fenced file's oracle is its own bytes, read without SQLite.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pytest
from psycopg import sql

from backend.app.edge_db.migration.cli import main
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.load import import_snapshot
from backend.app.edge_db.migration.mapping import DIAGNOSTICS_TABLES, diagnostics_schema_name
from backend.app.edge_db.migration.report import write_report
from backend.app.edge_db.migration.rollback import rollback_check
from backend.app.edge_db.migration.snapshot import export_snapshot, sidecar_paths
from backend.app.edge_db.migration.sqlite_fence import (
    fence_sqlite,
    preserved_path,
    read_fence_receipt,
)
from backend.app.edge_db.migration.unfence import RESTORED, unfence_sqlite
from backend.app.features.diagnostics.postgres_database import DIAGNOSTICS_POOL_BUDGET
from tests_support.postgres_migration import (
    MigrationTarget,
    add_incident,
    authority_file_token,
    ingest_live_record,
    open_source_writer,
    runtime_role_database,
    source_and_destination,
    table_counts,
)

pytest_plugins = ("tests_support.postgres_migration",)

# The provisioned target is generation 1, so the fence stamps 1_000_000 + 1.
SENTINEL = 1_000_001
PROG = "python -m backend.app.edge_db.migration rollback-check"


@dataclass(frozen=True)
class _Imported:
    source: Path
    snapshot: Path
    receipt: Path


def _import(target: MigrationTarget, root: Path) -> _Imported:
    """Import a snapshot of the seeded old database; nothing is fenced yet."""
    source, destination = source_and_destination(root)
    snapshot = export_snapshot(source, destination).path
    import_snapshot(target.database, schema=target.schema, snapshot_path=snapshot)
    receipts = root / "receipts"
    receipts.mkdir(mode=0o700)
    return _Imported(source, snapshot, receipts / "fence.json")


def _fence(target: MigrationTarget, source: Path, snapshot: Path | None, receipt: Path) -> None:
    generation, _ = authority_file_token(target.authority_path)
    fence_sqlite(source, snapshot=snapshot, generation=generation, receipt=receipt)


def _fenced(target: MigrationTarget, root: Path) -> _Imported:
    imported = _import(target, root)
    _fence(target, imported.source, imported.snapshot, imported.receipt)
    return imported


def _check(target: MigrationTarget, imported: _Imported) -> dict[str, object]:
    return rollback_check(
        target.database,
        schema=target.schema,
        snapshot_path=imported.snapshot,
        source=imported.source,
        fence_receipt=imported.receipt,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_diagnostics(target: MigrationTarget) -> dict[str, int]:
    """One live record through the runtime's own ingest; return the rows it left."""
    diagnostics = diagnostics_schema_name(target.schema)
    with runtime_role_database(
        target.dsn, diagnostics, target.runtime_role, DIAGNOSTICS_POOL_BUDGET
    ) as database:
        ingest_live_record(database, "live-1")
    written = table_counts(target.admin, diagnostics)
    del written["schema_migrations"]
    return written


def _unchanged(source: Path) -> None:
    del source


def _flip_last_byte(source: Path) -> None:
    data = bytearray(source.read_bytes())
    data[-1] ^= 0xFF
    source.write_bytes(bytes(data))


def _write_wal(source: Path) -> None:
    sidecar_paths(source)[0].write_bytes(b"\x00" * 32)


def _write_shm(source: Path) -> None:
    sidecar_paths(source)[1].write_bytes(b"\x00" * 32)


def _create_journal(source: Path) -> None:
    sidecar_paths(source)[2].touch()


def _remove(source: Path) -> None:
    source.unlink()


def test_rollback_check_allows_an_untouched_fence_over_fresh_diagnostics(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    imported = _fenced(target, tmp_path)
    fenced = _sha(imported.source)

    decision = _check(target, imported)

    assert (decision["result"], decision["reasons"]) == ("ALLOW", [])
    assert decision["sqlite"] == {
        "generation": 1,
        "source_present": True,
        "fenced_sha256": fenced,
        "live_sha256": fenced,
        "user_version": SENTINEL,
        "wal_bytes": 0,
        "shm_bytes": 0,
        "journal": False,
    }
    assert decision["diagnostics"] == {
        "schema": diagnostics_schema_name(target.schema),
        "mode": "live",
        "reconciled": False,
        "ledger": "PASS",
        "tables": "PASS",
        "rows": dict.fromkeys(DIAGNOSTICS_TABLES, 0),
        "result": "PASS",
    }


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param(_flip_last_byte, "sqlite:live_changed", id="bytes"),
        pytest.param(_write_wal, "sqlite:wal_content", id="wal"),
        pytest.param(_write_shm, "sqlite:shm_content", id="shm"),
        pytest.param(_create_journal, "sqlite:journal", id="journal"),
        pytest.param(_remove, "sqlite:source_missing", id="missing"),
    ],
)
def test_rollback_check_denies_a_fenced_file_that_changed(
    migration_target: MigrationTarget,
    tmp_path: Path,
    change: Callable[[Path], None],
    reason: str,
) -> None:
    target = migration_target
    imported = _fenced(target, tmp_path)
    change(imported.source)

    decision = _check(target, imported)

    assert (decision["result"], decision["reasons"]) == ("DENY", [reason])


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_unchanged, id="untouched"),
        pytest.param(_create_journal, id="journal"),
    ],
)
def test_rollback_check_denies_a_fence_that_found_no_source_without_inspecting_it(
    migration_target: MigrationTarget, tmp_path: Path, change: Callable[[Path], None]
) -> None:
    target = migration_target
    imported = _import(target, tmp_path)
    fresh = tmp_path / "fresh-state"
    fresh.mkdir(mode=0o700)
    absent = dataclasses.replace(imported, source=fresh / "edge.sqlite3")
    _fence(target, absent.source, None, absent.receipt)
    stamped = _sha(absent.source)
    change(absent.source)

    decision = _check(target, absent)

    assert (decision["result"], decision["reasons"]) == ("DENY", ["sqlite:source_absent"])
    assert decision["sqlite"] == {
        "generation": 1,
        "source_present": False,
        "fenced_sha256": stamped,
    }


def test_rollback_check_denies_a_fence_taken_against_another_snapshot(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    imported = _import(target, tmp_path)
    with closing(open_source_writer(imported.source)) as writer:
        add_incident(writer, 900)
    later = export_snapshot(imported.source, imported.snapshot.with_name("later.sqlite3")).path
    _fence(target, imported.source, later, imported.receipt)

    decision = _check(target, imported)

    assert (decision["result"], decision["reasons"]) == (
        "DENY",
        ["sqlite:receipt_snapshot_mismatch"],
    )


def test_rollback_check_denies_diagnostics_history_the_snapshot_never_held(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    imported = _fenced(target, tmp_path)
    written = _live_diagnostics(target)

    decision = _check(target, imported)

    assert written["execution_records"] == 1
    assert decision["diagnostics"]["rows"] == written
    assert (decision["result"], decision["reasons"]) == (
        "DENY",
        [f"diagnostics_history:{table}" for table in sorted(written) if written[table]],
    )


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param("DROP SCHEMA {schema} CASCADE", id="dropped"),
        pytest.param("UPDATE {ledger} SET checksum = repeat('b', 64)", id="foreign-ledger"),
    ],
)
def test_rollback_check_denies_diagnostics_it_cannot_trust_by_schema_alone(
    migration_target: MigrationTarget, tmp_path: Path, statement: str
) -> None:
    target = migration_target
    imported = _fenced(target, tmp_path)
    diagnostics = diagnostics_schema_name(target.schema)
    _live_diagnostics(target)
    target.admin.execute(
        sql.SQL(statement).format(
            schema=sql.Identifier(diagnostics),
            ledger=sql.Identifier(diagnostics, "schema_migrations"),
        )
    )

    decision = _check(target, imported)

    assert (decision["result"], decision["reasons"]) == ("DENY", ["diagnostics:schema"])


@pytest.mark.parametrize(
    ("omitted", "message"),
    [
        pytest.param(("source",), "1 required keyword-only argument: 'source'", id="source"),
        pytest.param(
            ("fence_receipt",),
            "1 required keyword-only argument: 'fence_receipt'",
            id="fence-receipt",
        ),
        pytest.param(
            ("source", "fence_receipt"),
            "2 required keyword-only arguments: 'source' and 'fence_receipt'",
            id="both",
        ),
    ],
)
def test_rollback_check_requires_the_source_and_its_fence_receipt(
    migration_target: MigrationTarget, tmp_path: Path, omitted: tuple[str, ...], message: str
) -> None:
    target = migration_target
    given = {"source": tmp_path / "edge.sqlite3", "fence_receipt": tmp_path / "fence.json"}
    for name in omitted:
        del given[name]

    with pytest.raises(TypeError, match=rf"^rollback_check\(\) missing {message}$"):
        rollback_check(
            target.database, schema=target.schema, snapshot_path=tmp_path / "snapshot", **given
        )


@pytest.mark.parametrize(
    ("given", "missing"),
    [
        pytest.param(["--fence-receipt", "fence.json"], "--source", id="source"),
        pytest.param(["--source", "edge.sqlite3"], "--fence-receipt", id="fence-receipt"),
        pytest.param([], "--source, --fence-receipt", id="both"),
    ],
)
def test_cli_rollback_check_requires_the_source_and_its_fence_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], given: list[str], missing: str
) -> None:
    argv = [
        "rollback-check",
        "--owner-dsn-file",
        str(tmp_path / "owner.dsn"),
        "--snapshot",
        str(tmp_path / "snapshot"),
        *given,
    ]

    with pytest.raises(SystemExit) as exited:
        main(argv)

    captured = capsys.readouterr()
    assert (exited.value.code, captured.out) == (2, "")
    assert captured.err.splitlines()[-1] == (
        f"{PROG}: error: the following arguments are required: {missing}"
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param(_flip_last_byte, "sqlite:live_changed", id="bytes"),
        pytest.param(_write_wal, "sqlite:wal_content", id="wal"),
    ],
)
def test_unfence_rechecks_a_source_changed_after_the_rollback_allowed(
    migration_target: MigrationTarget,
    tmp_path: Path,
    change: Callable[[Path], None],
    reason: str,
) -> None:
    target = migration_target
    imported = _fenced(target, tmp_path)
    report = tmp_path / "rollback.json"
    decision = _check(target, imported)
    write_report(report, decision)
    change(imported.source)
    changed = imported.source.read_bytes()

    with pytest.raises(MigrationError, match=f"^fenced source changed: {reason}$"):
        unfence_sqlite(imported.source, receipt=imported.receipt, rollback_report=report)

    assert decision["result"] == "ALLOW"
    assert imported.source.read_bytes() == changed
    assert _sha(preserved_path(imported.receipt)) == (
        read_fence_receipt(imported.receipt).pre_fence_sha256
    )


def test_unfence_restores_the_source_a_real_rollback_check_allowed(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    imported = _import(target, tmp_path)
    before = _sha(imported.source)
    _fence(target, imported.source, imported.snapshot, imported.receipt)
    report = tmp_path / "rollback.json"
    write_report(report, _check(target, imported))

    restored = unfence_sqlite(imported.source, receipt=imported.receipt, rollback_report=report)

    assert (restored["result"], restored["restored_sha256"]) == (RESTORED, before)
    assert _sha(imported.source) == before
