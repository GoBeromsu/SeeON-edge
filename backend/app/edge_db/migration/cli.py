"""Operator commands for the one-time SQLite to PostgreSQL cutover."""

from __future__ import annotations

import argparse
import os
import sqlite3
import stat
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, TypeVar

import psycopg
from psycopg.conninfo import conninfo_to_dict

from backend.app.edge_db.authority import AuthorityFenced
from backend.app.edge_db.migration.compatibility import EdgeDatabaseError
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.load import import_snapshot
from backend.app.edge_db.migration.mapping import diagnostics_schema_name
from backend.app.edge_db.migration.provision import provision, set_runtime_password
from backend.app.edge_db.migration.reconcile import PASS, reconcile, write_report
from backend.app.edge_db.migration.rollback import ALLOW, rollback_check
from backend.app.edge_db.migration.snapshot import export_snapshot
from backend.app.edge_db.migration.transfer import freeze, transfer
from backend.app.edge_db.migration.unfence import unfence_sqlite
from backend.app.edge_db.migration.worker_state import queue_digest
from backend.app.edge_db.paths import EDGE_DATABASE_PATH
from backend.app.edge_db.postgres import PoolBudget, PostgresDatabase, PostgresError

DEFAULT_SCHEMA: Final = "seeon_edge"
DEFAULT_RUNTIME_ROLE: Final = "seeon_edge_runtime"
_PREFIX: Final = "EDGE_PG_MIGRATION"
# One command at a time; the second slot covers a read beside an open scope.
_POOL_CONNECTIONS: Final = 2
_POOL_WAITING: Final = 1
_CONNECT_TIMEOUT_SEC: Final = 10.0
_CLOSE_TIMEOUT_SEC: Final = 10.0
_FAILURES: Final = (
    AuthorityFenced,
    EdgeDatabaseError,
    MigrationError,
    OSError,
    PostgresError,
    ValueError,
    psycopg.Error,
    sqlite3.Error,
)

T = TypeVar("T")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.app.edge_db.migration",
        description="Provision PostgreSQL and move a fenced SQLite edge database into it once",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def target(name: str, help_text: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--owner-dsn-file",
            type=Path,
            required=True,
            help="owner-only file holding the schema owner's libpq DSN",
        )
        command.add_argument("--schema", default=DEFAULT_SCHEMA)
        command.add_argument("--statement-timeout-ms", type=int, default=600_000)
        command.add_argument("--lock-timeout-ms", type=int, default=10_000)
        return command

    command = target("provision", "create the versioned schema, runtime role and authority file")
    command.add_argument("--runtime-role", default=DEFAULT_RUNTIME_ROLE)
    command.add_argument("--authority-file", type=Path, required=True)
    command.add_argument(
        "--runtime-dsn-file",
        type=Path,
        help="owner-only file holding the runtime role's libpq DSN; sets its password",
    )

    command = commands.add_parser("export", help="copy the fenced SQLite database to a snapshot")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--snapshot", type=Path, required=True)

    command = target("import", "load a snapshot into the empty schema in one transaction")
    command.add_argument("--snapshot", type=Path, required=True)

    command = target("reconcile", "compare the snapshot, source and target; write a report")
    command.add_argument("--snapshot", type=Path, required=True)
    command.add_argument("--report", type=Path, required=True)
    command.add_argument("--source", type=Path)
    command.add_argument("--worker-state-dir", type=Path)
    command.add_argument("--expect-delivery-queue-sha256")
    command.add_argument("--after-transfer", action="store_true")
    command.add_argument("--fence-receipt", type=Path)

    command = target("freeze", "stop accepting and egress for the authority in the file")
    command.add_argument("--authority-file", type=Path, required=True)

    command = target("transfer", "fence and move the authority to its next generation once")
    command.add_argument("--authority-file", type=Path, required=True)
    command.add_argument("--worker-state-dir", type=Path)
    command.add_argument(
        "--fresh-install",
        action="store_true",
        help="open an empty, never-imported target; refused while legacy SQLite data exists",
    )
    command.add_argument(
        "--source",
        type=Path,
        help=f"legacy SQLite path that must be absent (default {EDGE_DATABASE_PATH})",
    )

    command = target("rollback-check", "decide whether restoring the snapshot loses nothing")
    command.add_argument("--snapshot", type=Path, required=True)
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--fence-receipt", type=Path, required=True)
    command.add_argument("--report", type=Path)

    command = commands.add_parser("queue-digest", help="digest the stopped worker's queue")
    command.add_argument("--worker-state-dir", type=Path, required=True)

    command = commands.add_parser(
        "unfence-sqlite", help="restore the pre-fence SQLite bytes after an ALLOW rollback check"
    )
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--fence-receipt", type=Path, required=True)
    command.add_argument("--rollback-report", type=Path, required=True)
    command.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "transfer" and args.source is not None and not args.fresh_install:
        parser.error("transfer --source requires --fresh-install")
    if args.command == "reconcile" and args.fence_receipt is not None and args.source is None:
        parser.error("reconcile --fence-receipt requires --source")
    label = f"{_PREFIX}_{args.command.upper().replace('-', '_')}"
    try:
        return _run(args, label)
    except _FAILURES as error:
        print(f"{label}_FAILED: {_describe(error)}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace, label: str) -> int:
    if args.command == "export":
        snapshot = export_snapshot(args.source, args.snapshot)
        print(
            f"{label}_OK snapshot={snapshot.path} sha256={snapshot.sha256} "
            f"source_schema={snapshot.schema_version}"
        )
        return 0
    if args.command == "queue-digest":
        digest = queue_digest(args.worker_state_dir)
        print(
            f"{label}_OK queued={digest.queued} temporary={digest.temporary} "
            f"dead_lettered={digest.dead_lettered} sha256={digest.sha256}"
        )
        return 0
    if args.command == "unfence-sqlite":
        restored = unfence_sqlite(
            args.source, receipt=args.fence_receipt, rollback_report=args.rollback_report
        )
        if args.report is not None:
            write_report(args.report, restored)
        print(
            f"{label}_OK result={restored['result']} generation={restored['generation']} "
            f"sha256={restored['restored_sha256']}"
        )
        return 0
    if args.command == "transfer" and args.fresh_install:
        _require_no_canonical_source()
    conninfo = _read_dsn(args.owner_dsn_file, "owner DSN file")
    if args.command == "provision":
        runtime_password = _runtime_password(args.runtime_dsn_file, args.runtime_role)
        result = provision(
            conninfo,
            schema=args.schema,
            runtime_role=args.runtime_role,
            authority_path=args.authority_file,
            statement_timeout_ms=args.statement_timeout_ms,
            lock_timeout_ms=args.lock_timeout_ms,
        )
        # Its own transaction after provision commits: the role it names now exists.
        password_set = runtime_password is not None and set_runtime_password(
            conninfo,
            schema=args.schema,
            runtime_role=args.runtime_role,
            password=runtime_password,
            statement_timeout_ms=args.statement_timeout_ms,
            lock_timeout_ms=args.lock_timeout_ms,
        )
        print(
            f"{label}_OK schema={args.schema} "
            f"schema_created={_flag(result.schema_created)} "
            f"diagnostics_schema={diagnostics_schema_name(args.schema)} "
            f"diagnostics_schema_created={_flag(result.diagnostics_schema_created)} "
            f"role_created={_flag(result.role_created)} "
            f"authority_created={_flag(result.authority_created)} "
            f"generation={result.authority_generation} "
            f"runtime_password_set={_flag(password_set)}"
        )
        return 0
    return _with_database(args, conninfo, lambda database: _run_on(database, args, label))


def _run_on(database: PostgresDatabase, args: argparse.Namespace, label: str) -> int:
    if args.command == "import":
        imported = import_snapshot(database, schema=args.schema, snapshot_path=args.snapshot)
        print(
            f"{label}_OK tables={len(imported.rows)} rows={sum(imported.rows.values())} "
            f"source_db_sha256={imported.source_db_sha256} "
            f"reconciliation_sha256={imported.reconciliation_sha256}"
        )
        return 0
    if args.command == "reconcile":
        report = reconcile(
            database,
            schema=args.schema,
            snapshot_path=args.snapshot,
            source_path=args.source,
            worker_state_dir=args.worker_state_dir,
            expected_queue_sha256=args.expect_delivery_queue_sha256,
            after_transfer=args.after_transfer,
            fence_receipt=args.fence_receipt,
        )
        write_report(args.report, report)
        if report["result"] != PASS:
            print(
                f"{label}_FAILED: result={report['result']} report={args.report}", file=sys.stderr
            )
            return 1
        print(
            f"{label}_OK result={report['result']} report={args.report} "
            f"diagnostics_schema={diagnostics_schema_name(args.schema)}"
        )
        return 0
    if args.command == "freeze":
        generation = freeze(database, args.authority_file)
        print(f"{label}_OK generation={generation} accepting=false egress_enabled=false")
        return 0
    if args.command == "transfer":
        token = transfer(
            database,
            args.authority_file,
            schema=args.schema,
            worker_state_dir=args.worker_state_dir,
            fresh_install_source=_fresh_install_source(args),
        )
        print(f"{label}_OK generation={token.generation} accepting=true egress_enabled=true")
        return 0
    decision = rollback_check(
        database,
        schema=args.schema,
        snapshot_path=args.snapshot,
        source=args.source,
        fence_receipt=args.fence_receipt,
    )
    if args.report is not None:
        write_report(args.report, decision)
    reasons = ",".join(decision["reasons"]) or "none"
    if decision["result"] != ALLOW:
        print(f"{label}_FAILED: result={decision['result']} reasons={reasons}", file=sys.stderr)
        return 1
    print(f"{label}_OK result={decision['result']} reasons={reasons}")
    return 0


def _fresh_install_source(args: argparse.Namespace) -> Path | None:
    if not args.fresh_install:
        return None
    return EDGE_DATABASE_PATH if args.source is None else args.source


def _require_no_canonical_source() -> None:
    # A --source elsewhere must not skip legacy data left where the product kept it;
    # transfer() checks the path it is given, so this covers the canonical one.
    canonical = EDGE_DATABASE_PATH
    for path in (canonical, Path(f"{canonical}-wal"), Path(f"{canonical}-journal")):
        if path.exists() or path.is_symlink():
            raise MigrationError(
                f"legacy SQLite source exists at {canonical}; export and import it instead"
            )


def _with_database(
    args: argparse.Namespace, conninfo: str, action: Callable[[PostgresDatabase], T]
) -> T:
    budget = PoolBudget(
        max_connections=_POOL_CONNECTIONS,
        max_waiting=_POOL_WAITING,
        acquire_timeout_sec=_CONNECT_TIMEOUT_SEC,
        statement_timeout_ms=args.statement_timeout_ms,
        lock_timeout_ms=args.lock_timeout_ms,
        startup_timeout_sec=_CONNECT_TIMEOUT_SEC,
    )
    database = PostgresDatabase(conninfo, args.schema, budget)
    database.start()
    try:
        return action(database)
    finally:
        database.close(timeout_sec=_CLOSE_TIMEOUT_SEC)


def _read_dsn(path: Path, label: str) -> str:
    """Read the DSN from an owner-only regular file; it never appears in argv.

    Errors name the file by its path and never quote what it holds.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as error:
        raise MigrationError(f"{label} {path} cannot be opened ({error.strerror})") from None
    with os.fdopen(descriptor, "rb") as handle:
        status = os.fstat(handle.fileno())
        if not stat.S_ISREG(status.st_mode) or status.st_mode & 0o077:
            raise MigrationError(
                f"{label} {path} must be a regular file readable only by its owner"
            )
        body = handle.read()
    try:
        conninfo = body.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise MigrationError(f"{label} {path} is not UTF-8") from None
    if not conninfo or "\n" in conninfo:
        raise MigrationError(f"{label} {path} must hold exactly one DSN")
    return conninfo


def _runtime_password(path: Path | None, runtime_role: str) -> str | None:
    """Take the password from the DSN the runtime itself will use, so the two cannot drift."""
    if path is None:
        return None
    label = "runtime DSN file"
    try:
        fields = conninfo_to_dict(_read_dsn(path, label))
    except psycopg.Error:
        raise MigrationError(f"{label} {path} is not a libpq DSN") from None
    if fields.get("user") != runtime_role:
        raise MigrationError(f"{label} {path} user must be the runtime role")
    password = fields.get("password") or ""
    # set_runtime_password refuses these too, but only after provision has committed.
    if not (password and password.isascii() and password.isprintable()):
        raise MigrationError(f"{label} {path} must carry a printable ASCII password")
    return password


def _describe(error: BaseException) -> str:
    # Server messages and details can quote row values; name the class and state only.
    if isinstance(error, psycopg.Error):
        return f"{type(error).__name__} (sqlstate {error.sqlstate or 'none'})"
    return str(error) or type(error).__name__


def _flag(value: bool) -> str:
    return str(value).lower()


__all__ = ["DEFAULT_RUNTIME_ROLE", "DEFAULT_SCHEMA", "main"]
