"""Fence the provisioning authority and transfer it once to a new writer generation."""

from __future__ import annotations

import uuid
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg import sql

from backend.app.edge_db.authority import AuthorityFenced, AuthorityToken, freeze_authority
from backend.app.edge_db.migration.authority_file import (
    discard,
    publish_authority_file,
    read_authority_file,
    stage_authority_file,
)
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.load import lock_all, require_empty, require_unimported
from backend.app.edge_db.migration.mapping import DELIVERY_TABLES, require_identifier
from backend.app.edge_db.migration.worker_state import worker_stopped
from backend.app.edge_db.postgres import CommitOutcomeUnknown, PostgresDatabase, PostgresError

# Only the fenced provisioning generation may be transferred, and only once.
_PROVISIONED_GENERATION = 1


def pending_authority_path(authority_path: Path) -> Path:
    """The staged next token; it survives a crash between commit and publish."""
    return authority_path.with_name(f".{authority_path.name}.pending")


def freeze(database: PostgresDatabase, authority_path: Path) -> int:
    """Stop accepting and egress for the authority named by the file."""
    return freeze_authority(database, read_authority_file(authority_path))


def transfer(
    database: PostgresDatabase,
    authority_path: Path,
    *,
    schema: str,
    worker_state_dir: Path | None = None,
    fresh_install_source: Path | None = None,
) -> AuthorityToken:
    """Fence, then commit one generation bump with a new token and open the gates.

    With ``fresh_install_source`` the target needs no imported snapshot; it must
    instead be empty and unimported, and the named legacy SQLite path must not exist.
    """
    if fresh_install_source is not None:
        _require_no_legacy_source(fresh_install_source)
    require_identifier(schema, "schema")
    probe: AbstractContextManager[None] = (
        nullcontext() if worker_state_dir is None else worker_stopped(worker_state_dir)
    )
    with probe:
        pending = pending_authority_path(authority_path)
        if pending.exists() or pending.is_symlink():
            resolved = _resolve(database, schema, authority_path, pending)
            if resolved is not None:
                return resolved
        current = read_authority_file(authority_path)
        _require_transferable(database, schema, current)
        freeze_authority(database, current)
        successor = AuthorityToken(generation=current.generation + 1, writer_token=uuid.uuid4())
        staged = stage_authority_file(authority_path, successor)
        try:
            publish_authority_file(staged, pending, replace=False)
        finally:
            discard(staged)
        fresh = fresh_install_source is not None
        try:
            database.transact(
                lambda connection: _advance(connection, schema, current, successor, fresh=fresh)
            )
        except CommitOutcomeUnknown as error:
            # The pending file stays unless the database proves which token it holds.
            if _resolve(database, schema, authority_path, pending) is None:
                raise MigrationError(
                    "transfer did not commit; the authority is unchanged"
                ) from error
            return successor
        except (AuthorityFenced, MigrationError, PostgresError, psycopg.Error):
            discard(pending)
            raise
        publish_authority_file(pending, authority_path, replace=True)
        return successor


def _require_no_legacy_source(source: Path) -> None:
    # A host that still holds legacy data must export and import it, never start empty.
    for path in (source, Path(f"{source}-wal"), Path(f"{source}-journal")):
        if path.exists() or path.is_symlink():
            raise MigrationError("legacy SQLite source exists; export and import it instead")


def _require_transferable(database: PostgresDatabase, schema: str, current: AuthorityToken) -> None:
    # Checked before freezing so a repeated command cannot fence a live deployment.
    row = database.read(lambda connection: _authority_row(connection, schema, lock=False))
    if row[:2] != (current.generation, current.writer_token):
        raise AuthorityFenced("cannot transfer a different persistence authority")
    if current.generation != _PROVISIONED_GENERATION:
        raise MigrationError("authority was already transferred")


def _advance(
    connection: psycopg.Connection,
    schema: str,
    current: AuthorityToken,
    successor: AuthorityToken,
    *,
    fresh: bool,
) -> None:
    if fresh:
        # The same locks and guards as an import, so no row can land before the gates open.
        lock_all(connection, schema)
    row = _authority_row(connection, schema, lock=True)
    if row != (current.generation, current.writer_token, False, False):
        raise AuthorityFenced("persistence authority changed after it was fenced")
    if fresh:
        require_unimported(connection, schema)
        require_empty(connection, schema)
    else:
        _require_imported_without_delivery(connection, schema)
    # Serving reads the edge_site singleton strictly, so activation owns its bootstrap
    # row; an imported row is left exactly as the snapshot had it.
    activated_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    connection.execute(
        sql.SQL(
            "INSERT INTO {} (id, updated_at) VALUES (1, %s) ON CONFLICT (id) DO NOTHING"
        ).format(sql.Identifier(schema, "edge_site")),
        (activated_at,),
    )
    updated = connection.execute(
        sql.SQL(
            "UPDATE {} SET generation = %s, writer_token = %s, accepting = true, "
            "egress_enabled = true WHERE singleton = 1"
        ).format(sql.Identifier(schema, "deployment_authority")),
        (successor.generation, successor.writer_token),
    ).rowcount
    if updated != 1:
        raise MigrationError("deployment authority is not a singleton")


def _require_imported_without_delivery(connection: psycopg.Connection, schema: str) -> None:
    ledger = connection.execute(
        sql.SQL("SELECT count(*) FILTER (WHERE source_db_sha256 IS NOT NULL) FROM {}").format(
            sql.Identifier(schema, "schema_migrations")
        )
    ).fetchone()[0]
    if ledger != 1:
        raise MigrationError("target holds no imported snapshot")
    occupied = [
        table
        for table in DELIVERY_TABLES
        if connection.execute(
            sql.SQL("SELECT EXISTS (SELECT 1 FROM {})").format(sql.Identifier(schema, table))
        ).fetchone()[0]
    ]
    if occupied:
        raise MigrationError(f"delivery tables are not empty: {', '.join(occupied)}")


def _resolve(
    database: PostgresDatabase, schema: str, authority_path: Path, pending: Path
) -> AuthorityToken | None:
    """Publish the pending token if it committed, drop it if not, else refuse."""
    staged = read_authority_file(pending)
    current = read_authority_file(authority_path)
    generation, writer_token, _, _ = database.read(
        lambda connection: _authority_row(connection, schema, lock=False)
    )
    if (generation, writer_token) == (staged.generation, staged.writer_token):
        publish_authority_file(pending, authority_path, replace=True)
        return staged
    if (generation, writer_token) == (current.generation, current.writer_token):
        discard(pending)
        return None
    raise AuthorityFenced("pending authority matches neither the database nor the file")


def _authority_row(
    connection: psycopg.Connection, schema: str, *, lock: bool
) -> tuple[int, uuid.UUID, bool, bool]:
    rows = connection.execute(
        sql.SQL(
            "SELECT generation, writer_token, accepting, egress_enabled FROM {} "
            "WHERE singleton = 1{}"
        ).format(
            sql.Identifier(schema, "deployment_authority"),
            sql.SQL(" FOR UPDATE" if lock else ""),
        )
    ).fetchall()
    if len(rows) != 1:
        raise MigrationError("deployment authority is not a singleton")
    return tuple(rows[0])


__all__ = ["freeze", "pending_authority_path", "transfer"]
