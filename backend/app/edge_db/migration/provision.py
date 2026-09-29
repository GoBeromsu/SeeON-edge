"""Versioned, non-destructive PostgreSQL schemas and least-privilege role provisioning."""

from __future__ import annotations

import base64
import hashlib
import hmac
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Final

import psycopg
from psycopg import sql

from backend.app.edge_db.authority import AuthorityToken
from backend.app.edge_db.migration import authority_file
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.mapping import (
    DIAGNOSTICS_TARGET_TABLES,
    EXPECTED_TARGET_TABLES,
    diagnostics_schema_name,
    postgres_table_names,
    require_identifier,
)

SCHEMA_VERSION: Final = 1
SCHEMA_NAME: Final = "seeon_edge_postgres_v1"
MIN_SERVER_VERSION_NUM: Final = 180000
DDL_FILES: Final = ("postgres_product.sql", "postgres_diagnostics.sql", "postgres_delivery.sql")
# The live diagnostics schema carries the same execution_* DDL under its own ledger.
DIAGNOSTICS_SCHEMA_VERSION: Final = 1
DIAGNOSTICS_SCHEMA_NAME: Final = "seeon_edge_diagnostics_postgres_v1"
DIAGNOSTICS_DDL_FILE: Final = "postgres_diagnostics.sql"
_DIAGNOSTICS_LEDGER: Final = b"""
CREATE TABLE schema_migrations (
    version bigint PRIMARY KEY CHECK (version > 0),
    name text NOT NULL UNIQUE,
    checksum text NOT NULL CHECK (length(checksum) = 64),
    applied_at text NOT NULL
);
"""

# Least privilege by table: the runtime never runs DDL and owns nothing.
_SELECT_ONLY: Final = ("schema_migrations",)
_APPEND_ONLY: Final = (
    "audit_events",
    "event_delivery_attempts",
    "event_delivery_results",
    "event_delivery_observations",
)
_NO_DELETE: Final = ("event_outbox", "incidents", "artifacts")
_AUTHORITY: Final = "deployment_authority"


@dataclass(frozen=True, slots=True)
class ProvisionResult:
    schema_created: bool
    diagnostics_schema_created: bool
    role_created: bool
    authority_created: bool
    authority_generation: int


def schema_sources() -> tuple[bytes, ...]:
    package = resources.files("backend.app.edge_db")
    return tuple(package.joinpath(name).read_bytes() for name in DDL_FILES)


def schema_checksum(sources: tuple[bytes, ...] | None = None) -> str:
    digest = hashlib.sha256()
    for source in sources if sources is not None else schema_sources():
        digest.update(source)
    return digest.hexdigest()


def diagnostics_sources() -> tuple[bytes, ...]:
    package = resources.files("backend.app.edge_db")
    return (package.joinpath(DIAGNOSTICS_DDL_FILE).read_bytes(), _DIAGNOSTICS_LEDGER)


def diagnostics_checksum() -> str:
    return schema_checksum(diagnostics_sources())


@dataclass(frozen=True, slots=True)
class _Layout:
    label: str
    version: int
    name: str
    tables: frozenset[str]
    sources: tuple[bytes, ...]
    checksum: str


def _layout(
    label: str, version: int, name: str, tables: frozenset[str], sources: tuple[bytes, ...]
) -> _Layout:
    return _Layout(label, version, name, tables, sources, schema_checksum(sources))


def provision(
    owner_conninfo: str,
    *,
    schema: str,
    runtime_role: str,
    authority_path: Path,
    statement_timeout_ms: int,
    lock_timeout_ms: int,
) -> ProvisionResult:
    """Create or verify both schemas, the runtime role and the fenced authority."""
    diagnostics = diagnostics_schema_name(schema)
    require_identifier(runtime_role, "runtime role")
    product = _layout(
        "schema", SCHEMA_VERSION, SCHEMA_NAME, EXPECTED_TARGET_TABLES, schema_sources()
    )
    live = _layout(
        "diagnostics schema",
        DIAGNOSTICS_SCHEMA_VERSION,
        DIAGNOSTICS_SCHEMA_NAME,
        DIAGNOSTICS_TARGET_TABLES,
        diagnostics_sources(),
    )
    staged: Path | None = None
    token: AuthorityToken | None = None
    try:
        with psycopg.connect(owner_conninfo, autocommit=True, connect_timeout=10) as connection:
            with connection.transaction():
                _prepare(connection, schema, statement_timeout_ms, lock_timeout_ms)
                schema_created = _ensure_schema(connection, schema, product)
                diagnostics_created = _ensure_schema(connection, diagnostics, live)
                role_created = _ensure_role(connection, (schema, diagnostics), runtime_role)
                _grant(connection, schema, runtime_role, EXPECTED_TARGET_TABLES)
                _grant(connection, diagnostics, runtime_role, DIAGNOSTICS_TARGET_TABLES)
                existing = _authority_row(connection, schema)
                generation = existing[0] if existing is not None else 1
                if existing is None:
                    if authority_path.exists() or authority_path.is_symlink():
                        raise MigrationError(
                            "authority file already exists but the schema has no authority"
                        )
                    token = AuthorityToken(generation=1, writer_token=uuid.uuid4())
                    connection.execute(
                        sql.SQL(
                            "INSERT INTO {} (singleton, generation, writer_token, accepting, "
                            "egress_enabled) VALUES (1, %s, %s, false, false)"
                        ).format(sql.Identifier(schema, _AUTHORITY)),
                        (token.generation, token.writer_token),
                    )
                    staged = authority_file.stage_authority_file(authority_path, token)
                else:
                    current = authority_file.read_authority_file(authority_path)
                    if (current.generation, current.writer_token) != existing:
                        raise MigrationError("authority file does not match the database authority")
        if staged is not None:
            authority_file.publish_authority_file(staged, authority_path, replace=False)
            staged = None
    finally:
        if staged is not None:
            authority_file.discard(staged)
    return ProvisionResult(
        schema_created=schema_created,
        diagnostics_schema_created=diagnostics_created,
        role_created=role_created,
        authority_created=token is not None,
        authority_generation=generation,
    )


def set_runtime_password(
    owner_conninfo: str,
    *,
    schema: str,
    runtime_role: str,
    password: str,
    statement_timeout_ms: int,
    lock_timeout_ms: int,
) -> bool:
    """Store the password of a provisioned runtime role as a SCRAM verifier.

    The verifier is rewritten only when the password differs; the result says whether it was.
    """
    diagnostics = diagnostics_schema_name(schema)
    require_identifier(runtime_role, "runtime role")
    if not (password and password.isascii() and password.isprintable()):
        # Without SASLprep, only printable ASCII hashes the same on the client and server.
        raise MigrationError("runtime password must be non-empty printable ASCII")
    with psycopg.connect(owner_conninfo, autocommit=True, connect_timeout=10) as connection:
        with connection.transaction():
            _prepare(connection, schema, statement_timeout_ms, lock_timeout_ms)
            row = _role_row(connection, runtime_role)
            if row is None:
                raise MigrationError("runtime role is not provisioned")
            _check_role_row(connection, (schema, diagnostics), row)
            return _ensure_password(connection, runtime_role, password)


def _prepare(
    connection: psycopg.Connection, schema: str, statement_timeout_ms: int, lock_timeout_ms: int
) -> None:
    connection.execute(
        "SELECT set_config('statement_timeout', %s, true), set_config('lock_timeout', %s, true)",
        (str(statement_timeout_ms), str(lock_timeout_ms)),
    )
    row = connection.execute("SELECT current_setting('server_version_num')::integer").fetchone()
    if row is None or int(row[0]) < MIN_SERVER_VERSION_NUM:
        raise MigrationError("PostgreSQL 18 or newer is required")
    key = int.from_bytes(
        hashlib.sha256(b"seeon-edge-provision:" + schema.encode("ascii")).digest()[:8],
        "big",
        signed=True,
    )
    connection.execute("SELECT pg_advisory_xact_lock(%s)", (key,))


def _ensure_schema(connection: psycopg.Connection, schema: str, layout: _Layout) -> bool:
    row = connection.execute(
        "SELECT n.oid, pg_catalog.pg_get_userbyid(n.nspowner) = current_user "
        "FROM pg_catalog.pg_namespace n WHERE n.nspname = %s",
        (schema,),
    ).fetchone()
    if row is None:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        _apply(connection, schema, layout)
        return True
    namespace, owned = row
    if not owned:
        raise MigrationError(f"{layout.label} exists and is owned by another role")
    objects = connection.execute(
        "SELECT (SELECT count(*) FROM pg_catalog.pg_class WHERE relnamespace = %s) "
        "+ (SELECT count(*) FROM pg_catalog.pg_proc WHERE pronamespace = %s) "
        "+ (SELECT count(*) FROM pg_catalog.pg_type WHERE typnamespace = %s)",
        (namespace, namespace, namespace),
    ).fetchone()
    if objects is not None and int(objects[0]) == 0:
        _apply(connection, schema, layout)
        return True
    if "schema_migrations" not in postgres_table_names(connection, schema):
        raise MigrationError(f"{layout.label} has objects but no migration ledger")
    ledger = connection.execute(
        sql.SQL("SELECT version, name, checksum FROM {} ORDER BY version").format(
            sql.Identifier(schema, "schema_migrations")
        )
    ).fetchall()
    if any(int(version) > layout.version for version, _name, _checksum in ledger):
        raise MigrationError(f"{layout.label} is newer than this tool supports")
    if [tuple(entry) for entry in ledger] != [(layout.version, layout.name, layout.checksum)]:
        raise MigrationError(f"{layout.label} ledger does not match this tool's schema")
    if postgres_table_names(connection, schema) != layout.tables:
        raise MigrationError(f"{layout.label} tables drifted from the provisioned set")
    return False


def _apply(connection: psycopg.Connection, schema: str, layout: _Layout) -> None:
    # Guard functions pin `search_path FROM CURRENT`; the schema must lead it here.
    connection.execute(
        sql.SQL("SET LOCAL search_path TO {}, pg_catalog, pg_temp").format(sql.Identifier(schema))
    )
    for source in layout.sources:
        connection.execute(source.decode("utf-8"), prepare=False)
    applied_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    connection.execute(
        sql.SQL(
            "INSERT INTO {} (version, name, checksum, applied_at) VALUES (%s, %s, %s, %s)"
        ).format(sql.Identifier(schema, "schema_migrations")),
        (layout.version, layout.name, layout.checksum, applied_at),
    )


def _ensure_role(connection: psycopg.Connection, schemas: tuple[str, ...], role: str) -> bool:
    created = _role_row(connection, role) is None
    if created:
        connection.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION "
                "NOBYPASSRLS NOINHERIT"
            ).format(sql.Identifier(role))
        )
    row = _role_row(connection, role)
    if row is None:
        raise MigrationError("runtime role was not created")
    _check_role_row(connection, schemas, row)
    return created


def _ensure_password(connection: psycopg.Connection, role: str, password: str) -> bool:
    # pg_authid is superuser-only; a non-superuser owner fails here with 42501.
    row = connection.execute(
        "SELECT rolpassword FROM pg_catalog.pg_authid WHERE rolname = %s", (role,)
    ).fetchone()
    if row is None:
        raise MigrationError("runtime role is not provisioned")
    if row[0] is not None and _scram_matches(row[0], password):
        return False
    # Hash on the client so the plaintext never reaches the server log.
    verifier = connection.pgconn.encrypt_password(
        password.encode("ascii"), role.encode("ascii"), algorithm=b"scram-sha-256"
    )
    connection.execute(
        sql.SQL("ALTER ROLE {} PASSWORD {}").format(
            sql.Identifier(role), sql.Literal(verifier.decode("ascii"))
        )
    )
    return True


def _scram_matches(verifier: str, password: str) -> bool:
    """Check a `SCRAM-SHA-256$<iterations>:<salt>$<StoredKey>:<ServerKey>` verifier."""
    method, _, secret = verifier.partition("$")
    if method != "SCRAM-SHA-256":
        return False
    try:
        parameters, keys = secret.split("$")
        iterations, salt = parameters.split(":")
        stored_key, server_key = (base64.b64decode(key, validate=True) for key in keys.split(":"))
        salted = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("ascii"),
            base64.b64decode(salt, validate=True),
            int(iterations),
        )
    except ValueError:
        return False
    client_key = hmac.digest(salted, b"Client Key", "sha256")
    return hmac.compare_digest(hashlib.sha256(client_key).digest(), stored_key) and (
        hmac.compare_digest(hmac.digest(salted, b"Server Key", "sha256"), server_key)
    )


def _role_row(connection: psycopg.Connection, role: str) -> tuple[object, ...] | None:
    return connection.execute(
        "SELECT oid, rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls, "
        "rolcanlogin, rolname = current_user FROM pg_catalog.pg_roles WHERE rolname = %s",
        (role,),
    ).fetchone()


def _check_role_row(
    connection: psycopg.Connection, schemas: tuple[str, ...], row: tuple[object, ...]
) -> None:
    oid, superuser, createdb, createrole, replication, bypassrls, can_login, is_owner = row
    if is_owner:
        raise MigrationError("runtime role must differ from the schema owner")
    if superuser or createdb or createrole or replication or bypassrls:
        raise MigrationError("runtime role carries administrative attributes")
    if not can_login:
        raise MigrationError("runtime role cannot log in")
    memberships = connection.execute(
        "SELECT count(*) FROM pg_catalog.pg_auth_members WHERE member = %s", (oid,)
    ).fetchone()
    if memberships is None or int(memberships[0]) != 0:
        raise MigrationError("runtime role must not inherit other roles")
    owned = connection.execute(
        "SELECT (SELECT count(*) FROM pg_catalog.pg_namespace WHERE nspowner = %s "
        "AND nspname = ANY(%s)) + (SELECT count(*) FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "WHERE c.relowner = %s AND n.nspname = ANY(%s))",
        (oid, list(schemas), oid, list(schemas)),
    ).fetchone()
    if owned is None or int(owned[0]) != 0:
        raise MigrationError("runtime role must not own schema objects")


def _grant(connection: psycopg.Connection, schema: str, role: str, tables: frozenset[str]) -> None:
    grantee = sql.Identifier(role)
    namespace = sql.Identifier(schema)
    connection.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(namespace, grantee))
    connection.execute(
        sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}").format(
            namespace, grantee
        )
    )
    for table in sorted(tables):
        if table in _SELECT_ONLY:
            privileges = "SELECT"
        elif table == _AUTHORITY:
            # Row locks (FOR SHARE) need UPDATE on one column; the key cannot change.
            privileges = "SELECT, UPDATE (singleton)"
        elif table in _APPEND_ONLY:
            privileges = "SELECT, INSERT"
        elif table in _NO_DELETE:
            privileges = "SELECT, INSERT, UPDATE"
        else:
            privileges = "SELECT, INSERT, UPDATE, DELETE"
        connection.execute(
            sql.SQL("GRANT {} ON TABLE {} TO {}").format(
                sql.SQL(privileges), sql.Identifier(schema, table), grantee
            )
        )


def _authority_row(connection: psycopg.Connection, schema: str) -> tuple[int, uuid.UUID] | None:
    rows = connection.execute(
        sql.SQL("SELECT generation, writer_token FROM {}").format(
            sql.Identifier(schema, _AUTHORITY)
        )
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise MigrationError("deployment authority is not a singleton")
    generation, writer_token = rows[0]
    return int(generation), writer_token


__all__ = [
    "DDL_FILES",
    "DIAGNOSTICS_DDL_FILE",
    "DIAGNOSTICS_SCHEMA_NAME",
    "DIAGNOSTICS_SCHEMA_VERSION",
    "MIN_SERVER_VERSION_NUM",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "ProvisionResult",
    "diagnostics_checksum",
    "diagnostics_sources",
    "provision",
    "schema_checksum",
    "schema_sources",
]
