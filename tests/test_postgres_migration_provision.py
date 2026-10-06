"""Provisioning the PostgreSQL schema, runtime role and fenced authority on real PG18."""

from __future__ import annotations

import stat
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.provision import ProvisionResult
from backend.app.edge_db.migration.transfer import transfer
from tests_support.postgres_migration import (
    MigrationNames,
    MigrationTarget,
    authority_file_token,
    authority_row,
    ledger,
    provision_target,
    runtime_verifier,
    scram_verifier_accepts,
    set_target_runtime_password,
)

pytest_plugins = ("tests_support.postgres_migration",)


def _authority_path(root: Path) -> Path:
    directory = root / "authority"
    directory.mkdir(mode=0o700)
    return directory / "authority.json"


def test_provision_rerun_changes_nothing(migration_names: MigrationNames, tmp_path: Path) -> None:
    authority_path = _authority_path(tmp_path)

    first = provision_target(migration_names, authority_path)
    authority_bytes = authority_path.read_bytes()
    ledger_before = ledger(migration_names.admin, migration_names.schema)
    second = provision_target(migration_names, authority_path)

    assert first == ProvisionResult(
        schema_created=True,
        diagnostics_schema_created=True,
        role_created=True,
        authority_created=True,
        authority_generation=1,
    )
    assert second == ProvisionResult(
        schema_created=False,
        diagnostics_schema_created=False,
        role_created=False,
        authority_created=False,
        authority_generation=1,
    )
    assert authority_path.read_bytes() == authority_bytes
    assert ledger(migration_names.admin, migration_names.schema) == ledger_before
    assert len(ledger_before) == 1
    generation, token = authority_file_token(authority_path)
    assert authority_row(migration_names.admin, migration_names.schema) == (
        generation,
        token,
        False,
        False,
    )
    assert generation == 1
    assert stat.S_IMODE(authority_path.stat().st_mode) == 0o600


def test_provision_rerun_after_fresh_install_is_noop(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    source_directory = tmp_path / "fresh-state"
    source_directory.mkdir()
    successor = transfer(
        target.database,
        target.authority_path,
        schema=target.schema,
        fresh_install_source=source_directory / "edge.sqlite3",
    )
    authority_bytes = target.authority_path.read_bytes()
    ledger_before = ledger(target.admin, target.schema)
    names = MigrationNames(
        admin=target.admin, dsn=target.dsn, schema=target.schema, runtime_role=target.runtime_role
    )

    rerun = provision_target(names, target.authority_path)

    assert rerun == ProvisionResult(
        schema_created=False,
        diagnostics_schema_created=False,
        role_created=False,
        authority_created=False,
        authority_generation=2,
    )
    assert target.authority_path.read_bytes() == authority_bytes
    assert authority_row(target.admin, target.schema) == (
        2,
        successor.writer_token,
        True,
        True,
    )
    assert ledger(target.admin, target.schema) == ledger_before


def test_provision_refuses_a_newer_schema(migration_target: MigrationTarget) -> None:
    target = migration_target
    target.admin.execute(
        sql.SQL("UPDATE {} SET version = 2").format(
            sql.Identifier(target.schema, "schema_migrations")
        )
    )
    authority_bytes = target.authority_path.read_bytes()
    names = MigrationNames(
        admin=target.admin, dsn=target.dsn, schema=target.schema, runtime_role=target.runtime_role
    )

    with pytest.raises(MigrationError, match="^schema is newer than this tool supports$"):
        provision_target(names, target.authority_path)

    assert target.authority_path.read_bytes() == authority_bytes
    assert [entry[0] for entry in ledger(target.admin, target.schema)] == [2]


@pytest.mark.parametrize(
    ("statement", "table"),
    [
        ("DELETE FROM {}", "audit_events"),
        ("UPDATE {} SET version = version", "schema_migrations"),
        ("DELETE FROM {}", "incidents"),
    ],
    ids=["erase-audit", "rewrite-ledger", "erase-incidents"],
)
def test_runtime_role_is_least_privilege(
    migration_target: MigrationTarget, statement: str, table: str
) -> None:
    target = migration_target
    admin = target.admin

    with admin.transaction():
        admin.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(target.runtime_role)))
        (visible,) = admin.execute(
            sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(target.schema, "incidents"))
        ).fetchone()
        with pytest.raises(psycopg.errors.InsufficientPrivilege), admin.transaction():
            admin.execute(sql.SQL(statement).format(sql.Identifier(target.schema, table)))

    assert visible == 0


def test_set_runtime_password_stores_only_a_scram_verifier_after_provision(
    migration_names: MigrationNames, tmp_path: Path
) -> None:
    password = "synthetic-runtime-password-1"
    provision_target(migration_names, _authority_path(tmp_path))
    before = runtime_verifier(migration_names.admin, migration_names.runtime_role)

    changed = set_target_runtime_password(migration_names, password)

    verifier = runtime_verifier(migration_names.admin, migration_names.runtime_role)
    assert (before, changed) == (None, True)
    assert verifier is not None
    assert scram_verifier_accepts(verifier, password)
    assert not scram_verifier_accepts(verifier, "synthetic-runtime-password-2")


def test_set_runtime_password_sends_the_verifier_not_the_plaintext(
    migration_names: MigrationNames, tmp_path: Path
) -> None:
    """The server hashes a plaintext PASSWORD with the session's password_encryption
    and stores a pre-hashed one as sent, so under md5 a SCRAM verifier proves the
    statement, and any statement log, never carried the plaintext."""
    provision_target(migration_names, _authority_path(tmp_path))
    md5_session = replace(
        migration_names,
        dsn=make_conninfo(migration_names.dsn, options="-c password_encryption=md5"),
    )

    set_target_runtime_password(md5_session, "synthetic-runtime-password-1")

    verifier = runtime_verifier(migration_names.admin, migration_names.runtime_role)
    assert verifier is not None
    assert verifier.startswith("SCRAM-SHA-256$")
    assert scram_verifier_accepts(verifier, "synthetic-runtime-password-1")


def test_set_runtime_password_rerun_keeps_the_verifier(
    migration_names: MigrationNames, tmp_path: Path
) -> None:
    authority_path = _authority_path(tmp_path)
    password = "synthetic-runtime-password-1"
    provision_target(migration_names, authority_path)
    set_target_runtime_password(migration_names, password)
    verifier = runtime_verifier(migration_names.admin, migration_names.runtime_role)

    again = set_target_runtime_password(migration_names, password)
    provision_target(migration_names, authority_path)

    assert again is False
    assert runtime_verifier(migration_names.admin, migration_names.runtime_role) == verifier


def test_set_runtime_password_rotates_a_changed_password(
    migration_names: MigrationNames, tmp_path: Path
) -> None:
    provision_target(migration_names, _authority_path(tmp_path))
    set_target_runtime_password(migration_names, "synthetic-runtime-password-1")

    rotated = set_target_runtime_password(migration_names, "synthetic-runtime-password-2")

    verifier = runtime_verifier(migration_names.admin, migration_names.runtime_role)
    assert rotated is True
    assert verifier is not None
    assert scram_verifier_accepts(verifier, "synthetic-runtime-password-2")
    assert not scram_verifier_accepts(verifier, "synthetic-runtime-password-1")


@pytest.mark.parametrize(
    "password",
    ["", "synthetic-pässword", "synthetic\npassword"],
    ids=["empty", "non-ascii", "control-character"],
)
def test_set_runtime_password_refuses_a_password_outside_printable_ascii(
    migration_names: MigrationNames, tmp_path: Path, password: str
) -> None:
    provision_target(migration_names, _authority_path(tmp_path))

    with pytest.raises(
        MigrationError, match="^runtime password must be non-empty printable ASCII$"
    ):
        set_target_runtime_password(migration_names, password)

    assert runtime_verifier(migration_names.admin, migration_names.runtime_role) is None


def test_set_runtime_password_refuses_a_role_that_provision_did_not_create(
    migration_names: MigrationNames,
) -> None:
    with pytest.raises(MigrationError, match="^runtime role is not provisioned$"):
        set_target_runtime_password(migration_names, "synthetic-runtime-password-1")


def test_set_runtime_password_refuses_to_rewrite_the_schema_owner(
    migration_names: MigrationNames, tmp_path: Path
) -> None:
    provision_target(migration_names, _authority_path(tmp_path))
    (owner,) = migration_names.admin.execute("SELECT current_user").fetchone()
    before = runtime_verifier(migration_names.admin, owner)
    owner_names = replace(migration_names, runtime_role=owner)

    with pytest.raises(MigrationError, match="^runtime role must differ from the schema owner$"):
        set_target_runtime_password(owner_names, "synthetic-runtime-password-1")

    assert runtime_verifier(migration_names.admin, owner) == before
