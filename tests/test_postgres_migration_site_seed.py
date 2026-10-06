"""Activation owns the edge_site singleton; the rollback predicate tells its seed from a write."""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from psycopg import sql

from backend.app.edge_db.authority import AuthorityToken
from backend.app.edge_db.migration.load import import_snapshot
from backend.app.edge_db.migration.rollback import rollback_check
from backend.app.edge_db.migration.snapshot import export_snapshot
from backend.app.edge_db.migration.sqlite_fence import fence_sqlite
from backend.app.edge_db.migration.transfer import freeze, transfer
from backend.app.features.connection.repository import (
    ConnectionSettingsNotInitialized,
    ConnectionValue,
)
from backend.app.features.connection.store import (
    API_BACKEND_BASE_URL_ENV,
    ConnectionSettings,
    ConnectionSettingsStore,
)
from tests_support.postgres_migration import (
    NOW,
    MigrationTarget,
    authority_file_token,
    runtime_role_database,
    source_and_destination,
)
from tests_support.sqlite_source import open_source_writer

pytest_plugins = ("tests_support.postgres_migration",)

_NO_SITE = "DELETE FROM edge_site"
_CUSTOM_SITE = {
    "facility_code": "FAC-SEED-01",
    "client_installation_ref": "client-seed-01",
    "facility_id": "facility-seed-01",
    "facility_token": "token-seed-01",
    "edge_installation_id": "edge-seed-01",
    "enrollment_generation": 3,
    "enrollment_created_at": NOW,
    "enrollment_updated_at": NOW,
    "clip_store_subdir": "capture/local",
}
_ENROLLMENT: dict[str, ConnectionValue] = {
    "facility_code": "FAC-SAVE-01",
    "client_installation_ref": "client-save-01",
    "facility_id": "facility-save-01",
    "facility_token": "token-save-01",
    "edge_installation_id": "edge-save-01",
    "enrollment_generation": 1,
}


def _snapshot(
    root: Path, statement: str | None = None, params: tuple[object, ...] = ()
) -> tuple[Path, Path]:
    """Export the seeded old database after one optional raw statement on it; return both."""
    root.mkdir()
    source, destination = source_and_destination(root)
    if statement is not None:
        with closing(open_source_writer(source)) as writer:
            writer.execute(statement, params)
    return source, export_snapshot(source, destination).path


def _custom_site_snapshot(root: Path) -> Path:
    assignments = ", ".join(f"{column} = ?" for column in _CUSTOM_SITE)
    statement = f"UPDATE edge_site SET {assignments} WHERE id = 1"
    return _snapshot(root, statement, tuple(_CUSTOM_SITE.values()))[1]


def _activate(target: MigrationTarget, root: Path, route: str) -> AuthorityToken:
    """Transfer the fenced provisioning authority by the named activation route."""
    if route == "fresh":
        directory = root / "fresh-state"
        directory.mkdir()
        return transfer(
            target.database,
            target.authority_path,
            schema=target.schema,
            fresh_install_source=directory / "edge.sqlite3",
        )
    return transfer(target.database, target.authority_path, schema=target.schema)


def _import(target: MigrationTarget, snapshot: Path) -> None:
    import_snapshot(target.database, schema=target.schema, snapshot_path=snapshot)


def _prepare(target: MigrationTarget, root: Path, route: str) -> tuple[Path, Path]:
    """Import what the route needs; return the old database path and the snapshot to restore.

    A fresh install has no old database, so its path names the file activation checks.
    """
    if route == "fresh":
        return root / "fresh-state" / "edge.sqlite3", _snapshot(root / "reference")[1]
    if route == "no-site":
        source, snapshot = _snapshot(root / "no-site", _NO_SITE)
    else:
        source, snapshot = _snapshot(root / "site")
    _import(target, snapshot)
    return source, snapshot


def _fence(target: MigrationTarget, source: Path, snapshot: Path, root: Path) -> Path:
    """Fence the old database at the current generation and return its receipt.

    An absent old database is fenced without a snapshot, as on a fresh install.
    """
    receipts = root / "receipts"
    receipts.mkdir(mode=0o700)
    receipt = receipts / "fence.json"
    generation, _ = authority_file_token(target.authority_path)
    present = snapshot if source.exists() else None
    fence_sqlite(source, snapshot=present, generation=generation, receipt=receipt)
    return receipt


def _load(target: MigrationTarget, token: AuthorityToken) -> ConnectionSettings:
    with runtime_role_database(target.dsn, target.schema, target.runtime_role) as database:
        return ConnectionSettingsStore(database, token).load()


def _save(
    target: MigrationTarget, token: AuthorityToken, updates: dict[str, ConnectionValue]
) -> None:
    with runtime_role_database(target.dsn, target.schema, target.runtime_role) as database:
        ConnectionSettingsStore(database, token).save(updates)


def _site_rows(target: MigrationTarget) -> list[dict[str, object]]:
    query = sql.SQL("SELECT to_jsonb(site) FROM {} AS site ORDER BY site.id").format(
        sql.Identifier(target.schema, "edge_site")
    )
    return [row[0] for row in target.admin.execute(query).fetchall()]


@pytest.mark.parametrize("route", ["fresh", "no-site"])
def test_activation_leaves_a_readable_edge_site_singleton(
    migration_target: MigrationTarget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    target = migration_target
    monkeypatch.delenv(API_BACKEND_BASE_URL_ENV, raising=False)
    _prepare(target, tmp_path, route)
    generation, writer_token = authority_file_token(target.authority_path)
    provisioning = AuthorityToken(generation=generation, writer_token=writer_token)
    with pytest.raises(ConnectionSettingsNotInitialized):
        _load(target, provisioning)

    before = datetime.now(UTC)
    token = _activate(target, tmp_path, route)
    after = datetime.now(UTC)
    settings = _load(target, token)

    assert settings == ConnectionSettings(
        events_url=None,
        config_url=None,
        facility_id=None,
        facility_token=None,
        updated_at=settings.updated_at,
    )
    assert settings.updated_at is not None and settings.updated_at.endswith("Z")
    activated_at = datetime.fromisoformat(settings.updated_at)
    assert before.replace(microsecond=before.microsecond // 1000 * 1000) <= activated_at <= after


def test_activation_keeps_an_imported_customized_edge_site(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    _import(target, _custom_site_snapshot(tmp_path / "custom"))
    imported = _site_rows(target)

    _activate(target, tmp_path, "imported")

    assert _site_rows(target) == imported
    assert len(imported) == 1
    assert {name: imported[0][name] for name in (*_CUSTOM_SITE, "updated_at")} == {
        **_CUSTOM_SITE,
        "updated_at": NOW,
    }


# A fresh install has no SQLite rollback: unfence refuses a receipt that records no source.
@pytest.mark.parametrize(
    ("route", "updates", "expected"),
    [
        ("fresh", None, ("DENY", ["sqlite:source_absent"])),
        ("fresh", _ENROLLMENT, ("DENY", ["unimported_rows:edge_site", "sqlite:source_absent"])),
        ("no-site", None, ("ALLOW", [])),
        ("no-site", _ENROLLMENT, ("DENY", ["target_history:edge_site"])),
        ("imported", {}, ("DENY", ["target_history:edge_site"])),
    ],
    ids=[
        "fresh-seed-only",
        "fresh-enrolled",
        "no-site-seed-only",
        "no-site-enrolled",
        "imported-site-resaved",
    ],
)
def test_rollback_check_tells_the_activation_seed_from_a_later_write(
    migration_target: MigrationTarget,
    tmp_path: Path,
    route: str,
    updates: dict[str, ConnectionValue] | None,
    expected: tuple[str, list[str]],
) -> None:
    target = migration_target
    source, snapshot = _prepare(target, tmp_path, route)
    token = _activate(target, tmp_path, route)
    if updates is not None:
        _save(target, token, updates)
    freeze(target.database, target.authority_path)
    # After activation: a fresh transfer refuses a path that already holds a file.
    receipt = _fence(target, source, snapshot, tmp_path)

    decision = rollback_check(
        target.database,
        schema=target.schema,
        snapshot_path=snapshot,
        source=source,
        fence_receipt=receipt,
    )

    assert len(_site_rows(target)) == 1
    assert (decision["result"], decision["reasons"]) == expected
