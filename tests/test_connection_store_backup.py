from __future__ import annotations

import hashlib
import sqlite3
import stat
from pathlib import Path
from sqlite3 import Connection
from typing import cast

import pytest

from backend.app.features.connection.sqlite_store import ConnectionStoreDatabase
from backend.app.features.connection.store import ConnectionSettingsStore


def _legacy_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        _ = connection.execute(
            """CREATE TABLE connection_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            events_url TEXT, config_url TEXT, facility_id TEXT,
            facility_token TEXT, updated_at TEXT) STRICT"""
        )
        _ = connection.execute(
            """INSERT INTO connection_settings
            (id, events_url, config_url, facility_id, facility_token, updated_at)
            VALUES (1, ?, ?, ?, ?, ?)""",
            (
                "https://saved.example/events",
                "https://saved.example/ml-config",
                "legacy-facility",
                "synthetic-token-1234",
                "2026-08-10T00:00:00.000Z",
            ),
        )


def test_legacy_migration_creates_integrity_checked_owner_only_backup(tmp_path: Path) -> None:
    database = tmp_path / "connection-settings.sqlite3"
    _legacy_database(database)
    store = ConnectionSettingsStore(database)

    _ = store.load()

    backups = list(store.rollback_directory.glob("connection-settings.sqlite3.pre-v1.*"))
    assert len(backups) == 1
    backup = backups[0]
    assert stat.S_IMODE(store.rollback_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert store.integrity_check(backup) == "ok"
    with backup.open("rb") as backup_file:
        digest = hashlib.file_digest(backup_file, "sha256").hexdigest()
    with sqlite3.connect(database) as connection:
        metadata = cast(
            tuple[str, str, int] | None,
            connection.execute(
                "SELECT backup_filename, backup_sha256, backup_size_bytes "
                + "FROM connection_store_migrations WHERE version = 1"
            ).fetchone(),
        )
    assert len(backup.name.rsplit(".", maxsplit=1)[-1]) == 64
    assert metadata == (backup.name, digest, backup.stat().st_size)


def test_online_backup_includes_committed_wal_state(tmp_path: Path) -> None:
    database = tmp_path / "connection-settings.sqlite3"
    store = ConnectionSettingsStore(database)
    _ = store.save({"events_url": "https://before.example/events"})
    with sqlite3.connect(database) as writer:
        _ = writer.execute("PRAGMA journal_mode = WAL")
        _ = writer.execute(
            "UPDATE connection_settings SET events_url = ? WHERE id = 1",
            ("https://wal.example/events",),
        )

    backup = store.create_pre_v1_backup()

    with sqlite3.connect(backup.path) as connection:
        row = cast(
            tuple[str] | None,
            connection.execute(
                "SELECT events_url FROM connection_settings WHERE id = 1"
            ).fetchone(),
        )
    assert row == ("https://wal.example/events",)


def test_restore_discards_only_post_snapshot_connection_state(tmp_path: Path) -> None:
    database = tmp_path / "connection-settings.sqlite3"
    store = ConnectionSettingsStore(database)
    _ = store.save({"events_url": "https://before.example/events"})
    backup = store.create_pre_v1_backup()
    _ = store.save(
        {
            "events_url": "https://after.example/events",
            "facility_code": "NH-7H2K9M4QXP",
            "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            "facility_token": "synthetic-token-after-snapshot",
            "facility_id": "canonical-facility-id",
            "edge_installation_id": "edge-installation-id",
            "enrollment_generation": 2,
        }
    )
    with sqlite3.connect(database) as connection:
        _ = connection.execute("CREATE TABLE unrelated_outbox (id INTEGER PRIMARY KEY, body TEXT)")
        _ = connection.execute("INSERT INTO unrelated_outbox (body) VALUES ('preserve-me')")

    store.restore_pre_v1_backup(backup.path)

    restored = ConnectionSettingsStore(database).load()
    with sqlite3.connect(database) as connection:
        unrelated = cast(
            tuple[str] | None,
            connection.execute("SELECT body FROM unrelated_outbox").fetchone(),
        )
    assert restored.events_url == "https://before.example/events"
    assert restored.facility_code is None
    assert restored.facility_token is None
    assert restored.edge_installation_id is None
    assert restored.enrollment_generation is None
    assert unrelated == ("preserve-me",)


def test_corrupt_backup_restore_leaves_original_database_intact(tmp_path: Path) -> None:
    store = ConnectionSettingsStore(tmp_path / "connection-settings.sqlite3")
    _ = store.save({"events_url": "https://original.example/events"})
    corrupt_backup = tmp_path / "connection-settings.sqlite3.pre-v1.corrupt"
    _ = corrupt_backup.write_bytes(b"not sqlite")

    with pytest.raises(sqlite3.DatabaseError):
        store.restore_pre_v1_backup(corrupt_backup)

    assert ConnectionSettingsStore(store.path).load().events_url == (
        "https://original.example/events"
    )


def test_interrupted_backup_cleans_partial_file_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConnectionSettingsStore(tmp_path / "connection-settings.sqlite3")
    _ = store.save({"events_url": "https://original.example/events"})

    def interrupt(_source: Connection, destination: Connection) -> None:
        _ = destination.execute("CREATE TABLE partial (id INTEGER)")
        raise KeyboardInterrupt

    monkeypatch.setattr(ConnectionStoreDatabase, "_copy_database", interrupt)

    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            _ = store.create_pre_v1_backup()

    assert list(store.rollback_directory.glob("*.tmp")) == []
    assert ConnectionSettingsStore(store.path).load().events_url == (
        "https://original.example/events"
    )
