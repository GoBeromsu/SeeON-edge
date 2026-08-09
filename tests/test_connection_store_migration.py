from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from backend.app.features.connection.store import ConnectionSettingsStore

if TYPE_CHECKING:
    from sqlite3 import Connection


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


class TestPreMigrationCharacterization:
    def test_current_url_row_is_readable(self, tmp_path: Path) -> None:
        database = tmp_path / "connection-settings.sqlite3"
        _legacy_database(database)

        settings = ConnectionSettingsStore(database).load()

        assert settings.events_url == "https://saved.example/events"
        assert settings.config_url == "https://saved.example/ml-config"

    def test_current_public_view_masks_token(self, tmp_path: Path) -> None:
        database = tmp_path / "connection-settings.sqlite3"
        _legacy_database(database)

        public = ConnectionSettingsStore(database).masked()

        assert public["facility_token_masked"] == "****1234"
        assert "synthetic-token-1234" not in json.dumps(public)

    def test_current_write_uses_owner_only_file_mode(self, tmp_path: Path) -> None:
        store = ConnectionSettingsStore(tmp_path / "connection-settings.sqlite3")

        _ = store.save({"events_url": "https://saved.example/events"})

        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600

    def test_current_corrupt_database_falls_back_to_empty(self, tmp_path: Path) -> None:
        database = tmp_path / "connection-settings.sqlite3"
        _ = database.write_bytes(b"not sqlite")

        settings = ConnectionSettingsStore(database).load()

        assert settings.events_url is None
        assert settings.facility_id is None

    def test_current_legacy_identity_row_is_readable(self, tmp_path: Path) -> None:
        database = tmp_path / "connection-settings.sqlite3"
        _legacy_database(database)

        settings = ConnectionSettingsStore(database).load()

        assert settings.facility_id == "legacy-facility"
        assert settings.facility_token == "synthetic-token-1234"


def test_additive_migration_preserves_legacy_reader_columns(tmp_path: Path) -> None:
    database = tmp_path / "connection-settings.sqlite3"
    _legacy_database(database)

    _ = ConnectionSettingsStore(database).load()

    with sqlite3.connect(database) as connection:
        schema_rows: list[tuple[int, str, str, int, str | None, int]] = connection.execute(
            "PRAGMA table_info(connection_settings)"
        ).fetchall()
        columns = {row[1] for row in schema_rows}
        legacy_row: tuple[str, str, str, str, str] | None = connection.execute(
            "SELECT events_url, config_url, facility_id, facility_token, updated_at "
            + "FROM connection_settings WHERE id = 1"
        ).fetchone()
    assert {
        "facility_code",
        "edge_installation_id",
        "enrollment_generation",
        "enrollment_created_at",
        "enrollment_updated_at",
    } <= columns
    assert legacy_row == (
        "https://saved.example/events",
        "https://saved.example/ml-config",
        "legacy-facility",
        "synthetic-token-1234",
        "2026-08-10T00:00:00.000Z",
    )


def test_runtime_enrollment_survives_restart_and_is_masked(tmp_path: Path) -> None:
    database = tmp_path / "connection-settings.sqlite3"
    store = ConnectionSettingsStore(database)

    saved = store.save(
        {
            "facility_code": "NH-7H2K9M4QXP",
            "facility_token": "synthetic-enrollment-token-9876",
            "facility_id": "canonical-facility-id",
            "edge_installation_id": "edge-installation-id",
            "enrollment_generation": 3,
        }
    )
    reopened = ConnectionSettingsStore(database)

    assert saved.facility_code == "NH-7H2K9M4QXP"
    assert reopened.load().edge_installation_id == "edge-installation-id"
    assert reopened.load().enrollment_generation == 3
    assert reopened.load().enrollment_created_at is not None
    assert reopened.load().enrollment_updated_at is not None
    assert reopened.masked()["facility_token_masked"] == "****9876"
    assert "synthetic-enrollment-token-9876" not in json.dumps(reopened.masked())
    assert "synthetic-enrollment-token-9876" not in repr(reopened.load())


@pytest.mark.parametrize(
    "updates",
    [
        {"facility_code": ""},
        {"edge_installation_id": ""},
        {"enrollment_generation": 0},
        {"enrollment_generation": -1},
    ],
)
def test_invalid_enrollment_fields_are_rejected_atomically(
    tmp_path: Path, updates: dict[str, str | int | None]
) -> None:
    store = ConnectionSettingsStore(tmp_path / "connection-settings.sqlite3")
    _ = store.save({"facility_id": "legacy-facility"})

    with pytest.raises(ValueError):
        _ = store.save(updates)

    assert store.load().facility_id == "legacy-facility"


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
    assert len(backup.name.rsplit(".", maxsplit=1)[-1]) == 64
    with backup.open("rb") as backup_file:
        digest = hashlib.file_digest(backup_file, "sha256").hexdigest()
    with sqlite3.connect(database) as connection:
        metadata: tuple[str, str, int] | None = connection.execute(
            "SELECT backup_filename, backup_sha256, backup_size_bytes "
            + "FROM connection_store_migrations WHERE version = 1"
        ).fetchone()
    assert metadata == (backup.name, digest, backup.stat().st_size)


def test_readonly_migration_failure_keeps_legacy_row_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "connection-settings.sqlite3"
    _legacy_database(database)

    def deny_backup(
        _store: ConnectionSettingsStore, _source: Connection
    ) -> None:
        raise PermissionError

    monkeypatch.setattr(ConnectionSettingsStore, "_create_pre_v1_backup_unlocked", deny_backup)

    settings = ConnectionSettingsStore(database).load()

    assert settings.events_url == "https://saved.example/events"
    assert settings.facility_id == "legacy-facility"
    with sqlite3.connect(database) as connection:
        schema_rows: list[tuple[int, str, str, int, str | None, int]] = connection.execute(
            "PRAGMA table_info(connection_settings)"
        ).fetchall()
    assert "facility_code" not in {row[1] for row in schema_rows}


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
        row: tuple[str] | None = connection.execute(
            "SELECT events_url FROM connection_settings WHERE id = 1"
        ).fetchone()
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
        unrelated: tuple[str] | None = connection.execute(
            "SELECT body FROM unrelated_outbox"
        ).fetchone()
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

    monkeypatch.setattr(ConnectionSettingsStore, "_copy_database_unlocked", interrupt)

    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            _ = store.create_pre_v1_backup()

    assert list(store.rollback_directory.glob("*.tmp")) == []
    assert ConnectionSettingsStore(store.path).load().events_url == (
        "https://original.example/events"
    )
