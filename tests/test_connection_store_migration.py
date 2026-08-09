from __future__ import annotations

import json
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
        legacy_row = cast(
            tuple[str, str, str, str, str] | None,
            connection.execute(
                "SELECT events_url, config_url, facility_id, facility_token, updated_at "
                + "FROM connection_settings WHERE id = 1"
            ).fetchone(),
        )
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
    "required_field",
    [
        "facility_code",
        "facility_token",
        "facility_id",
        "edge_installation_id",
        "enrollment_generation",
    ],
)
def test_complete_enrollment_rejects_clearing_each_required_field(
    tmp_path: Path, required_field: str
) -> None:
    store = ConnectionSettingsStore(tmp_path / "connection-settings.sqlite3")
    before = store.save(
        {
            "facility_code": "NH-7H2K9M4QXP",
            "facility_token": "synthetic-enrollment-token-9876",
            "facility_id": "canonical-facility-id",
            "edge_installation_id": "edge-installation-id",
            "enrollment_generation": 3,
        }
    )

    with pytest.raises(ValueError):
        _ = store.save({required_field: None})

    assert ConnectionSettingsStore(store.path).load() == before


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


def test_readonly_migration_failure_keeps_legacy_row_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "connection-settings.sqlite3"
    _legacy_database(database)

    def deny_backup(_database: ConnectionStoreDatabase, _source: Connection) -> None:
        raise PermissionError

    monkeypatch.setattr(ConnectionStoreDatabase, "_create_backup", deny_backup)

    settings = ConnectionSettingsStore(database).load()

    assert settings.events_url == "https://saved.example/events"
    assert settings.facility_id == "legacy-facility"
    with sqlite3.connect(database) as connection:
        schema_rows = cast(
            list[tuple[int, str, str, int, str | None, int]],
            connection.execute("PRAGMA table_info(connection_settings)").fetchall(),
        )
    assert "facility_code" not in {row[1] for row in schema_rows}
