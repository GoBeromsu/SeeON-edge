from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.compatibility import MigrationRequiredError
from backend.app.features.connection.store import (
    ConnectionSettingsStore,
    InvalidConnectionSettingError,
)
from tests_support.compact_authority_db import prepare_compact_database

REQUIRED_FIELDS = (
    "facility_code",
    "client_installation_ref",
    "facility_id",
    "facility_token",
    "edge_installation_id",
    "enrollment_generation",
)


def _enrollment() -> dict[str, str | int]:
    return {
        "facility_code": "NH-7H2K9M4QXP",
        "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
        "facility_id": "facility-1",
        "facility_token": "token-1234",
        "edge_installation_id": "edge-1",
        "enrollment_generation": 3,
    }


def _store(tmp_path: Path) -> ConnectionSettingsStore:
    path = prepare_compact_database(tmp_path / "edge.sqlite3")
    return ConnectionSettingsStore(path)


def test_constructor_requires_external_schema18_migration(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite3"

    with pytest.raises(MigrationRequiredError):
        ConnectionSettingsStore(path).load()

    assert not path.exists()


def test_fresh_schema18_connection_authority_is_empty(tmp_path: Path) -> None:
    settings = _store(tmp_path).load()

    assert settings.facility_id is None
    assert settings.facility_token is None
    assert settings.events_url is None
    assert settings.config_url is None


def test_runtime_enrollment_survives_restart_and_is_masked(tmp_path: Path) -> None:
    store = _store(tmp_path)
    saved = store.save(_enrollment())

    restarted = ConnectionSettingsStore(store.path)

    assert restarted.load() == saved
    assert restarted.masked() == {
        "events_url": None,
        "config_url": None,
        "facility_id": "facility-1",
        "facility_token_masked": "****1234",
        "facility_token_set": True,
        "updated_at": saved.updated_at,
    }


@pytest.mark.parametrize("field_name", REQUIRED_FIELDS)
def test_complete_enrollment_rejects_clearing_each_required_field_atomically(
    tmp_path: Path, field_name: str
) -> None:
    store = _store(tmp_path)
    before = store.save(_enrollment())

    with pytest.raises(InvalidConnectionSettingError, match="saved atomically"):
        store.save({field_name: None})

    assert store.load() == before


@pytest.mark.parametrize(
    "updates",
    [
        {"facility_code": ""},
        {"client_installation_ref": " "},
        {"enrollment_generation": 0},
        {"enrollment_generation": True},
        {"events_url": "https://stored.example/events"},
        {"config_url": "https://stored.example/config"},
    ],
)
def test_invalid_or_retired_enrollment_fields_are_rejected_atomically(
    tmp_path: Path, updates: dict[str, str | int | bool]
) -> None:
    store = _store(tmp_path)
    before = store.save(_enrollment())

    with pytest.raises(InvalidConnectionSettingError):
        store.save(updates)

    assert store.load() == before


def test_schema18_connection_write_changes_only_enrollment_columns(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO edge_site(id,clip_export_enabled,runtime_settings_version,updated_at) "
            "VALUES (1,1,7,'2026-08-24T00:00:00Z')"
        )

    store.save(_enrollment())

    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT clip_export_enabled,runtime_settings_version,facility_id FROM edge_site "
            "WHERE id=1"
        ).fetchone()
        tables = {
            str(item[0])
            for item in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert row == (1, 7, "facility-1")
    assert "connection_settings" not in tables
    assert "connection_store_migrations" not in tables
