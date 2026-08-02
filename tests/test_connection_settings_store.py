from __future__ import annotations

import json
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from backend.app.features.connection.store import (
    API_CONNECTION_SETTINGS_PATH_ENV,
    DEFAULT_CONNECTION_SETTINGS_PATH,
    ConnectionSettings,
    ConnectionSettingsStore,
    mask_facility_token,
)
from backend.app.lifespan import (
    API_BACKEND_CONFIG_URL_ENV,
    API_BACKEND_EVENTS_URL_ENV,
    API_FACILITY_ID_ENV,
    EDGE_FACILITY_TOKEN_ENV,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        API_CONNECTION_SETTINGS_PATH_ENV,
        API_BACKEND_EVENTS_URL_ENV,
        API_BACKEND_CONFIG_URL_ENV,
        API_FACILITY_ID_ENV,
        EDGE_FACILITY_TOKEN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _store(tmp_path: Path) -> ConnectionSettingsStore:
    return ConnectionSettingsStore(tmp_path / "connection_settings.json")


class TestFromEnv:
    def test_default_path_matches_camera_store_directory(self) -> None:
        assert DEFAULT_CONNECTION_SETTINGS_PATH == "/var/lib/ml-api/connection_settings.json"

    def test_from_env_uses_default_path_when_unset(self) -> None:
        store = ConnectionSettingsStore.from_env()
        assert store.path == Path(DEFAULT_CONNECTION_SETTINGS_PATH)

    def test_from_env_honors_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        override = tmp_path / "custom" / "settings.json"
        monkeypatch.setenv(API_CONNECTION_SETTINGS_PATH_ENV, str(override))
        store = ConnectionSettingsStore.from_env()
        assert store.path == override


class TestLoadPrecedence:
    def test_empty_file_and_env_yields_all_none(self, tmp_path: Path) -> None:
        settings = _store(tmp_path).load()
        assert settings == ConnectionSettings(
            events_url=None,
            config_url=None,
            facility_id=None,
            facility_token=None,
            updated_at=None,
        )

    def test_env_seeds_when_no_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(API_BACKEND_EVENTS_URL_ENV, "https://backend.example/events")
        monkeypatch.setenv(API_BACKEND_CONFIG_URL_ENV, "https://backend.example/ml-config")
        monkeypatch.setenv(API_FACILITY_ID_ENV, "facility-42")
        monkeypatch.setenv(EDGE_FACILITY_TOKEN_ENV, "supersecrettoken")

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://backend.example/events"
        assert settings.config_url == "https://backend.example/ml-config"
        assert settings.facility_id == "facility-42"
        assert settings.facility_token == "supersecrettoken"
        assert settings.updated_at is None

    def test_saved_file_wins_over_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_EVENTS_URL_ENV, "https://env.example/events")
        monkeypatch.setenv(EDGE_FACILITY_TOKEN_ENV, "env-token")
        store = _store(tmp_path)

        store.save({"events_url": "https://saved.example/events", "facility_token": "saved-token"})
        settings = store.load()

        assert settings.events_url == "https://saved.example/events"
        assert settings.facility_token == "saved-token"

    def test_saved_file_wins_across_new_store_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "connection_settings.json"
        ConnectionSettingsStore(path).save({"facility_id": "facility-1"})

        reloaded = ConnectionSettingsStore(path).load()

        assert reloaded.facility_id == "facility-1"

    def test_unsaved_field_still_falls_back_to_env_after_a_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_FACILITY_ID_ENV, "facility-from-env")
        store = _store(tmp_path)

        store.save({"events_url": "https://saved.example/events"})
        settings = store.load()

        assert settings.events_url == "https://saved.example/events"
        assert settings.facility_id == "facility-from-env"


class TestSavePartialUpdate:
    def test_partial_save_preserves_other_fields(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save(
            {
                "events_url": "https://backend.example/events",
                "config_url": "https://backend.example/ml-config",
                "facility_id": "facility-42",
                "facility_token": "token-abcd",
            }
        )

        store.save({"facility_id": "facility-99"})
        settings = store.load()

        assert settings.facility_id == "facility-99"
        assert settings.events_url == "https://backend.example/events"
        assert settings.config_url == "https://backend.example/ml-config"
        assert settings.facility_token == "token-abcd"

    def test_explicit_none_clears_a_saved_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_FACILITY_ID_ENV, "facility-from-env")
        store = _store(tmp_path)
        store.save({"facility_id": "facility-saved"})

        store.save({"facility_id": None})
        settings = store.load()

        assert settings.facility_id == "facility-from-env"

    def test_unknown_field_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown connection setting field"):
            _store(tmp_path).save({"timeout_sec": "5"})

    def test_updated_at_stamped_on_save(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.load().updated_at is None

        settings = store.save({"facility_id": "facility-1"})

        assert settings.updated_at is not None
        assert settings.updated_at.endswith("Z")

    def test_updated_at_advances_on_subsequent_save(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        first = store.save({"facility_id": "facility-1"})
        second = store.save({"facility_id": "facility-2"})

        assert second.updated_at is not None
        assert second.updated_at >= first.updated_at


class TestAtomicWriteAndCorruption:
    def test_write_leaves_final_file_0600(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save({"facility_id": "facility-1"})

        mode = stat.S_IMODE(store.path.stat().st_mode)
        assert mode == 0o600

    def test_write_leaves_no_leftover_tmp_file(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save({"facility_id": "facility-1"})

        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_corrupt_json_file_treated_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "connection_settings.json"
        path.write_text("{not valid json", encoding="utf-8")

        settings = ConnectionSettingsStore(path).load()

        assert settings == ConnectionSettings(
            events_url=None,
            config_url=None,
            facility_id=None,
            facility_token=None,
            updated_at=None,
        )

    def test_non_object_json_file_treated_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "connection_settings.json"
        path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

        settings = ConnectionSettingsStore(path).load()

        assert settings.events_url is None

    def test_missing_file_does_not_crash_and_save_still_works(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "connection_settings.json"
        store = ConnectionSettingsStore(path)

        assert store.load().facility_id is None
        store.save({"facility_id": "facility-1"})

        assert store.load().facility_id == "facility-1"


class TestMasking:
    def test_mask_facility_token_short_token_fully_masked(self) -> None:
        assert mask_facility_token("abcd") == "****"
        assert mask_facility_token("ab") == "****"

    def test_mask_facility_token_long_token_shows_last_four(self) -> None:
        assert mask_facility_token("supersecrettoken1234") == "****1234"

    def test_mask_facility_token_none_or_empty(self) -> None:
        assert mask_facility_token(None) is None
        assert mask_facility_token("") is None

    def test_masked_dict_never_contains_raw_token(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save({"facility_token": "supersecrettoken1234"})

        public = store.masked()

        assert public["facility_token_masked"] == "****1234"
        assert public["facility_token_set"] is True
        assert "supersecrettoken1234" not in json.dumps(public)
        assert "facility_token" not in public

    def test_masked_dict_when_token_unset(self, tmp_path: Path) -> None:
        public = _store(tmp_path).masked()

        assert public["facility_token_masked"] is None
        assert public["facility_token_set"] is False

    def test_masked_dict_reflects_other_fields(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save(
            {
                "events_url": "https://backend.example/events",
                "config_url": "https://backend.example/ml-config",
                "facility_id": "facility-42",
            }
        )

        public = store.masked()

        assert public["events_url"] == "https://backend.example/events"
        assert public["config_url"] == "https://backend.example/ml-config"
        assert public["facility_id"] == "facility-42"
        assert public["updated_at"] is not None
