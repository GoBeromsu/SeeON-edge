from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from backend.app.core.config import reject_retired_backend_environment
from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.features.connection import store as connection_store_module
from backend.app.features.connection.store import (
    API_BACKEND_BASE_URL_ENV,
    API_CONNECTION_SETTINGS_PATH_ENV,
    DEFAULT_CONNECTION_SETTINGS_PATH,
    ConnectionSettings,
    ConnectionSettingsStore,
)
from backend.app.lifespan import (
    API_BACKEND_CONFIG_URL_ENV,
    API_BACKEND_EVENTS_URL_ENV,
    EDGE_FACILITY_TOKEN_ENV,
)

_production_from_env = ConnectionSettingsStore.from_env


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        API_CONNECTION_SETTINGS_PATH_ENV,
        API_BACKEND_EVENTS_URL_ENV,
        API_BACKEND_CONFIG_URL_ENV,
        "API_FACILITY_ID",
        "EDGE_FACILITY_TOKEN",
        API_BACKEND_BASE_URL_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _store(tmp_path: Path) -> ConnectionSettingsStore:
    return ConnectionSettingsStore(tmp_path / "connection_settings.sqlite3")


class TestFromEnv:
    def test_default_path_is_the_central_edge_database(self) -> None:
        assert DEFAULT_CONNECTION_SETTINGS_PATH == str(EDGE_DATABASE_PATH)
        assert Path(DEFAULT_CONNECTION_SETTINGS_PATH).name == "edge.sqlite3"

    def test_from_env_uses_the_isolated_central_path(self) -> None:
        store = _production_from_env()

        assert store.path == connection_store_module.EDGE_DATABASE_PATH
        assert store.path.name == "edge.sqlite3"

    def test_retired_path_override_is_rejected_without_recreating_legacy_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        override = tmp_path / "custom" / "connection-settings.sqlite3"
        monkeypatch.setenv(API_CONNECTION_SETTINGS_PATH_ENV, str(override))

        with pytest.raises(ValueError, match=API_CONNECTION_SETTINGS_PATH_ENV):
            reject_retired_backend_environment({API_CONNECTION_SETTINGS_PATH_ENV: str(override)})

        store = _production_from_env()
        assert store.path == connection_store_module.EDGE_DATABASE_PATH
        assert not override.exists()


class TestLoadPrecedence:
    def test_empty_db_and_env_yields_all_none(self, tmp_path: Path) -> None:
        settings = _store(tmp_path).load()
        assert settings == ConnectionSettings(
            events_url=None,
            config_url=None,
            facility_id=None,
            facility_token=None,
            updated_at=None,
        )

    def test_retired_url_and_identity_env_do_not_seed_when_no_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retired = {
            API_BACKEND_EVENTS_URL_ENV: "https://backend.example/events",
            API_BACKEND_CONFIG_URL_ENV: "https://backend.example/ml-config",
            "API_FACILITY_ID": "facility-42",
            "EDGE_FACILITY_TOKEN": "supersecrettoken",
        }
        for name, value in retired.items():
            monkeypatch.setenv(name, value)

        with pytest.raises(ValueError, match="retired edge environment key"):
            reject_retired_backend_environment(retired)

        settings = _store(tmp_path).load()

        assert settings.events_url is None
        assert settings.config_url is None
        assert settings.facility_id is None
        assert settings.facility_token is None
        assert settings.updated_at is None

    def test_legacy_saved_row_remains_readable_despite_retired_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_EVENTS_URL_ENV, "https://env.example/events")
        monkeypatch.setenv("EDGE_FACILITY_TOKEN", "env-token")
        store = _store(tmp_path)

        _ = store.save(
            {"events_url": "https://saved.example/events", "facility_token": "saved-token"}
        )
        settings = store.load()

        assert settings.events_url == "https://saved.example/events"
        assert settings.facility_token == "saved-token"

    def test_saved_row_wins_across_new_store_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "connection_settings.sqlite3"
        _ = ConnectionSettingsStore(path).save({"facility_id": "facility-1"})

        reloaded = ConnectionSettingsStore(path).load()

        assert reloaded.facility_id == "facility-1"

    def test_complete_enrollment_persists_client_ref_across_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "connection_settings.sqlite3"
        ConnectionSettingsStore(path).save(
            {
                "facility_code": "NH-7H2K9M4QXP",
                "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
                "facility_id": "87d79f24-b32f-49a3-b534-19f0af7d9135",
                "facility_token": "stored-token",
                "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
                "enrollment_generation": 3,
            }
        )

        reloaded = ConnectionSettingsStore(path).load()

        assert reloaded.client_installation_ref == "aa83ea3f-6e5f-4f45-a401-fb36c38835b6"
        assert reloaded.enrollment_generation == 3

    def test_unsaved_identity_does_not_fall_back_to_env_after_a_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(EDGE_FACILITY_TOKEN_ENV, "token-from-env")
        store = _store(tmp_path)

        _ = store.save({"events_url": "https://saved.example/events"})
        settings = store.load()

        assert settings.events_url == "https://saved.example/events"
        # Neither facility_id nor facility_token has an env seed/gap-fill:
        # both are DB-only, so saving an unrelated field (events_url) must
        # not cause the env-set token to leak in.
        assert settings.facility_id is None
        assert settings.facility_token is None


class TestLoadBaseUrlPrecedence:
    def test_base_url_seeds_both_events_and_config_when_no_row_or_specific_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example")

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://backend.example/api/v1/events"
        assert settings.config_url == "https://backend.example/api/v1/ml-config"

    def test_retired_specific_env_cannot_override_base_derivation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example")
        monkeypatch.setenv(API_BACKEND_EVENTS_URL_ENV, "https://override.example/events")

        with pytest.raises(ValueError, match=API_BACKEND_EVENTS_URL_ENV):
            reject_retired_backend_environment(
                {API_BACKEND_EVENTS_URL_ENV: "https://override.example/events"}
            )

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://backend.example/api/v1/events"
        assert settings.config_url == "https://backend.example/api/v1/ml-config"

    def test_base_url_trailing_slash_is_normalized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example/")

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://backend.example/api/v1/events"
        assert settings.config_url == "https://backend.example/api/v1/ml-config"

    def test_base_url_wins_over_legacy_saved_urls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example")
        store = _store(tmp_path)

        _ = store.save(
            {
                "events_url": "https://saved.example/events",
                "config_url": "https://saved.example/ml-config",
            }
        )
        settings = store.load()

        assert settings.events_url == "https://backend.example/api/v1/events"
        assert settings.config_url == "https://backend.example/api/v1/ml-config"


class TestSavePartialUpdate:
    def test_partial_save_preserves_other_fields(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _ = store.save(
            {
                "events_url": "https://backend.example/events",
                "config_url": "https://backend.example/ml-config",
                "facility_id": "facility-42",
                "facility_token": "token-abcd",
            }
        )

        _ = store.save({"facility_id": "facility-99"})
        settings = store.load()

        assert settings.facility_id == "facility-99"
        assert settings.events_url == "https://backend.example/events"
        assert settings.config_url == "https://backend.example/ml-config"
        assert settings.facility_token == "token-abcd"

    def test_explicit_none_clears_a_saved_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("API_FACILITY_ID", "facility-from-env")
        store = _store(tmp_path)
        _ = store.save({"facility_id": "facility-saved"})

        _ = store.save({"facility_id": None})
        settings = store.load()

        # Cleared DB field stays None even when env API_FACILITY_ID is set.
        assert settings.facility_id is None

    def test_unknown_field_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown connection setting field"):
            _ = _store(tmp_path).save({"timeout_sec": "5"})

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

        assert first.updated_at is not None
        assert second.updated_at is not None
        assert second.updated_at >= first.updated_at


class TestApiPrefixNormalization:
    """I9: 호스트 base에서 파생한 URL이 NestJS 전역 prefix를 포함해야 한다.

    실제 heartbeat 경로는 ``/api/v1/events/heartbeat``다. ``/api``가 빠지면
    404가 나고, 엣지는 그걸 조용한 실패로 넘겨서 카메라가 계속 online으로
    남는다 -- 프로덕션에서 7대가 이틀 동안 그 상태였다.
    """

    def test_base_url_gains_api_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example")

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://backend.example/api/v1/events"
        assert settings.config_url == "https://backend.example/api/v1/ml-config"

    def test_existing_api_prefix_is_not_duplicated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example/api")

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://backend.example/api/v1/events"
        assert "/api/api/" not in (settings.config_url or "")

    def test_trailing_slash_with_api_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example/api/")

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://backend.example/api/v1/events"

    def test_retired_explicit_url_does_not_bypass_normalization(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example")
        monkeypatch.setenv(API_BACKEND_EVENTS_URL_ENV, "https://custom.example/v1/events")

        with pytest.raises(ValueError, match=API_BACKEND_EVENTS_URL_ENV):
            reject_retired_backend_environment(
                {API_BACKEND_EVENTS_URL_ENV: "https://custom.example/v1/events"}
            )

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://backend.example/api/v1/events"

    def test_empty_base_stays_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "   ")

        settings = _store(tmp_path).load()

        assert settings.events_url is None


def test_compose_edge_persists_only_the_central_connection_database() -> None:
    """The runtime mounts edge.sqlite3 and never recreates the retired store."""
    compose_path = Path(__file__).resolve().parents[1] / "compose.edge.yaml"
    compose = compose_path.read_text()

    # Every service mounting edge-state must agree on the container path, and a
    # mount may carry an access-mode suffix (`:ro`/`:rw`) that is not part of it.
    # Selected by the volume alone, deliberately: filtering on the expected path
    # would hide a service that mounts edge-state somewhere else, which is the
    # disagreement this test exists to catch.
    mount_lines = [
        line
        for line in compose.splitlines()
        if line.strip().startswith("- edge-state:")
    ]
    assert mount_lines, "no service mounts edge-state"

    container_dirs = set()
    for line in mount_lines:
        target = line.split("edge-state:", 1)[1].strip()
        for mode in (":ro", ":rw"):
            target = target.removesuffix(mode)
        container_dirs.add(Path(target))

    assert len(container_dirs) == 1, (
        f"services disagree on where edge-state is mounted: {sorted(container_dirs)}"
    )
    container_dir = container_dirs.pop()

    assert Path(DEFAULT_CONNECTION_SETTINGS_PATH) == container_dir / "edge.sqlite3"
    assert f"{API_CONNECTION_SETTINGS_PATH_ENV}:" not in compose
    assert "connection-settings.sqlite3" not in compose
