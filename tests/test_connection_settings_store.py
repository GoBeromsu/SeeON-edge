from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

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
        API_BACKEND_BASE_URL_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _store(tmp_path: Path) -> ConnectionSettingsStore:
    return ConnectionSettingsStore(tmp_path / "connection_settings.sqlite3")


class TestFromEnv:
    def test_default_path_matches_expected_sqlite_location(self) -> None:
        assert DEFAULT_CONNECTION_SETTINGS_PATH == "/var/lib/ml-api/connection-settings.sqlite3"

    def test_from_env_uses_default_path_when_unset(self) -> None:
        store = ConnectionSettingsStore.from_env()
        assert store.path == Path(DEFAULT_CONNECTION_SETTINGS_PATH)

    def test_from_env_honors_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        override = tmp_path / "custom" / "settings.sqlite3"
        monkeypatch.setenv(API_CONNECTION_SETTINGS_PATH_ENV, str(override))
        store = ConnectionSettingsStore.from_env()
        assert store.path == override


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

    def test_only_url_env_seeds_when_no_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_EVENTS_URL_ENV, "https://backend.example/events")
        monkeypatch.setenv(API_BACKEND_CONFIG_URL_ENV, "https://backend.example/ml-config")
        monkeypatch.setenv(API_FACILITY_ID_ENV, "facility-42")
        monkeypatch.setenv(EDGE_FACILITY_TOKEN_ENV, "supersecrettoken")

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://backend.example/events"
        assert settings.config_url == "https://backend.example/ml-config"
        assert settings.facility_id is None
        assert settings.facility_token is None
        assert settings.updated_at is None

    def test_saved_row_wins_over_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_EVENTS_URL_ENV, "https://env.example/events")
        monkeypatch.setenv(EDGE_FACILITY_TOKEN_ENV, "env-token")
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

    def test_unsaved_identity_does_not_fall_back_to_env_after_a_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_FACILITY_ID_ENV, "facility-from-env")
        store = _store(tmp_path)

        _ = store.save({"events_url": "https://saved.example/events"})
        settings = store.load()

        assert settings.events_url == "https://saved.example/events"
        assert settings.facility_id is None


class TestLoadBaseUrlPrecedence:
    def test_base_url_seeds_both_events_and_config_when_no_row_or_specific_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example")

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://backend.example/api/v1/events"
        assert settings.config_url == "https://backend.example/api/v1/ml-config"

    def test_specific_env_var_overrides_base_derivation_per_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example")
        monkeypatch.setenv(API_BACKEND_EVENTS_URL_ENV, "https://override.example/events")

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://override.example/events"
        # config_url has no specific override set, so it still falls back to base.
        assert settings.config_url == "https://backend.example/api/v1/ml-config"

    def test_base_url_trailing_slash_is_normalized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example/")

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://backend.example/api/v1/events"
        assert settings.config_url == "https://backend.example/api/v1/ml-config"

    def test_saved_row_still_wins_over_base_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example")
        store = _store(tmp_path)

        _ = store.save({"events_url": "https://saved.example/events"})
        settings = store.load()

        assert settings.events_url == "https://saved.example/events"
        # config_url was never saved, so it still falls back to the base var.
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
        monkeypatch.setenv(API_FACILITY_ID_ENV, "facility-from-env")
        store = _store(tmp_path)
        _ = store.save({"facility_id": "facility-saved"})

        _ = store.save({"facility_id": None})
        settings = store.load()

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

    def test_explicit_urls_are_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 운영자가 직접 지정한 URL에는 손대지 않는다.
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example")
        monkeypatch.setenv(API_BACKEND_EVENTS_URL_ENV, "https://custom.example/v1/events")

        settings = _store(tmp_path).load()

        assert settings.events_url == "https://custom.example/v1/events"

    def test_empty_base_stays_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "   ")

        settings = _store(tmp_path).load()

        assert settings.events_url is None


def test_compose_edge_puts_settings_db_inside_the_persisted_volume() -> None:
    """연결 설정이 컨테이너 재생성에도 살아남는지 배포 파일에서 고정한다.

    ``DEFAULT_CONNECTION_SETTINGS_PATH``는 ``/var/lib/ml-api/``인데
    ``compose.edge.yaml``의 ml-api 볼륨은 ``/root/.local/state/ml-api``에
    마운트된다. 기본값을 그대로 쓰면 sqlite가 볼륨 밖에 생겨,
    기사님이 대시보드에서 저장한 이벤트 URL/시설 토큰이 컨테이너를
    다시 만들 때 사라진다. 읽기 실패는 조용히 빈 값으로 떨어지므로
    (``store.py``의 ``connection settings store unreadable`` 경고)
    현장에서는 "설정이 왜 초기화됐지"로만 보인다.
    """
    compose_path = Path(__file__).resolve().parents[1] / "compose.edge.yaml"
    compose = compose_path.read_text()

    mount_line = next(
        line for line in compose.splitlines() if "ml-api-state:" in line and ":/" in line
    )
    container_dir = mount_line.split(":", 1)[1].strip()

    setting_line = next(
        line for line in compose.splitlines() if "API_CONNECTION_SETTINGS_PATH:" in line
    )
    settings_path = setting_line.split(":", 1)[1].strip()

    assert settings_path.startswith(f"{container_dir}/"), (
        f"connection settings db {settings_path!r} must live inside the persisted "
        f"volume mount {container_dir!r}"
    )
