"""Hub base / enrollment URL transport policy (HTTPS production contract)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from backend.app.features.connection.enrollment import (
    EnrollmentCredentials,
    EnrollmentVerificationFailure,
    enrollment_endpoint,
    verify_enrollment,
)
from backend.app.features.connection.hub_url import (
    API_BACKEND_ALLOW_INSECURE_HTTP_ENV,
    hub_url_transport_allowed,
)
from backend.app.features.connection.store import (
    API_BACKEND_BASE_URL_ENV,
    ConnectionSettingsStore,
    InvalidConnectionSettingError,
)
from tests_support.compact_authority_db import prepare_compact_database

_CREDS = EnrollmentCredentials(
    facility_code="NH-7H2K9M4QXP",
    client_installation_ref="aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
    facility_token="facility-bearer-secret",
)


@pytest.fixture(autouse=True)
def production_hub_transport_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """HTTPS policy tests must not inherit the suite's insecure-HTTP opt-in."""

    monkeypatch.delenv(API_BACKEND_ALLOW_INSECURE_HTTP_ENV, raising=False)
    prepare_compact_database(tmp_path / "c.sqlite3")


class TestHubUrlPolicy:
    def test_https_public_origins_are_allowed(self) -> None:
        assert hub_url_transport_allowed("https://hub.example.com")
        assert hub_url_transport_allowed("https://49.247.204.81")

    def test_loopback_http_is_permitted_without_opt_in(self) -> None:
        assert hub_url_transport_allowed("http://127.0.0.1:8000/api/v1/events")
        assert hub_url_transport_allowed("http://localhost/api/v1/events")
        assert hub_url_transport_allowed("http://[::1]:9/api/v1/events")

    def test_cleartext_public_ip_and_hostname_are_rejected(self) -> None:
        assert not hub_url_transport_allowed("http://49.247.204.81")
        assert not hub_url_transport_allowed("http://hub.example.com/api/v1/events")
        assert not hub_url_transport_allowed("http://backend.example/api/v1/events")

    def test_explicit_dev_contract_permits_non_loopback_http(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_ALLOW_INSECURE_HTTP_ENV, "1")
        assert hub_url_transport_allowed("http://backend.example/api/v1/events")


class TestStoreRejectsInsecureHubUrls:
    def test_base_url_http_public_does_not_seed_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "http://hub.example.com")
        settings = ConnectionSettingsStore(tmp_path / "c.sqlite3").load()
        assert settings.events_url is None
        assert settings.config_url is None

    def test_base_url_http_public_ip_does_not_seed_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "http://49.247.204.81")
        settings = ConnectionSettingsStore(tmp_path / "c.sqlite3").load()
        assert settings.events_url is None
        assert settings.config_url is None

    def test_base_url_https_seeds_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://hub.example.com")
        settings = ConnectionSettingsStore(tmp_path / "c.sqlite3").load()
        assert settings.events_url == "https://hub.example.com/api/v1/events"
        assert settings.config_url == "https://hub.example.com/api/v1/ml-config"

    def test_save_rejects_retired_editable_events_url(self, tmp_path: Path) -> None:
        store = ConnectionSettingsStore(tmp_path / "c.sqlite3")
        with pytest.raises(InvalidConnectionSettingError) as exc:
            store.save({"events_url": "http://49.247.204.81/api/v1/events"})
        assert exc.value.field_name == "events_url"
        assert "unknown connection setting" in exc.value.reason


class TestEnrollmentNeverSendsBearerToRejectedOrigin:
    def test_enrollment_endpoint_none_for_cleartext_public(self) -> None:
        assert enrollment_endpoint("http://49.247.204.81/api/v1/events") is None
        assert enrollment_endpoint("http://hub.example.com/api/v1/events") is None
        assert enrollment_endpoint("https://hub.example.com/api/v1/events") == (
            "https://hub.example.com/api/v1/edge/enrollments/verify"
        )

    def test_verify_enrollment_does_not_call_urlopen_for_http_public(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[object] = []

        def _boom(request: object, timeout: float = 0) -> object:  # noqa: ARG001
            calls.append(request)
            raise AssertionError("urlopen must not run for rejected Hub origin")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        with pytest.raises(EnrollmentVerificationFailure) as exc:
            verify_enrollment(
                "http://hub.example.com/api/v1/events",
                _CREDS,
                timeout_sec=0.2,
            )
        assert exc.value.error_class == "unreachable"
        assert calls == []

    def test_loopback_http_endpoint_is_formed(self) -> None:
        assert enrollment_endpoint("http://127.0.0.1:9/api/v1/events") == (
            "http://127.0.0.1:9/api/v1/edge/enrollments/verify"
        )
