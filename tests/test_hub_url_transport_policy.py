"""Hub base / enrollment URL transport policy (HTTPS production contract)."""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

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
from backend.app.features.connection.repository import ConnectionData
from backend.app.features.connection.store import (
    API_BACKEND_BASE_URL_ENV,
    ConnectionSettingsStore,
    InvalidConnectionSettingError,
)

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_CREDS = EnrollmentCredentials(
    facility_code="NH-7H2K9M4QXP",
    client_installation_ref="aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
    facility_token="facility-bearer-secret",
)


@pytest.fixture(autouse=True)
def production_hub_transport_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTPS policy tests must not inherit the suite's insecure-HTTP opt-in."""

    monkeypatch.delenv(API_BACKEND_ALLOW_INSECURE_HTTP_ENV, raising=False)


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


def _enrollment() -> ConnectionData:
    return {
        "facility_code": "NH-1234",
        "client_installation_ref": "install-1",
        "facility_id": "facility-1",
        "facility_token": "synthetic-client-token-1234",
        "edge_installation_id": "edge-1",
        "enrollment_generation": 1,
    }


class TestConnectionStoreTransportPolicy:
    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            ("https://hub.example", "https://hub.example/api"),
            (" https://hub.example/ ", "https://hub.example/api"),
            ("https://hub.example/api/", "https://hub.example/api"),
            ("http://hub.example", None),
            ("http://127.0.0.1:8000", "http://127.0.0.1:8000/api"),
            ("http://[::1]:8000/api", "http://[::1]:8000/api"),
            ("https://[invalid", None),
            ("ftp://hub.example", None),
            ("   ", None),
        ],
    )
    def test_base_url_is_canonical_deployment_authority_with_https_policy(
        self,
        postgres_product_sandbox: ProductSandbox,
        monkeypatch: pytest.MonkeyPatch,
        base: str,
        expected: str | None,
    ) -> None:
        sandbox = postgres_product_sandbox
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, base)
        monkeypatch.setenv("API_BACKEND_EVENTS_URL", "https://retired.example/events")
        monkeypatch.setenv("API_BACKEND_CONFIG_URL", "https://retired.example/config")
        store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
        saved = store.save(_enrollment())
        assert saved.events_url == (f"{expected}/v1/events" if expected else None)
        assert saved.config_url == (f"{expected}/v1/ml-config" if expected else None)
        assert store.load() == saved
        with pytest.raises(InvalidConnectionSettingError, match="unknown connection setting"):
            store.save({"events_url": "https://site-override.example/events"})
        assert store.load() == saved

    def test_http_requires_explicit_development_opt_in_for_public_host(
        self, postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = postgres_product_sandbox
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "http://hub.example")
        monkeypatch.setenv(API_BACKEND_ALLOW_INSECURE_HTTP_ENV, "1")
        store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
        assert store.load().events_url == "http://hub.example/api/v1/events"
        monkeypatch.setenv(API_BACKEND_ALLOW_INSECURE_HTTP_ENV, "0")
        assert store.load().events_url is None
