"""Unit coverage for the shared RTSP destination policy."""

from __future__ import annotations

import ipaddress
import socket

import pytest

from shared.rtsp_url_policy import (
    ALLOW_LOCAL_RTSP_ENV,
    ALLOW_PRIVATE_RTSP_ENV,
    assert_rtsp_endpoint_allowed,
    assert_rtsp_url_allowed,
    reject_resolved_addresses_reason,
    reject_rtsp_url_reason,
    resolve_rtsp_endpoint,
)


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://camera.example/live",
        "rtsps://camera.example:322/stream",
        "rtsp://user:pass@camera.example:8554/path?subtype=0",
        "rtsp://8.8.8.8/live",
    ],
)
def test_public_and_named_hosts_are_allowed(url: str) -> None:
    assert reject_rtsp_url_reason(url) is None
    assert assert_rtsp_url_allowed(url) == url.strip()


@pytest.mark.parametrize(
    "url",
    [
        "http://camera.example/live",
        "rtsp:///no-host",
        "rtsp://",
        "file:///etc/passwd",
        "",
        "rtsp://camera.example/live#frag",
        "rtsp://camera.example/li\x00ve",
    ],
)
def test_unsupported_schemes_and_syntax_are_rejected(url: str) -> None:
    assert reject_rtsp_url_reason(url) is not None
    with pytest.raises(ValueError):
        assert_rtsp_url_allowed(url)


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://127.0.0.1/live",
        "rtsp://localhost/live",
        "rtsp://[::1]/live",
        "rtsp://169.254.169.254/latest/meta-data",
        "rtsp://169.254.1.1/live",
        "rtsp://10.0.0.9/live",
        "rtsp://192.168.1.20/live",
        "rtsp://172.16.5.5/live",
        "rtsp://100.64.0.1/live",
        "rtsp://metadata.google.internal/live",
    ],
)
def test_unsafe_destinations_rejected_by_default(url: str) -> None:
    assert reject_rtsp_url_reason(url) is not None


def test_private_allowance_admits_lan_but_not_loopback_or_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ALLOW_PRIVATE_RTSP_ENV, "1")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    assert reject_rtsp_url_reason("rtsp://10.0.0.9/live") is None
    assert reject_rtsp_url_reason("rtsp://127.0.0.1/live") is not None
    assert reject_rtsp_url_reason("rtsp://169.254.169.254/live") is not None


def test_local_allowance_admits_loopback_link_local_and_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ALLOW_LOCAL_RTSP_ENV, "1")
    monkeypatch.delenv(ALLOW_PRIVATE_RTSP_ENV, raising=False)
    assert reject_rtsp_url_reason("rtsp://127.0.0.1:8554/live") is None
    assert reject_rtsp_url_reason("rtsp://localhost/live") is None
    assert reject_rtsp_url_reason("rtsp://10.0.0.9/live") is None
    assert reject_rtsp_url_reason("rtsp://169.254.1.1/live") is None
    # Metadata remains denied even under local fixture allowance.
    assert reject_rtsp_url_reason("rtsp://169.254.169.254/live") is not None
    assert reject_rtsp_url_reason("rtsp://metadata.google.internal/live") is not None


def test_resolve_pins_public_ipv4_and_preserves_userinfo_port_path() -> None:
    endpoint = resolve_rtsp_endpoint(
        "rtsp://user:pass@cam.example:8554/live?x=1",
        resolver=lambda _host: ("8.8.8.8",),
    )
    assert endpoint.original_url == "rtsp://user:pass@cam.example:8554/live?x=1"
    assert endpoint.pinned_url == "rtsp://user:pass@8.8.8.8:8554/live?x=1"
    assert endpoint.addresses == ("8.8.8.8",)
    assert endpoint.selected_address == "8.8.8.8"


def test_resolve_pins_public_ipv6_with_brackets() -> None:
    endpoint = resolve_rtsp_endpoint(
        "rtsps://cam.example/secure",
        resolver=lambda _host: ("2001:4860:4860::8888",),
    )
    assert endpoint.pinned_url == "rtsps://[2001:4860:4860::8888]/secure"
    assert endpoint.selected_address == "2001:4860:4860::8888"


def test_resolve_rejects_metadata_ipv4_answer() -> None:
    with pytest.raises(ValueError, match="metadata"):
        resolve_rtsp_endpoint(
            "rtsp://cam.example/live",
            resolver=lambda _host: ("169.254.169.254",),
        )


def test_resolve_rejects_metadata_ipv6_answer() -> None:
    with pytest.raises(ValueError, match="metadata"):
        resolve_rtsp_endpoint(
            "rtsp://cam.example/live",
            resolver=lambda _host: ("fd00:ec2::254",),
        )


def test_resolve_rejects_private_answer_by_default() -> None:
    with pytest.raises(ValueError, match="private"):
        resolve_rtsp_endpoint(
            "rtsp://cam.example/live",
            resolver=lambda _host: ("10.0.0.9",),
        )


def test_resolve_rejects_loopback_answer_by_default() -> None:
    with pytest.raises(ValueError, match="loopback"):
        resolve_rtsp_endpoint(
            "rtsp://cam.example/live",
            resolver=lambda _host: ("127.0.0.1",),
        )


def test_resolve_rejects_loopback_ipv6_answer_by_default() -> None:
    with pytest.raises(ValueError, match="loopback"):
        resolve_rtsp_endpoint(
            "rtsp://cam.example/live",
            resolver=lambda _host: ("::1",),
        )


def test_resolve_rejects_link_local_even_when_private_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ALLOW_PRIVATE_RTSP_ENV, "1")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    with pytest.raises(ValueError, match="link-local"):
        resolve_rtsp_endpoint(
            "rtsp://cam.example/live",
            resolver=lambda _host: ("169.254.1.1",),
        )
    with pytest.raises(ValueError, match="metadata"):
        resolve_rtsp_endpoint(
            "rtsp://cam.example/live",
            resolver=lambda _host: ("169.254.169.254",),
        )
    # Facility LAN still admitted under PRIVATE=1.
    endpoint = resolve_rtsp_endpoint(
        "rtsp://cam.example/live",
        resolver=lambda _host: ("10.0.0.9",),
    )
    assert endpoint.pinned_url == "rtsp://10.0.0.9/live"


def test_resolve_rejects_if_any_answer_is_blocked_mixed_set() -> None:
    # Public + metadata must not be admissible: any blocked answer fails closed.
    with pytest.raises(ValueError, match="metadata"):
        resolve_rtsp_endpoint(
            "rtsp://cam.example/live",
            resolver=lambda _host: ("8.8.8.8", "169.254.169.254"),
        )
    with pytest.raises(ValueError, match="private"):
        resolve_rtsp_endpoint(
            "rtsp://cam.example/live",
            resolver=lambda _host: ("8.8.8.8", "10.0.0.9"),
        )


def test_resolve_admits_local_fixture_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ALLOW_LOCAL_RTSP_ENV, "1")
    monkeypatch.delenv(ALLOW_PRIVATE_RTSP_ENV, raising=False)
    loopback = resolve_rtsp_endpoint(
        "rtsp://fixture.local:8554/live",
        resolver=lambda _host: ("127.0.0.1",),
    )
    assert loopback.pinned_url == "rtsp://127.0.0.1:8554/live"
    link_local = resolve_rtsp_endpoint(
        "rtsp://fixture.local/live",
        resolver=lambda _host: ("169.254.1.1",),
    )
    assert link_local.pinned_url == "rtsp://169.254.1.1/live"
    # Metadata still denied under LOCAL=1.
    with pytest.raises(ValueError, match="metadata"):
        resolve_rtsp_endpoint(
            "rtsp://fixture.local/live",
            resolver=lambda _host: ("169.254.169.254",),
        )


def test_resolve_empty_or_failing_resolver_is_rejected() -> None:
    with pytest.raises(ValueError, match="could not be resolved"):
        resolve_rtsp_endpoint(
            "rtsp://cam.example/live",
            resolver=lambda _host: (),
        )

    def _boom(_host: str) -> tuple[str, ...]:
        raise socket.gaierror(socket.EAI_NONAME, "not found")

    with pytest.raises(ValueError, match="could not be resolved"):
        resolve_rtsp_endpoint("rtsp://cam.example/live", resolver=_boom)


def test_resolve_literal_ip_does_not_call_resolver() -> None:
    calls: list[str] = []

    def _resolver(host: str) -> tuple[str, ...]:
        calls.append(host)
        return ("8.8.8.8",)

    endpoint = resolve_rtsp_endpoint("rtsp://8.8.8.8/live", resolver=_resolver)
    assert endpoint.pinned_url == "rtsp://8.8.8.8/live"
    assert calls == []


def test_assert_rtsp_endpoint_allowed_matches_resolve() -> None:
    endpoint = assert_rtsp_endpoint_allowed(
        "rtsp://cam.example/live",
        resolver=lambda _host: ("1.1.1.1", "8.8.4.4"),
    )
    assert endpoint.addresses == ("1.1.1.1", "8.8.4.4")
    assert endpoint.selected_address == "1.1.1.1"
    assert endpoint.pinned_url == "rtsp://1.1.1.1/live"


def test_reject_resolved_addresses_reason_covers_empty_and_mixed() -> None:
    assert reject_resolved_addresses_reason(()) is not None
    assert reject_resolved_addresses_reason(("8.8.8.8",)) is None
    assert reject_resolved_addresses_reason(("8.8.8.8", "10.0.0.1")) is not None


def test_real_getaddrinfo_driver_resolves_localhost_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call the real platform resolver (not the suite DNS stub)."""

    # Import the live module function after restoring the real implementation.
    import shared.rtsp_url_policy as policy

    monkeypatch.setattr(policy, "resolve_host_a_aaaa", _real_resolve_host_a_aaaa)

    addresses = policy.resolve_host_a_aaaa("localhost")
    assert addresses
    assert all(ipaddress.ip_address(item).is_loopback for item in addresses)

    monkeypatch.setenv(ALLOW_LOCAL_RTSP_ENV, "1")
    endpoint = policy.resolve_rtsp_endpoint("rtsp://localhost/live")
    assert ipaddress.ip_address(endpoint.selected_address).is_loopback
    assert "localhost" not in endpoint.pinned_url


def _real_resolve_host_a_aaaa(hostname: str) -> tuple[str, ...]:
    """Direct getaddrinfo driver used by the real-resolver characterization test."""

    cleaned = hostname.strip()
    if not cleaned:
        return ()
    infos = socket.getaddrinfo(
        cleaned,
        None,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
    )
    addresses: list[str] = []
    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        raw = sockaddr[0]
        try:
            rendered = str(ipaddress.ip_address(raw))
        except ValueError:
            continue
        if rendered in seen:
            continue
        seen.add(rendered)
        addresses.append(rendered)
    return tuple(addresses)
