"""Hub / enrollment URL transport policy.

Production and any public Hub origin must use HTTPS with normal certificate
validation. Cleartext HTTP is permitted only for loopback destinations, or
when the explicit development/test contract
``API_BACKEND_ALLOW_INSECURE_HTTP=1`` is set (never a production default).
"""

from __future__ import annotations

import ipaddress
import os
import urllib.parse
from typing import Final

API_BACKEND_ALLOW_INSECURE_HTTP_ENV: Final = "API_BACKEND_ALLOW_INSECURE_HTTP"


def allow_insecure_http_from_env() -> bool:
    """True only when the explicit dev/test HTTP contract is opted in."""

    return os.environ.get(API_BACKEND_ALLOW_INSECURE_HTTP_ENV, "").strip() == "1"


def _hostname(netloc: str) -> str:
    """Extract host from a URL netloc (userinfo and port stripped)."""

    _, _, hostport = netloc.rpartition("@")
    host = hostport
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[1:end]
        return host.strip("[]")
    return host.rsplit(":", 1)[0]


def is_loopback_host(host: str) -> bool:
    cleaned = host.strip().lower().rstrip(".")
    if cleaned in {"localhost", "localhost."}:
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def hub_url_transport_allowed(url: str, *, allow_insecure_http: bool | None = None) -> bool:
    """Return whether ``url`` may be used as a Hub/enrollment origin.

    ``allow_insecure_http`` defaults to the process env contract when omitted.
    """

    cleaned = url.strip()
    if not cleaned:
        return False
    parsed = urllib.parse.urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    if parsed.scheme == "https":
        return True
    host = _hostname(parsed.netloc)
    if not host:
        return False
    if is_loopback_host(host):
        return True
    permitted = (
        allow_insecure_http
        if allow_insecure_http is not None
        else allow_insecure_http_from_env()
    )
    return bool(permitted)


def reject_hub_url_reason(url: str, *, allow_insecure_http: bool | None = None) -> str | None:
    """Human-readable rejection reason, or ``None`` when the URL is allowed."""

    cleaned = (url or "").strip()
    if not cleaned:
        return "hub URL is empty"
    parsed = urllib.parse.urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "hub URL must be an absolute http(s) URL"
    if hub_url_transport_allowed(cleaned, allow_insecure_http=allow_insecure_http):
        return None
    if parsed.scheme == "http":
        return (
            "cleartext http hub URL is not permitted for non-loopback hosts; "
            "use https, or set API_BACKEND_ALLOW_INSECURE_HTTP=1 only for local fixtures"
        )
    return "hub URL is not permitted"


__all__ = [
    "API_BACKEND_ALLOW_INSECURE_HTTP_ENV",
    "allow_insecure_http_from_env",
    "hub_url_transport_allowed",
    "is_loopback_host",
    "reject_hub_url_reason",
]
