"""Refuse a worker boot against a mixed schema-identity API image."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from shared.release_identity import require_peer_schema_identity

UrlOpen = Callable[..., Any]


def require_api_release_identity(
    relay_url: str,
    *,
    urlopen: UrlOpen | None = None,
    timeout_sec: float = 5.0,
) -> None:
    """Fetch the API release identity and refuse a 17/18 pair."""
    opener = urllib.request.urlopen if urlopen is None else urlopen
    request = urllib.request.Request(
        f"{relay_url.rstrip('/')}/health/release-identity",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with opener(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            require_peer_schema_identity(17)
        raise
    require_peer_schema_identity(int(payload["edge_database_schema_version"]))


__all__ = ["require_api_release_identity"]
