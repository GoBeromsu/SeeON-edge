"""Worker relay transport for stored-clip analysis."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Protocol

_RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnalysisRelaySettings:
    origin: str
    token: str | None
    timeout: float


@dataclass(frozen=True, slots=True)
class AnalysisRelayResponse:
    body: bytes
    status_code: int


class AnalysisRelayError(Exception):
    def __init__(self, status_code: int, detail: str | None = None) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail or HTTPStatus(status_code).phrase)


class AnalysisTransport(Protocol):
    def __call__(
        self,
        clip_id: str,
        suffix: str,
        *,
        body: Mapping[str, str] | None,
        accepted: frozenset[HTTPStatus | int],
        method: str = "POST",
    ) -> AnalysisRelayResponse: ...


def relay(
    settings: AnalysisRelaySettings,
    clip_id: str,
    suffix: str,
    *,
    body: Mapping[str, str] | None,
    accepted: frozenset[HTTPStatus | int],
    method: str = "POST",
) -> AnalysisRelayResponse:
    origin = settings.origin.strip().rstrip("/")
    if not origin:
        raise AnalysisRelayError(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="worker_unreachable",
        )
    relay_token = settings.token
    if not relay_token:
        raise AnalysisRelayError(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="worker_unreachable",
        )
    url = f"{origin}/clips/{urllib.parse.quote(clip_id, safe='')}/{suffix}"
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    headers = {_RELAY_TOKEN_HEADER: relay_token}
    if data is not None:
        headers["Content-Type"] = "application/json"
    upstream_request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        upstream = urllib.request.urlopen(
            upstream_request, timeout=settings.timeout
        )
    except urllib.error.HTTPError as exc:
        if exc.code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            _LOG.warning("%s", exc.__class__.__name__)
            raise AnalysisRelayError(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="worker_unreachable",
            ) from exc
        if HTTPStatus(exc.code) not in accepted:
            raise AnalysisRelayError(status_code=exc.code) from exc
        raw = exc.read()
        return AnalysisRelayResponse(body=raw, status_code=exc.code)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AnalysisRelayError(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="worker_unreachable",
        ) from exc
    try:
        raw = upstream.read()
        upstream_status = HTTPStatus(upstream.status)
    except (OSError, TimeoutError, ValueError) as exc:
        raise AnalysisRelayError(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="worker_unreachable",
        ) from exc
    finally:
        upstream.close()
    if upstream_status not in accepted:
        raise AnalysisRelayError(status_code=int(upstream_status))
    return AnalysisRelayResponse(body=raw, status_code=int(upstream_status))
