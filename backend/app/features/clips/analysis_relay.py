"""Worker relay transport for stored-clip analysis."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus

from fastapi import HTTPException, Request, Response, status

from backend.app.core.config import get_settings

_RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"
_LOG = logging.getLogger(__name__)


def relay(
    request: Request,
    clip_id: str,
    suffix: str,
    *,
    body: dict[str, str] | dict[object, object] | None,
    accepted: frozenset[HTTPStatus | int],
    method: str = "POST",
) -> Response:
    settings = get_settings()
    origin = settings.worker_stream_origin.strip().rstrip("/")
    if not origin:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker_unreachable",
        )
    relay_token = request.app.state.edge_relay_token
    if not isinstance(relay_token, str) or not relay_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
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
            upstream_request, timeout=settings.worker_stream_timeout_s
        )
    except urllib.error.HTTPError as exc:
        if exc.code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            _LOG.warning("%s", exc.__class__.__name__)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="worker_unreachable",
            ) from exc
        if HTTPStatus(exc.code) not in accepted:
            raise HTTPException(status_code=exc.code) from exc
        raw = exc.read()
        return Response(content=raw, status_code=exc.code, media_type="application/json")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker_unreachable",
        ) from exc
    try:
        raw = upstream.read()
        upstream_status = HTTPStatus(upstream.status)
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker_unreachable",
        ) from exc
    finally:
        upstream.close()
    if upstream_status not in accepted:
        raise HTTPException(status_code=int(upstream_status))
    return Response(content=raw, status_code=int(upstream_status), media_type="application/json")
