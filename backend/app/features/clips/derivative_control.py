from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Literal

from fastapi import HTTPException, Request, status

from backend.app.core.config import get_settings

DerivativeAction = Literal["request", "cancel", "status"]


def control_derivative(
    request: Request,
    clip_id: str,
    kind: Literal["still", "video"],
    action: DerivativeAction,
) -> dict[str, object]:
    settings = get_settings()
    origin = settings.worker_stream_origin.strip().rstrip("/")
    if not origin:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker derivative origin is not configured",
        )
    encoded = urllib.parse.quote(clip_id, safe="")
    url = f"{origin}/derivatives/{encoded}/{kind.upper()}"
    token = getattr(request.app.state, "edge_relay_token", None)
    headers = {}
    if isinstance(token, str) and token:
        headers["X-Edge-Relay-Token"] = token
    method = {"request": "POST", "cancel": "DELETE", "status": "GET"}[action]
    upstream_request = urllib.request.Request(
        url,
        data=b"" if method == "POST" else None,
        method=method,
        headers=headers,
    )
    try:
        upstream = urllib.request.urlopen(
            upstream_request,
            timeout=settings.worker_stream_timeout_s,
        )
    except urllib.error.HTTPError as error:
        code = status.HTTP_404_NOT_FOUND if error.code == 404 else status.HTTP_409_CONFLICT
        raise HTTPException(status_code=code, detail="worker derivative request failed") from error
    except (OSError, urllib.error.URLError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker derivative runtime unavailable",
        ) from error
    try:
        payload = upstream.read(64 * 1024 + 1)
    finally:
        upstream.close()
    if len(payload) > 64 * 1024:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="worker derivative response is invalid",
        )
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="worker derivative response is invalid",
        ) from error
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="worker derivative response is invalid",
        )
    return {str(key): value for key, value in parsed.items()}


__all__ = ["control_derivative"]
