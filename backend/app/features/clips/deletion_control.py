from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from fastapi import HTTPException, Request, status

from backend.app.core.config import get_settings


def control_clip_deletion(request: Request, clip_id: str) -> dict[str, object]:
    """Forward an operator's clip-delete command to the worker over the same
    authenticated control seam used by derivative requests
    (``backend.app.features.clips.derivative_control``): one plain-HTTP call
    to the worker's shared MJPEG/derivative-control listener, gated by the
    ``X-Edge-Relay-Token`` the backend already holds.
    """
    settings = get_settings()
    origin = settings.worker_stream_origin.strip().rstrip("/")
    if not origin:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker derivative origin is not configured",
        )
    encoded = urllib.parse.quote(clip_id, safe="")
    url = f"{origin}/clips/{encoded}"
    token = getattr(request.app.state, "edge_relay_token", None)
    headers = {}
    if isinstance(token, str) and token:
        headers["X-Edge-Relay-Token"] = token
    upstream_request = urllib.request.Request(url, method="DELETE", headers=headers)
    try:
        upstream = urllib.request.urlopen(
            upstream_request,
            timeout=settings.worker_stream_timeout_s,
        )
    except urllib.error.HTTPError as error:
        code = error.code if error.code in {403, 404, 409, 503} else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(
            status_code=code, detail="worker clip deletion request failed"
        ) from error
    except (OSError, urllib.error.URLError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker clip deletion runtime unavailable",
        ) from error
    try:
        payload = upstream.read(64 * 1024 + 1)
    finally:
        upstream.close()
    if len(payload) > 64 * 1024:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="worker clip deletion response is invalid",
        )
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="worker clip deletion response is invalid",
        ) from error
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="worker clip deletion response is invalid",
        )
    return {str(key): value for key, value in parsed.items()}


__all__ = ["control_clip_deletion"]
