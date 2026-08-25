"""Authenticated backend-to-worker clip deletion commands."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from fastapi import FastAPI, HTTPException, Request, status

from backend.app.core.config import get_settings


def preflight_clip_deletion(target: FastAPI | Request, clip_id: str) -> dict[str, object]:
    """Ask the worker to verify hold, ownership, and containment without mutation."""
    return _command(target, clip_id, method="GET", suffix="/deletion-preflight")


def control_clip_deletion(target: FastAPI | Request, clip_id: str) -> dict[str, object]:
    """Command deletion only after the backend has committed durable intent."""
    return _command(target, clip_id, method="DELETE", suffix="")


def _command(
    target: FastAPI | Request,
    clip_id: str,
    *,
    method: str,
    suffix: str,
) -> dict[str, object]:
    settings = get_settings()
    origin = settings.worker_stream_origin.strip().rstrip("/")
    if not origin:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker clip deletion origin is not configured",
        )
    encoded = urllib.parse.quote(clip_id, safe="")
    url = f"{origin}/clips/{encoded}{suffix}"
    holder = target.app if isinstance(target, Request) else target
    token = getattr(holder.state, "edge_relay_token", None)
    headers = {"X-Edge-Relay-Token": token} if isinstance(token, str) and token else {}
    upstream_request = urllib.request.Request(url, method=method, headers=headers)
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
    except (TimeoutError, OSError, urllib.error.URLError) as error:
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


__all__ = ["control_clip_deletion", "preflight_clip_deletion"]
