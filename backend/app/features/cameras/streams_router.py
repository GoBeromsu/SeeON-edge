from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Annotated, Literal, Protocol, get_args

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from backend.app.core.config import get_settings
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(tags=["streams"])

_DEFAULT_MEDIA_TYPE = "multipart/x-mixed-replace; boundary=frame"
_STREAM_CHUNK_SIZE = 64 * 1024


class _ResponseHeaders(Protocol):
    def get(self, name: str, default: str | None = None) -> str | None: ...


class _ReadableResponse(Protocol):
    status: int
    headers: _ResponseHeaders

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...


OverlayMode = Literal["none", "bedexit", "fall"]

_OVERLAY_MODES: frozenset[str] = frozenset(get_args(OverlayMode))


class PoseOverlayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: OverlayMode


class PoseOverlayResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: OverlayMode


@router.get("/streams/{camera_id}")
def camera_stream(
    camera_id: str,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    token: Annotated[str | None, Query()] = None,
) -> StreamingResponse:
    _authorize(request, authorization, query_token=token)
    settings = get_settings()
    upstream_url = _stream_url(settings.worker_stream_origin, camera_id)
    upstream_request = urllib.request.Request(upstream_url, method="GET")

    try:
        upstream: _ReadableResponse = urllib.request.urlopen(
            upstream_request,
            timeout=settings.worker_stream_timeout_s,
        )
    except urllib.error.HTTPError as exc:
        raise _upstream_unavailable(exc.code) from exc
    except urllib.error.URLError as exc:
        raise _upstream_unavailable(status.HTTP_503_SERVICE_UNAVAILABLE) from exc

    upstream_status = int(getattr(upstream, "status", status.HTTP_200_OK))
    if upstream_status != status.HTTP_200_OK:
        upstream.close()
        raise _upstream_unavailable(upstream_status)

    return StreamingResponse(
        _iter_upstream(upstream),
        media_type=_content_type(upstream),
    )


@router.get("/streams/{camera_id}/snapshot")
def camera_snapshot(
    camera_id: str,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    token: Annotated[str | None, Query()] = None,
) -> Response:
    _authorize(request, authorization, query_token=token)
    settings = get_settings()
    upstream_url = _snapshot_url(settings.worker_stream_origin, camera_id)
    upstream_request = urllib.request.Request(upstream_url, method="GET")

    try:
        upstream: _ReadableResponse = urllib.request.urlopen(
            upstream_request,
            timeout=settings.worker_stream_timeout_s,
        )
    except urllib.error.HTTPError as exc:
        raise _upstream_unavailable(exc.code) from exc
    except urllib.error.URLError as exc:
        raise _upstream_unavailable(status.HTTP_503_SERVICE_UNAVAILABLE) from exc

    upstream_status = int(getattr(upstream, "status", status.HTTP_200_OK))
    if upstream_status != status.HTTP_200_OK:
        upstream.close()
        raise _upstream_unavailable(upstream_status)

    try:
        body = upstream.read()
        content_type = _content_type(upstream)
    finally:
        upstream.close()

    return Response(content=body, media_type=content_type, headers={"Cache-Control": "no-store"})


@router.get("/streams/{camera_id}/pose")
def camera_pose_get(
    camera_id: str,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    token: Annotated[str | None, Query()] = None,
) -> PoseOverlayResponse:
    _authorize(request, authorization, query_token=token)
    settings = get_settings()
    upstream_url = _pose_url(settings.worker_stream_origin, camera_id)
    return _pose_request(upstream_url, settings.worker_stream_timeout_s)


@router.post("/streams/{camera_id}/pose")
def camera_pose_set(
    camera_id: str,
    payload: PoseOverlayRequest,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    token: Annotated[str | None, Query()] = None,
) -> PoseOverlayResponse:
    _authorize(request, authorization, query_token=token)
    settings = get_settings()
    upstream_url = _pose_url(settings.worker_stream_origin, camera_id)
    body = json.dumps({"mode": payload.mode}).encode("utf-8")
    return _pose_request(upstream_url, settings.worker_stream_timeout_s, body=body)


def _worker_url(origin: str, segment: str, camera_id: str, *, suffix: str = "") -> str:
    base = origin.strip().rstrip("/")
    if not base:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker stream origin is not configured",
        )
    encoded_camera_id = urllib.parse.quote(camera_id, safe="")
    return f"{base}/{segment}/{encoded_camera_id}{suffix}"


def _stream_url(origin: str, camera_id: str) -> str:
    return _worker_url(origin, "stream", camera_id)


def _snapshot_url(origin: str, camera_id: str) -> str:
    return _worker_url(origin, "snapshot", camera_id)


def _pose_url(origin: str, camera_id: str) -> str:
    return _worker_url(origin, "overlay", camera_id, suffix="/pose")


def _pose_request(
    upstream_url: str,
    timeout_s: float,
    *,
    body: bytes | None = None,
) -> PoseOverlayResponse:
    method = "GET" if body is None else "POST"
    headers = {} if body is None else {"Content-Type": "application/json"}
    upstream_request = urllib.request.Request(
        upstream_url, data=body, method=method, headers=headers
    )

    try:
        upstream: _ReadableResponse = urllib.request.urlopen(
            upstream_request,
            timeout=timeout_s,
        )
    except urllib.error.HTTPError as exc:
        raise _upstream_unavailable(exc.code) from exc
    except urllib.error.URLError as exc:
        raise _upstream_unavailable(status.HTTP_503_SERVICE_UNAVAILABLE) from exc

    try:
        upstream_status = int(getattr(upstream, "status", status.HTTP_200_OK))
        raw = upstream.read()
    finally:
        upstream.close()
    if upstream_status != status.HTTP_200_OK:
        raise _upstream_unavailable(upstream_status)

    try:
        parsed = json.loads(raw)
        mode = parsed["mode"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise _upstream_unavailable(status.HTTP_503_SERVICE_UNAVAILABLE) from exc
    if not isinstance(mode, str) or mode not in _OVERLAY_MODES:
        raise _upstream_unavailable(status.HTTP_503_SERVICE_UNAVAILABLE)
    return PoseOverlayResponse(mode=mode)  # type: ignore[arg-type]


def _iter_upstream(upstream: _ReadableResponse) -> Iterator[bytes]:
    try:
        while True:
            chunk = upstream.read(_STREAM_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk
    finally:
        upstream.close()


def _content_type(upstream: _ReadableResponse) -> str:
    value = upstream.headers.get("Content-Type")
    if value is not None and value.strip():
        return value
    return _DEFAULT_MEDIA_TYPE


def _authorize(
    request: Request,
    authorization: str | None,
    *,
    query_token: str | None = None,
) -> None:
    supplied = _bearer_token(authorization) or query_token
    authorize_dashboard(request, legacy_token=supplied)


def _bearer_token(value: str | None) -> str | None:
    if value is None or not value.startswith("Bearer "):
        return None
    return value.removeprefix("Bearer ").strip() or None

def _upstream_unavailable(status_code: int) -> HTTPException:
    if status_code not in {
        status.HTTP_404_NOT_FOUND,
        status.HTTP_503_SERVICE_UNAVAILABLE,
    }:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HTTPException(status_code=status_code, detail="worker stream unavailable")


__all__ = ["router"]
