from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Literal, Protocol, get_args

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask

from backend.app.core.config import get_settings
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(tags=["streams"])

_DEFAULT_MEDIA_TYPE = "multipart/x-mixed-replace; boundary=frame"
_STREAM_CHUNK_SIZE = 64 * 1024
# Mirrors cameras/router.py's RELAY_TOKEN_HEADER -- the worker's dev MJPEG
# server now gates POST/GET /overlay/{camera_id}/pose with the same relay
# token as /probe (issue #71). Duplicated here rather than importing from
# cameras/router.py to avoid coupling this module to that router's private
# helpers/ownership.
_RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"


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
async def camera_stream(
    camera_id: str,
    request: Request,
) -> StreamingResponse:
    _authorize(request)
    settings = get_settings()
    upstream_url = _stream_url(settings.worker_stream_origin, _worker_camera_id(request, camera_id))

    # MJPEG 스트림은 장시간 연결이고 worker가 1초 간격 하트비트 프레임을
    # 보내므로, 본문(read)에는 타임아웃을 걸지 않는다(read=None). 최초 응답
    # 헤더까지는 아래 asyncio.wait_for로 worker_stream_timeout_s를 별도로
    # 씌워 무한 대기를 막는다.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=settings.worker_stream_timeout_s,
            read=None,
            write=settings.worker_stream_timeout_s,
            pool=settings.worker_stream_timeout_s,
        )
    )
    # Worker :8090 gates /stream with the same relay token as /probe
    # (security finding #3). Forward server-side only -- never surface it to
    # the browser (dashboard session already authenticated this request).
    stream_ctx = client.stream("GET", upstream_url, headers=_worker_relay_headers(request))
    handed_off = False
    try:
        try:
            upstream = await asyncio.wait_for(
                stream_ctx.__aenter__(), timeout=settings.worker_stream_timeout_s
            )
        except (TimeoutError, httpx.HTTPError) as exc:
            raise _upstream_unavailable(status.HTTP_503_SERVICE_UNAVAILABLE) from exc

        if upstream.status_code != status.HTTP_200_OK:
            upstream_status = upstream.status_code
            await stream_ctx.__aexit__(None, None, None)
            raise _upstream_unavailable(upstream_status)

        media_type = _stream_content_type(upstream)
        # 여기서부터는 커넥션/클라이언트 정리 책임이 closer로 넘어간다 --
        # 클라이언트가 스트리밍 도중 끊으면(CancelledError) 블로킹 스레드 없이
        # 이 태스크 자체가 즉시 취소되므로 _iter_upstream의 finally가 upstream
        # 소켓을 지연 없이 닫는다. 그런데 "헤더 응답 직후 ~ Starlette가
        # body_iterator 순회를 시작하기 전" 사이의 좁은 창에서 끊기면(SIGTERM
        # 재현 패턴에서 실측됨) 그 async 제너레이터가 한 번도 시작되지 않아
        # finally가 절대 실행되지 않는다 -- 시작한 적 없는 제너레이터는 닫아도
        # 내부 코드가 돌지 않는다. StreamingResponse.__call__은 (uvicorn의
        # h11/httptools 경로가 쓰는 spec_version 2.3에서는) body_iterator 순회가
        # disconnect로 취소된 뒤에도 background를 항상 실행하므로, 같은 closer를
        # background에도 걸어 두 경로 중 어느 쪽이 실제로 실행되든 정리되게 한다.
        closer = _UpstreamCloser(client, stream_ctx)
        handed_off = True
        return StreamingResponse(
            _iter_upstream(closer, upstream),
            media_type=media_type,
            background=BackgroundTask(closer.aclose),
        )
    finally:
        if not handed_off:
            await client.aclose()


@router.get("/streams/{camera_id}/snapshot")
def camera_snapshot(
    camera_id: str,
    request: Request,
) -> Response:
    _authorize(request)
    settings = get_settings()
    upstream_url = _snapshot_url(
        settings.worker_stream_origin, _worker_camera_id(request, camera_id)
    )
    upstream_request = urllib.request.Request(
        upstream_url,
        method="GET",
        headers=_worker_relay_headers(request),
    )

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
) -> PoseOverlayResponse:
    _authorize(request)
    settings = get_settings()
    upstream_url = _pose_url(settings.worker_stream_origin, _worker_camera_id(request, camera_id))
    return _pose_request(
        upstream_url, settings.worker_stream_timeout_s, relay_token=_relay_token(request)
    )


@router.post("/streams/{camera_id}/pose")
def camera_pose_set(
    camera_id: str,
    payload: PoseOverlayRequest,
    request: Request,
) -> PoseOverlayResponse:
    _authorize(request)
    settings = get_settings()
    upstream_url = _pose_url(settings.worker_stream_origin, _worker_camera_id(request, camera_id))
    body = json.dumps({"mode": payload.mode}).encode("utf-8")
    return _pose_request(
        upstream_url,
        settings.worker_stream_timeout_s,
        body=body,
        relay_token=_relay_token(request),
    )


def _worker_url(origin: str, segment: str, camera_id: str, *, suffix: str = "") -> str:
    base = origin.strip().rstrip("/")
    if not base:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker stream origin is not configured",
        )
    encoded_camera_id = urllib.parse.quote(camera_id, safe="")
    return f"{base}/{segment}/{encoded_camera_id}{suffix}"


def _worker_camera_id(request: Request, camera_id: str) -> str:
    store = getattr(request.app.state, "camera_registry", None)
    if not isinstance(store, CameraRegistryStore):
        return camera_id
    cameras = store.snapshot().get("cameras")
    if not isinstance(cameras, list):
        return camera_id
    for record in cameras:
        if not isinstance(record, dict):
            continue
        local_id = record.get("id")
        backend_id = record.get("backend_camera_id")
        if camera_id in {local_id, backend_id}:
            return backend_id if isinstance(backend_id, str) and backend_id else camera_id
    return camera_id


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
    relay_token: str | None = None,
) -> PoseOverlayResponse:
    method = "GET" if body is None else "POST"
    headers: dict[str, str] = {} if body is None else {"Content-Type": "application/json"}
    if relay_token:
        headers[_RELAY_TOKEN_HEADER] = relay_token
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

    return _parse_pose_payload(raw)


def _parse_pose_payload(raw: bytes) -> PoseOverlayResponse:
    """The worker's ``{"mode": ...}`` body; anything else is an upstream failure."""
    try:
        parsed = json.loads(raw)
        mode = parsed["mode"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise _upstream_unavailable(status.HTTP_503_SERVICE_UNAVAILABLE) from exc
    if not isinstance(mode, str) or mode not in _OVERLAY_MODES:
        raise _upstream_unavailable(status.HTTP_503_SERVICE_UNAVAILABLE)
    return PoseOverlayResponse(mode=mode)  # type: ignore[arg-type]


class _UpstreamCloser:
    """upstream 응답과 커넥션 풀을 정확히 한 번만 닫는다.

    닫는 경로가 두 곳이라 멱등성이 필요하다: (1) 스트림을 순회하던 중 취소되면
    ``_iter_upstream``의 finally가 부르고, (2) 그 제너레이터가 한 번도
    순회되지 않은 채(예: 헤더 응답 직후 ~ Starlette가 body_iterator 순회를
    시작하기 전 사이에 클라이언트가 끊기는 경우, SIGTERM 재현 패턴에서 실측됨)
    응답이 버려지면 ``StreamingResponse``의 background가 대신 부른다. 둘 중
    어느 쪽이 먼저(또는 유일하게) 실행되든 안전하도록 ``_closed`` 플래그로
    이중 close를 막는다.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        stream_ctx: AbstractAsyncContextManager[httpx.Response],
    ) -> None:
        self._client = client
        self._stream_ctx = stream_ctx
        self._closed = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._stream_ctx.__aexit__(None, None, None)
        await self._client.aclose()


async def _iter_upstream(
    closer: _UpstreamCloser,
    upstream: httpx.Response,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in upstream.aiter_bytes(_STREAM_CHUNK_SIZE):
            yield chunk
    finally:
        # 클라이언트 disconnect는 asyncio.CancelledError로 이 제너레이터의
        # 현재 await 지점(aiter_bytes 내부의 소켓 read)에 곧바로 꽂힌다 --
        # 스레드풀로 감싼 블로킹 read와 달리 여기선 대기 중인 작업 자체가
        # 취소되므로, 이 finally가 지연 없이 실행돼 upstream 응답과 커넥션
        # 풀을 닫는다. (이 제너레이터가 아예 시작되지 않는 경로는
        # camera_stream의 background=BackgroundTask(closer.aclose)가 대신
        # 담당한다 -- 그 경우 이 finally 자체가 실행될 기회가 없다.)
        await closer.aclose()


def _stream_content_type(upstream: httpx.Response) -> str:
    value = upstream.headers.get("Content-Type")
    if value is not None and value.strip():
        return value
    return _DEFAULT_MEDIA_TYPE


def _content_type(upstream: _ReadableResponse) -> str:
    value = upstream.headers.get("Content-Type")
    if value is not None and value.strip():
        return value
    return _DEFAULT_MEDIA_TYPE


def _authorize(request: Request) -> None:
    authorize_dashboard(request)


def _relay_token(request: Request) -> str | None:
    """The same worker-relay token cameras/router.py's ``_expected_relay_token``
    resolves for the ``/probe`` connection-test call (app.state.edge_relay_token,
    set at boot from ``API_EDGE_RELAY_TOKEN`` -- see
    lifespan._configure_backend_ingest, which is the only place that reads the
    env var). Duplicated here (rather
    than importing the private helper from cameras/router.py) per issue #71's
    worker-side auth fix: the worker's pose GET/POST overlay routes now gate on
    this same token, so it must be forwarded on those upstream calls too.

    Stream, snapshot, and bed-zone recognition on worker :8090 use the same
    gate (security finding #3); this helper is the sole server-side source for
    the header those proxies forward.
    """
    # `app.state.edge_relay_token`이 유일한 출처다. 부팅 때
    # `lifespan._configure_backend_ingest`가 `API_EDGE_RELAY_TOKEN`에서 한 번
    # 채운다(`lifespan.py:105`가 반드시 부른다). 여기서 env를 또 읽으면
    # state에 `None`을 명시적으로 넣어 미설정을 재현하려는 경우까지
    # env가 조용히 덮어써서, 실제로 무엇이 유효한지 두 곳을 봐야 한다.
    expected = getattr(request.app.state, "edge_relay_token", None)
    return expected if isinstance(expected, str) and expected else None


def _worker_relay_headers(request: Request) -> dict[str, str]:
    """Headers the backend attaches when calling worker :8090 media routes.

    Empty when no relay token is configured (upstream then fails closed with
    403, which this proxy maps to 503). Never written into browser-facing
    responses or query strings.
    """
    token = _relay_token(request)
    if token is None:
        return {}
    return {_RELAY_TOKEN_HEADER: token}


def _upstream_unavailable(status_code: int) -> HTTPException:
    if status_code not in {
        status.HTTP_404_NOT_FOUND,
        status.HTTP_503_SERVICE_UNAVAILABLE,
    }:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HTTPException(status_code=status_code, detail="worker stream unavailable")


__all__ = ["router"]
