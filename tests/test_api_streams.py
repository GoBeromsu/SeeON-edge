from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Callable, Iterator
from email.message import Message
from types import TracebackType
from typing import NoReturn, Self, TypedDict, cast

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.features.cameras import streams_router
from backend.app.features.cameras.streams_router import _iter_upstream, _UpstreamCloser
from backend.app.main import LifespanFactory, create_app, no_lifespan

AUTH = {"Authorization": "Bearer relay-token"}
NO_LIFESPAN: LifespanFactory = no_lifespan

# Dashboard auth now always resolves to a session store (persisted file > env
# > the built-in admin/admin default, see backend/app/shared/dashboard_auth.py),
# so a bare worker relay/bearer token or query token is never sufficient on
# its own -- these tests log in as the zero-config default and rely on the
# TestClient's cookie jar to carry the session across subsequent calls.


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


class UrlopenCall(TypedDict):
    url: str
    timeout: float
    method: str


class UrlopenCallWithHeaders(TypedDict):
    url: str
    method: str
    headers: dict[str, str]


class FiniteStreamResponse:
    status: int = 200
    headers: dict[str, str] = {"Content-Type": "multipart/x-mixed-replace; boundary=frame"}

    def __init__(self, body: bytes) -> None:
        self._body: bytes = body
        self.closed: bool = False

    def read(self, size: int = -1) -> bytes:
        if not self._body:
            return b""
        if size < 0:
            size = len(self._body)
        chunk = self._body[:size]
        self._body = self._body[size:]
        return chunk

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()


@pytest.fixture(autouse=True)
def stream_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", "http://worker.local:8090")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class StreamCall(TypedDict):
    url: str
    method: str


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """camera_stream now proxies via httpx.AsyncClient (see streams_router.py)
    instead of urllib, so these tests inject an httpx.MockTransport wherever
    the router constructs its client -- ``httpx.AsyncClient(...)`` is looked
    up on the module at call time, so patching the module attribute is
    enough."""
    real_async_client = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", _factory)


def test_stream_proxy_forwards_mjpeg_with_a_dashboard_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        b"\xff\xd8camera-jpeg\xff\xd9\r\n"
    )
    calls: list[StreamCall] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "method": request.method})
        return httpx.Response(
            200,
            headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame"},
            content=body,
        )

    _install_mock_transport(monkeypatch, handler)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.get("/api/v1/streams/cam_sp_201")

    assert response.status_code == 200
    assert response.content == body
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert calls == [{"url": "http://worker.local:8090/stream/cam_sp_201", "method": "GET"}]


def test_stream_proxy_requires_a_dashboard_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker relay bearer token or a query token is never a substitute for
    a real dashboard session -- the legacy bypass is unreachable now that
    dashboard auth always resolves (persisted file > env > built-in default)."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            200,
            headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame"},
            content=b"--frame\r\n",
        )

    _install_mock_transport(monkeypatch, handler)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        missing = client.get("/api/v1/streams/cam_sp_201")
        wrong_query_token = client.get("/api/v1/streams/cam_sp_201", params={"token": "wrong"})
        bearer_without_session = client.get("/api/v1/streams/cam_sp_201", headers=AUTH)
        _login(client)
        authorized = client.get("/api/v1/streams/cam_sp_201")

    assert missing.status_code == 401
    assert wrong_query_token.status_code == 401
    assert bearer_without_session.status_code == 401
    assert authorized.status_code == 200
    assert calls == ["http://worker.local:8090/stream/cam_sp_201"]


@pytest.mark.parametrize("code", [404, 503])
def test_stream_proxy_preserves_upstream_404_and_503(
    code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(code)

    _install_mock_transport(monkeypatch, handler)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.get("/api/v1/streams/missing")

    assert response.status_code == code
    assert response.json()["detail"] == "worker stream unavailable"


def test_stream_proxy_reports_connection_failure_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> NoReturn:
        raise httpx.ConnectError("connection refused", request=request)

    _install_mock_transport(monkeypatch, handler)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.get("/api/v1/streams/cam_sp_201")

    assert response.status_code == 503
    assert response.json()["detail"] == "worker stream unavailable"


def test_stream_proxy_closes_upstream_response_and_client_on_cancellation() -> None:
    """핵심 회귀(연결 누수 버그): 클라이언트가 스트림을 중간에 끊으면 이 태스크가
    asyncio.CancelledError로 취소되고, ``_iter_upstream``의 finally가 지연 없이
    실행돼 upstream 응답(``aclose``)과 커넥션 풀(``client.aclose``)을 닫아야
    한다. 실제 프로덕션에서는 uvicorn이 클라이언트 disconnect 시 요청 처리
    태스크를 취소하는데, 그 취소가 (스레드풀로 감싼 블로킹 read와 달리) 이
    async 제너레이터의 현재 await 지점에 곧바로 전달되는 것이 이번 수정의
    핵심이다."""

    class _StubUpstreamStream(httpx.AsyncByteStream):
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks
            self.closed = False
            self.yielded_first_chunk = asyncio.Event()

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for chunk in self._chunks:
                yield chunk
                self.yielded_first_chunk.set()
                # 실제 MJPEG 스트림처럼 다음 프레임을 무기한 기다린다 --
                # 취소는 바로 이 await 지점에 꽂혀야 한다.
                await asyncio.sleep(3600)

        async def aclose(self) -> None:
            self.closed = True

    async def scenario() -> None:
        upstream_stream = _StubUpstreamStream([b"--frame\r\njpeg-bytes\r\n"])
        stream_request = httpx.Request("GET", "http://worker.local:8090/stream/cam_sp_201")
        response = httpx.Response(200, stream=upstream_stream, request=stream_request)

        def handler(_: httpx.Request) -> httpx.Response:
            return response

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stream_ctx = client.stream("GET", "http://worker.local:8090/stream/cam_sp_201")
        upstream = await stream_ctx.__aenter__()
        closer = _UpstreamCloser(client, stream_ctx)

        gen = _iter_upstream(closer, upstream)

        async def consume() -> None:
            async for _chunk in gen:
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(upstream_stream.yielded_first_chunk.wait(), timeout=1.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert upstream_stream.closed is True
        assert client.is_closed is True

    asyncio.run(scenario())


def test_stream_proxy_closes_upstream_via_background_when_never_iterated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """핵심 회귀(연결 누수 버그, 2차): 헤더 응답 직후 ~ Starlette가
    ``body_iterator`` 순회를 시작하기 전 사이의 좁은 창에서 클라이언트가
    끊기면(curl을 SIGTERM으로 죽이는 패턴에서 실측됨), 요청 처리 태스크가
    한 번도 실행되지 못한 채 취소돼 ``_iter_upstream`` 제너레이터는 시작조차
    되지 않는다 -- 시작한 적 없는 제너레이터는 닫아도 그 finally가 돌지
    않으므로, 그 경로 하나에만 정리를 맡기면 upstream/client가 영영 새는
    것이 실제로 재현됐다. camera_stream이 같은 closer를
    ``StreamingResponse(background=...)``에도 걸어 두므로, body_iterator를
    전혀 건드리지 않고 background만 실행해도 정리가 되는지 고정한다."""

    closed = {"stream": False}
    created_clients: list[httpx.AsyncClient] = []
    real_async_client = httpx.AsyncClient

    class _StubUpstreamStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"--frame\r\n"

        async def aclose(self) -> None:
            closed["stream"] = True

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_StubUpstreamStream(), request=request)

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        created_client = real_async_client(*args, **kwargs)  # type: ignore[arg-type]
        created_clients.append(created_client)
        return created_client

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    # camera_stream 자체의 배선(타임아웃 설정, closer/background 연결)을
    # 실제로 거치도록 라우트 함수를 직접 호출한다 -- 여기선 대시보드 세션
    # 검증이 관심사가 아니므로 _authorize만 이 모듈 안에서 no-op으로 바꾼다.
    monkeypatch.setattr(streams_router, "_authorize", lambda *args, **kwargs: None)

    async def scenario() -> None:
        response = await streams_router.camera_stream(
            "cam_sp_201",
            request=cast(Request, object()),
            authorization=None,
            token=None,
        )

        assert response.background is not None
        # body_iterator는 절대 건드리지 않는다 -- SIGTERM 재현 패턴에서
        # Starlette가 실제로 밟는, "제너레이터를 한 번도 순회하지 않은 채
        # background만 실행"하는 경로 그대로다.
        await response.background()

    asyncio.run(scenario())

    assert closed["stream"] is True
    assert len(created_clients) == 1
    assert created_clients[0].is_closed is True


def test_snapshot_proxy_forwards_jpeg_with_a_dashboard_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"\xff\xd8camera-jpeg\xff\xd9"
    calls: list[UrlopenCall] = []

    class JpegResponse(FiniteStreamResponse):
        headers: dict[str, str] = {"Content-Type": "image/jpeg"}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> JpegResponse:
        calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "method": request.get_method(),
            }
        )
        return JpegResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.get("/api/v1/streams/cam_sp_201/snapshot")

    assert response.status_code == 200
    assert response.content == body
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert calls == [
        {
            "url": "http://worker.local:8090/snapshot/cam_sp_201",
            "timeout": 3.0,
            "method": "GET",
        }
    ]


def test_snapshot_proxy_requires_a_dashboard_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FiniteStreamResponse:
        del timeout
        calls.append(request.full_url)
        return FiniteStreamResponse(b"\xff\xd8jpeg\xff\xd9")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        missing = client.get("/api/v1/streams/cam_sp_201/snapshot")
        wrong_query_token = client.get(
            "/api/v1/streams/cam_sp_201/snapshot", params={"token": "wrong"}
        )
        bearer_without_session = client.get(
            "/api/v1/streams/cam_sp_201/snapshot", headers=AUTH
        )
        _login(client)
        authorized = client.get("/api/v1/streams/cam_sp_201/snapshot")

    assert missing.status_code == 401
    assert wrong_query_token.status_code == 401
    assert bearer_without_session.status_code == 401
    assert authorized.status_code == 200
    assert calls == ["http://worker.local:8090/snapshot/cam_sp_201"]


@pytest.mark.parametrize("code", [404, 503])
def test_snapshot_proxy_preserves_upstream_404_and_503(
    code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> NoReturn:
        del timeout
        raise urllib.error.HTTPError(
            request.full_url,
            code,
            "upstream status",
            hdrs=Message(),
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.get("/api/v1/streams/missing/snapshot")

    assert response.status_code == code
    assert response.json()["detail"] == "worker stream unavailable"


def test_snapshot_proxy_reports_connection_failure_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> NoReturn:
        del request, timeout
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.get("/api/v1/streams/cam_sp_201/snapshot")

    assert response.status_code == 503
    assert response.json()["detail"] == "worker stream unavailable"


class PoseJsonResponse(FiniteStreamResponse):
    headers: dict[str, str] = {"Content-Type": "application/json"}


def test_pose_get_forwards_and_returns_current_state_with_a_dashboard_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[UrlopenCall] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> PoseJsonResponse:
        calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "method": request.get_method(),
            }
        )
        return PoseJsonResponse(json.dumps({"mode": "fall"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.get("/api/v1/streams/cam_sp_201/pose")

    assert response.status_code == 200
    assert response.json() == {"mode": "fall"}
    assert calls == [
        {
            "url": "http://worker.local:8090/overlay/cam_sp_201/pose",
            "timeout": 3.0,
            "method": "GET",
        }
    ]


def test_pose_set_forwards_the_requested_value_with_a_dashboard_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> PoseJsonResponse:
        calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "method": request.get_method(),
                "body": request.data,
            }
        )
        return PoseJsonResponse(json.dumps({"mode": "bedexit"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.post("/api/v1/streams/cam_sp_201/pose", json={"mode": "bedexit"})

    assert response.status_code == 200
    assert response.json() == {"mode": "bedexit"}
    assert calls == [
        {
            "url": "http://worker.local:8090/overlay/cam_sp_201/pose",
            "timeout": 3.0,
            "method": "POST",
            "body": json.dumps({"mode": "bedexit"}).encode("utf-8"),
        }
    ]


def test_pose_get_forwards_the_relay_token_to_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #71: the worker now gates GET /overlay/{camera_id}/pose on the
    same relay token as /probe, so the proxy must forward it (mirroring
    cameras/router.py's `/probe` connection-test call) or every dashboard
    pose read would start 403ing against a worker with a token configured.
    """
    calls: list[UrlopenCallWithHeaders] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> PoseJsonResponse:
        del timeout
        calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": dict(request.headers),
            }
        )
        return PoseJsonResponse(json.dumps({"mode": "none"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.get("/api/v1/streams/cam_sp_201/pose")

    assert response.status_code == 200
    assert calls == [
        {
            "url": "http://worker.local:8090/overlay/cam_sp_201/pose",
            "method": "GET",
            "headers": {"X-edge-relay-token": "relay-token"},
        }
    ]


def test_pose_set_forwards_the_relay_token_to_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #71: same as the GET case above, but for POST /overlay/{camera_id}/pose."""
    calls: list[UrlopenCallWithHeaders] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> PoseJsonResponse:
        del timeout
        calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": dict(request.headers),
            }
        )
        return PoseJsonResponse(json.dumps({"mode": "fall"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.post("/api/v1/streams/cam_sp_201/pose", json={"mode": "fall"})

    assert response.status_code == 200
    assert calls == [
        {
            "url": "http://worker.local:8090/overlay/cam_sp_201/pose",
            "method": "POST",
            "headers": {
                "Content-type": "application/json",
                "X-edge-relay-token": "relay-token",
            },
        }
    ]


def test_pose_get_and_set_require_a_dashboard_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> PoseJsonResponse:
        del timeout
        return PoseJsonResponse(json.dumps({"mode": "none"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        missing = client.get("/api/v1/streams/cam_sp_201/pose")
        missing_post = client.post("/api/v1/streams/cam_sp_201/pose", json={"mode": "fall"})
        _login(client)
        authorized = client.get("/api/v1/streams/cam_sp_201/pose")

    assert missing.status_code == 401
    assert missing_post.status_code == 401
    assert authorized.status_code == 200


def test_pose_set_rejects_unknown_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> NoReturn:
        del request, timeout
        raise AssertionError("upstream must not be called for a rejected payload")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.post(
            "/api/v1/streams/cam_sp_201/pose",
            json={"mode": "fall", "unexpected": "field"},
        )

    assert response.status_code == 422


def test_pose_set_rejects_unknown_mode_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> NoReturn:
        del request, timeout
        raise AssertionError("upstream must not be called for a rejected payload")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.post(
            "/api/v1/streams/cam_sp_201/pose",
            json={"mode": "show_pose"},
        )

    assert response.status_code == 422


@pytest.mark.parametrize("code", [404, 503])
def test_pose_get_preserves_upstream_404_and_503(
    code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> NoReturn:
        del timeout
        raise urllib.error.HTTPError(
            request.full_url, code, "upstream status", hdrs=Message(), fp=None
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.get("/api/v1/streams/missing/pose")

    assert response.status_code == code
    assert response.json()["detail"] == "worker stream unavailable"
