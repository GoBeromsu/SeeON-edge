from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from email.message import Message
from types import TracebackType
from typing import NoReturn, Self, TypedDict

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
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


def test_stream_proxy_forwards_mjpeg_with_a_dashboard_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        b"\xff\xd8camera-jpeg\xff\xd9\r\n"
    )
    calls: list[UrlopenCall] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FiniteStreamResponse:
        calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "method": request.get_method(),
            }
        )
        return FiniteStreamResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.get("/api/v1/streams/cam_sp_201")

    assert response.status_code == 200
    assert response.content == body
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert calls == [
        {
            "url": "http://worker.local:8090/stream/cam_sp_201",
            "timeout": 3.0,
            "method": "GET",
        }
    ]


def test_stream_proxy_requires_a_dashboard_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker relay bearer token or a query token is never a substitute for
    a real dashboard session -- the legacy bypass is unreachable now that
    dashboard auth always resolves (persisted file > env > built-in default)."""
    calls: list[str] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FiniteStreamResponse:
        del timeout
        calls.append(request.full_url)
        return FiniteStreamResponse(b"--frame\r\n")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

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
        response = client.get("/api/v1/streams/missing")

    assert response.status_code == code
    assert response.json()["detail"] == "worker stream unavailable"


def test_stream_proxy_reports_connection_failure_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> NoReturn:
        del request, timeout
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=NO_LIFESPAN)) as client:
        _login(client)
        response = client.get("/api/v1/streams/cam_sp_201")

    assert response.status_code == 503
    assert response.json()["detail"] == "worker stream unavailable"


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
