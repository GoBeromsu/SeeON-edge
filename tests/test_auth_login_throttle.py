"""Bounded login attempt throttling without timing sleeps."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.features.auth import router as auth_router
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.main import create_app, no_lifespan


def _client(tmp_path):
    app = create_app(lifespan=no_lifespan)
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    app.state.dashboard_username = "operator"
    app.state.dashboard_password = "correct-horse"
    return TestClient(app)


def test_auth_204_routes_declare_no_response_model() -> None:
    # Regression: with `from __future__ import annotations`, a bare `-> None`
    # return hint is a string; on FastAPI >=0.115 that made 204 endpoints trip
    # "Status code 204 must not have a response body" at import. Pinning
    # response_model=None keeps the router importable across the supported floor.
    no_content_routes = [
        route
        for route in auth_router.router.routes
        if getattr(route, "status_code", None) == 204
    ]
    assert no_content_routes
    for route in no_content_routes:
        assert route.response_model is None


def test_login_throttle_returns_429_after_bounded_failures(tmp_path, monkeypatch) -> None:
    # Deterministic window: inject fixed monotonic stamps via record/allow path.
    throttle = auth_router._LoginThrottle()
    monkeypatch.setattr(auth_router, "_LOGIN_THROTTLE", throttle)
    monkeypatch.setattr(auth_router, "_LOGIN_MAX_FAILURES_PER_KEY", 3)
    monkeypatch.setattr(auth_router, "_LOGIN_WINDOW_SECONDS", 60.0)

    now = 1_000.0
    key = "testclient\0operator"
    assert throttle.allow(key, now=now)
    throttle.record_failure(key, now=now)
    throttle.record_failure(key, now=now + 1)
    throttle.record_failure(key, now=now + 2)
    assert throttle.allow(key, now=now + 3) is False
    # After the window elapses, attempts are admitted again without sleeping.
    assert throttle.allow(key, now=now + 61) is True

    # The unit phase above injects absolute stamps (1000..1002) that are unrelated
    # to the real ``time.monotonic`` clock the HTTP phase below uses. On a fresh
    # runner monotonic can be < 1000, so those stamps read as future, never prune,
    # and would trip 429 before the first 401. Clear the shared key so the HTTP
    # phase starts from an empty window.
    throttle.clear(key)

    with _client(tmp_path) as client:
        for _ in range(3):
            denied = client.post(
                "/api/v1/auth/session",
                json={"username": "operator", "password": "wrong"},
            )
            assert denied.status_code == 401
        limited = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "wrong"},
        )
        assert limited.status_code == 429
        assert limited.headers.get("retry-after") is not None
        # Successful auth after a clear still works when under the limit.
        throttle.clear(key)
        ok = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "correct-horse"},
        )
        assert ok.status_code == 204
