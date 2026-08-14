"""Server-side dashboard login, logout, and credential-rotation routes."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.shared.dashboard_auth import (
    DASHBOARD_SESSION_COOKIE,
    DashboardSessionStore,
    authorize_dashboard,
    dashboard_sessions,
    rotate_dashboard_credentials,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# Sliding-window login throttle: bound credential-guessing without sleep delays.
_LOGIN_WINDOW_SECONDS = 60.0
_LOGIN_MAX_FAILURES_PER_KEY = 10
_LOGIN_MAX_TRACKED_KEYS = 4096


class DashboardLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class DashboardCredentialsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str | None = Field(default=None, min_length=1, max_length=128)
    new_password: str = Field(min_length=4, max_length=512)


@dataclass
class _LoginThrottle:
    """Process-local failed-login window keyed by client + username."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _failures: dict[str, deque[float]] = field(default_factory=dict)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        stamp = time.monotonic() if now is None else now
        cutoff = stamp - _LOGIN_WINDOW_SECONDS
        with self._lock:
            self._prune_locked(cutoff)
            bucket = self._failures.get(key)
            if bucket is None:
                return True
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if not bucket:
                del self._failures[key]
                return True
            return len(bucket) < _LOGIN_MAX_FAILURES_PER_KEY

    def record_failure(self, key: str, *, now: float | None = None) -> None:
        stamp = time.monotonic() if now is None else now
        cutoff = stamp - _LOGIN_WINDOW_SECONDS
        with self._lock:
            self._prune_locked(cutoff)
            bucket = self._failures.get(key)
            if bucket is None:
                if len(self._failures) >= _LOGIN_MAX_TRACKED_KEYS:
                    # Drop the oldest key by first failure timestamp.
                    oldest_key = min(
                        self._failures,
                        key=lambda item: self._failures[item][0] if self._failures[item] else stamp,
                    )
                    del self._failures[oldest_key]
                bucket = deque()
                self._failures[key] = bucket
            bucket.append(stamp)

    def clear(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)

    def _prune_locked(self, cutoff: float) -> None:
        stale = [
            key for key, bucket in self._failures.items() if not bucket or bucket[-1] <= cutoff
        ]
        for key in stale:
            del self._failures[key]


_LOGIN_THROTTLE = _LoginThrottle()


def _set_session_cookie(
    response: Response, request: Request, sessions: DashboardSessionStore, token: str
) -> None:
    response.set_cookie(
        DASHBOARD_SESSION_COOKIE,
        token,
        max_age=sessions.ttl_seconds,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )


def _client_key(request: Request, username: str) -> str:
    client = request.client.host if request.client is not None else "unknown"
    return f"{client}\0{username.strip().lower()}"


@router.post("/session", status_code=status.HTTP_204_NO_CONTENT)
def login(payload: DashboardLoginRequest, request: Request, response: Response) -> None:
    key = _client_key(request, payload.username)
    if not _LOGIN_THROTTLE.allow(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="too many login attempts",
            headers={"Retry-After": str(int(_LOGIN_WINDOW_SECONDS))},
        )
    sessions = dashboard_sessions(request)
    token = sessions.authenticate(payload.username, payload.password)
    if token is None:
        _LOGIN_THROTTLE.record_failure(key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid dashboard credentials",
        )
    _LOGIN_THROTTLE.clear(key)
    _set_session_cookie(response, request, sessions, token)


@router.get("/session", status_code=status.HTTP_204_NO_CONTENT)
def session(request: Request) -> None:
    authorize_dashboard(request)


@router.delete("/session", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response) -> None:
    dashboard_sessions(request).revoke(request.cookies.get(DASHBOARD_SESSION_COOKIE))
    response.delete_cookie(DASHBOARD_SESSION_COOKIE, path="/", samesite="strict")


@router.put("/credentials", status_code=status.HTTP_204_NO_CONTENT)
def update_credentials(
    payload: DashboardCredentialsUpdateRequest, request: Request, response: Response
) -> None:
    authorize_dashboard(request)
    token = rotate_dashboard_credentials(
        request,
        new_username=payload.username,
        new_password=payload.new_password,
    )
    _set_session_cookie(response, request, dashboard_sessions(request), token)


__all__ = ["router"]
