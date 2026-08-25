"""Server-validated dashboard sessions, separate from worker relay authority."""

from __future__ import annotations

import hmac
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from fastapi import HTTPException, Request, status

from backend.app.shared.dashboard_credentials import (
    DashboardCredentialsStore,
    DashboardCredentialsStoreError,
    PersistedDashboardCredentials,
)

API_DASHBOARD_USERNAME_ENV = "API_DASHBOARD_USERNAME"
API_DASHBOARD_PASSWORD_ENV = "API_DASHBOARD_PASSWORD"
DASHBOARD_SESSION_COOKIE = "ml_dashboard_session"
DASHBOARD_SESSION_TTL_SECONDS = 12 * 60 * 60

# Known insecure pair. Never used as a runtime authority fallback. Preflight
# rejects this pair in deployment env files; tests may still set it explicitly
# via API_DASHBOARD_* env fixtures for disposable bootstrap ergonomics.
DEFAULT_DASHBOARD_USERNAME = "admin"
DEFAULT_DASHBOARD_PASSWORD = "admin"
KNOWN_DEFAULT_DASHBOARD_USERNAME = DEFAULT_DASHBOARD_USERNAME
KNOWN_DEFAULT_DASHBOARD_PASSWORD = DEFAULT_DASHBOARD_PASSWORD

_SESSION_STORE_INIT_LOCK = threading.Lock()


def _compare_str(candidate: str, expected: str) -> bool:
    """Constant-time compare of two ``str`` values that may contain non-ASCII
    text (e.g. a Korean username or password).

    ``hmac.compare_digest`` raises ``TypeError`` for non-ASCII ``str``
    arguments -- it only accepts ASCII ``str`` or ``bytes``/``bytes``-like
    objects. Encoding to UTF-8 bytes first keeps the comparison constant-time
    while supporting any username/password the product's Korean-language UI
    can produce.
    """

    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


class DashboardCredentials(Protocol):
    """Something that knows one dashboard username and can verify a login."""

    @property
    def username(self) -> str: ...

    def verify(self, username: str, password: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class PlaintextDashboardCredentials:
    """Env-var or built-in-default credentials: compared directly."""

    username: str
    password: str

    def verify(self, username: str, password: str) -> bool:
        return _compare_str(username, self.username) and _compare_str(password, self.password)


@dataclass(frozen=True, slots=True)
class HashedDashboardCredentials:
    """Persisted-file credentials: password compared via scrypt verify."""

    persisted: PersistedDashboardCredentials

    @property
    def username(self) -> str:
        return self.persisted.username

    def verify(self, username: str, password: str) -> bool:
        return _compare_str(username, self.persisted.username) and (
            self.persisted.verify_password(password)
        )


@dataclass(slots=True)
class DashboardSessionStore:
    credentials: DashboardCredentials
    ttl_seconds: int = DASHBOARD_SESSION_TTL_SECONDS
    _sessions: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def username(self) -> str:
        return self.credentials.username

    def authenticate(self, username: str, password: str) -> str | None:
        if not self.credentials.verify(username, password):
            return None
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._prune_locked()
            self._sessions[token] = time.monotonic() + self.ttl_seconds
        return token

    def actor(self, token: str | None) -> str | None:
        if token is None:
            return None
        with self._lock:
            self._prune_locked()
            if token not in self._sessions:
                return None
        return self.username

    def revoke(self, token: str | None) -> None:
        if token is None:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def revoke_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    def rotate_credentials(self, persisted: PersistedDashboardCredentials) -> None:
        """Swap in newly-persisted credentials and revoke every existing session.

        Called after a successful ``PUT /auth/credentials``; the caller is
        expected to hold ``_SESSION_STORE_INIT_LOCK`` while calling this so the
        swap is observed atomically by concurrent requests resolving the
        session store for the first time.
        """
        self.credentials = HashedDashboardCredentials(persisted)
        self.revoke_all()

    def _prune_locked(self) -> None:
        now = time.monotonic()
        expired = [token for token, deadline in self._sessions.items() if deadline <= now]
        for token in expired:
            self._sessions.pop(token, None)


def dashboard_credentials_store(request: Request) -> DashboardCredentialsStore:
    existing = getattr(request.app.state, "dashboard_credentials_store", None)
    if isinstance(existing, DashboardCredentialsStore):
        return existing
    store = DashboardCredentialsStore.from_env()
    request.app.state.dashboard_credentials_store = store
    return store


def _resolve_credentials(request: Request) -> DashboardCredentials:
    store = dashboard_credentials_store(request)
    try:
        persisted = store.load()
    except DashboardCredentialsStoreError as exc:
        # Fail closed: a corrupt/unreadable store after rotation must not
        # resurrect env or known-default credentials.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="dashboard credentials store is unreadable",
        ) from exc
    if persisted is not None:
        return HashedDashboardCredentials(persisted)

    username = str(
        getattr(request.app.state, "dashboard_username", "")
        or os.environ.get(API_DASHBOARD_USERNAME_ENV, "")
    ).strip()
    password = str(
        getattr(request.app.state, "dashboard_password", "")
        or os.environ.get(API_DASHBOARD_PASSWORD_ENV, "")
    )
    if not username and not password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="dashboard credentials are not configured",
        )
    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="dashboard credentials are incompletely configured",
        )
    return PlaintextDashboardCredentials(username=username, password=password)


def dashboard_sessions(request: Request) -> DashboardSessionStore:
    """Return the app-wide dashboard session store, resolving credentials once.

    Resolution order (highest wins): a persisted credentials row, then a
    fully-set deployment bootstrap pair (``API_DASHBOARD_USERNAME``/``_PASSWORD``
    or matching ``app.state`` attributes). There is no built-in password
    default — missing or corrupt authority fails closed with HTTP 503.
    """

    existing = getattr(request.app.state, "dashboard_sessions", None)
    if isinstance(existing, DashboardSessionStore):
        return existing
    with _SESSION_STORE_INIT_LOCK:
        existing = getattr(request.app.state, "dashboard_sessions", None)
        if isinstance(existing, DashboardSessionStore):
            return existing
        credentials = _resolve_credentials(request)
        store = DashboardSessionStore(credentials=credentials)
        request.app.state.dashboard_sessions = store
        return store


def rotate_dashboard_credentials(
    request: Request,
    *,
    new_username: str | None,
    new_password: str,
    after_write: Callable[[sqlite3.Connection], None] | None = None,
) -> str:
    """Persist the new credentials, revoke every existing session, and return
    a fresh session token for the caller (so the PUT response can carry a
    live cookie).

    The caller must already hold a valid dashboard session -- callers reach
    this function only after ``authorize_dashboard`` -- so no current-password
    check is performed here; the session cookie is the sole auth gate.

    The whole persist -> swap -> mint sequence runs under
    ``_SESSION_STORE_INIT_LOCK`` so two concurrent rotations can't interleave
    (e.g. two admin tabs submitting at once).
    """

    sessions = dashboard_sessions(request)
    store = dashboard_credentials_store(request)

    with _SESSION_STORE_INIT_LOCK:
        resolved_username = (new_username or "").strip() or sessions.username
        persisted = store.save(
            username=resolved_username, password=new_password, after_write=after_write
        )

        sessions.rotate_credentials(persisted)

        token = sessions.authenticate(resolved_username, new_password)
        if token is None:
            # Defensive: rotate_credentials() just persisted this exact pair,
            # so authenticate() against it must succeed.
            raise RuntimeError("dashboard credential rotation produced an unauthenticated store")
        return token


def authorize_dashboard(request: Request) -> str:
    """Authorize only a server-validated dashboard session."""

    sessions = dashboard_sessions(request)
    actor = sessions.actor(request.cookies.get(DASHBOARD_SESSION_COOKIE))
    if actor is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="dashboard session required",
        )
    return actor


__all__ = [
    "API_DASHBOARD_PASSWORD_ENV",
    "API_DASHBOARD_USERNAME_ENV",
    "DASHBOARD_SESSION_COOKIE",
    "DASHBOARD_SESSION_TTL_SECONDS",
    "DEFAULT_DASHBOARD_PASSWORD",
    "DEFAULT_DASHBOARD_USERNAME",
    "KNOWN_DEFAULT_DASHBOARD_PASSWORD",
    "KNOWN_DEFAULT_DASHBOARD_USERNAME",
    "DashboardCredentials",
    "DashboardSessionStore",
    "HashedDashboardCredentials",
    "PlaintextDashboardCredentials",
    "authorize_dashboard",
    "dashboard_credentials_store",
    "dashboard_sessions",
    "rotate_dashboard_credentials",
]
