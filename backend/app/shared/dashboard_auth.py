"""Server-validated dashboard sessions, separate from worker relay authority."""

from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from fastapi import HTTPException, Request, status

from backend.app.shared.dashboard_credentials import (
    PersistedDashboardCredentials,
)
from backend.app.shared.postgres_dashboard_credentials import PostgresDashboardCredentialsStore

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
    """Persisted credentials: password compared via scrypt verify."""

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
    _active: bool = field(default=True, init=False, repr=False)

    @property
    def username(self) -> str:
        return self.credentials.username

    def authenticate(self, username: str, password: str) -> str | None:
        with self._lock:
            if not self._active or not self.credentials.verify(username, password):
                return None
            token = secrets.token_urlsafe(32)
            self._prune_locked()
            self._sessions[token] = time.monotonic() + self.ttl_seconds
        return token

    def actor(self, token: str | None) -> str | None:
        if token is None:
            return None
        with self._lock:
            self._prune_locked()
            if not self._active or token not in self._sessions:
                return None
            return self.credentials.username

    def revoke(self, token: str | None) -> None:
        if token is None:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def revoke_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    def invalidate(self) -> None:
        """Retire even held references after an uncertain credential write."""
        with self._lock:
            self._active = False
            self._sessions.clear()

    def rotate_credentials(self, persisted: PersistedDashboardCredentials) -> None:
        """Swap in newly-persisted credentials and revoke every existing session.

        Called after credential persistence and audit publication; the caller
        must hold ``_SESSION_STORE_INIT_LOCK`` while calling this so the
        swap is observed atomically by concurrent requests resolving the
        session store for the first time.
        """
        with self._lock:
            if not self._active:
                raise RuntimeError("dashboard session store is inactive")
            self.credentials = HashedDashboardCredentials(persisted)
            self._sessions.clear()

    def _prune_locked(self) -> None:
        now = time.monotonic()
        expired = [token for token, deadline in self._sessions.items() if deadline <= now]
        for token in expired:
            self._sessions.pop(token, None)


def dashboard_credentials_store(request: Request) -> PostgresDashboardCredentialsStore:
    existing = getattr(request.app.state, "dashboard_credentials_store", None)
    if existing is None:
        raise RuntimeError("dashboard credentials store is not injected")
    if not isinstance(existing, PostgresDashboardCredentialsStore):
        raise TypeError("dashboard credentials store has invalid type")
    return existing


def _resolve_credentials(request: Request) -> DashboardCredentials:
    store = dashboard_credentials_store(request)
    persisted = store.load()
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
    persist: Callable[[PostgresDashboardCredentialsStore, str, str], PersistedDashboardCredentials],
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

    ``persist`` must return only after the complete native save owner and its
    audit publication return. Failures retire cached authority, not committed
    database state. Authorization is checked again after acquiring the lock.
    """

    sessions = dashboard_sessions(request)
    store = dashboard_credentials_store(request)

    with _SESSION_STORE_INIT_LOCK:
        if (
            getattr(request.app.state, "dashboard_sessions", None) is not sessions
            or sessions.actor(request.cookies.get(DASHBOARD_SESSION_COOKIE)) is None
        ):
            raise HTTPException(status_code=401, detail="dashboard session required")
        resolved_username = (new_username or "").strip() or sessions.username
        try:
            persisted = persist(store, resolved_username, new_password)
            return _mint_rotated_session(sessions, persisted, new_password)
        except BaseException:
            # A failed complete owner return may already have committed. Never
            # keep authenticating against possibly superseded cached credentials.
            sessions.invalidate()
            if getattr(request.app.state, "dashboard_sessions", None) is sessions:
                del request.app.state.dashboard_sessions
            raise


def _mint_rotated_session(
    sessions: DashboardSessionStore, persisted: PersistedDashboardCredentials, password: str
) -> str:
    sessions.rotate_credentials(persisted)
    token = sessions.authenticate(persisted.username, password)
    if token is None:
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
