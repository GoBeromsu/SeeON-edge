"""Server-validated dashboard sessions, separate from worker relay authority."""

from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from fastapi import HTTPException, Request, status

from backend.app.shared.dashboard_credentials import (
    DashboardCredentialsStore,
    PersistedDashboardCredentials,
)

API_DASHBOARD_USERNAME_ENV = "API_DASHBOARD_USERNAME"
API_DASHBOARD_PASSWORD_ENV = "API_DASHBOARD_PASSWORD"
API_ALLOW_LEGACY_DASHBOARD_AUTH_ENV = "API_ALLOW_LEGACY_DASHBOARD_AUTH"
API_EDGE_RELAY_TOKEN_ENV = "API_EDGE_RELAY_TOKEN"
DASHBOARD_SESSION_COOKIE = "ml_dashboard_session"
DASHBOARD_SESSION_TTL_SECONDS = 12 * 60 * 60

# Zero-config installer default: an edge box with no env vars and no persisted
# credentials file accepts this pair so the installer never has to touch a
# config file before logging in. Changeable from the dashboard settings; the
# change persists to disk and wins over this default forever after.
DEFAULT_DASHBOARD_USERNAME = "admin"
DEFAULT_DASHBOARD_PASSWORD = "admin"

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

    username: str

    def verify(self, username: str, password: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class PlaintextDashboardCredentials:
    """Env-var or built-in-default credentials: compared directly."""

    username: str
    password: str

    def verify(self, username: str, password: str) -> bool:
        return _compare_str(username, self.username) and _compare_str(
            password, self.password
        )


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


class WrongCurrentPasswordError(Exception):
    """The current_password supplied to a credential rotation didn't match."""


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
    if username or password:
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="dashboard credentials are incompletely configured",
            )
        return PlaintextDashboardCredentials(username=username, password=password)
    return PlaintextDashboardCredentials(
        username=DEFAULT_DASHBOARD_USERNAME, password=DEFAULT_DASHBOARD_PASSWORD
    )


def dashboard_sessions(request: Request) -> DashboardSessionStore:
    """Return the app-wide dashboard session store, resolving credentials once.

    Resolution order (highest wins): a persisted credentials file, then a
    fully-set env var pair (``API_DASHBOARD_USERNAME``/``_PASSWORD``, or the
    matching ``app.state`` attributes tests inject), then the built-in
    ``admin``/``admin`` default. This never returns ``None`` -- dashboard auth
    is always configured, even on a fresh edge box with zero setup.
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
    current_password: str,
    new_username: str | None,
    new_password: str,
) -> str:
    """Verify ``current_password`` against the live session, persist the new
    credentials, revoke every existing session, and return a fresh session
    token for the caller (so the PUT response can carry a live cookie).

    Raises ``WrongCurrentPasswordError`` without touching the credentials file
    when ``current_password`` doesn't match the active session's password.

    The whole verify -> persist -> swap -> mint sequence runs under
    ``_SESSION_STORE_INIT_LOCK`` so two concurrent rotations can't interleave
    (e.g. two admin tabs submitting at once): the second call re-verifies
    ``current_password`` against whatever the first call already swapped in,
    so at most one of two racing rotations against the same prior password
    succeeds -- the other gets a clean ``WrongCurrentPasswordError`` instead
    of a silently-overwritten file or a 204 whose cookie is already dead.
    """

    sessions = dashboard_sessions(request)
    store = dashboard_credentials_store(request)

    with _SESSION_STORE_INIT_LOCK:
        if not sessions.credentials.verify(sessions.username, current_password):
            raise WrongCurrentPasswordError()

        resolved_username = (new_username or "").strip() or sessions.username
        persisted = store.save(username=resolved_username, password=new_password)

        sessions.rotate_credentials(persisted)

        token = sessions.authenticate(resolved_username, new_password)
        if token is None:
            # Defensive: rotate_credentials() just persisted this exact pair,
            # so authenticate() against it must succeed.
            raise RuntimeError(
                "dashboard credential rotation produced an unauthenticated store"
            )
        return token


def authorize_dashboard(request: Request, *, legacy_token: str | None = None) -> str:
    """Authorize only a server-validated dashboard session."""

    sessions = dashboard_sessions(request)
    if sessions is not None:
        actor = sessions.actor(request.cookies.get(DASHBOARD_SESSION_COOKIE))
        if actor is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="dashboard session required",
            )
        return actor

    # dashboard_sessions() always resolves to a store (persisted file > env >
    # built-in default), so this branch is unreachable in practice today. Kept
    # so a future opt-out of the built-in default doesn't require re-deriving
    # the legacy worker-relay-token bypass from scratch.
    allow_legacy = os.environ.get(API_ALLOW_LEGACY_DASHBOARD_AUTH_ENV, "").strip() == "1"
    if not allow_legacy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="dashboard authentication is not configured",
        )
    expected = getattr(request.app.state, "edge_relay_token", None) or os.environ.get(
        API_EDGE_RELAY_TOKEN_ENV
    )
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="legacy dashboard authentication is not configured",
        )
    if legacy_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="dashboard credential required",
        )
    if not _compare_str(legacy_token, str(expected)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="dashboard credential mismatch",
        )
    return "legacy-dashboard"


__all__ = [
    "API_ALLOW_LEGACY_DASHBOARD_AUTH_ENV",
    "API_DASHBOARD_PASSWORD_ENV",
    "API_DASHBOARD_USERNAME_ENV",
    "DASHBOARD_SESSION_COOKIE",
    "DASHBOARD_SESSION_TTL_SECONDS",
    "DEFAULT_DASHBOARD_PASSWORD",
    "DEFAULT_DASHBOARD_USERNAME",
    "DashboardCredentials",
    "DashboardSessionStore",
    "HashedDashboardCredentials",
    "PlaintextDashboardCredentials",
    "WrongCurrentPasswordError",
    "authorize_dashboard",
    "dashboard_credentials_store",
    "dashboard_sessions",
    "rotate_dashboard_credentials",
]
