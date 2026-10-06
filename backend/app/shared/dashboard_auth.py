"""Server-validated dashboard sessions, separate from worker relay authority."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

from fastapi import HTTPException, Request, status

from backend.app.shared.dashboard_credential_rotation import (
    DashboardSessionRequired,
    rotate_credentials,
)
from backend.app.shared.dashboard_credentials import (
    PersistedDashboardCredentials,
)
from backend.app.shared.dashboard_sessions import (
    DASHBOARD_SESSION_TTL_SECONDS,
    DashboardCredentials,
    DashboardSessionStore,
    HashedDashboardCredentials,
    PlaintextDashboardCredentials,
)
from backend.app.shared.postgres_dashboard_credentials import PostgresDashboardCredentialsStore

API_DASHBOARD_USERNAME_ENV = "API_DASHBOARD_USERNAME"
API_DASHBOARD_PASSWORD_ENV = "API_DASHBOARD_PASSWORD"
DASHBOARD_SESSION_COOKIE = "ml_dashboard_session"

# Known insecure pair. Never used as a runtime authority fallback. Preflight
# rejects this pair in deployment env files; tests may still set it explicitly
# via API_DASHBOARD_* env fixtures for disposable bootstrap ergonomics.
DEFAULT_DASHBOARD_USERNAME = "admin"
DEFAULT_DASHBOARD_PASSWORD = "admin"
KNOWN_DEFAULT_DASHBOARD_USERNAME = DEFAULT_DASHBOARD_USERNAME
KNOWN_DEFAULT_DASHBOARD_PASSWORD = DEFAULT_DASHBOARD_PASSWORD

_SESSION_STORE_INIT_LOCK = threading.Lock()


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
    """Adapt the request and serialize complete credential rotations."""
    sessions = dashboard_sessions(request)
    store = dashboard_credentials_store(request)

    with _SESSION_STORE_INIT_LOCK:
        if getattr(request.app.state, "dashboard_sessions", None) is not sessions:
            raise HTTPException(status_code=401, detail="dashboard session required")
        try:
            return rotate_credentials(
                sessions,
                store,
                token=request.cookies.get(DASHBOARD_SESSION_COOKIE),
                new_username=new_username,
                new_password=new_password,
                persist=persist,
            )
        except DashboardSessionRequired as error:
            raise HTTPException(status_code=401, detail="dashboard session required") from error
        except BaseException:
            if getattr(request.app.state, "dashboard_sessions", None) is sessions:
                del request.app.state.dashboard_sessions
            raise


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
