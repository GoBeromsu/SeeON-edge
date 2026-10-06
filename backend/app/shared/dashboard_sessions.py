"""In-memory dashboard credential and session authority."""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from backend.app.shared.dashboard_credentials import PersistedDashboardCredentials

DASHBOARD_SESSION_TTL_SECONDS = 12 * 60 * 60


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
    """Process-local session authority; credential and token mutation is its purpose."""

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


__all__ = [
    "DASHBOARD_SESSION_TTL_SECONDS",
    "DashboardCredentials",
    "DashboardSessionStore",
    "HashedDashboardCredentials",
    "PlaintextDashboardCredentials",
]
