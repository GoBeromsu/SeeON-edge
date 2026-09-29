"""Shared helpers for observability E2E and real-stack load measurements."""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import uvicorn

from backend.app.core.config import get_settings
from backend.app.edge_db.postgres import PoolBudget, PostgresDatabase
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.diagnostics.retention import RetentionBudget
from backend.app.features.diagnostics.store import ExecutionRecordStore
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

_QUERY_PATH: Final = "/api/v1/diagnostics/executions"
_SESSION_PATH: Final = "/api/v1/auth/session"
_BUILD_REVISION: Final = "observability-rev-1"
_DASHBOARD_USERNAME: Final = "admin"
_DASHBOARD_PASSWORD: Final = "admin"
_POLL_SEC: Final = 0.01
_DIAGNOSTICS_POOL: Final = PoolBudget(
    max_connections=4,
    max_waiting=8,
    acquire_timeout_sec=1.0,
    statement_timeout_ms=5000,
    lock_timeout_ms=3000,
    startup_timeout_sec=5.0,
)


def _free_tcp_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_until(predicate: Callable[[], bool], *, timeout: float, what: str) -> None:
    """Poll ``predicate`` until true or raise ``AssertionError`` naming ``what``."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(_POLL_SEC)


def mediamtx_available() -> bool:
    return shutil.which("mediamtx") is not None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def deepstream_available() -> bool:
    return importlib.util.find_spec("pyservicemaker") is not None


@dataclass
class BackendUnderTest:
    """Live uvicorn Backend with execution-record ingest and dashboard query."""

    base_url: str
    relay_token: str
    dashboard_username: str
    dashboard_password: str

    def dashboard_session(self) -> httpx.Client:
        client = httpx.Client(base_url=self.base_url)
        response = client.post(
            _SESSION_PATH,
            json={"username": self.dashboard_username, "password": self.dashboard_password},
        )
        if response.status_code != 204:
            client.close()
            raise AssertionError(f"dashboard login failed: {response.status_code} {response.text}")
        return client

    def query(
        self,
        camera_id: str,
        from_ns: int,
        to_ns: int,
        *,
        limit: int = 500,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, str | int] = {
            "camera_id": camera_id,
            "from_ns": from_ns,
            "to_ns": to_ns,
            "limit": limit,
        }
        if cursor is not None:
            params["cursor"] = cursor
        client = self.dashboard_session()
        try:
            response = client.get(_QUERY_PATH, params=params)
            if response.status_code != 200:
                raise AssertionError(
                    f"executions query failed: {response.status_code} {response.text}"
                )
            payload = response.json()
        finally:
            client.close()
        if not isinstance(payload, dict):
            raise TypeError("executions query did not return a JSON object")
        return payload


def _restore_environ(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextmanager
def serve_backend(
    tmp_path: Path,
    *,
    budget_bytes: int,
    relay_token: str,
    sandbox: ProductSandbox,
    audit_runtime: PostgresAuditRuntime,
    diagnostics_schema: str,
) -> Iterator[BackendUnderTest]:
    """Serve the no-lifespan app on the PostgreSQL sandbox root on a free loopback port."""
    previous = {
        key: os.environ.get(key)
        for key in (
            "ML_API_EXECUTION_RECORDS_ENABLED",
            "ML_API_EXECUTION_RECORDS_BUDGET_BYTES",
            "ML_API_BUILD_REVISION",
            "API_EDGE_RELAY_TOKEN",
            "API_DASHBOARD_USERNAME",
            "API_DASHBOARD_PASSWORD",
            "API_BACKEND_HEARTBEAT_RELAY_SEC",
        )
    }
    os.environ["ML_API_EXECUTION_RECORDS_ENABLED"] = "1"
    os.environ["ML_API_EXECUTION_RECORDS_BUDGET_BYTES"] = str(budget_bytes)
    os.environ["ML_API_BUILD_REVISION"] = _BUILD_REVISION
    os.environ["API_EDGE_RELAY_TOKEN"] = relay_token
    os.environ["API_DASHBOARD_USERNAME"] = _DASHBOARD_USERNAME
    os.environ["API_DASHBOARD_PASSWORD"] = _DASHBOARD_PASSWORD
    os.environ["API_BACKEND_HEARTBEAT_RELAY_SEC"] = "0"
    get_settings.cache_clear()
    port = _free_tcp_port()
    diagnostics = ExitStack()
    try:
        app = postgres_api_app(sandbox, audit_runtime)
        app.state.edge_relay_token = relay_token
        app.state.backend_build_revision = _BUILD_REVISION
        # The caller owns the schema, so a second serve reopens the same rows.
        database = PostgresDatabase(sandbox.dsn, diagnostics_schema, _DIAGNOSTICS_POOL)
        database.start()
        diagnostics.callback(database.close, timeout_sec=3.0)
        app.state.execution_record_store = ExecutionRecordStore(
            database, RetentionBudget(total_bytes=budget_bytes)
        )
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True, name="observability-backend")
        thread.start()
        try:
            wait_until(lambda: server.started, timeout=10.0, what="observability uvicorn startup")
            yield BackendUnderTest(
                base_url=f"http://127.0.0.1:{port}",
                relay_token=relay_token,
                dashboard_username=_DASHBOARD_USERNAME,
                dashboard_password=_DASHBOARD_PASSWORD,
            )
        finally:
            server.should_exit = True
            thread.join(timeout=10.0)
    finally:
        diagnostics.close()
        _restore_environ(previous)
        get_settings.cache_clear()


__all__ = [
    "BackendUnderTest",
    "deepstream_available",
    "ffmpeg_available",
    "mediamtx_available",
    "serve_backend",
    "wait_until",
]
