"""Shared helpers for observability E2E and real-stack load measurements."""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import uvicorn

from backend.app.core.config import get_settings
from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.features.diagnostics.retention import RetentionBudget
from backend.app.features.diagnostics.store import ExecutionRecordStore
from backend.app.main import create_app, no_lifespan

_QUERY_PATH: Final = "/api/v1/diagnostics/executions"
_SESSION_PATH: Final = "/api/v1/auth/session"
_BUILD_REVISION: Final = "observability-rev-1"
_DASHBOARD_USERNAME: Final = "admin"
_DASHBOARD_PASSWORD: Final = "admin"
_POLL_SEC: Final = 0.01


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
    database_path: Path
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
    tmp_path: Path, *, budget_bytes: int, relay_token: str
) -> Iterator[BackendUnderTest]:
    """Bootstrap schema-19 sqlite and serve ``create_app()`` on a free loopback port."""
    database = tmp_path / "observability-edge.sqlite3"
    bootstrap_database(database)
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
    try:
        app = create_app(lifespan=no_lifespan)
        app.state.edge_relay_token = relay_token
        app.state.backend_build_revision = _BUILD_REVISION
        app.state.execution_record_store = ExecutionRecordStore(
            lambda: open_runtime_database(
                database, actor=RuntimeActor.API, check_same_thread=False
            ),
            RetentionBudget(total_bytes=budget_bytes),
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
                database_path=database,
                dashboard_username=_DASHBOARD_USERNAME,
                dashboard_password=_DASHBOARD_PASSWORD,
            )
        finally:
            server.should_exit = True
            thread.join(timeout=10.0)
    finally:
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
