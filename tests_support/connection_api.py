from __future__ import annotations

import json
import socket
import time
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Protocol, TypeAlias, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from typing_extensions import override

from backend.app.core.config import get_settings
from backend.app.features.connection.store import (
    API_BACKEND_BASE_URL_ENV,
    API_BACKEND_CONFIG_URL_ENV,
    API_BACKEND_EVENTS_URL_ENV,
    API_CONNECTION_SETTINGS_PATH_ENV,
    ConnectionSettingsStore,
)
from backend.app.lifespan import API_EDGE_RELAY_TOKEN_ENV
from backend.app.main import create_app, no_lifespan

DASHBOARD_LOGIN: Mapping[str, JsonValue] = {
    "username": "admin",
    "password": "admin",
}
TEST_TIMEOUT_ENV = "ML_API_CONNECTION_TEST_TIMEOUT_S"

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None


class ConnectionTestClient(Protocol):
    app: FastAPI

    def get(self, url: str) -> Response: ...

    def post(self, url: str, *, json: Mapping[str, JsonValue]) -> Response: ...

    def put(self, url: str, *, json: Mapping[str, JsonValue]) -> Response: ...


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        API_CONNECTION_SETTINGS_PATH_ENV,
        API_BACKEND_BASE_URL_ENV,
        API_BACKEND_EVENTS_URL_ENV,
        API_BACKEND_CONFIG_URL_ENV,
        "API_FACILITY_ID",
        "EDGE_FACILITY_TOKEN",
        API_EDGE_RELAY_TOKEN_ENV,
        TEST_TIMEOUT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def login(client: ConnectionTestClient) -> None:
    response = client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN)
    assert response.status_code == 204


def connection_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, test_timeout_s: float = 0.3
) -> ConnectionTestClient:
    monkeypatch.setenv(API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection_settings.json"))
    monkeypatch.setenv(TEST_TIMEOUT_ENV, str(test_timeout_s))
    get_settings.cache_clear()
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    return cast(ConnectionTestClient, TestClient(app))


def connection_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ConnectionSettingsStore:
    monkeypatch.setenv(API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection_settings.json"))
    return ConnectionSettingsStore.from_env()


def response_json(response: Response) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], response.json())


class OKHandler(BaseHTTPRequestHandler):
    received_auth: list[str | None] = []

    def do_GET(self) -> None:  # noqa: N802
        self.__class__.received_auth.append(self.headers.get("Authorization"))
        body = json.dumps({"configVersion": 1, "cameras": []}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        _ = self.wfile.write(body)

    @override
    def log_message(self, format: str, *args: str) -> None:
        return


class AuthFailHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(401)
        self.end_headers()

    @override
    def log_message(self, format: str, *args: str) -> None:
        return


class EnrollmentVerifyHandler(BaseHTTPRequestHandler):
    received_auth: list[str | None] = []
    received_paths: list[str] = []
    received_bodies: list[dict[str, JsonValue]] = []
    response_status = 200

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = cast(dict[str, JsonValue], json.loads(self.rfile.read(length)))
        self.__class__.received_auth.append(self.headers.get("Authorization"))
        self.__class__.received_paths.append(self.path)
        self.__class__.received_bodies.append(body)
        response = {
            "schemaVersion": 1,
            "edgeInstallationId": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
            "enrollmentGeneration": 3,
            "facility": {
                "id": "87d79f24-b32f-49a3-b534-19f0af7d9135",
                "displayName": "Test Facility",
            },
            "serverRevision": 7,
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(self.__class__.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        _ = self.wfile.write(encoded)

    @classmethod
    def reset(cls) -> None:
        cls.received_auth = []
        cls.received_paths = []
        cls.received_bodies = []
        cls.response_status = 200

    @override
    def log_message(self, format: str, *args: str) -> None:
        return


class HangHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        time.sleep(5)

    @override
    def log_message(self, format: str, *args: str) -> None:
        return


def run_server(server: ThreadingHTTPServer) -> Thread:
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        address = cast(tuple[str, int], sock.getsockname())
        return address[1]
