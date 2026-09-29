"""The PostgreSQL composition root: refuse missing wiring, serve through PG, close on shutdown."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.postgres import PoolBudget, PostgresError
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.main import create_app
from backend.app.postgres_root import (
    API_POSTGRES_AUTHORITY_FILE_ENV,
    API_POSTGRES_DSN_FILE_ENV,
    API_POSTGRES_SCHEMA_ENV,
    PostgresRoot,
    PostgresRootError,
    open_postgres_root,
)
from tests_support.connection_api import closed_port
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox", "tests_support.postgres_app_env")

FAST_BUDGET = PoolBudget(
    max_connections=1,
    max_waiting=1,
    acquire_timeout_sec=1.0,
    statement_timeout_ms=1000,
    lock_timeout_ms=1000,
    startup_timeout_sec=1.0,
)
PASSWORD_MARKER = "root-test-password-marker"
ROOT_ENV_NAMES = (
    API_POSTGRES_DSN_FILE_ENV,
    API_POSTGRES_AUTHORITY_FILE_ENV,
    API_POSTGRES_SCHEMA_ENV,
)


def _secret(directory: Path, name: str, text: str) -> str:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _authority_text(generation: int, writer_token: str) -> str:
    return json.dumps({"generation": generation, "writer_token": writer_token})


def _unreachable_dsn() -> str:
    return f"postgresql://seeon:{PASSWORD_MARKER}@127.0.0.1:{closed_port()}/seeon?connect_timeout=1"


def _assert_unchained(error: PostgresRootError) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize(
    ("dsn_file", "expected"),
    [
        (None, "not configured; set API_POSTGRES_DSN_FILE"),
        ("missing", "named by API_POSTGRES_DSN_FILE is unreadable"),
        ("", "named by API_POSTGRES_DSN_FILE is empty"),
    ],
)
def test_startup_refuses_without_a_readable_dsn_file(
    tmp_path: Path, dsn_file: str | None, expected: str
) -> None:
    environ = {
        API_POSTGRES_AUTHORITY_FILE_ENV: _secret(
            tmp_path, "authority.json", _authority_text(1, str(uuid4()))
        )
    }
    if dsn_file == "missing":
        environ[API_POSTGRES_DSN_FILE_ENV] = str(tmp_path / "absent.dsn")
    elif dsn_file is not None:
        environ[API_POSTGRES_DSN_FILE_ENV] = _secret(tmp_path, "postgres.dsn", dsn_file)

    with pytest.raises(PostgresRootError, match=expected):
        open_postgres_root(environ, budget=FAST_BUDGET)


@pytest.mark.parametrize(
    "authority_text",
    [
        "not json",
        json.dumps({"generation": 1}),
        _authority_text(1, "not-a-uuid"),
        json.dumps({"generation": 1, "writer_token": str(uuid4()), "extra": True}),
    ],
)
def test_startup_refuses_a_malformed_authority_file(tmp_path: Path, authority_text: str) -> None:
    environ = {
        API_POSTGRES_DSN_FILE_ENV: _secret(tmp_path, "postgres.dsn", _unreachable_dsn()),
        API_POSTGRES_AUTHORITY_FILE_ENV: _secret(tmp_path, "authority.json", authority_text),
    }

    with pytest.raises(PostgresRootError, match="API_POSTGRES_AUTHORITY_FILE is invalid") as info:
        open_postgres_root(environ, budget=FAST_BUDGET)

    assert authority_text not in str(info.value)


def test_startup_refuses_an_unreachable_server_without_leaking_the_dsn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dsn = _unreachable_dsn()
    environ = {
        API_POSTGRES_DSN_FILE_ENV: _secret(tmp_path, "postgres.dsn", dsn),
        API_POSTGRES_AUTHORITY_FILE_ENV: _secret(
            tmp_path, "authority.json", _authority_text(1, str(uuid4()))
        ),
    }
    caplog.set_level(logging.DEBUG)

    with pytest.raises(PostgresRootError, match="unreachable within the startup budget") as info:
        open_postgres_root(environ, budget=FAST_BUDGET)

    assert PASSWORD_MARKER not in str(info.value)
    assert dsn not in str(info.value)
    _assert_unchained(info.value)
    assert PASSWORD_MARKER not in caplog.text


def test_startup_refuses_a_schema_provisioned_for_another_authority(
    tmp_path: Path, postgres_product_sandbox: ProductSandbox
) -> None:
    sandbox = postgres_product_sandbox
    foreign_token = str(uuid4())
    environ = {
        API_POSTGRES_DSN_FILE_ENV: _secret(tmp_path, "postgres.dsn", sandbox.dsn),
        API_POSTGRES_AUTHORITY_FILE_ENV: _secret(
            tmp_path, "authority.json", _authority_text(sandbox.authority.generation, foreign_token)
        ),
        API_POSTGRES_SCHEMA_ENV: sandbox.schema,
    }

    with pytest.raises(PostgresRootError, match="not provisioned for this deployment") as info:
        open_postgres_root(environ, budget=FAST_BUDGET)

    assert sandbox.dsn not in str(info.value)
    assert foreign_token not in str(info.value)
    _assert_unchained(info.value)


def test_real_app_startup_refuses_without_postgres_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ROOT_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    app = create_app()

    with pytest.raises(PostgresRootError, match="set API_POSTGRES_DSN_FILE"), TestClient(app):
        pass

    assert not hasattr(app.state, "camera_registry")


def test_assembled_app_serves_a_camera_seeded_through_the_postgres_store(
    monkeypatch: pytest.MonkeyPatch, postgres_app_env: ProductSandbox
) -> None:
    CameraRegistryStore(postgres_app_env.database, postgres_app_env.authority).create(
        camera_id="root-camera",
        label="Root Camera",
        rtsp_url="rtsp://root/stream",
        space_id=None,
        status="online",
    )
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")

    with TestClient(create_app()) as client:
        response = client.get(
            "/api/v1/cameras/worker-config", headers={"X-Edge-Relay-Token": "relay-token"}
        )

    assert response.status_code == 200
    assert [(camera["camera_id"], camera["rtsp_url"]) for camera in response.json()["cameras"]] == [
        ("root-camera", "rtsp://root/stream")
    ]


@pytest.mark.usefixtures("postgres_app_env")
def test_shutdown_closes_the_owned_pool_and_withdraws_the_root() -> None:
    app = create_app()

    with TestClient(app):
        root = app.state.postgres_root
        assert root.database.read(lambda connection: connection.execute("SELECT 1").fetchone())

    with pytest.raises(PostgresError):
        root.database.read(lambda connection: connection.execute("SELECT 1").fetchone())
    assert not hasattr(app.state, "postgres_root")
    assert not hasattr(app.state, "camera_registry")


def test_shutdown_leaves_an_injected_root_open_for_its_owner(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    app = create_app()
    app.state.postgres_root = PostgresRoot(sandbox.database, sandbox.authority)

    with TestClient(app):
        pass

    assert sandbox.database.read(lambda connection: connection.execute("SELECT 1").fetchone()) == (
        1,
    )
