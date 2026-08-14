from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Final, Protocol, Self, cast

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from backend.app.features.connection.store import ConnectionSettingsStore

_CLOUD_ENV = (
    "CLOUD_EDGE_ML_URL",
    "CLOUD_EDGE_RELAY_TOKEN",
    "CLOUD_EDGE_ML_CATALOG_PATH",
    "CLOUD_EDGE_PRE_V1_BACKUP_PATH",
    "CLOUD_EDGE_SECRET_HANDOFF_PATH",
)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        any(not os.environ.get(name) for name in _CLOUD_ENV),
        reason="cloud edge integration environment is not configured",
    ),
]

_RELAY_HEADER: Final = "X-Edge-Relay-Token"
_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])


class UrlResponse(Protocol):
    headers: Message

    def read(self) -> bytes: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class SecretHandoff(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    facilityCode: str
    token: str
    installationRef: str


def test_wal_safe_restore_reenrolls_and_reconciles_server_revision() -> None:
    # Given: the real ML API is enrolled and has accepted topology state.
    ml_url = _required("CLOUD_EDGE_ML_URL")
    relay_token = _required("CLOUD_EDGE_RELAY_TOKEN")
    dashboard_cookie = _dashboard_cookie(ml_url)
    database = Path(_required("CLOUD_EDGE_ML_CATALOG_PATH"))
    backup = Path(_required("CLOUD_EDGE_PRE_V1_BACKUP_PATH"))
    handoff = SecretHandoff.model_validate_json(
        Path(_required("CLOUD_EDGE_SECRET_HANDOFF_PATH")).read_text()
    )
    before = _request_json(
        f"{ml_url}/api/v1/connection",
        relay_token=relay_token,
        dashboard_cookie=dashboard_cookie,
    )
    assert before["enrolled"] is True
    with sqlite3.connect(database) as connection:
        table_rows = cast(
            list[tuple[str]],
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall(),
        )
        tables = {row[0] for row in table_rows}
        assert "edge_topology_sync_state" in tables, (
            f"topology state missing from {database}: {sorted(tables)}"
        )
        _ = connection.execute(
            "CREATE TABLE IF NOT EXISTS task15_unrelated (value TEXT NOT NULL)"
        )
        _ = connection.execute("DELETE FROM task15_unrelated")
        _ = connection.execute(
            "INSERT INTO task15_unrelated(value) VALUES ('preserved')"
        )
        state = cast(
            tuple[int] | None,
            connection.execute(
                "SELECT server_revision FROM edge_topology_sync_state WHERE id = 1"
            ).fetchone(),
        )
    assert state is not None
    server_revision = int(state[0])

    # When: the exact pre-v1 online backup is selectively restored.
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    expected_hash = _required("CLOUD_EDGE_PRE_V1_BACKUP_SHA256")
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == expected_hash
    ConnectionSettingsStore(database).restore_pre_v1_backup(backup)

    # Then: unrelated catalog state survives and runtime re-enrollment converges.
    restored = ConnectionSettingsStore(database).load()
    assert restored.facility_code is None
    assert restored.facility_token is None
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT value FROM task15_unrelated").fetchone() == (
            "preserved",
        )
    payload = {
        "facility_code": handoff.facilityCode,
        "facility_token": handoff.token,
        "client_installation_ref": handoff.installationRef,
    }
    enrolled = _request_json(
        f"{ml_url}/api/v1/connection",
        method="PUT",
        relay_token=relay_token,
        dashboard_cookie=dashboard_cookie,
        payload=payload,
    )
    assert enrolled["enrolled"] is True
    sync = _request_json(
        f"{ml_url}/api/v1/connection/sync-cameras",
        method="POST",
        relay_token=relay_token,
        dashboard_cookie=dashboard_cookie,
    )
    assert sync["status"] in {"synced", "pending"}
    with sqlite3.connect(database) as connection:
        reconciled = cast(
            tuple[int] | None,
            connection.execute(
                "SELECT server_revision FROM edge_topology_sync_state WHERE id = 1"
            ).fetchone(),
        )
    assert reconciled is not None
    assert int(reconciled[0]) >= server_revision


def _request_json(
    url: str,
    *,
    method: str = "GET",
    relay_token: str,
    dashboard_cookie: str,
    payload: dict[str, str] | None = None,
) -> dict[str, JsonValue]:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            _RELAY_HEADER: relay_token,
            "Cookie": dashboard_cookie,
            "Content-Type": "application/json",
        },
    )
    try:
        response = cast(
            UrlResponse,
            urllib.request.urlopen(request, timeout=10),  # noqa: S310
        )
        with response:
            parsed = _JSON_OBJECT_ADAPTER.validate_json(response.read())
    except urllib.error.HTTPError as error:
        error_body: bytes = error.read()
        pytest.fail(
            f"{method} {url} failed with HTTP {error.code}: {error_body.decode()}",
            pytrace=False,
        )
    return parsed


def _dashboard_cookie(ml_url: str) -> str:
    payload = json.dumps(
        {
            "username": _required("CLOUD_EDGE_ML_DASHBOARD_USERNAME"),
            "password": _required("CLOUD_EDGE_ML_DASHBOARD_PASSWORD"),
        }
    ).encode()
    request = urllib.request.Request(
        f"{ml_url}/api/v1/auth/session",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    response = cast(
        UrlResponse,
        urllib.request.urlopen(request, timeout=10),  # noqa: S310
    )
    with response:
        header = response.headers.get("Set-Cookie")
    if header is None:
        raise RuntimeError("dashboard session cookie missing")
    return header.split(";", maxsplit=1)[0]


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"{name} is required")
    return value
