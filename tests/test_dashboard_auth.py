import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.main import create_app, no_lifespan
from backend.app.shared.dashboard_auth import (
    API_DASHBOARD_PASSWORD_ENV,
    API_DASHBOARD_USERNAME_ENV,
    DEFAULT_DASHBOARD_PASSWORD,
    DEFAULT_DASHBOARD_USERNAME,
)
from backend.app.shared.dashboard_credentials import (
    DashboardCredentialsStore,
    DashboardCredentialsStoreError,
)


@pytest.fixture(autouse=True)
def _migrated_compact_database(tmp_path: Path) -> None:
    migrate_database(tmp_path / "catalog.sqlite3")


def _app(tmp_path, **state):
    app = create_app(lifespan=no_lifespan)
    registry = state.pop("camera_registry", None)
    app.state.camera_registry = (
        registry
        if isinstance(registry, CameraRegistryStore)
        else CameraRegistryStore(tmp_path / "catalog.sqlite3")
    )
    app.state.edge_relay_token = "worker-secret"
    for key, value in state.items():
        setattr(app.state, key, value)
    return app


def _client(tmp_path):
    return TestClient(
        _app(tmp_path, dashboard_username="operator", dashboard_password="correct horse")
    )


def test_dashboard_login_uses_httponly_cookie_for_protected_routes(tmp_path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "correct horse"},
        )

        assert response.status_code == 204
        cookie = response.headers["set-cookie"]
        assert "ml_dashboard_session=" in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=strict" in cookie
        assert client.get("/api/v1/cameras").status_code == 200


def test_dashboard_rejects_invalid_login_and_worker_token(tmp_path) -> None:
    with _client(tmp_path) as client:
        invalid = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "wrong"},
        )
        worker_token = client.get(
            "/api/v1/cameras",
            headers={"Authorization": "Bearer worker-secret"},
        )

        assert invalid.status_code == 401
        assert worker_token.status_code == 401


def test_worker_config_keeps_dedicated_relay_auth_when_dashboard_sessions_are_enabled(
    tmp_path,
) -> None:
    with _client(tmp_path) as client:
        accepted = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "worker-secret"},
        )
        dashboard_route = client.get(
            "/api/v1/cameras",
            headers={"Authorization": "Bearer worker-secret"},
        )

        assert accepted.status_code == 200
        assert dashboard_route.status_code == 401


def test_dashboard_routes_ignore_relay_credentials_even_with_legacy_opt_in(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("API_ALLOW_LEGACY_DASHBOARD_AUTH", "1")
    app = _app(tmp_path)

    with TestClient(app) as client:
        bearer_response = client.get(
            "/api/v1/cameras",
            headers={"Authorization": "Bearer worker-secret"},
        )
        relay_header_response = client.get(
            "/api/v1/cameras",
            headers={"X-Edge-Relay-Token": "worker-secret"},
        )

    assert bearer_response.status_code == 401
    assert relay_header_response.status_code == 401


def test_concurrent_first_logins_share_one_session_store(tmp_path) -> None:
    app = _app(tmp_path, dashboard_username="operator", dashboard_password="correct horse")

    def login_and_query(_index: int) -> tuple[int, int]:
        with TestClient(app) as client:
            login = client.post(
                "/api/v1/auth/session",
                json={"username": "operator", "password": "correct horse"},
            )
            query = client.get("/api/v1/cameras")
            return login.status_code, query.status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(login_and_query, range(16)))

    assert statuses == [(204, 200)] * 16


def test_dashboard_logout_revokes_session(tmp_path) -> None:
    with _client(tmp_path) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "correct horse"},
        )
        assert login.status_code == 204
        assert client.delete("/api/v1/auth/session").status_code == 204

        assert client.get("/api/v1/cameras").status_code == 401


def test_credentials_store_returns_none_on_zero_rows(tmp_path) -> None:
    store = DashboardCredentialsStore(tmp_path / "catalog.sqlite3")

    assert store.load() is None

    store.save(username="operator", password="bootstrap-secret")
    persisted = store.load()
    assert persisted is not None
    assert persisted.username == "operator"
    assert persisted.verify_password("bootstrap-secret")


def test_zero_config_edge_box_refuses_built_in_admin_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(API_DASHBOARD_USERNAME_ENV, raising=False)
    monkeypatch.delenv(API_DASHBOARD_PASSWORD_ENV, raising=False)
    app = _app(tmp_path)
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={
                "username": DEFAULT_DASHBOARD_USERNAME,
                "password": DEFAULT_DASHBOARD_PASSWORD,
            },
        )
        assert login.status_code == 503
        assert "not configured" in login.json()["detail"]


def test_explicit_env_bootstrap_pair_accepts_login(tmp_path) -> None:
    app = _app(tmp_path, dashboard_username="operator", dashboard_password="correct horse")
    with TestClient(app) as client:
        default_rejected = client.post(
            "/api/v1/auth/session",
            json={
                "username": DEFAULT_DASHBOARD_USERNAME,
                "password": DEFAULT_DASHBOARD_PASSWORD,
            },
        )
        env_accepted = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "correct horse"},
        )

        assert default_rejected.status_code == 401
        assert env_accepted.status_code == 204


def test_credential_rotation_changes_login_and_revokes_other_sessions(tmp_path) -> None:
    store_path = tmp_path / "catalog.sqlite3"
    app = _app(
        tmp_path,
        dashboard_credentials_store=DashboardCredentialsStore(store_path),
        dashboard_username="operator",
        dashboard_password="bootstrap-secret",
    )

    with TestClient(app) as bystander:
        bystander_login = bystander.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "bootstrap-secret"},
        )
        assert bystander_login.status_code == 204
        assert bystander.get("/api/v1/cameras").status_code == 200

        with TestClient(app) as client:
            login = client.post(
                "/api/v1/auth/session",
                json={"username": "operator", "password": "bootstrap-secret"},
            )
            assert login.status_code == 204

            rotate = client.put(
                "/api/v1/auth/credentials",
                json={
                    "username": "operator",
                    "new_password": "new-secret-pw",
                },
            )
            assert rotate.status_code == 204
            assert "ml_dashboard_session=" in rotate.headers["set-cookie"]
            assert client.get("/api/v1/cameras").status_code == 200

        with TestClient(app) as retry:
            old_login = retry.post(
                "/api/v1/auth/session",
                json={"username": "operator", "password": "bootstrap-secret"},
            )
            assert old_login.status_code == 401
            new_login = retry.post(
                "/api/v1/auth/session",
                json={"username": "operator", "password": "new-secret-pw"},
            )
            assert new_login.status_code == 204

        assert bystander.get("/api/v1/cameras").status_code == 401


def test_rotation_to_a_non_ascii_username_logs_in_and_survives_a_restart(tmp_path) -> None:
    store_path = tmp_path / "catalog.sqlite3"
    korean_username = "관리자"

    with TestClient(
        _app(
            tmp_path,
            dashboard_credentials_store=DashboardCredentialsStore(store_path),
            dashboard_username="operator",
            dashboard_password="bootstrap-secret",
        )
    ) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "bootstrap-secret"},
        )
        assert login.status_code == 204

        rotate = client.put(
            "/api/v1/auth/credentials",
            json={
                "username": korean_username,
                "new_password": "new-secret-pw",
            },
        )
        assert rotate.status_code == 204

        new_username_login = client.post(
            "/api/v1/auth/session",
            json={"username": korean_username, "password": "new-secret-pw"},
        )
        assert new_username_login.status_code == 204

        stale_bootstrap = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "bootstrap-secret"},
        )
        assert stale_bootstrap.status_code == 401

    with TestClient(
        _app(
            tmp_path,
            dashboard_credentials_store=DashboardCredentialsStore(store_path),
            dashboard_username="operator",
            dashboard_password="bootstrap-secret",
        )
    ) as client:
        rejected = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "bootstrap-secret"},
        )
        accepted = client.post(
            "/api/v1/auth/session",
            json={"username": korean_username, "password": "new-secret-pw"},
        )

        assert rejected.status_code == 401
        assert accepted.status_code == 204


def test_unauthenticated_credential_rotation_is_rejected_without_touching_the_store(
    tmp_path,
) -> None:
    store_path = tmp_path / "catalog.sqlite3"
    app = _app(
        tmp_path,
        dashboard_credentials_store=DashboardCredentialsStore(store_path),
        dashboard_username="operator",
        dashboard_password="bootstrap-secret",
    )

    with TestClient(app) as client:
        rejected = client.put(
            "/api/v1/auth/credentials",
            json={"new_password": "new-secret-pw"},
        )
        assert rejected.status_code == 401
        connection = sqlite3.connect(store_path)
        try:
            count = connection.execute("SELECT COUNT(*) FROM credentials").fetchone()[0]
        finally:
            connection.close()
        assert count == 0

        still_bootstrap = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "bootstrap-secret"},
        )
        assert still_bootstrap.status_code == 204


def test_persisted_file_wins_over_env_after_store_reresolution(tmp_path) -> None:
    store_path = tmp_path / "catalog.sqlite3"

    with TestClient(
        _app(
            tmp_path,
            dashboard_credentials_store=DashboardCredentialsStore(store_path),
            dashboard_username="operator",
            dashboard_password="correct horse",
        )
    ) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "correct horse"},
        )
        assert login.status_code == 204
        rotate = client.put(
            "/api/v1/auth/credentials",
            json={"new_password": "rotated-pw"},
        )
        assert rotate.status_code == 204

    with TestClient(
        _app(
            tmp_path,
            dashboard_credentials_store=DashboardCredentialsStore(store_path),
            dashboard_username="operator",
            dashboard_password="correct horse",
        )
    ) as client:
        env_rejected = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "correct horse"},
        )
        file_accepted = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "rotated-pw"},
        )

        assert env_rejected.status_code == 401
        assert file_accepted.status_code == 204


def test_persisted_credentials_file_is_written_with_mode_0600(tmp_path) -> None:
    store_path = tmp_path / "catalog.sqlite3"
    app = _app(
        tmp_path,
        dashboard_credentials_store=DashboardCredentialsStore(store_path),
        dashboard_username="operator",
        dashboard_password="bootstrap-secret",
    )

    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "bootstrap-secret"},
        )
        assert login.status_code == 204
        rotate = client.put(
            "/api/v1/auth/credentials",
            json={"new_password": "new-secret-pw"},
        )
        assert rotate.status_code == 204

    assert stat.S_IMODE(store_path.stat().st_mode) == 0o600


def test_corrupt_credentials_store_fails_closed_without_env_fallback(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "catalog.sqlite3"
    store_path.write_bytes(b"not a sqlite database")
    monkeypatch.setenv(API_DASHBOARD_USERNAME_ENV, "operator")
    monkeypatch.setenv(API_DASHBOARD_PASSWORD_ENV, "env-secret-should-not-apply")
    app = _app(
        tmp_path,
        dashboard_credentials_store=DashboardCredentialsStore(store_path),
        camera_registry=CameraRegistryStore(tmp_path / ".central-fixture" / "edge.sqlite3"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "env-secret-should-not-apply"},
        )

    assert response.status_code == 503
    assert "unreadable" in response.json()["detail"]


def test_credentials_store_load_raises_on_corrupt_file(tmp_path) -> None:
    store_path = tmp_path / "catalog.sqlite3"
    store_path.write_bytes(b"not a sqlite database")
    store = DashboardCredentialsStore(store_path)
    try:
        store.load()
        raised = False
    except DashboardCredentialsStoreError:
        raised = True
    assert raised


def test_compose_edge_requires_dashboard_and_rtsp_policy_flags() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (root / "compose.edge.yaml").read_text()
    example = (root / ".env.edge.prod.example").read_text()

    for var in ("API_DASHBOARD_USERNAME", "API_DASHBOARD_PASSWORD"):
        assert f"${{{var}:?" in compose

    assert "ML_RTSP_ALLOW_PRIVATE_DESTINATIONS" in compose
    assert "ML_RTSP_ALLOW_LOCAL_DESTINATIONS" in compose
    assert "API_DASHBOARD_USERNAME=admin\nAPI_DASHBOARD_PASSWORD=admin" not in example
    assert "<random-bootstrap-password>" in example
    assert "ML_RTSP_ALLOW_PRIVATE_DESTINATIONS=1" in example
    assert "ML_RTSP_ALLOW_LOCAL_DESTINATIONS=0" in example
