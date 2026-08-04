import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.main import create_app, no_lifespan
from backend.app.shared.dashboard_auth import (
    DEFAULT_DASHBOARD_PASSWORD,
    DEFAULT_DASHBOARD_USERNAME,
)
from backend.app.shared.dashboard_credentials import DashboardCredentialsStore


def _app(tmp_path, **state):
    app = create_app(lifespan=no_lifespan)
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    app.state.edge_relay_token = "worker-secret"
    for key, value in state.items():
        setattr(app.state, key, value)
    return app


def _client(tmp_path):
    """An edge box with an env-equivalent username/password pair configured
    (matches the pre-issue-#23 fixture shape: no persisted file)."""
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


def test_dashboard_routes_require_a_real_session_even_without_legacy_opt_in(
    tmp_path, monkeypatch
) -> None:
    """Dashboard auth now always resolves to a store (persisted file > env >
    the built-in admin/admin default), so the legacy worker-bearer-token
    dashboard bypass is unreachable whether or not legacy auth is opted into:
    a bare worker relay token is never a valid dashboard credential."""
    monkeypatch.delenv("API_ALLOW_LEGACY_DASHBOARD_AUTH", raising=False)
    app = _app(tmp_path)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/cameras",
            headers={"Authorization": "Bearer worker-secret"},
        )

    assert response.status_code == 401


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


# -- Default credentials + persisted rotation (issue #23) --------------------


def test_credentials_store_returns_none_on_zero_rows(tmp_path) -> None:
    """A brand-new (or freshly-migrated) ``credentials`` table has zero rows
    -- ``DashboardCredentialsStore.load()`` must return ``None`` (never
    raise), preserving the exact contract that feeds the admin/admin
    default-and-rotate login flow (issue #23 / PR #32)."""
    store = DashboardCredentialsStore(tmp_path / "catalog.sqlite3")

    assert store.load() is None

    # Persisting a credential flips the contract: subsequent loads return it.
    store.save(username="admin", password="admin")
    persisted = store.load()
    assert persisted is not None
    assert persisted.username == "admin"
    assert persisted.verify_password("admin")


def test_zero_config_edge_box_accepts_built_in_admin_default(tmp_path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": DEFAULT_DASHBOARD_USERNAME, "password": DEFAULT_DASHBOARD_PASSWORD},
        )
        assert login.status_code == 204
        assert "ml_dashboard_session=" in login.headers["set-cookie"]

        wrong = client.post(
            "/api/v1/auth/session",
            json={"username": DEFAULT_DASHBOARD_USERNAME, "password": "not-admin"},
        )
        assert wrong.status_code == 401


def test_env_pair_wins_over_built_in_default_when_no_file_is_persisted(tmp_path) -> None:
    app = _app(tmp_path, dashboard_username="operator", dashboard_password="correct horse")
    with TestClient(app) as client:
        default_rejected = client.post(
            "/api/v1/auth/session",
            json={"username": DEFAULT_DASHBOARD_USERNAME, "password": DEFAULT_DASHBOARD_PASSWORD},
        )
        env_accepted = client.post(
            "/api/v1/auth/session",
            json={"username": "operator", "password": "correct horse"},
        )

        assert default_rejected.status_code == 401
        assert env_accepted.status_code == 204


def test_credential_rotation_changes_login_and_revokes_other_sessions(tmp_path) -> None:
    store_path = tmp_path / "catalog.sqlite3"
    app = _app(tmp_path, dashboard_credentials_store=DashboardCredentialsStore(store_path))

    with TestClient(app) as bystander:
        bystander_login = bystander.post(
            "/api/v1/auth/session",
            json={"username": DEFAULT_DASHBOARD_USERNAME, "password": DEFAULT_DASHBOARD_PASSWORD},
        )
        assert bystander_login.status_code == 204
        assert bystander.get("/api/v1/cameras").status_code == 200

        with TestClient(app) as client:
            login = client.post(
                "/api/v1/auth/session",
                json={
                    "username": DEFAULT_DASHBOARD_USERNAME,
                    "password": DEFAULT_DASHBOARD_PASSWORD,
                },
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

            # TestClient persists Set-Cookie across requests on the same
            # instance, so the PUT response's cookie is already live here.
            assert client.get("/api/v1/cameras").status_code == 200

        # Old default credentials no longer work; the new pair does.
        with TestClient(app) as retry:
            old_login = retry.post(
                "/api/v1/auth/session",
                json={
                    "username": DEFAULT_DASHBOARD_USERNAME,
                    "password": DEFAULT_DASHBOARD_PASSWORD,
                },
            )
            assert old_login.status_code == 401
            new_login = retry.post(
                "/api/v1/auth/session",
                json={"username": "operator", "password": "new-secret-pw"},
            )
            assert new_login.status_code == 204

        # The bystander's pre-rotation session was revoked by the rotation.
        assert bystander.get("/api/v1/cameras").status_code == 401


def test_rotation_to_a_non_ascii_username_logs_in_and_survives_a_restart(tmp_path) -> None:
    """Regression for a non-ASCII (Korean) username crashing every future
    login with a 500: hmac.compare_digest raises TypeError for non-ASCII
    `str` arguments, so both PlaintextDashboardCredentials.verify() and
    HashedDashboardCredentials.verify() must compare UTF-8-encoded bytes,
    not `str`, directly. Exercises rotation to "관리자", a login with the new
    Korean username, a 401 (not 500) for the stale default, and a simulated
    restart (fresh store re-resolution from the same persisted file)."""
    store_path = tmp_path / "catalog.sqlite3"
    korean_username = "관리자"

    with TestClient(
        _app(tmp_path, dashboard_credentials_store=DashboardCredentialsStore(store_path))
    ) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": DEFAULT_DASHBOARD_USERNAME, "password": DEFAULT_DASHBOARD_PASSWORD},
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

        stale_default = client.post(
            "/api/v1/auth/session",
            json={"username": DEFAULT_DASHBOARD_USERNAME, "password": DEFAULT_DASHBOARD_PASSWORD},
        )
        assert stale_default.status_code == 401

    # Simulate a restart: a brand-new app/session-store resolves the Korean
    # username fresh from the same on-disk file, without crashing.
    with TestClient(
        _app(tmp_path, dashboard_credentials_store=DashboardCredentialsStore(store_path))
    ) as client:
        rejected = client.post(
            "/api/v1/auth/session",
            json={"username": DEFAULT_DASHBOARD_USERNAME, "password": DEFAULT_DASHBOARD_PASSWORD},
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
    """``PUT /auth/credentials`` no longer takes a ``current_password`` -- the
    session cookie is the sole auth gate. A request with no (or an invalid)
    session must still be rejected, and must never write a persisted
    credential row."""
    store_path = tmp_path / "catalog.sqlite3"
    app = _app(tmp_path, dashboard_credentials_store=DashboardCredentialsStore(store_path))

    with TestClient(app) as client:
        rejected = client.put(
            "/api/v1/auth/credentials",
            json={"new_password": "new-secret-pw"},
        )
        assert rejected.status_code == 401
        # The credentials table exists (created on first connect while
        # resolving credentials for the session check above) but must have
        # zero rows: a rejected rotation must never write a persisted
        # credential row.
        connection = sqlite3.connect(store_path)
        try:
            count = connection.execute("SELECT COUNT(*) FROM credentials").fetchone()[0]
        finally:
            connection.close()
        assert count == 0

        still_default = client.post(
            "/api/v1/auth/session",
            json={"username": DEFAULT_DASHBOARD_USERNAME, "password": DEFAULT_DASHBOARD_PASSWORD},
        )
        assert still_default.status_code == 204


def test_persisted_file_wins_over_env_and_default_after_store_reresolution(tmp_path) -> None:
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

    # Simulate a restart: a brand-new app/session-store resolves credentials
    # fresh from the same on-disk file plus the same env-equivalent pair.
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
    app = _app(tmp_path, dashboard_credentials_store=DashboardCredentialsStore(store_path))

    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": DEFAULT_DASHBOARD_USERNAME, "password": DEFAULT_DASHBOARD_PASSWORD},
        )
        assert login.status_code == 204
        rotate = client.put(
            "/api/v1/auth/credentials",
            json={"new_password": "new-secret-pw"},
        )
        assert rotate.status_code == 204

    assert stat.S_IMODE(store_path.stat().st_mode) == 0o600


def test_corrupt_credentials_file_falls_back_without_a_500(tmp_path) -> None:
    store_path = tmp_path / "catalog.sqlite3"
    store_path.write_bytes(b"not a sqlite database")
    app = _app(tmp_path, dashboard_credentials_store=DashboardCredentialsStore(store_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/session",
            json={"username": DEFAULT_DASHBOARD_USERNAME, "password": DEFAULT_DASHBOARD_PASSWORD},
        )

    assert response.status_code == 204


def test_compose_edge_requires_dashboard_credentials_so_default_never_ships() -> None:
    """admin/admin 기본값이 프로덕션 엣지로 나가지 않음을 배포 파일에서 고정한다.

    ``dashboard_auth.py``는 env가 비었을 때만 ``admin/admin``으로 떨어진다
    (:160-177). 그 기본값 자체는 zero-config 설치를 위해 남겨두되, 실제
    배포 경로에서는 절대 도달하지 않아야 한다.

    ``compose.edge.yaml``이 두 변수를 ``:?``로 강제하므로 값이 없으면
    컨테이너가 아예 뜨지 않는다. 누가 그 ``:?``를 지우거나 기본값을 넣으면
    조용히 admin/admin으로 뜨는 엣지가 요양원에 설치될 수 있다.
    """
    compose = (Path(__file__).resolve().parents[1] / "compose.edge.yaml").read_text()

    for var in ("API_DASHBOARD_USERNAME", "API_DASHBOARD_PASSWORD"):
        assert f"${{{var}:?" in compose, (
            f"{var} must stay required (`:?`) in compose.edge.yaml — "
            "otherwise the admin/admin fallback can reach a real edge box"
        )
