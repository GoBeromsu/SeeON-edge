"""Native auth HTTP and cache fencing with explicit owners; no lifespan claim."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Lock
from types import SimpleNamespace

import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient

from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.catalog import AuditAction
from backend.app.features.auth.router import (
    DashboardCredentialsUpdateRequest,
    DashboardLoginRequest,
    login,
    update_credentials,
)
from backend.app.features.clips.router import clip_video
from backend.app.features.clips.store import ClipStore
from backend.app.main import create_app, no_lifespan
from backend.app.shared import dashboard_auth
from backend.app.shared.dashboard_auth import DASHBOARD_SESSION_COOKIE, DashboardSessionStore
from backend.app.shared.dashboard_credentials import PersistedDashboardCredentials
from backend.app.shared.postgres_dashboard_credentials import PostgresDashboardCredentialsStore

pytest_plugins = ("tests_support.postgres_sandbox",)
_USER, _PASSWORD, _NEW = "auth-operator", "bootstrap-password", "rotated-password"


@pytest.fixture
def setup(postgres_product_sandbox, postgres_audit_runtime, monkeypatch):
    sandbox = postgres_product_sandbox
    app = create_app(lifespan=no_lifespan)
    store = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    app.state.dashboard_credentials_store = store
    app.state.audit_runtime = postgres_audit_runtime
    app.state.dashboard_username, app.state.dashboard_password = _USER, _PASSWORD

    def no_sqlite(*args, **kwargs):
        pytest.fail("native authentication attempted a SQLite connection")

    with TestClient(app) as client, monkeypatch.context() as guard:
        guard.setattr(sqlite3, "connect", no_sqlite)
        yield SimpleNamespace(
            sandbox=sandbox, runtime=postgres_audit_runtime, app=app, client=client, store=store
        )


def _request(setup, token=None, *, method="PUT", path="/api/v1/auth/credentials"):
    return Request(
        {
            "type": "http",
            "app": setup.app,
            "method": method,
            "path": path,
            "scheme": "http",
            "headers": []
            if token is None
            else [(b"cookie", f"{DASHBOARD_SESSION_COOKIE}={token}".encode())],
        }
    )


def _login(setup, username=_USER, password=_PASSWORD):
    response = setup.client.post(
        "/api/v1/auth/session", json={"username": username, "password": password}
    )
    assert response.status_code == 204
    return setup.client.cookies.get(DASHBOARD_SESSION_COOKIE)


def _actions(setup):
    return setup.sandbox.admin.execute(
        "SELECT action FROM audit_events ORDER BY audit_id"
    ).fetchall()


def _rotate(setup, **overrides):
    return setup.client.put("/api/v1/auth/credentials", json={"new_password": _NEW, **overrides})


def test_native_login_session_logout_do_not_start_or_heal_other_owners(setup):
    readiness = {"ready": False, "reason": "unrelated subsystem"}
    setup.app.state.readiness = readiness
    token = _login(setup)
    assert token and setup.client.get("/api/v1/auth/session").status_code == 204
    assert setup.client.delete("/api/v1/auth/session").status_code == 204
    assert setup.client.get("/api/v1/auth/session").status_code == 401
    assert _actions(setup) == [
        (action,)
        for action in (
            AuditAction.AUDIT_SESSION_START,
            AuditAction.AUTH_LOGIN,
            AuditAction.AUTH_SESSION_READ,
            AuditAction.AUTH_LOGOUT,
        )
    ]
    assert setup.app.state.readiness is readiness
    for name in ("audit_store", "audit_readiness", "audit_checkpoint"):
        assert not hasattr(setup.app.state, name)


def test_rotation_publishes_after_owner_return_before_local_swap_and_mint(setup, monkeypatch):
    old_token = _login(setup)
    sessions, trace = setup.app.state.dashboard_sessions, []
    save, publish = setup.store.save, setup.runtime.publish_committed
    swap, authenticate = (
        DashboardSessionStore.rotate_credentials,
        DashboardSessionStore.authenticate,
    )

    def saving(**kwargs):
        append = kwargs["after_write"]

        def hook(connection):
            assert (
                setup.sandbox.admin.execute("SELECT username FROM credentials").fetchone() is None
            )
            trace.append("hook")
            append(connection)

        kwargs["after_write"] = hook
        result = save(**kwargs)
        trace.append("owner-return")
        return result

    def publication(token):
        assert trace == ["hook", "owner-return"]
        assert setup.sandbox.admin.execute("SELECT username FROM credentials").fetchone() == (
            "new-operator",
        )
        assert _actions(setup)[-1] == (AuditAction.CREDENTIAL_ROTATE,)
        trace.append("publication")
        return publish(token)

    def swapping(self, persisted):
        assert trace == ["hook", "owner-return", "publication"]
        trace.append("swap")
        return swap(self, persisted)

    def minting(self, username, password):
        assert trace == ["hook", "owner-return", "publication", "swap"]
        trace.append("mint")
        return authenticate(self, username, password)

    monkeypatch.setattr(setup.store, "save", saving)
    monkeypatch.setattr(setup.runtime, "publish_committed", publication)
    monkeypatch.setattr(DashboardSessionStore, "rotate_credentials", swapping)
    monkeypatch.setattr(DashboardSessionStore, "authenticate", minting)
    response = _rotate(setup, username="new-operator")
    assert response.status_code == 204
    assert trace == ["hook", "owner-return", "publication", "swap", "mint"]
    new_token = setup.client.cookies.get(DASHBOARD_SESSION_COOKIE)
    assert new_token != old_token and sessions.actor(old_token) is None
    assert sessions.actor(new_token) == "new-operator"
    assert setup.store.load().verify_password(_NEW)
    assert not setup.runtime._pending


@pytest.mark.parametrize("mode", ["missing", "stopped", "expired", "foreign"])
def test_login_failure_revokes_only_the_new_unpublished_token(setup, monkeypatch, mode):
    old_token = _login(setup)
    sessions, before = setup.app.state.dashboard_sessions, _actions(setup)
    if mode == "missing":
        del setup.app.state.audit_runtime
    elif mode == "stopped":
        setup.runtime.stop()
    elif mode == "expired":
        monkeypatch.setattr(setup.runtime, "_clock", lambda: 11.0)
    else:
        setup.app.state.dashboard_credentials_store = PostgresDashboardCredentialsStore(
            setup.sandbox.database, replace(setup.sandbox.authority, generation=2)
        )
    if mode == "foreign":
        with pytest.raises(ValueError, match="share database and authority"):
            _login(setup)
    else:
        response = setup.client.post(
            "/api/v1/auth/session", json={"username": _USER, "password": _PASSWORD}
        )
        assert (response.status_code, response.content) == (503, b"")
        assert "set-cookie" not in response.headers
    assert set(sessions._sessions) == {old_token}
    assert _actions(setup) == before
    assert not hasattr(setup.app.state, "audit_store")


@pytest.mark.parametrize("mode", ["schema_missing", "malformed_row"])
def test_unreadable_credentials_never_fall_back_to_bootstrap(setup, mode):
    if mode == "schema_missing":
        setup.sandbox.admin.execute("ALTER TABLE credentials RENAME TO detached_credentials")
    else:
        setup.store.save(username="persisted-operator", password=_NEW)
        # Bypass only the CHECK for a negative read-integrity control.
        setup.sandbox.admin.execute(
            "ALTER TABLE credentials DROP CONSTRAINT credentials_algorithm_check"
        )
        setup.sandbox.admin.execute("UPDATE credentials SET algorithm='invalid'")
    before = _actions(setup)
    response = setup.client.post(
        "/api/v1/auth/session", json={"username": _USER, "password": _PASSWORD}
    )
    assert (response.status_code, response.content) == (503, b"")
    assert "set-cookie" not in response.headers and _actions(setup) == before
    assert not hasattr(setup.app.state, "dashboard_sessions")
    assert not setup.runtime.snapshot().ready


def test_persisted_native_credentials_override_bootstrap(setup):
    setup.store.save(username="persisted-operator", password=_NEW)
    denied = setup.client.post(
        "/api/v1/auth/session", json={"username": _USER, "password": _PASSWORD}
    )
    assert denied.status_code == 401
    assert _login(setup, "persisted-operator", _NEW)
    assert _actions(setup) == [(AuditAction.AUDIT_SESSION_START,), (AuditAction.AUTH_LOGIN,)]


@pytest.mark.parametrize("mode", ["rollback", "unknown", "ordinary", "cancel"])
def test_failed_rotation_retires_cached_authority_without_claiming_rollback(
    setup, monkeypatch, mode
):
    old_token = _login(setup)
    old_sessions, before = setup.app.state.dashboard_sessions, _actions(setup)
    transact, reached = setup.sandbox.database.transact, []
    error = (
        CommitOutcomeUnknown()
        if mode == "unknown"
        else (
            OSError("owner exit") if mode == "ordinary" else KeyboardInterrupt("owner cancellation")
        )
    )
    if mode == "rollback":
        setup.sandbox.admin.execute(
            "CREATE FUNCTION reject_credentials() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'private credential detail' USING ERRCODE='23514'; END $$"
        )
        setup.sandbox.admin.execute(
            "CREATE CONSTRAINT TRIGGER reject_credentials AFTER INSERT OR UPDATE ON credentials "
            "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_credentials()"
        )
    else:

        def failed_owner(body):
            transact(body)
            reached.append(True)
            raise error

        monkeypatch.setattr(setup.sandbox.database, "transact", failed_owner)
    if mode in {"ordinary", "cancel"}:
        # Call the real synchronous handler to retain BaseException identity
        # without imposing ASGI task-group cancellation semantics on this check.
        request = _request(setup, old_token)
        with pytest.raises(type(error)) as caught:
            update_credentials(
                DashboardCredentialsUpdateRequest(new_password=_NEW), request, Response()
            )
        assert caught.value is error
    else:
        response = _rotate(setup)
        assert (response.status_code, response.content) == (503, b"")
        assert "set-cookie" not in response.headers
    assert old_sessions.actor(old_token) is None
    assert old_sessions.authenticate(_USER, _PASSWORD) is None
    assert not hasattr(setup.app.state, "dashboard_sessions")
    stored = setup.store.load()
    if mode == "rollback":
        assert stored is None and _actions(setup) == before and not reached
        setup.sandbox.admin.execute("DROP TRIGGER reject_credentials ON credentials")
        assert setup.runtime.verify_once()
        assert _login(setup)
    else:
        assert stored.verify_password(_NEW)
        assert _actions(setup) == [*before, (AuditAction.CREDENTIAL_ROTATE,)]
        assert reached == [True]
    assert not setup.runtime._pending
    if mode == "unknown":
        assert setup.runtime.snapshot().indeterminate


@pytest.mark.parametrize("stage", ["swap", "mint"])
@pytest.mark.parametrize("cancel", [False, True])
def test_local_rotation_failure_keeps_commit_but_retires_every_cached_token(
    setup, monkeypatch, stage, cancel
):
    old_token = _login(setup)
    sessions, minted, before = setup.app.state.dashboard_sessions, [], _actions(setup)
    error = KeyboardInterrupt("local cancellation") if cancel else RuntimeError("local failure")
    swap, authenticate = (
        DashboardSessionStore.rotate_credentials,
        DashboardSessionStore.authenticate,
    )

    def failed_swap(self, persisted):
        swap(self, persisted)
        raise error

    def failed_mint(self, username, password):
        minted.append(authenticate(self, username, password))
        raise error

    with monkeypatch.context() as faults:
        faults.setattr(
            DashboardSessionStore,
            "rotate_credentials" if stage == "swap" else "authenticate",
            failed_swap if stage == "swap" else failed_mint,
        )
        response = Response()
        with pytest.raises(type(error)) as caught:
            update_credentials(
                DashboardCredentialsUpdateRequest(new_password=_NEW),
                _request(setup, old_token),
                response,
            )
        assert caught.value is error and "set-cookie" not in response.headers
    assert sessions.actor(old_token) is None and not sessions._sessions
    assert sessions.authenticate(_USER, _PASSWORD) is None
    assert not hasattr(setup.app.state, "dashboard_sessions")
    assert setup.store.load().verify_password(_NEW)
    assert _actions(setup) == [*before, (AuditAction.CREDENTIAL_ROTATE,)]
    assert setup.runtime.snapshot().ready and not setup.runtime._pending
    assert len(minted) == (1 if stage == "mint" else 0)
    assert all(token is not None and sessions.actor(token) is None for token in minted)
    assert _login(setup, password=_NEW)


@pytest.mark.parametrize("committed", [False, True])
def test_login_cancellation_revokes_token_without_rewriting_committed_audit(
    setup, monkeypatch, committed
):
    old_token = _login(setup)
    sessions, before = setup.app.state.dashboard_sessions, _actions(setup)
    original, error = setup.runtime.append_owned, KeyboardInterrupt("governed cancellation")

    def interrupted(event):
        if committed:
            original(event)
        raise error

    monkeypatch.setattr(setup.runtime, "append_owned", interrupted)
    response = Response()
    with pytest.raises(KeyboardInterrupt) as caught:
        login(
            DashboardLoginRequest(username=_USER, password=_PASSWORD),
            _request(setup, method="POST", path="/api/v1/auth/session"),
            response,
        )
    assert caught.value is error and "set-cookie" not in response.headers
    assert set(sessions._sessions) == {old_token}
    assert _actions(setup) == ([*before, (AuditAction.AUTH_LOGIN,)] if committed else before)


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("failure", ["stopped", "unknown", "cancel"])
def test_governed_playback_failure_closes_unreturned_media(
    setup, tmp_path, monkeypatch, method, failure
):
    old_token = _login(setup)
    root = tmp_path / "media"
    folder = root / "clips" / "auth-clip"
    folder.mkdir(parents=True)
    (folder / "clip.mp4").write_bytes(b"synthetic media")
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "auth-clip",
                "camera_id": "camera-1",
                "event_ref": "event-1",
                "event_type": "fall",
                "started_at": "2026-07-06T00:00:00Z",
                "duration_s": 1.0,
                "codec": "h264",
                "path": "clips/auth-clip",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    setup.app.state.clip_store = ClipStore(root)
    opened, writes, before = [], [], _actions(setup)
    original_open = ClipStore.open_located_playback_identity
    transact = setup.sandbox.database.transact
    error = KeyboardInterrupt("governed playback cancellation")

    def observe_open(self, located):
        identity = original_open(self, located)
        opened.append(identity.opened.handle)
        return identity

    def unknown_commit(body):
        result = transact(body)
        writes.append(result)
        raise CommitOutcomeUnknown()

    def cancel_append(event):
        raise error

    monkeypatch.setattr(ClipStore, "open_located_playback_identity", observe_open)
    if failure == "stopped":
        setup.runtime.stop()
    elif failure == "unknown":
        monkeypatch.setattr(setup.sandbox.database, "transact", unknown_commit)
    else:
        monkeypatch.setattr(setup.runtime, "append_owned", cancel_append)
    path = "/api/v1/clips/auth-clip/video"
    if failure == "cancel":
        with pytest.raises(KeyboardInterrupt) as caught:
            clip_video("auth-clip", _request(setup, old_token, method=method, path=path))
        assert caught.value is error
    else:
        response = setup.client.request(method, path)
        assert (response.status_code, response.content) == (503, b"")
    assert len(opened) == 1 and opened[0].closed
    assert _actions(setup) == (
        [*before, (AuditAction.CLIP_PLAY,)] if failure == "unknown" else before
    )
    assert len(writes) == (1 if failure == "unknown" else 0)
    if failure == "unknown":
        assert setup.runtime.snapshot().indeterminate
    assert not setup.runtime._pending


class _ObservedLock:
    def __init__(self):
        self.inner, self.blocked = Lock(), Event()

    def __enter__(self):
        if not self.inner.acquire(blocking=False):
            self.blocked.set()
            assert self.inner.acquire(timeout=3.0)
        return self

    def __exit__(self, *args):
        self.inner.release()


def test_waiting_rotation_rechecks_revoked_authorization(setup, monkeypatch):
    token = _login(setup)
    entered, release, lock, writes = Event(), Event(), _ObservedLock(), []
    save = setup.store.save

    def held_save(**kwargs):
        writes.append(kwargs["username"])
        entered.set()
        assert release.wait(3.0)
        return save(**kwargs)

    monkeypatch.setattr(setup.store, "save", held_save)
    monkeypatch.setattr(dashboard_auth, "_SESSION_STORE_INIT_LOCK", lock)

    def rotate(username):
        with TestClient(setup.app) as client:
            client.cookies.set(DASHBOARD_SESSION_COOKIE, token)
            return client.put(
                "/api/v1/auth/credentials", json={"username": username, "new_password": _NEW}
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(rotate, "first-operator")
        try:
            assert entered.wait(3.0)
            second = executor.submit(rotate, "second-operator")
            assert lock.blocked.wait(3.0)
        finally:
            release.set()
        assert first.result(timeout=3.0).status_code == 204
        assert second.result(timeout=3.0).status_code == 401
    assert writes == ["first-operator"]
    assert setup.store.load().username == "first-operator"
    assert _actions(setup).count((AuditAction.CREDENTIAL_ROTATE,)) == 1


def test_authentication_and_rotation_share_the_same_session_lock():
    entered, release = Event(), Event()
    original = dashboard_auth.PlaintextDashboardCredentials(_USER, _PASSWORD)

    class HeldCredentials:
        username = _USER

        def verify(self, username, password):
            entered.set()
            assert release.wait(3.0)
            return original.verify(username, password)

    lock = _ObservedLock()
    sessions = DashboardSessionStore(HeldCredentials(), _lock=lock)
    persisted = PersistedDashboardCredentials.from_password(username="new-operator", password=_NEW)
    with ThreadPoolExecutor(max_workers=2) as executor:
        authentication = executor.submit(sessions.authenticate, _USER, _PASSWORD)
        try:
            assert entered.wait(3.0)
            rotation = executor.submit(sessions.rotate_credentials, persisted)
            assert lock.blocked.wait(3.0)
        finally:
            release.set()
        token = authentication.result(timeout=3.0)
        rotation.result(timeout=3.0)
    assert token is not None and sessions.actor(token) is None
    assert sessions.authenticate(_USER, _PASSWORD) is None
    assert sessions.actor(sessions.authenticate("new-operator", _NEW)) == "new-operator"
