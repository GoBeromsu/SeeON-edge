"""Cross-suite compatibility fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def enable_legacy_dashboard_auth_for_pre_session_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep old API contract tests explicit while production fails closed by default."""

    monkeypatch.setenv("API_ALLOW_LEGACY_DASHBOARD_AUTH", "1")
    yield


@pytest.fixture(autouse=True)
def default_dashboard_credentials_store_to_tmp_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Point ``DashboardCredentialsStore.from_env()`` at a per-test tmp path.

    Any test that logs in via the dashboard session route without explicitly
    setting ``app.state.dashboard_credentials_store`` falls through to
    ``from_env()``, which otherwise defaults to the real
    ``/var/lib/ml-api/dashboard_credentials.json``. That path is read-only
    and absent on developer/CI machines today, so tests pass, but it is an
    ambient-filesystem read that shouldn't happen from the suite at all.
    Defaulting every test to an isolated tmp path removes that dependency.
    """

    monkeypatch.setenv(
        "API_DASHBOARD_CREDENTIALS_STORE", str(tmp_path / "dashboard_credentials.json")
    )
    yield
