"""Cross-suite compatibility fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from backend.app.shared.dashboard_credentials import DashboardCredentialsStore


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
    ``from_env()``, which otherwise resolves to the real
    ``~/.local/state/ml-api/catalog.sqlite3`` (no env override exists anymore
    for this path). That path is an ambient-filesystem read that shouldn't
    happen from the suite at all, so ``from_env()`` itself is patched
    (constructor injection, same isolated-path guarantee as before, just no
    longer routed through an environment variable) to build the store from a
    per-test tmp path instead.
    """

    monkeypatch.setattr(
        DashboardCredentialsStore,
        "from_env",
        classmethod(lambda cls: cls(tmp_path / "catalog.sqlite3")),
    )
    yield
