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


@pytest.fixture(autouse=True)
def isolate_state_dir_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """``resolve_state_dir``(``backend/app/shared/state_dir.py``,
    ``worker/runtime/state_dir.py``)가 참조하는 ``Path.home()``을 테스트별
    tmp로 격리한다.

    ``resolve_state_dir``은 의도적으로 "단일 규칙, override 없음"으로
    설계돼 있어 (이슈 #153) 환경변수 주입 지점이 없다 -- 그래서
    ``resolve_state_dir`` 자체는 건드리지 않고, 그 유일한 입력원인
    ``Path.home()``을 픽스처에서 리다이렉트한다 (``test_clips_catalog.py``의
    ``test_catalog_from_env_resolves_under_state_dir_and_is_queryable``가
    이미 쓰던 것과 동일한 패턴을 전역 autouse로 승격한 것).

    ``HOME`` 환경변수 자체를 바꾸는 대신 ``Path.home``만 monkeypatch하는
    이유: ``HOME``을 바꾸면 uv 캐시, git config, 서브프로세스 등 pytest와
    무관한 것들까지 영향받는다. ``Path.home()`` 클래스메서드만 리다이렉트
    하면 ``resolve_state_dir``이 참조하는 경로만 격리되고, ``os.path.
    expanduser`` 등 다른 경로 해석 경로는 그대로 실제 홈을 본다.

    이게 없으면 ``app.state.camera_registry``를 tmp_path로 주입하지 않는
    테스트가 개발자의 실제 ``~/.local/state/ml-api/catalog.sqlite3``를
    읽고 쓴다 (dev 스택에 카메라가 등록돼 있으면 카메라/설정 테스트가
    무더기로 거짓 실패한다 -- 이슈 #153).
    """

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    yield


@pytest.fixture(autouse=True)
def default_connection_settings_store_to_tmp_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """``ConnectionSettingsStore``의 기본 경로를 테스트별 tmp로 돌린다.

    기본값은 ``/var/lib/ml-api/connection-settings.sqlite3``라 로컬에서
    열리지 않는다. 스토어는 실패를 삼키고 빈 값을 돌려주므로 테스트는
    통과하지만, 매 실행마다 ``connection settings store unreadable`` 경고가
    찍히고 **연결 설정을 실제로 저장·복원하는 경로가 한 번도 검증되지
    않는다.** 위의 dashboard credentials 픽스처와 같은 이유다.
    """

    monkeypatch.setenv(
        "API_CONNECTION_SETTINGS_PATH",
        str(tmp_path / "connection-settings.sqlite3"),
    )
    yield
