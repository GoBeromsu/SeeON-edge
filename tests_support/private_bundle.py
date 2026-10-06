"""Private fall bundle gate: marked and selected, never conditionally skipped.

The packaged fall bundle (models/fall/pose-bbox56-gru) lives in a private Hugging
Face repository and CI downloads no model weights. Tests that read it carry the
``private_bundle`` marker; CI's test job deselects it with ``-m``, and any run that
selects one without the bundle fails at setup, naming the missing file. Provision
models/ with scripts/fetch-models.sh or deselect with ``-m "not private_bundle"``.

The marker is added at collection so ``-m`` sees it. The module list below names
only worker tests that Rust pass 4 deletes or ports; a surviving test that needs
the bundle declares ``pytestmark = pytest.mark.private_bundle`` itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_PRIVATE_BUNDLE_SENTINEL = Path("models/fall/pose-bbox56-gru/model.onnx")
_PRIVATE_BUNDLE_MODULES = frozenset(
    {
        "test_episode_metric.py",
        "test_fall_model_family_registry.py",
        "test_fall_contract_fixtures.py",
        "test_fetch_models.py",
        "test_golden_toolchain.py",
        "test_local_env_defaults.py",
        "test_ort_pose_bbox56_runner.py",
        "test_pose_bbox56_bundle_runner.py",
        "test_runtime_manifest.py",
        "test_worker_config_lifecycle.py",
        "test_worker_config_local_overrides.py",
        "test_worker_fall_model_selection.py",
        "test_worker_real_warmup_no_stub.py",
        "test_worker_startup_config_resolution.py",
    }
)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.name in _PRIVATE_BUNDLE_MODULES:
            item.add_marker(pytest.mark.private_bundle)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("private_bundle") is None:
        return
    sentinel = item.config.rootpath / _PRIVATE_BUNDLE_SENTINEL
    if not sentinel.is_file():
        pytest.fail(
            f"private fall bundle missing: {sentinel}; provision models/ with "
            "scripts/fetch-models.sh or deselect with -m 'not private_bundle'",
            pytrace=False,
        )
