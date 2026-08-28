"""Contract-symbol export tests.

Architectural import boundaries are enforced declaratively by import-linter
(``uv run --group lint lint-imports``; see ``[tool.importlinter]`` in
pyproject.toml), which replaced the old hand-rolled AST import-ladder walker.
What import-linter cannot express — that the shared ``contracts`` package keeps
exporting the exact symbols the runtime depends on — stays here as pytest.
"""

from __future__ import annotations

from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_RUNNER_CONTRACT_SYMBOLS = {
    "RunnerResult",
    "PoseRunnerResult",
    "PersonRunnerResult",
    "BedRunnerResult",
    "DetectionRunnerResult",
    "pose_result",
    "person_result",
    "bed_result",
    "detection_result",
}
EXPECTED_TRACKER_CONTRACT_SYMBOLS = {"TrackerProtocol"}
EXPECTED_WORKER_CONFIG_CONTRACT_SYMBOLS = {
    "WORKER_CONFIG_PATH",
    "WORKER_RESTART_PATH",
    "CONFIG_VERSION_KEY",
    "RESTART_EPOCH_KEY",
    "PulledCameraConfig",
    "PulledNightWindow",
    "PulledWorkerConfig",
}


def _exported_symbols(module_relpath: str) -> set[str]:
    namespace: dict[str, object] = {}
    exec((ML_ROOT / module_relpath).read_text(encoding="utf-8"), namespace)  # noqa: S102
    return set(namespace["__all__"])


def test_runner_contract_exports_tagged_result_symbols() -> None:
    assert _exported_symbols("contracts/runner.py") >= EXPECTED_RUNNER_CONTRACT_SYMBOLS


def test_tracker_contract_exports_protocol_symbol() -> None:
    assert _exported_symbols("contracts/tracker.py") >= EXPECTED_TRACKER_CONTRACT_SYMBOLS


def test_worker_config_contract_exports_pull_symbols() -> None:
    exported = _exported_symbols("contracts/worker_config.py")
    assert exported >= EXPECTED_WORKER_CONFIG_CONTRACT_SYMBOLS
