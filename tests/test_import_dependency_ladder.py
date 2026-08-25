from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
IMPORT_LINTER_CONFIG: Final = REPO_ROOT / "pyproject.toml"
FIXTURE_PACKAGES: Final = (
    "backend",
    "backend.app",
    "backend.app.core",
    "backend.app.shared",
    "backend.app.features",
    "backend.app.routes",
    "backend.app.main",
    "backend.app.lifespan",
    "backend.app.edge_db",
    "backend.app.edge_db.compatibility",
    "backend.app.edge_db.schema18_manifest",
    "contracts",
    "shared",
    "shared.events",
    "worker",
    "worker.types",
    "worker.interfaces",
    "worker.adapters",
    "worker.pipeline",
    "worker.pipeline.bus",
    "worker.pipeline.ingest",
    "worker.pipeline.perception",
    "worker.pipeline.perception.features",
    "worker.pipeline.decision",
    "worker.pipeline.output",
    "worker.domains",
    "worker.runtime",
)


def _write_import_fixture(
    fixture_root: Path,
    forbidden_import: tuple[str, str] | None = None,
) -> None:
    sources = {
        "worker.types": "import contracts\n",
        "worker.interfaces": "import contracts\nimport worker.types\n",
    }
    if forbidden_import is not None:
        source_module, target_module = forbidden_import
        sources[source_module] = f"{sources.get(source_module, '')}import {target_module}\n"

    for module_name in FIXTURE_PACKAGES:
        package_dir = fixture_root.joinpath(*module_name.split("."))
        package_dir.mkdir(parents=True, exist_ok=True)
        _ = (package_dir / "__init__.py").write_text(
            sources.get(module_name, ""),
            encoding="utf-8",
        )


def _lint_fixture(fixture_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(REPO_ROOT),
            "--frozen",
            "--group",
            "lint",
            "lint-imports",
            "--config",
            str(IMPORT_LINTER_CONFIG),
            "--no-cache",
        ],
        cwd=fixture_root,
        env={**os.environ, "PYTHONPATH": str(fixture_root)},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_contracts_types_interfaces_dependency_ladder_is_allowed(tmp_path: Path) -> None:
    fixture_root = tmp_path / "allowed"
    _write_import_fixture(fixture_root)

    result = _lint_fixture(fixture_root)

    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    ("source_module", "target_module"),
    [
        pytest.param("backend", "worker", id="backend-to-worker"),
        pytest.param("contracts", "worker.types", id="contracts-to-worker"),
        pytest.param("worker.types", "worker.interfaces", id="types-to-interfaces"),
        pytest.param("worker.interfaces", "worker.adapters", id="interfaces-to-adapters"),
        pytest.param(
            "worker.pipeline.perception.features",
            "worker.adapters",
            id="perception-features-to-adapters",
        ),
        pytest.param("worker.adapters", "worker.runtime", id="adapters-to-runtime"),
        pytest.param(
            "worker.domains",
            "worker.pipeline.output",
            id="domains-to-pipeline-output",
        ),
        pytest.param("worker.pipeline.bus", "worker.runtime", id="pipeline-to-runtime"),
    ],
)
def test_worker_firewall_rejects_forbidden_imports(
    tmp_path: Path,
    source_module: str,
    target_module: str,
) -> None:
    fixture_root = tmp_path / "forbidden"
    _write_import_fixture(fixture_root, (source_module, target_module))

    result = _lint_fixture(fixture_root)

    assert result.returncode != 0, result.stdout
    assert source_module in result.stdout
    assert target_module in result.stdout
