"""Retired worker compatibility surfaces must stay off the shipped tree."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = ROOT / "worker"
DOCKERFILE = ROOT / "Dockerfile.edge"

RETIRED_RELATIVE_PATHS = (
    "worker/pipeline/output/evidence/evidence_outbox_database.py",
    "worker/pipeline/output/evidence/evidence_outbox_schema.py",
    "worker/pipeline/output/evidence/outbox_transaction.py",
    "worker/pipeline/output/evidence/evidence_outbox.py",
    "worker/pipeline/output/evidence/evidence_reconciliation.py",
    "worker/pipeline/output/evidence/evidence_records.py",
    "worker/pipeline/output/evidence/evidence_record_models.py",
    "worker/pipeline/output/evidence/event_identity.py",
    "worker/pipeline/output/evidence/evidence_manifest_validation.py",
    "worker/pipeline/output/evidence/snapshot_reconciliation.py",
    "worker/pipeline/ingest/camera_probe.py",
    "worker/pipeline/ingest/frame_source.py",
    "worker/interfaces/device_batch.py",
    "worker/runtime/telemetry/device_residency.py",
)
RETIRED_MODULES = tuple(
    path.removesuffix(".py").replace("/", ".") for path in RETIRED_RELATIVE_PATHS
)
ACTIVE_COMPOSITION_MODULES = (
    "shared.events.delivery_queue",
    "worker.pipeline.output.evidence.evidence_sender",
    "worker.pipeline.output.evidence.evidence_stager",
    "worker.pipeline.output.evidence.evidence_runtime",
    "worker.pipeline.output.evidence.snapshot_store",
    "worker.pipeline.output.evidence.clip_recorder",
    "worker.pipeline.decision.event_identity",
    "worker.pipeline.decision.incident_manager",
    "worker.pipeline.trace",
    "worker.runtime.telemetry.runtime_status_sender",
)
COMPOSITION_ROOTS = (
    ROOT / "worker" / "__main__.py",
    ROOT / "worker" / "runtime" / "worker.py",
)


def _module_path(module_name: str) -> Path | None:
    relative = Path(*module_name.split("."))
    as_file = ROOT / relative.with_suffix(".py")
    as_package = ROOT / relative / "__init__.py"
    if as_file.is_file():
        return as_file
    if as_package.is_file():
        return as_package
    return None


def _imported_modules(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return frozenset(names)


def _reachable_modules(roots: tuple[Path, ...]) -> frozenset[str]:
    pending = [path for path in roots if path.is_file()]
    seen_paths = set(pending)
    reachable: set[str] = set()
    while pending:
        path = pending.pop()
        for module_name in _imported_modules(path):
            if not module_name.startswith(("worker.", "shared.", "contracts.")):
                continue
            reachable.add(module_name)
            target = _module_path(module_name)
            if target is not None and target not in seen_paths:
                seen_paths.add(target)
                pending.append(target)
    return frozenset(reachable)


def _defined_function_names(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )


def test_retired_worker_files_are_absent_from_the_source_tree() -> None:
    present = [path for path in RETIRED_RELATIVE_PATHS if (ROOT / path).exists()]
    assert present == []


def test_retired_worker_modules_are_unimportable() -> None:
    for module_name in RETIRED_MODULES:
        with pytest.raises(ModuleNotFoundError):
            _ = importlib.import_module(module_name)


def test_worker_image_copy_tree_cannot_ship_retired_files() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY worker ./worker" in dockerfile
    shipped = [
        path
        for path in RETIRED_RELATIVE_PATHS
        if path.startswith("worker/") and (ROOT / path).exists()
    ]
    assert shipped == []


def test_refuse_only_catalog_helpers_are_absent_while_validators_remain() -> None:
    runtime_manifest = (
        WORKER_ROOT / "pipeline" / "output" / "evidence" / "runtime_manifest_reference.py"
    )
    decision_trace = (
        WORKER_ROOT / "pipeline" / "output" / "evidence" / "decision_trace_reference.py"
    )
    assert runtime_manifest.is_file()
    assert decision_trace.is_file()
    assert "require_runtime_manifest_contents" not in _defined_function_names(runtime_manifest)
    assert "require_decision_trace" not in _defined_function_names(decision_trace)

    runtime_module = importlib.import_module(
        "worker.pipeline.output.evidence.runtime_manifest_reference"
    )
    decision_module = importlib.import_module(
        "worker.pipeline.output.evidence.decision_trace_reference"
    )
    assert hasattr(runtime_module, "RuntimeManifestReferenceError")
    assert hasattr(runtime_module, "RuntimeManifestReferenceFailure")
    assert hasattr(decision_module, "DecisionTraceReferenceError")
    assert hasattr(decision_module, "validate_decision_trace_id")
    assert not hasattr(runtime_module, "require_runtime_manifest_contents")
    assert not hasattr(decision_module, "require_decision_trace")


def test_composition_roots_do_not_reach_retired_modules() -> None:
    reachable = _reachable_modules(COMPOSITION_ROOTS)
    assert reachable.isdisjoint(RETIRED_MODULES)


def test_composition_roots_still_wire_active_delivery_and_identity() -> None:
    reachable = _reachable_modules(COMPOSITION_ROOTS)
    missing = [
        module_name for module_name in ACTIVE_COMPOSITION_MODULES if module_name not in reachable
    ]
    assert missing == []
    worker_imports = _imported_modules(ROOT / "worker" / "runtime" / "worker.py")
    assert "shared.events.delivery_queue" in worker_imports
    assert "worker.pipeline.decision.event_identity" in worker_imports
    assert "worker.pipeline.output.evidence.evidence_stager" in worker_imports
    runtime_imports = _imported_modules(
        ROOT / "worker" / "pipeline" / "output" / "evidence" / "evidence_runtime.py"
    )
    assert "worker.pipeline.output.evidence.evidence_sender" in runtime_imports


def test_active_worker_contracts_remain_importable() -> None:
    for module_name in ACTIVE_COMPOSITION_MODULES:
        module = importlib.import_module(module_name)
        assert module is not None
    identity = importlib.import_module("worker.pipeline.decision.event_identity")
    assert hasattr(identity, "EventIdentityStore")
    assert hasattr(identity, "event_identity_path")
