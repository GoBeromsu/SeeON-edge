"""Phase 6 remains a dark candidate: deployed LSTM behavior never changes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import worker.runtime.worker as worker_module
from worker.adapters.model.fall_family_registry import (
    DisabledFallModelTypeError,
    default_fall_model_family_registry,
)
from worker.runtime.config import local_env
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.worker_models import WorkerModelsConfig
from worker.runtime.provenance.model_bundle import DesiredModelBundle, ModelBundleProof
from worker.runtime.worker import WorkerRuntime


def _desired() -> DesiredModelBundle:
    return DesiredModelBundle(
        bundle_sha256="a" * 64,
        identities={
            "dataset": "dataset.v1",
            "evaluation": "evaluation.v1",
            "field": "field.v1",
            "seed": "seed.v1",
            "rule": "rule.v1",
            "calibration": "calibration.v1",
            "conformance": "conformance.v1",
            "class": "class.v1",
            "input": "pose-bbox56.v1",
            "policy": "fall.policy.v2",
            "config": "config.v1",
            "restart": "restart.v1",
            "worker": "worker.v1",
        },
    )


def test_default_family_registry_keeps_gru_dark() -> None:
    registry = default_fall_model_family_registry()

    assert registry.types() == ("lstm",)
    with pytest.raises(DisabledFallModelTypeError):
        registry.create("gru", object(), "cpu")


def test_candidate_refuses_person_boxes_before_camera_activation() -> None:
    with pytest.raises(ValueError, match="requires box_source=pose"):
        WorkerModelsConfig(
            box_source="person",
            candidate={
                "models_root": str(Path("/sealed/candidate")),
                "desired": _desired(),
                "observed_worker_image_digest": "b" * 64,
            },
        )


def test_candidate_config_is_not_an_active_fall_selection() -> None:
    models = WorkerModelsConfig(
        candidate={
            "models_root": str(Path("/sealed/candidate")),
            "desired": _desired(),
            "observed_worker_image_digest": "b" * 64,
        },
    )

    assert models.fall is None
    assert models.candidate is not None


def test_candidate_selection_is_only_reachable_through_a_pinned_local_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = tmp_path / "candidate.json"
    selection.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        local_env, "desired_model_bundle_from_selection_document", lambda raw: _desired()
    )

    candidate = local_env.fall_candidate_bundle_config_from_environment(
        {
            "FALL_CANDIDATE_SELECTION_PATH": str(selection),
            "FALL_CANDIDATE_MODELS_ROOT": "/app/models",
            "ML_WORKER_IMAGE": "registry.test/ml-worker@sha256:" + "b" * 64,
        }
    )

    assert candidate is not None
    assert candidate.models_root == Path("/app/models")
    assert candidate.observed_worker_image_digest == "b" * 64


@pytest.mark.parametrize(
    "environment",
    (
        {"FALL_CANDIDATE_SELECTION_PATH": "/sealed/candidate.json"},
        {
            "FALL_CANDIDATE_SELECTION_PATH": "/sealed/candidate.json",
            "FALL_CANDIDATE_MODELS_ROOT": "/app/models",
        },
        {
            "FALL_CANDIDATE_SELECTION_PATH": "/sealed/candidate.json",
            "FALL_CANDIDATE_MODELS_ROOT": "/app/models",
            "ML_WORKER_IMAGE": "registry.test/ml-worker:mutable",
        },
    ),
)
def test_candidate_selection_refuses_incomplete_or_mutable_environment(
    environment: dict[str, str],
) -> None:
    with pytest.raises(WorkerConfigError):
        local_env.fall_candidate_bundle_config_from_environment(environment)


def test_candidate_selection_refuses_noncanonical_json(tmp_path: Path) -> None:
    selection = tmp_path / "candidate.json"
    selection.write_text("{ }\n", encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="canonical JSON"):
        local_env.fall_candidate_bundle_config_from_environment(
            {
                "FALL_CANDIDATE_SELECTION_PATH": str(selection),
                "FALL_CANDIDATE_MODELS_ROOT": "/app/models",
                "ML_WORKER_IMAGE": "registry.test/ml-worker@sha256:" + "b" * 64,
            }
        )


def test_candidate_startup_telemetry_is_message_visible_and_digest_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(WorkerRuntime)
    runtime.config = SimpleNamespace(
        models=SimpleNamespace(
            fall=SimpleNamespace(type="lstm"),
            candidate=SimpleNamespace(desired=_desired()),
        )
    )
    messages: list[str] = []
    monkeypatch.setattr(
        worker_module.LOGGER,
        "info",
        lambda message, *args: messages.append(message % args),
    )

    runtime._log_fall_candidate_startup(  # noqa: SLF001
        ModelBundleProof(observed={"bundle_sha256": "a" * 64}, applied={})
    )

    expected_message = " ".join(
        (
            f"fall candidate startup desired={'a' * 64} observed={'a' * 64}",
            "match=True active=lstm/fall.v1/fall.policy.v1 candidate=disabled",
        )
    )
    assert messages == [expected_message]


def test_publish_runbook_records_flat_lstm_rollback_artifacts() -> None:
    runbook = (
        Path(__file__).resolve().parents[1] / "docs/runbooks/edge-image-publish.md"
    ).read_text(encoding="utf-8")

    assert "LSTM bundle SHA-256" not in runbook
    artifact_identities = (
        (
            "fall/lstm/model.pt",
            "889075695884742475b9713e3b86ba67085bb96979b64c51756ea3fd715ab57a",
        ),
        (
            "fall/lstm/metadata.upstream.json",
            "c0870223db642f9e773256ff90a52d9c3021aaf4e6981281d6e69772003b0f66",
        ),
        (
            "fall/lstm/metadata.yaml",
            "3f6aca78bf535d02873c753cd0600510bde8860d698af32479505d3856e3d509",
        ),
        (
            "fall/lstm/arch.json",
            "541100998c5627be4126dc3aed63a1546fc5f5a4862ceb9b1041de57e83de43b",
        ),
    )
    for path, digest in artifact_identities:
        assert f"{path} sha256={digest}" in runbook
    for identity in ("fall.v1", "fall.policy.v1", "[30,51]"):
        assert identity in runbook

    assert "dc run --rm --no-deps edge-model-fetch" in runbook
    assert "dc up -d --no-deps --force-recreate ml-worker" in runbook
    assert "Do not restart `ml-api` or any other service." in runbook
    assert "docker compose down -v" in runbook
    assert "hand-edit the model volume" in runbook
    assert "GRU candidate remains dark and inert" in runbook
