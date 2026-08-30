"""Phase 6 remains a dark candidate: deployed LSTM behavior never changes."""

from __future__ import annotations

import hashlib
import json
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

    candidate = local_env.fall_candidate_config_from_environment(
        {"ML_WORKER_IMAGE": "registry.test/ml-worker@sha256:" + "b" * 64},
        selection_path=selection,
        models_root=Path("/models"),
    )

    assert candidate is not None
    assert candidate.models_root == Path("/models")
    assert candidate.observed_worker_image_digest == "b" * 64


@pytest.mark.parametrize(
    "environment",
    (
        {},
        {"ML_WORKER_IMAGE": "registry.test/ml-worker:mutable"},
    ),
)
def test_candidate_selection_refuses_incomplete_or_mutable_environment(
    environment: dict[str, str], tmp_path: Path
) -> None:
    selection = tmp_path / "candidate.json"
    selection.write_text("{}", encoding="utf-8")
    with pytest.raises(WorkerConfigError):
        local_env.fall_candidate_config_from_environment(environment, selection_path=selection)


def test_candidate_selection_refuses_noncanonical_json(tmp_path: Path) -> None:
    selection = tmp_path / "candidate.json"
    selection.write_text("{ }\n", encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="canonical JSON"):
        local_env.fall_candidate_config_from_environment(
            {"ML_WORKER_IMAGE": "registry.test/ml-worker@sha256:" + "b" * 64},
            selection_path=selection,
        )


def test_candidate_environment_cannot_select_a_missing_image_document(
    tmp_path: Path,
) -> None:
    assert (
        local_env.fall_candidate_config_from_environment(
            {
                "FALL_CANDIDATE_SELECTION_PATH": str(tmp_path / "forged.json"),
                "FALL_CANDIDATE_MODELS_ROOT": "/forged",
                "ML_WORKER_IMAGE": "registry.test/ml-worker@sha256:" + "b" * 64,
            },
            selection_path=tmp_path / "missing-image-document.json",
        )
        is None
    )


def test_admitted_dark_gru_is_constructed_warmed_and_never_becomes_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worker.adapters.model import torch_gru_fall

    class FakeRunner:
        def __init__(self) -> None:
            self.manifest = SimpleNamespace(
                class_order=("background", "fall_transition", "fallen"),
                input_digest="i" * 64,
                policy_digest="p" * 64,
                calibration_digest="c" * 64,
                conformance_digest="n" * 64,
            )
            self.warmups = 0

        def warmup(self) -> None:
            self.warmups += 1

    runner = FakeRunner()
    class_digest = hashlib.sha256(
        json.dumps(list(runner.manifest.class_order), separators=(",", ":")).encode()
    ).hexdigest()
    desired = DesiredModelBundle(
        "a" * 64,
        {
            **dict(_desired().identities),
            "class": class_digest,
            "input": "i" * 64,
            "policy": "p" * 64,
            "calibration": "c" * 64,
            "conformance": "n" * 64,
        },
    )
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        torch_gru_fall.GruFallRunner,
        "from_artifact_dir",
        classmethod(lambda _cls, path, device: calls.append((path, device)) or runner),
    )
    runtime = object.__new__(WorkerRuntime)
    runtime.fall_model = "active-lstm"

    actual = runtime._construct_dark_candidate_runner(  # noqa: SLF001
        SimpleNamespace(models_root=Path("/models"), desired=desired)
    )

    assert actual is runner
    assert calls == [(Path("/models") / "bundles" / ("a" * 64), "cpu")]
    assert runner.warmups == 1
    assert runtime.fall_model == "active-lstm"


def test_dark_gru_identity_failure_is_fatal_before_active_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worker.adapters.model import torch_gru_fall

    monkeypatch.setattr(
        torch_gru_fall.GruFallRunner,
        "from_artifact_dir",
        classmethod(
            lambda _cls, _path, device: SimpleNamespace(
                manifest=SimpleNamespace(
                    class_order=("bad", "order", "x"),
                    input_digest="0" * 64,
                    policy_digest="0" * 64,
                    calibration_digest="0" * 64,
                    conformance_digest="0" * 64,
                ),
                warmup=lambda: None,
            )
        ),
    )
    runtime = object.__new__(WorkerRuntime)
    runtime.fall_model = "active-lstm"

    with pytest.raises(RuntimeError, match="manifest identities"):
        runtime._construct_dark_candidate_runner(  # noqa: SLF001
            SimpleNamespace(models_root=Path("/models"), desired=_desired())
        )

    assert runtime.fall_model == "active-lstm"


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
    assert "/var/lib/seeon-state/applied-model-manifest.json" not in runbook
    image_attestation = " ".join(
        (
            "RepoDigests={{json .RepoDigests}}",
            'source_revision={{index .Config.Labels "org.opencontainers.image.revision"}}',
        )
    )
    for assertion in (
        'configured_ref="$(docker inspect --format \'{{.Config.Image}}\' "$worker_id")"',
        'image_id="$(docker inspect --format \'{{.Image}}\' "$worker_id")"',
        image_attestation,
        "dual mounts of the same model volume",
        "/app/models",
        "/models/bundles/<digest>",
        "intentionally ships neither",
        "/app/model-selection.json",
        "nor a real candidate bundle",
        "human/Hugging Face evidence is blocked",
        "Dark support is unreachable by default",
        "no candidate receipt is fabricated",
        "A later G005-green sealed image may bake both",
        "Evaluation and field receipts are externally",
        "non-recursive receipts",
        "closed before assignment",
        "admit → construct → warm → persist",
        "while the candidate remains",
        "The durable queue remains canonical.",
        "/var/lib/seeon-state/applied-runtime-manifest.json",
        "st_mode & 0o777 == 0o600",
        'receipt["manifest_sha256"]',
        "canonical_json",
        "Candidate activation and Phase 7 are outside this runbook.",
    ):
        assert assertion in runbook


def test_unadmitted_candidate_never_projects_to_the_durable_manifest() -> None:
    runtime = object.__new__(WorkerRuntime)
    runtime._candidate_admission = None
    runtime.config = SimpleNamespace(models=SimpleNamespace(candidate=object()))

    assert runtime._dark_model_candidate_manifest_content() is None  # noqa: SLF001
