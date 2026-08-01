"""Acceptance criterion 1: real_warmup passes with no stub injection.

The real-stack e2e (``tests/test_e2e_night_bed_exit_relay.py``) deliberately
injects ``ScriptedServingClient`` so it can drive deterministic detections, and
that client's ``warmup()`` is a no-op. That makes the e2e authoritative for
decode, relay, and liveness -- but *not* for warmup.

This module closes that gap without faking it. It provisions models through the
production path -- ``InProcessServingClient`` over ``default_registry()`` -- and
runs each adapter's real ``warmup()``, which performs one genuine forward on a
synthetic frame of the configured shape. Nothing here is mocked: no injected
``model=``, no patched loader, no fake registry.

The tests skip only when a required weight file is genuinely absent, so a
machine without artifacts reports "skipped", never a false pass.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.in_process import InProcessServingClient
from worker.adapters.model.registry import default_registry
from worker.adapters.model.torch_lstm_fall import LstmFallRunner
from worker.runtime.model_composition import compose_yolo_extractors
from worker.runtime.worker import WorkerRuntime

REPO_ROOT = Path(__file__).resolve().parents[1]

# Task -> the weight file its production default points at.
_TASK_ARTIFACTS: dict[str, Path] = {
    "pose": REPO_ROOT / "models" / "pose" / "yolo26n-pose.pt",
    "person": REPO_ROOT / "models" / "person" / "yolo26n.pt",
    "bed": REPO_ROOT / "models" / "bed" / "yolo26m-seg.pt",
}


def _require(task: str) -> Path:
    artifact = _TASK_ARTIFACTS[task]
    if not artifact.is_file():
        pytest.skip(f"{task} weights are not present at {artifact}")
    return artifact


@pytest.mark.parametrize("task", sorted(_TASK_ARTIFACTS))
def test_real_warmup_runs_a_genuine_forward_through_the_production_serving_path(
    task: str,
) -> None:
    """Provision through the real serving client and run the real warmup.

    A stubbed warmup cannot fail this: the adapter loads actual weights and
    executes a real forward pass. If the artifact were missing or the model
    unloadable, ``warmup()`` raises rather than quietly succeeding.
    """
    _ = _require(task)

    serving = InProcessServingClient(registry=default_registry())
    adapter = serving.create(task, device="cpu")

    # The real forward. No injected model, no patched loader.
    adapter.warmup()


def test_production_warm_models_warms_every_configured_task_for_real() -> None:
    """Drive ``WorkerRuntime._warm_models()`` itself, not a hand-rolled copy.

    An earlier version of this test iterated pose/person/bed/fall inside the
    test body. That proved the adapters *can* warm, but it would still pass if
    production stopped warming fall or the stage wiring were cut, because the
    test controlled its own loop and then asserted on its own list. So this
    calls the production method and asserts on what production returns.

    ``_warm_models`` warms every extractor in ``shared_yolo`` plus
    ``fall_model`` and reports ``("pose", "person", "bed", "fall")``. Both
    collaborators are built the way production builds them:
    ``compose_yolo_extractors`` over the real ``InProcessServingClient``, and
    the real ``LstmFallRunner`` on the configured artifact directory. Nothing
    is stubbed -- real weights, real forwards, real warmup helper.
    """
    for task in sorted(_TASK_ARTIFACTS):
        _ = _require(task)
    artifact_dir = REPO_ROOT / "models" / "fall" / "lstm"
    if not (artifact_dir / "model.pt").is_file():
        pytest.skip(f"fall LSTM artifacts are not present at {artifact_dir}")

    serving = InProcessServingClient(registry=default_registry())

    runtime = WorkerRuntime.__new__(WorkerRuntime)
    runtime.shared_yolo = compose_yolo_extractors(serving, device="cpu")
    # This artifact declares neither schema_version nor preprocessing_identity,
    # so the manifest defaults apply and the example config's pins do not match
    # it -- pinned by the negative test below. Passing None keeps this test
    # about warmup rather than about that mismatch.
    fall = LstmFallRunner.from_artifact_dir(
        str(artifact_dir),
        device="cpu",
        expected_schema_version=None,
        expected_preprocessing_identity=None,
    )

    # Observe that warmup is actually *invoked* on each model, not just that
    # _warm_models reports it. The return tuple is a hard-coded literal, so
    # asserting on it alone would still pass if the fall warmup call were
    # deleted and the literal left untouched.
    warmed_calls: list[str] = []

    def _record(name: str, model: object) -> None:
        real = model.warmup  # type: ignore[attr-defined]

        def _counting() -> None:
            warmed_calls.append(name)
            real()

        model.warmup = _counting  # type: ignore[attr-defined]

    yolo_names = [e.module_name for e in runtime.shared_yolo.extractors]
    for extractor in runtime.shared_yolo.extractors:
        _record(extractor.module_name, extractor.runner)
    _record("fall", fall)

    runtime.fall_model = fall
    runtime._boot = SimpleNamespace(device="cpu")  # type: ignore[assignment]  # noqa: SLF001

    warmed = runtime._warm_models()  # noqa: SLF001

    # Production's own report...
    assert warmed == ("pose", "person", "bed", "fall")
    # ...and proof each model's real warmup ran, fall included.
    # Exact multiset, not just a count: a loose `len == 4` would still pass if
    # one model were warmed twice and another skipped.
    assert sorted(warmed_calls) == sorted([*yolo_names, "fall"])


def test_example_config_fall_contract_does_not_match_the_local_artifact() -> None:
    """Records a real mismatch found while proving no-stub warmup.

    ``worker/ml-worker.example.yaml`` pins the fall model to
    ``schema_version: 2`` and
    ``preprocessing_identity: coco17-xyc-frame-normalized-zero-fill-v1``, but
    the artifact under ``models/fall/lstm`` on this host declares neither, so
    its manifest defaults to schema_version 1. Loading the example contract
    against it raises ``ModelLoadError``.

    That is the loader behaving correctly -- ADR-0003 fail-closed, no silent
    downgrade -- so this test pins the behaviour rather than treating it as a
    defect to patch over. It is also why the production-warm test above passes
    ``None`` for both expectations.

    If the artifact is later re-exported with schema_version 2 and the coco17
    identity, this test starts failing and should be deleted along with those
    ``None`` expectations.
    """
    artifact_dir = REPO_ROOT / "models" / "fall" / "lstm"
    if not (artifact_dir / "model.pt").is_file():
        pytest.skip(f"fall LSTM artifacts are not present at {artifact_dir}")

    # Read the pins from the example config itself. Hard-coding them here would
    # make this test agree with a *copy* of the config rather than the config,
    # so editing the YAML would silently stop exercising the real contract.
    example = yaml.safe_load(
        (REPO_ROOT / "worker" / "ml-worker.example.yaml").read_text(encoding="utf-8")
    )
    fall_cfg = example["models"]["fall"]
    configured_schema = fall_cfg["schema_version"]
    configured_identity = fall_cfg["preprocessing_identity"]

    with pytest.raises(ModelLoadError) as captured:
        _ = LstmFallRunner.from_artifact_dir(
            str(artifact_dir),
            device="cpu",
            expected_schema_version=configured_schema,
            expected_preprocessing_identity=configured_identity,
        )

    message = str(captured.value)
    assert "does not match configured" in message
    # The mismatch is on the value the example config actually declares.
    assert str(configured_schema) in message
