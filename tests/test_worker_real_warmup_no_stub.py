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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from worker.adapters.model.in_process import InProcessServingClient
from worker.adapters.model.registry import default_registry
from worker.adapters.model.torch_lstm_fall import LstmFallRunner, build_lstm_module
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
    # box_source="person": this smoke test wants every YOLO extractor
    # (pose, person, bed) actually provisioned and warmed, not just the
    # box_source="pose" default's subset.
    runtime.shared_yolo = compose_yolo_extractors(serving, device="cpu", box_source="person")
    # This artifact declares neither schema_version nor preprocessing_identity,
    # so the manifest defaults apply. Passing None keeps this test about
    # warmup rather than about the example config's pins, which are covered
    # separately below.
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


def test_example_config_fall_contract_matches_the_local_artifact() -> None:
    """The example config's fall contract now matches the shipped artifact.

    ``worker/ml-worker.example.yaml`` used to pin the fall model to
    ``schema_version: 2`` and the current coco17
    ``preprocessing_identity``, but the artifact under
    ``models/fall/lstm`` on this host declares neither field, so its
    manifest defaults to ``schema_version: 1`` with the legacy identity --
    and ``eldercare-dataset-ops`` does not emit ``schema_version: 2`` for
    the fall family yet (see ``ml/training/_selftest_g006.py``), so no
    shipped artifact could ever satisfy that pin. The example was
    corrected to document the legacy contract training actually produces
    today; see ``docs/architecture.md`` for the full rationale
    (issue #8).

    That is a config fix, not a loader change: ADR-0003 fail-closed
    validation in ``_validate_expected_identity``
    (``worker/adapters/model/torch_lstm_fall.py``) stays exactly as-is, and
    this test proves it now succeeds because the *configured* contract
    agrees with the artifact, not because the check was loosened.

    This reads the pins from the example config itself -- not hard-coded
    copies -- so editing the YAML exercises the real contract rather than a
    frozen expectation of it. If the pins are later bumped back to
    schema_version 2 (once ``eldercare-dataset-ops`` exports that contract
    and ``models/fall/lstm`` is re-exported to match), this test keeps
    passing as long as both sides move together.
    """
    artifact_dir = REPO_ROOT / "models" / "fall" / "lstm"
    if not (artifact_dir / "model.pt").is_file():
        pytest.skip(f"fall LSTM artifacts are not present at {artifact_dir}")

    example = yaml.safe_load(
        (REPO_ROOT / "worker" / "ml-worker.example.yaml").read_text(encoding="utf-8")
    )
    fall_cfg = example["models"]["fall"]
    configured_schema = fall_cfg["schema_version"]
    configured_identity = fall_cfg["preprocessing_identity"]

    runner = LstmFallRunner.from_artifact_dir(
        str(artifact_dir),
        device="cpu",
        expected_schema_version=configured_schema,
        expected_preprocessing_identity=configured_identity,
    )

    assert runner.schema_version == configured_schema
    assert runner.preprocessing_identity == configured_identity


def test_example_config_fall_contract_boots_against_a_synthesized_artifact(
    tmp_path: Path,
) -> None:
    """CI-runnable companion to the local-artifact test above.

    ``models/`` is gitignored, so the test above only skips-or-runs
    depending on whether a real ``models/fall/lstm`` artifact happens to be
    checked out locally -- it never runs in CI. This test closes that gap by
    synthesizing a legacy-shaped artifact entirely in-process (a real
    ``_LstmNet``-shaped ``torch.nn.Module`` with its ``state_dict()`` saved
    to ``model.pt``, a matching ``arch.json``, and a ``metadata.yaml`` with
    all loader-required fields but -- critically -- no ``schema_version`` or
    ``preprocessing_identity`` keys, the same shape as the real shipped
    artifact and as ``eldercare-dataset-ops``'s
    ``ml/training/model_artifacts.py::build_fall_lstm_metadata``, which
    never writes those two fields) and then boots
    ``LstmFallRunner.from_artifact_dir`` against the pins read live from
    ``worker/ml-worker.example.yaml``.

    This has no ``real_stack`` marker and no dependency on ``models/`` being
    present, so it runs in default CI. It proves the shipped example config
    actually boots the contract it documents, and it demonstrably catches
    drift: mismatching either pin (verified manually while authoring this
    test, then reverted) makes ``from_artifact_dir`` raise the real
    ``ModelLoadError`` from ``_validate_expected_identity``
    (``worker/adapters/model/torch_lstm_fall.py``), not a mocked one.
    """
    example = yaml.safe_load(
        (REPO_ROOT / "worker" / "ml-worker.example.yaml").read_text(encoding="utf-8")
    )
    fall_cfg = example["models"]["fall"]
    configured_schema = fall_cfg["schema_version"]
    configured_identity = fall_cfg["preprocessing_identity"]
    window = fall_cfg["window"]

    artifact_dir = tmp_path / "fall-lstm"
    artifact_dir.mkdir()

    hidden, layers, dropout = 4, 1, 0.0
    module = build_lstm_module(hidden=hidden, layers=layers, dropout=dropout)
    torch.save(module.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "arch.json").write_text(
        json.dumps({"hidden": hidden, "layers": layers, "dropout": dropout}),
        encoding="utf-8",
    )
    metadata = {
        "type": "lstm",
        "framework": "pytorch",
        "mode": "sequence",
        "artifact_dir": str(artifact_dir),
        "weights": "model.pt",
        "architecture": "arch.json",
        "metadata": "metadata.yaml",
        "window": window,
        "stride": fall_cfg["stride"],
        "input_shape": [window, 51],
        "operating_threshold": fall_cfg["operating_threshold"],
        # Deliberately no schema_version / preprocessing_identity keys: the
        # manifest's legacy defaults must apply, matching the real shipped
        # artifact and build_fall_lstm_metadata's actual output.
    }
    (artifact_dir / "metadata.yaml").write_text(
        yaml.safe_dump(metadata), encoding="utf-8"
    )

    runner = LstmFallRunner.from_artifact_dir(
        str(artifact_dir),
        device="cpu",
        expected_schema_version=configured_schema,
        expected_preprocessing_identity=configured_identity,
    )

    assert runner.schema_version == configured_schema
    assert runner.preprocessing_identity == configured_identity
