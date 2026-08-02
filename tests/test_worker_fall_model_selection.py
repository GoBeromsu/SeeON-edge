"""Issue #43: the fall model has no implicit fallback.

Before this change, ``WorkerRuntime._create_fall_model`` fell through to
``self._serving.create("fall")`` whenever ``config.models.fall`` was unset --
silently swapping in whatever the process-wide model registry happened to
provide (historically a random-forest classifier). "Which model ran that
night" must never be answered by an unconfigured default, so the fallback is
gone entirely: an operator who omits ``models.fall`` now gets a refused boot
(``REFUSE_TO_START_EXIT_CODE``, not the generic runtime code), and the
configured LSTM path is otherwise unchanged.

This file covers three levels: (1) the unit-level refusal, proving it never
even reaches the serving client; (2) the unit-level configured path, proving
``LstmFallRunner.from_artifact_dir`` still receives exactly the arguments the
config declares; (3) the full ``WorkerRuntime.run()`` integration path,
proving the refusal surfaces as ``SystemExit`` with the refuse-to-start code
and activates zero cameras.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import pytest

import worker.runtime.worker as worker_module
from contracts.runner import Image, RunnerResult
from worker.runtime import bootstrap
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime


@final
class _ForbiddenServingClient:
    """Fails the test immediately if fall-model construction ever reaches the
    serving client -- proving the refusal happens before any fallback model
    could be requested."""

    def create(self, task: str, **_options: object) -> object:
        raise AssertionError(f"fall model refusal must not call serving.create({task!r})")


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 2
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


@final
class _FakeRunner:
    def __init__(self, task: str) -> None:
        self.task = task
        self.metadata = _FallMetadata()
        self.operating_threshold = 0.5
        self.warmup_count = 0

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("this test must not run model inference")

    def predict(self, _features: object) -> float:
        return 0.0

    def warmup(self) -> None:
        self.warmup_count += 1


@final
class _YoloOnlyServingClient:
    """Provides pose/person/bed runners (consumed before the fall model is
    ever constructed) but must never be asked for "fall": the refusal is
    unconditional, not something that reaches the registry."""

    def create(self, task: str, **_options: object) -> _FakeRunner:
        if task == "fall":
            raise AssertionError("fall model refusal must not call serving.create('fall')")
        return _FakeRunner(task)


def _config(*, with_fall: dict[str, object] | None = None) -> WorkerConfig:
    payload: dict[str, object] = {
        "version": 1,
        "relay": {"url": "http://relay.test", "token": "relay-token"},
        "cameras": [
            {
                "camera_id": "camera-a",
                "facility_id": "facility-a",
                "rtsp_url": "rtsp://example.test/camera-a",
                "heartbeat_interval_sec": 30.0,
            }
        ],
    }
    if with_fall is not None:
        payload["models"] = {"fall": with_fall}
    return WorkerConfig.model_validate(payload)


def _runtime(config: WorkerConfig, serving: object, state_dir: Path) -> WorkerRuntime:
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=serving,
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
    )


def test_create_fall_model_refuses_when_unconfigured_and_never_touches_serving(
    tmp_path: Path,
) -> None:
    runtime = _runtime(_config(), _ForbiddenServingClient(), tmp_path)

    with pytest.raises(RuntimeError, match="fall model must be explicitly configured"):
        runtime._create_fall_model("cpu")  # noqa: SLF001


def _write_fall_artifact(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(b"placeholder")
    (path / "arch.json").write_text('{"hidden":4,"layers":1,"dropout":0.0}', encoding="utf-8")
    (path / "metadata.yaml").write_text("type: lstm\n", encoding="utf-8")
    return path


def test_create_fall_model_uses_the_configured_lstm_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_dir = _write_fall_artifact(tmp_path / "models" / "fall" / "lstm")
    config = _config(
        with_fall={
            "type": "lstm",
            "framework": "pytorch",
            "mode": "sequence",
            "artifact_dir": str(artifact_dir),
            "weights": "model.pt",
            "architecture": "arch.json",
            "metadata": "metadata.yaml",
            "window": 3,
            "stride": 1,
            "input_shape": [3, 51],
            "operating_threshold": 0.5,
            "schema_version": 2,
            "preprocessing_identity": "v2-poses",
        }
    )
    calls: list[tuple[Path, str, int | None, str | None]] = []
    sentinel = object()

    def fake_from_artifact_dir(
        artifact_dir: Path,
        device: str = "cpu",
        *,
        expected_schema_version: int | None = None,
        expected_preprocessing_identity: str | None = None,
        expected_artifact_digest: str | None = None,
    ) -> object:
        del expected_artifact_digest
        calls.append(
            (Path(artifact_dir), device, expected_schema_version, expected_preprocessing_identity)
        )
        return sentinel

    monkeypatch.setattr(worker_module.LstmFallRunner, "from_artifact_dir", fake_from_artifact_dir)
    runtime = _runtime(config, _ForbiddenServingClient(), tmp_path)

    model = runtime._create_fall_model("cuda")  # noqa: SLF001

    assert model is sentinel
    fall_config = config.models.fall
    assert fall_config is not None
    assert calls == [
        (
            fall_config.artifact_dir,
            "cuda",
            fall_config.schema_version,
            fall_config.preprocessing_identity,
        )
    ]


def test_run_refuses_to_start_with_refuse_to_start_exit_code_when_fall_is_unconfigured(
    tmp_path: Path,
) -> None:
    runtime = _runtime(_config(), _YoloOnlyServingClient(), tmp_path)

    with pytest.raises(SystemExit) as exc:
        runtime.run()

    assert exc.value.code == bootstrap.REFUSE_TO_START_EXIT_CODE
    assert runtime.cameras == ()
