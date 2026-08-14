from __future__ import annotations

import json
import math
import pickle
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol, TypeAlias, final, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from typing_extensions import override

from worker.adapters.model.artifact import verify_artifact_digest
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.lstm_manifest import (
    CURRENT_PREPROCESSING_IDENTITY,
    CURRENT_SCHEMA_VERSION,
    KPT_VECTOR_DIM,
    LEGACY_PREPROCESSING_IDENTITY,
    LEGACY_SCHEMA_VERSION,
    SUPPORTED_PREPROCESSING_IDENTITIES,
    LstmFallManifest,
    LstmMetadata,
    YamlValue,
)
from worker.types import FallModelInput

StateDict: TypeAlias = Mapping[str, torch.Tensor]
_TORCH_LOAD: Final[Callable[..., StateDict]] = torch.load


@runtime_checkable
class _JsonModule(Protocol):
    def loads(self, value: str) -> YamlValue: ...


@runtime_checkable
class _TorchModule(Protocol):
    def from_numpy(self, values: NDArray[np.float32]) -> torch.Tensor: ...


class _LstmLayer(Protocol):
    def __call__(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...


class _TensorModule(Protocol):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...


def _run_lstm(
    layer: _LstmLayer, inputs: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    return layer(inputs)


def _run_tensor_module(module: _TensorModule, inputs: torch.Tensor) -> torch.Tensor:
    return module(inputs)


@final
class _LstmNet(nn.Module):
    def __init__(self, *, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=KPT_VECTOR_DIM,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 2)

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = _run_lstm(self.lstm, inputs)
        _, (hidden, _) = result
        return _run_tensor_module(self.fc, hidden[-1])


@final
class LstmFallRunner:
    name: Final = "fall-detector"
    version: Final = "lstm"

    def __init__(self, manifest: LstmFallManifest, module: nn.Module, device: str = "cpu") -> None:
        self.manifest = manifest
        self.metadata = LstmMetadata(
            window=manifest.window,
            stride=manifest.stride,
            mode=manifest.mode,
        )
        self.operating_threshold = manifest.operating_threshold
        self.preprocessing_identity = manifest.preprocessing_identity
        self.schema_version = manifest.schema_version
        self.artifact_digest = manifest.artifact_digest
        try:
            self.device = torch.device(device)
            self.model: nn.Module = module.to(self.device).eval()
        except (RuntimeError, TypeError) as exc:
            raise ModelLoadError(
                f"cannot place LSTM fall model on device {device!r}: {exc}"
            ) from exc

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        device: str = "cpu",
        *,
        expected_schema_version: int | None = None,
        expected_preprocessing_identity: str | None = None,
        expected_artifact_digest: str | None = None,
        operating_threshold: float | None = None,
    ) -> LstmFallRunner:
        manifest = LstmFallManifest.from_artifact_dir(artifact_dir)
        _validate_expected_identity(
            manifest,
            schema_version=expected_schema_version,
            preprocessing_identity=expected_preprocessing_identity,
        )
        declared_digest = manifest.artifact_digest
        if (
            declared_digest is not None
            and expected_artifact_digest is not None
            and declared_digest != expected_artifact_digest
        ):
            raise ModelLoadError(
                "configured artifact digest does not match metadata artifact digest"
            )
        digest = verify_artifact_digest(
            manifest.weights_path,
            expected_artifact_digest if expected_artifact_digest is not None else declared_digest,
        )
        # Issue #217: operating_threshold is a runtime-configured decision
        # boundary, not an artifact identity field like schema_version/
        # preprocessing_identity above -- so it overrides the packaged
        # manifest value outright instead of being validated against it.
        # ``None`` (no override supplied) leaves the manifest's own value in
        # place, which is what keeps a zero-config boot working: the
        # packaged default model must still produce a working detector with
        # no ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD set at all.
        resolved_manifest = replace(
            manifest,
            artifact_digest=digest,
            operating_threshold=(
                manifest.operating_threshold if operating_threshold is None else operating_threshold
            ),
        )
        module = build_lstm_module_from_arch(resolved_manifest.architecture_path)
        state_dict = _load_state_dict(resolved_manifest.weights_path)
        try:
            _ = module.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise ModelLoadError(f"cannot load model.pt state_dict: {exc}") from exc
        return cls(resolved_manifest, module, device=device)

    def predict(self, features: FallModelInput) -> float:
        sequence = np.asarray(features, dtype=np.float32)
        if sequence.shape != self.manifest.input_shape:
            message = (
                "unexpected input shape: "
                + f"expected {self.manifest.input_shape}, received {sequence.shape}"
            )
            raise ModelLoadError(message)
        with torch.no_grad():
            torch_module: ModuleType = torch
            if not isinstance(torch_module, _TorchModule):
                raise ModelLoadError("torch does not expose a compatible from_numpy")
            tensor = torch_module.from_numpy(sequence).unsqueeze(0).to(self.device)
            logits = _run_tensor_module(self.model, tensor)
            if logits.shape != (1, 2):
                raise ModelLoadError(
                    f"LSTM model must return shape (1, 2), received {tuple(logits.shape)}"
                )
            probabilities: torch.Tensor = torch.softmax(logits, dim=1)
        selected: torch.Tensor = probabilities[0, 1]
        scalar: Callable[[], float] = selected.item
        probability = scalar()
        if not math.isfinite(probability):
            raise ModelLoadError("LSTM model returned a non-finite probability")
        return min(max(probability, 0.0), 1.0)

    def warmup(self) -> None:
        window, row_size = self.manifest.input_shape
        zero_row = tuple(0.0 for _ in range(row_size))
        zero_input: FallModelInput = tuple(zero_row for _ in range(window))
        _ = self.predict(zero_input)


def _validate_expected_identity(
    manifest: LstmFallManifest,
    *,
    schema_version: int | None,
    preprocessing_identity: str | None,
) -> None:
    if schema_version is not None and manifest.schema_version != schema_version:
        message = (
            f"artifact schema_version {manifest.schema_version} does not match configured "
            + f"schema_version {schema_version}"
        )
        raise ModelLoadError(message)
    if (
        preprocessing_identity is not None
        and manifest.preprocessing_identity != preprocessing_identity
    ):
        message = (
            "artifact preprocessing_identity "
            + f"{manifest.preprocessing_identity!r} does not match configured "
            + f"preprocessing_identity {preprocessing_identity!r}"
        )
        raise ModelLoadError(message)


def build_lstm_module(*, hidden: int, layers: int, dropout: float) -> _LstmNet:
    return _LstmNet(hidden=hidden, layers=layers, dropout=dropout)


def build_lstm_module_from_arch(path: Path) -> _LstmNet:
    try:
        json_module: ModuleType = json
        if not isinstance(json_module, _JsonModule):
            raise ModelLoadError("json does not expose a compatible loads function")
        raw = json_module.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelLoadError(f"cannot read arch.json at {path}") from exc
    if not isinstance(raw, dict):
        raise ModelLoadError("arch.json must contain an object")
    dropout = raw.get("dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ModelLoadError("dropout must be numeric")
    return build_lstm_module(
        hidden=_positive_int(raw.get("hidden"), "hidden"),
        layers=_positive_int(raw.get("layers"), "layers"),
        dropout=float(dropout),
    )


def _positive_int(value: YamlValue, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelLoadError(f"{field_name} must be a positive integer")
    return value


def _load_state_dict(path: Path) -> StateDict:
    try:
        try:
            return _TORCH_LOAD(path, map_location="cpu", weights_only=True)
        except TypeError:
            return _TORCH_LOAD(path, map_location="cpu")
    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        raise ModelLoadError(f"cannot load model.pt at {path}") from exc


__all__ = [
    "CURRENT_PREPROCESSING_IDENTITY",
    "CURRENT_SCHEMA_VERSION",
    "LEGACY_PREPROCESSING_IDENTITY",
    "LEGACY_SCHEMA_VERSION",
    "LstmFallManifest",
    "LstmFallRunner",
    "LstmMetadata",
    "ModelLoadError",
    "SUPPORTED_PREPROCESSING_IDENTITIES",
    "build_lstm_module",
    "build_lstm_module_from_arch",
]
