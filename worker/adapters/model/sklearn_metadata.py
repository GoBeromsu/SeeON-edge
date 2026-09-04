from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from worker.adapters.model.errors import ModelLoadError

EXPECTED_WINDOW: Final = 30
EXPECTED_STRIDE: Final = 5
EXPECTED_FEATURE_DIM: Final = 45
DEFAULT_OPERATING_THRESHOLD: Final = 0.09

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    model_type: Literal["random-forest"]
    framework: Literal["sklearn"]
    mode: Literal["features"]
    window: int
    stride: int
    feature_dim: int
    name: str
    version: str
    operating_threshold: float | None
    source: str
    reacquire: str | None
    artifact_digest: str | None

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> ModelMetadata:
        model_type = _required_string(data, "model_type")
        framework = _required_string(data, "framework")
        if model_type != "random-forest":
            raise ModelLoadError(f"unsupported model_type {model_type!r}")
        if framework != "sklearn":
            raise ModelLoadError(f"unsupported framework {framework!r}")

        window = _required_positive_int(data, "window")
        stride = _required_positive_int(data, "stride")
        feature_dim = _required_positive_int(data, "feature_dim")
        if window != EXPECTED_WINDOW:
            raise ModelLoadError(f"metadata window must be {EXPECTED_WINDOW}, received {window}")
        if stride != EXPECTED_STRIDE:
            raise ModelLoadError(f"metadata stride must be {EXPECTED_STRIDE}, received {stride}")
        if feature_dim != EXPECTED_FEATURE_DIM:
            raise ModelLoadError(
                f"metadata feature_dim must be {EXPECTED_FEATURE_DIM}, received {feature_dim}"
            )

        return cls(
            model_type="random-forest",
            framework="sklearn",
            mode="features",
            window=window,
            stride=stride,
            feature_dim=feature_dim,
            name=_required_string(data, "name"),
            version=_required_string(data, "version"),
            operating_threshold=_optional_threshold(data.get("operating_threshold")),
            source=_source(data.get("source")),
            reacquire=_optional_string(data.get("reacquire")),
            artifact_digest=_optional_digest(data.get("artifact_digest")),
        )

    def asdict(self) -> dict[str, JsonValue]:
        return {
            "model_type": self.model_type,
            "framework": self.framework,
            "mode": self.mode,
            "window": self.window,
            "stride": self.stride,
            "feature_dim": self.feature_dim,
            "name": self.name,
            "version": self.version,
            "operating_threshold": self.operating_threshold,
            "source": self.source,
            "reacquire": self.reacquire,
            "artifact_digest": self.artifact_digest,
        }


def _required_string(data: dict[str, JsonValue], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ModelLoadError(f"metadata {key} must be a non-empty string")
    return value


def _optional_string(value: JsonValue, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str) or not value:
        raise ModelLoadError("metadata provenance values must be non-empty strings")
    return value


def _source(value: JsonValue) -> str:
    source = _optional_string(value, default="trained")
    if source is None:
        raise ModelLoadError("metadata source must be a non-empty string")
    return source


def _optional_digest(value: JsonValue) -> str | None:
    digest = _optional_string(value)
    if digest is None:
        return None
    if (
        len(digest) != 64
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ModelLoadError("metadata artifact_digest must be lowercase SHA-256")
    return digest


def _required_positive_int(data: dict[str, JsonValue], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelLoadError(f"metadata {key} must be a positive integer")
    return value


def _optional_threshold(value: JsonValue) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelLoadError("metadata operating_threshold must be numeric")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ModelLoadError("metadata operating_threshold must be between 0 and 1")
    return threshold


__all__ = [
    "DEFAULT_OPERATING_THRESHOLD",
    "EXPECTED_FEATURE_DIM",
    "EXPECTED_STRIDE",
    "EXPECTED_WINDOW",
    "JsonValue",
    "ModelMetadata",
]
