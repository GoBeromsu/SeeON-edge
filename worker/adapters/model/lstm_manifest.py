from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, Protocol, TypeAlias, overload, runtime_checkable

import yaml

from worker.adapters.model.errors import ModelLoadError

KPT_VECTOR_DIM: Final = 51
CURRENT_SCHEMA_VERSION: Final = 2
LEGACY_SCHEMA_VERSION: Final = 1
CURRENT_PREPROCESSING_IDENTITY: Final = "coco17-xyc-frame-normalized-zero-fill-v1"
LEGACY_PREPROCESSING_IDENTITY: Final = (
    "legacy-coco17-xyc-frame-normalized-zero-fill-v1"
)
SUPPORTED_PREPROCESSING_IDENTITIES: Final = frozenset(
    {CURRENT_PREPROCESSING_IDENTITY, LEGACY_PREPROCESSING_IDENTITY}
)
YamlValue: TypeAlias = (
    str | int | float | bool | None | list["YamlValue"] | dict[str, "YamlValue"]
)


@runtime_checkable
class _YamlModule(Protocol):
    def safe_load(self, stream: str) -> YamlValue: ...


@dataclass(frozen=True, slots=True)
class LstmMetadata:
    window: int
    stride: int
    mode: Literal["sequence"]


@dataclass(frozen=True, slots=True)
class LstmFallManifest:
    type: Literal["lstm"]
    framework: Literal["pytorch"]
    mode: Literal["sequence"]
    artifact_dir: Path
    weights: str
    architecture: str
    metadata: str
    window: int
    stride: int
    input_shape: tuple[int, int]
    operating_threshold: float
    schema_version: int
    preprocessing_identity: str
    training_fps: float | None = None
    artifact_digest: str | None = None

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> LstmFallManifest:
        root = Path(artifact_dir).expanduser().resolve()
        metadata_path = root / "metadata.yaml"
        if not metadata_path.is_file():
            raise ModelLoadError(f"missing metadata.yaml at {metadata_path}")
        return cls.from_yaml(metadata_path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> LstmFallManifest:
        metadata_path = Path(path).expanduser().resolve()
        try:
            module: ModuleType = yaml
            if not isinstance(module, _YamlModule):
                raise ModelLoadError("PyYAML does not expose a compatible safe_load")
            raw = module.safe_load(metadata_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ModelLoadError(f"cannot read metadata.yaml at {metadata_path}") from exc
        except yaml.YAMLError as exc:
            raise ModelLoadError(f"metadata.yaml is not valid YAML at {metadata_path}") from exc
        if not isinstance(raw, dict):
            raise ModelLoadError("metadata.yaml must contain an object")

        required = (
            "type",
            "framework",
            "mode",
            "artifact_dir",
            "weights",
            "architecture",
            "metadata",
            "window",
            "stride",
            "input_shape",
            "operating_threshold",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ModelLoadError(
                f"metadata.yaml missing required fields: {', '.join(missing)}"
            )

        schema_version = _schema_version(
            raw.get("schema_version", LEGACY_SCHEMA_VERSION)
        )
        manifest = cls(
            type=_literal(raw["type"], "lstm", "type"),
            framework=_literal(raw["framework"], "pytorch", "framework"),
            mode=_literal(raw["mode"], "sequence", "mode"),
            artifact_dir=_resolve_artifact_dir(raw["artifact_dir"], metadata_path.parent),
            weights=_required_string(raw["weights"], "weights"),
            architecture=_required_string(raw["architecture"], "architecture"),
            metadata=_required_string(raw["metadata"], "metadata"),
            window=_positive_int(raw["window"], "window"),
            stride=_positive_int(raw["stride"], "stride"),
            input_shape=_parse_input_shape(raw["input_shape"]),
            training_fps=_optional_positive_float(raw.get("training_fps"), "training_fps"),
            operating_threshold=_threshold(raw["operating_threshold"]),
            schema_version=schema_version,
            preprocessing_identity=_preprocessing_identity(raw, schema_version),
            artifact_digest=_optional_digest(raw.get("artifact_digest")),
        )
        manifest._validate()
        return manifest

    @property
    def weights_path(self) -> Path:
        return self.artifact_dir / self.weights

    @property
    def architecture_path(self) -> Path:
        return self.artifact_dir / self.architecture

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / self.metadata

    def _validate(self) -> None:
        if self.input_shape != (self.window, KPT_VECTOR_DIM):
            raise ModelLoadError(
                f"input_shape must be [{self.window}, {KPT_VECTOR_DIM}]"
            )
        if self.metadata != "metadata.yaml":
            raise ModelLoadError("metadata must be metadata.yaml")
        if self.schema_version not in (LEGACY_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION):
            message = (
                f"unsupported schema_version {self.schema_version}; supported versions are "
                + f"{LEGACY_SCHEMA_VERSION} and {CURRENT_SCHEMA_VERSION}"
            )
            raise ModelLoadError(message)
        if self.preprocessing_identity not in SUPPORTED_PREPROCESSING_IDENTITIES:
            message = (
                "unsupported preprocessing_identity "
                + f"{self.preprocessing_identity!r}; expected one of "
                + f"{sorted(SUPPORTED_PREPROCESSING_IDENTITIES)!r}"
            )
            raise ModelLoadError(message)
        if (
            self.schema_version == CURRENT_SCHEMA_VERSION
            and self.preprocessing_identity != CURRENT_PREPROCESSING_IDENTITY
        ):
            message = (
                "schema_version 2 requires preprocessing_identity "
                + f"{CURRENT_PREPROCESSING_IDENTITY!r}"
            )
            raise ModelLoadError(message)
        for artifact_path in (
            self.weights_path,
            self.architecture_path,
            self.metadata_path,
        ):
            if not artifact_path.is_file():
                raise ModelLoadError(
                    f"missing {artifact_path.name} at {artifact_path}"
                )


def _resolve_artifact_dir(raw: YamlValue, base: Path) -> Path:
    path = Path(str(raw)).expanduser()
    return (base / path if not path.is_absolute() else path).resolve()


@overload
def _literal(
    value: YamlValue, expected: Literal["lstm"], field_name: str
) -> Literal["lstm"]: ...


@overload
def _literal(
    value: YamlValue, expected: Literal["pytorch"], field_name: str
) -> Literal["pytorch"]: ...


@overload
def _literal(
    value: YamlValue, expected: Literal["sequence"], field_name: str
) -> Literal["sequence"]: ...


def _literal(value: YamlValue, expected: str, field_name: str) -> str:
    if value != expected:
        raise ModelLoadError(f"{field_name} must be {expected!r}")
    return expected


def _parse_input_shape(value: YamlValue) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ModelLoadError("input_shape must contain [window, 51]")
    return (
        _positive_int(value[0], "input_shape[0]"),
        _positive_int(value[1], "input_shape[1]"),
    )


def _positive_int(value: YamlValue, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelLoadError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_float(value: YamlValue, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelLoadError(f"{field_name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ModelLoadError(f"{field_name} must be finite and positive")
    return parsed


def _required_string(value: YamlValue, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelLoadError(f"{field_name} must be a non-empty string")
    return value


def _schema_version(value: YamlValue) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelLoadError("schema_version must be an integer")
    return value


def _threshold(value: YamlValue) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelLoadError("operating_threshold must be numeric")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ModelLoadError("operating_threshold must be between 0 and 1")
    return threshold


def _preprocessing_identity(raw: dict[str, YamlValue], schema_version: int) -> str:
    value = raw.get("preprocessing_identity")
    if value is None and schema_version == LEGACY_SCHEMA_VERSION:
        return LEGACY_PREPROCESSING_IDENTITY
    return _required_string(value, "preprocessing_identity")


def _optional_digest(value: YamlValue) -> str | None:
    if value is None:
        return None
    digest = _required_string(value, "artifact_digest")
    if (
        len(digest) != 64
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ModelLoadError("artifact_digest must be lowercase SHA-256")
    return digest


__all__ = [
    "CURRENT_PREPROCESSING_IDENTITY",
    "CURRENT_SCHEMA_VERSION",
    "KPT_VECTOR_DIM",
    "LEGACY_PREPROCESSING_IDENTITY",
    "LEGACY_SCHEMA_VERSION",
    "LstmFallManifest",
    "LstmMetadata",
    "SUPPORTED_PREPROCESSING_IDENTITIES",
]
