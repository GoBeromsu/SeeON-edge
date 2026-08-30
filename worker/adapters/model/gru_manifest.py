from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, Protocol, TypeAlias, runtime_checkable

import yaml

from worker.adapters.model.errors import ModelLoadError

GRU_WINDOW: Final = 30
GRU_INPUT_DIM: Final = 56
GRU_CLASS_ORDER: Final = ("background", "fall_transition", "fallen")
GRU_SCHEMA_VERSION: Final = 1
GRU_PREPROCESSING_IDENTITY: Final = "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1"
YamlValue: TypeAlias = str | int | float | bool | list["YamlValue"] | dict[str, "YamlValue"] | None


@runtime_checkable
class _YamlModule(Protocol):
    def safe_load(self, stream: str) -> YamlValue: ...


@dataclass(frozen=True, slots=True)
class GruFallManifest:
    type: Literal["gru"]
    framework: Literal["pytorch"]
    mode: Literal["sequence"]
    artifact_dir: Path
    weights: str
    architecture: str
    metadata: str
    input_contract: str
    policy: str
    calibration: str
    conformance: str
    input_shape: tuple[int, int]
    class_order: tuple[str, str, str]
    schema_version: int
    preprocessing_identity: str
    artifact_digest: str
    architecture_digest: str
    input_digest: str
    policy_digest: str
    calibration_digest: str
    conformance_digest: str

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> GruFallManifest:
        root = Path(artifact_dir).expanduser().resolve()
        metadata_path = root / "metadata.yaml"
        if not metadata_path.is_file():
            raise ModelLoadError(f"missing metadata.yaml at {metadata_path}")
        return cls.from_yaml(metadata_path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GruFallManifest:
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
        required = {
            "type",
            "framework",
            "mode",
            "artifact_dir",
            "weights",
            "architecture",
            "metadata",
            "input_contract",
            "policy",
            "calibration",
            "conformance",
            "input_shape",
            "class_order",
            "schema_version",
            "preprocessing_identity",
            "artifact_digest",
            "architecture_digest",
            "input_digest",
            "policy_digest",
            "calibration_digest",
            "conformance_digest",
        }
        missing = sorted(required - raw.keys())
        extra = sorted(raw.keys() - required)
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing required fields: {', '.join(missing)}")
            if extra:
                detail.append(f"unknown fields: {', '.join(extra)}")
            raise ModelLoadError("metadata.yaml " + "; ".join(detail))
        root = _artifact_dir(raw["artifact_dir"], metadata_path.parent)
        manifest = cls(
            type=_literal(raw["type"], "gru", "type"),
            framework=_literal(raw["framework"], "pytorch", "framework"),
            mode=_literal(raw["mode"], "sequence", "mode"),
            artifact_dir=root,
            weights=_file_name(raw["weights"], "weights"),
            architecture=_file_name(raw["architecture"], "architecture"),
            metadata=_file_name(raw["metadata"], "metadata"),
            input_contract=_file_name(raw["input_contract"], "input_contract"),
            policy=_file_name(raw["policy"], "policy"),
            calibration=_file_name(raw["calibration"], "calibration"),
            conformance=_file_name(raw["conformance"], "conformance"),
            input_shape=_input_shape(raw["input_shape"]),
            class_order=_class_order(raw["class_order"]),
            schema_version=_schema_version(raw["schema_version"]),
            preprocessing_identity=_required_string(
                raw["preprocessing_identity"], "preprocessing_identity"
            ),
            artifact_digest=_digest(raw["artifact_digest"], "artifact_digest"),
            architecture_digest=_digest(raw["architecture_digest"], "architecture_digest"),
            input_digest=_digest(raw["input_digest"], "input_digest"),
            policy_digest=_digest(raw["policy_digest"], "policy_digest"),
            calibration_digest=_digest(raw["calibration_digest"], "calibration_digest"),
            conformance_digest=_digest(raw["conformance_digest"], "conformance_digest"),
        )
        manifest._validate()
        return manifest

    def _validate(self) -> None:
        if self.schema_version != GRU_SCHEMA_VERSION:
            raise ModelLoadError(f"schema_version must be {GRU_SCHEMA_VERSION}")
        if self.input_shape != (GRU_WINDOW, GRU_INPUT_DIM):
            raise ModelLoadError(f"input_shape must be [{GRU_WINDOW}, {GRU_INPUT_DIM}]")
        if self.class_order != GRU_CLASS_ORDER:
            raise ModelLoadError(f"class_order must be {list(GRU_CLASS_ORDER)!r}")
        if self.preprocessing_identity != GRU_PREPROCESSING_IDENTITY:
            raise ModelLoadError("unsupported preprocessing_identity")
        if self.metadata != "metadata.yaml":
            raise ModelLoadError("metadata must be metadata.yaml")
        for path in self.bundle_paths:
            if not path.is_file():
                raise ModelLoadError(f"missing {path.name} at {path}")

    @property
    def weights_path(self) -> Path:
        return self.artifact_dir / self.weights

    @property
    def architecture_path(self) -> Path:
        return self.artifact_dir / self.architecture

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / self.metadata

    @property
    def input_contract_path(self) -> Path:
        return self.artifact_dir / self.input_contract

    @property
    def policy_path(self) -> Path:
        return self.artifact_dir / self.policy

    @property
    def calibration_path(self) -> Path:
        return self.artifact_dir / self.calibration

    @property
    def conformance_path(self) -> Path:
        return self.artifact_dir / self.conformance

    @property
    def bundle_paths(self) -> tuple[Path, ...]:
        return (
            self.weights_path,
            self.architecture_path,
            self.metadata_path,
            self.input_contract_path,
            self.policy_path,
            self.calibration_path,
            self.conformance_path,
        )


def _artifact_dir(value: YamlValue, base: Path) -> Path:
    path = Path(_required_string(value, "artifact_dir")).expanduser()
    return (base / path if not path.is_absolute() else path).resolve()


def _literal(value: YamlValue, expected: str, name: str) -> str:
    if value != expected:
        raise ModelLoadError(f"{name} must be {expected!r}")
    return expected


def _required_string(value: YamlValue, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelLoadError(f"{name} must be a non-empty string")
    return value


def _file_name(value: YamlValue, name: str) -> str:
    file_name = _required_string(value, name)
    if Path(file_name).name != file_name:
        raise ModelLoadError(f"{name} must be a file name")
    return file_name


def _input_shape(value: YamlValue) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ModelLoadError("input_shape must contain two integers")
    return value[0], value[1]


def _class_order(value: YamlValue) -> tuple[str, str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(item, str) for item in value)
    ):
        raise ModelLoadError("class_order must contain three strings")
    return value[0], value[1], value[2]


def _schema_version(value: YamlValue) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelLoadError("schema_version must be an integer")
    return value


def _digest(value: YamlValue, name: str) -> str:
    digest = _required_string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ModelLoadError(f"{name} must be lowercase SHA-256")
    return digest


__all__ = [
    "GRU_CLASS_ORDER",
    "GRU_INPUT_DIM",
    "GRU_PREPROCESSING_IDENTITY",
    "GRU_SCHEMA_VERSION",
    "GRU_WINDOW",
    "GruFallManifest",
]
