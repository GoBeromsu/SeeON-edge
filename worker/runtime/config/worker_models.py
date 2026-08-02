from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import ClassVar, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import override

from worker.runtime.config.camera_models import CameraRuntimeConfig, RelayConfig
from worker.runtime.config.domain_models import DomainsConfig
from worker.runtime.config.errors import ConfigValidationError, WorkerConfigError

RELAY_ALERTS_PATH: Final = "/api/v1/relay/alerts"
RELAY_HEARTBEAT_PATH: Final = "/api/v1/relay/heartbeat"

ConfigValue: TypeAlias = (
    str | int | float | bool | None | list["ConfigValue"] | dict[str, "ConfigValue"]
)


class WorkerRuntimeConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    max_failures: int = Field(default=30, gt=0)
    open_timeout_ms: int = Field(default=5000, gt=0)
    read_timeout_ms: int = Field(default=5000, gt=0)


class FallModelConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    type: Literal["lstm"]
    framework: Literal["pytorch"]
    mode: Literal["sequence"]
    artifact_dir: Path
    weights: str = Field(default="model.pt", min_length=1)
    architecture: str = Field(default="arch.json", min_length=1)
    metadata: str = Field(default="metadata.yaml", min_length=1)
    window: int = Field(gt=0)
    stride: int = Field(gt=0)
    input_shape: tuple[int, int]
    operating_threshold: float = Field(ge=0.0, le=1.0)
    schema_version: int | None = Field(default=None, ge=1)
    preprocessing_identity: str | None = Field(default=None, min_length=1)

    @field_validator("artifact_dir")
    @classmethod
    def _expand_artifact_dir(cls, value: Path) -> Path:
        return Path(os.path.expanduser(str(value))).resolve()

    @field_validator("metadata")
    @classmethod
    def _require_metadata_yaml(cls, value: str) -> str:
        if value != "metadata.yaml":
            raise ConfigValidationError("metadata must be metadata.yaml")
        return value

    @model_validator(mode="after")
    def _validate_artifact_contract(self) -> FallModelConfig:
        if self.input_shape != (self.window, 51):
            raise ConfigValidationError("input_shape must be [window, 51]")
        for relative in (self.weights, self.architecture, self.metadata):
            if not (self.artifact_dir / relative).exists():
                raise ConfigValidationError(
                    f"missing {relative} at configured artifact directory"
                )
        return self


class WorkerModelsConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    fall: FallModelConfig | None = None


class DevMjpegConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8090, gt=0, le=65535)


class ClipRecordingConfig(BaseModel):
    """Whether this worker records evidence clips at all.

    Default off. Clip recording is an opt-in capability: when it is disabled
    the worker still detects, stages, and relays events -- it simply does not
    build a ``ClipRecorder`` or any per-camera clip feeder, so no video is
    captured or retained.

    This is independent of, and takes no precedence over, the
    ``ML_WORKER_EVENT_CLIP_EXPORT_ENABLED`` environment gate. That gate is the
    export *sender* opt-in (whether staged evidence is shipped to the relay);
    this flag controls whether a *recorder* exists in the first place. All four
    combinations are valid, and evidence delivery is composed either way.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False


class WorkerConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    relay: RelayConfig
    runtime: WorkerRuntimeConfig = Field(default_factory=WorkerRuntimeConfig)
    models: WorkerModelsConfig = Field(default_factory=WorkerModelsConfig)
    domains: DomainsConfig = Field(default_factory=DomainsConfig)
    dev_mjpeg: DevMjpegConfig = Field(default_factory=DevMjpegConfig)
    clip: ClipRecordingConfig = Field(default_factory=ClipRecordingConfig)
    cameras: tuple[CameraRuntimeConfig, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_backend_ingest(cls, data: ConfigValue) -> ConfigValue:
        if not isinstance(data, dict):
            return data
        legacy_fields = [
            name for name in ("ingest", "alert_api_url", "heartbeat_api_url") if name in data
        ]
        raw_cameras = data.get("cameras")
        if isinstance(raw_cameras, list):
            for index, camera in enumerate(raw_cameras):
                if not isinstance(camera, dict):
                    continue
                for name in ("ingest_key_id", "ingest_secret"):
                    if name in camera:
                        legacy_fields.append(f"cameras.{index}.{name}")
        if legacy_fields:
            raise ConfigValidationError(
                "worker config must use relay only; backend ingest fields are forbidden: "
                + ", ".join(legacy_fields)
            )
        return data

    @override
    def model_post_init(self, __context: None) -> None:
        duplicate_ids = sorted(
            camera_id
            for camera_id, count in Counter(camera.camera_id for camera in self.cameras).items()
            if count > 1
        )
        if duplicate_ids:
            raise WorkerConfigError("duplicate camera_id: " + ", ".join(duplicate_ids))

    @property
    def enabled_domains(self) -> tuple[str, ...] | None:
        if "domains" not in self.model_fields_set:
            return None
        return self.domains.enabled_domains or ()

    @property
    def relay_alert_url(self) -> str:
        return f"{self.relay.url}{RELAY_ALERTS_PATH}"

    @property
    def relay_heartbeat_url(self) -> str:
        return f"{self.relay.url}{RELAY_HEARTBEAT_PATH}"


__all__ = [
    "RELAY_ALERTS_PATH",
    "RELAY_HEARTBEAT_PATH",
    "ClipRecordingConfig",
    "ConfigValue",
    "DevMjpegConfig",
    "FallModelConfig",
    "WorkerConfig",
    "WorkerConfigError",
    "WorkerModelsConfig",
    "WorkerRuntimeConfig",
]
