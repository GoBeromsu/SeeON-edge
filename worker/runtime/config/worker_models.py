from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import override

from shared.detection_policies import PolicyBundle, default_policy_bundle
from worker.domains.registry import DOMAIN_REGISTRY
from worker.runtime.config.camera_models import CameraRuntimeConfig, RelayConfig
from worker.runtime.config.domain_models import DomainsConfig
from worker.runtime.config.errors import ConfigValidationError, WorkerConfigError
from worker.runtime.provenance.model_bundle import DesiredModelBundle

RELAY_ALERTS_PATH: Final = "/api/v1/relay/alerts"
RELAY_HEARTBEAT_PATH: Final = "/api/v1/relay/heartbeat"

ConfigValue: TypeAlias = (
    str | int | float | bool | list["ConfigValue"] | dict[str, "ConfigValue"] | None
)


class WorkerRuntimeConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    max_failures: int = Field(default=30, gt=0)
    open_timeout_ms: int = Field(default=5000, gt=0)
    read_timeout_ms: int = Field(default=5000, gt=0)


class FallModelConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    # Issue #65: the fall-model family is config/metadata-driven, not code-pinned.
    # A brand-new AI model family (a different architecture, not a same-family
    # weights version-up) is added by implementing FallV2ModelProtocol and
    # registering a factory in
    # ``worker.adapters.model.fall_family_registry.DEFAULT_FALL_MODEL_FAMILY_REGISTRY``
    # under the same string used here -- no edits to this Literal, and no edits
    # to ``WorkerRuntime._create_fall_model``. An unregistered ``type`` value
    # refuses to boot (fail-closed, ADR-0002); this field only rejects empty
    # strings, the registry decides which values are actually valid.
    type: str = Field(min_length=1)
    framework: Literal["pytorch", "onnxruntime"]
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
        if self.input_shape != (self.window, 56):
            raise ConfigValidationError("input_shape must be [window, 56]")
        for relative in (self.weights, self.architecture, self.metadata):
            if not (self.artifact_dir / relative).exists():
                raise ConfigValidationError(f"missing {relative} at configured artifact directory")
        if self.framework == "onnxruntime" and not (self.artifact_dir / "model.onnx").exists():
            raise ConfigValidationError("missing model.onnx at configured artifact directory")
        return self


class SelectedFallBundleConfig(BaseModel):
    """An admitted selection that replaces the packaged fall model."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    models_root: Path
    desired: DesiredModelBundle

    @field_validator("models_root")
    @classmethod
    def _expand_models_root(cls, value: Path) -> Path:
        return Path(os.path.expanduser(str(value))).resolve()


class WorkerModelsConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    fall: FallModelConfig | None = None
    selected: SelectedFallBundleConfig | None = None
    # Issue #44: which extraction module's boxes are authoritative for the
    # person bounding box consumed downstream. Explicit and defaulted (never
    # an implicit dict-insertion-order winner) -- "pose" only schedules and
    # provisions the pose model; "person" additionally schedules/provisions
    # the person model and its boxes take over in the merge stage.
    box_source: Literal["pose", "person"] = "pose"

    @model_validator(mode="after")
    def _validate_fall_model_source(self) -> WorkerModelsConfig:
        if (self.fall is None) == (self.selected is None):
            raise ConfigValidationError(
                "no fall model configured"
                if self.fall is None
                else "packaged and selected fall models cannot coexist"
            )
        if self.selected is not None and self.box_source != "pose":
            raise ConfigValidationError("selected fall bundle requires box_source=pose")
        return self


class DevMjpegConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8090, gt=0, le=65535)


class ClipRecordingConfig(BaseModel):
    """Whether this worker records evidence clips at all.

    Default on. Clip recording is always-on by default: when disabled the
    worker still detects, stages, and relays events -- it simply does not
    build a ``ClipRecorder`` or any per-camera clip feeder, so no video is
    captured or retained. If the recorder fails to start, the worker keeps
    running (delivery still works); that failure degrades visibly through
    runtime diagnostics (``set_clip_recorder_status``) instead of silently
    disabling clips.

    This is independent of the persisted live clip-export policy: this flag
    controls whether a recorder exists, while the runtime setting controls
    whether already-durable clips may be claimed for relay. Event delivery is
    composed in either state.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    # Legacy field retained for older local YAML/tests. Live clip export is
    # owned by WorkerConfig.clip_export_* + LiveClipExportPolicy; event delivery
    # is always composed when relay credentials are valid.
    delivery_enabled: bool = True
    # Backend-selected clip storage location, relative to the fixed
    # ``CLIP_STORE_DIR`` volume (see ``worker.runtime.worker
    # ._resolved_clip_store_dir``); ``None`` keeps clips at the store root.
    # Populated from the pulled ``clip_store_subdir`` (see
    # ``pull_models.BackendWorkerConfigPayload``), which already validates it
    # (relative, no ``..`` traversal) before it reaches this field -- this
    # validator re-checks anyway since this value ultimately drives filesystem
    # path construction.
    store_subdir: str | None = Field(default=None, min_length=1)

    @field_validator("store_subdir")
    @classmethod
    def _validate_store_subdir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ConfigValidationError(
                "clip.store_subdir must be a relative path without .. segments"
            )
        return value


class WorkerConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    relay: RelayConfig
    runtime: WorkerRuntimeConfig = Field(default_factory=WorkerRuntimeConfig)
    # Config loading settles this local overlay after parsing the optional
    # YAML hatch. A composed runtime always receives a WorkerModelsConfig.
    models: WorkerModelsConfig | None = None
    domains: DomainsConfig = Field(default_factory=DomainsConfig)
    detection_policies: PolicyBundle = Field(default_factory=default_policy_bundle)
    dev_mjpeg: DevMjpegConfig = Field(default_factory=DevMjpegConfig)
    clip: ClipRecordingConfig = Field(default_factory=ClipRecordingConfig)
    clip_export_enabled: bool = False
    clip_export_version: int = Field(default=0, ge=0)
    # Issue #150: an empty roster is a valid boot state, not a config error --
    # a fresh install has zero cameras until an operator registers one, and
    # the worker must still pass its boot gates (profile/device, decode
    # preflight, model load) and stay up so the RTSP probe/MJPEG server is
    # reachable *before* the first camera exists. This is deliberately
    # distinct from "no config at all" (an unreadable/malformed YAML, or a
    # relay pull with neither a fresh payload nor a last-known-good cache),
    # which still refuses to boot exactly as issue #43 intended -- that gate
    # lives upstream of this model, in `load_worker_config`/`config_pull.py`.
    cameras: tuple[CameraRuntimeConfig, ...] = ()

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
                legacy_fields.extend(
                    f"cameras.{index}.{name}"
                    for name in ("ingest_key_id", "ingest_secret")
                    if name in camera
                )
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
    def selected_module_versions(self) -> Mapping[str, int]:
        return self.domains.selected_versions()

    @property
    def enabled_domains(self) -> tuple[str, ...]:
        """Active domain names: the registry's own defaults, overlaid by
        whatever per-domain overrides ``self.domains`` carries (see
        ``DomainsConfig.resolved_overrides``).

        Always resolves to a concrete tuple -- ``DOMAIN_REGISTRY`` is
        iterated directly (never ``self.domains``) so the set of *known*
        domains can never come from config, and "no config at all" still
        yields every registry entry whose own default is enabled rather than
        an empty or undefined set. This is the boot floor: a worker with an
        unreachable relay and zero domain config still resolves fall and
        bed_exit active, because the registry -- not config presence -- is
        what "active" is measured against now. There is no longer a
        fail-open sentinel to fall back to; config that names no override
        for a domain simply defers to the registry, unconditionally.
        """
        if self.domains.versions is not None:
            return tuple(self.domains.versions)
        overrides = self.domains.resolved_overrides()
        return tuple(
            name
            for name, registration in DOMAIN_REGISTRY.items()
            if overrides.get(name, registration.enabled)
        )

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
    "SelectedFallBundleConfig",
    "WorkerConfig",
    "WorkerConfigError",
    "WorkerModelsConfig",
    "WorkerRuntimeConfig",
]
