from __future__ import annotations

import sys
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from contracts.worker_config import (
    PulledCameraConfig,
    PulledNightWindow,
    PulledWorkerConfig,
    detection_window_validation_error,
)
from worker.runtime.config.camera_models import CameraRuntimeConfig, RelayConfig
from worker.runtime.config.domain_models import DomainsConfig, NightWindowConfig
from worker.runtime.config.errors import ConfigValidationError, WorkerConfigError
from worker.runtime.config.restart import RestartDirective
from worker.runtime.config.worker_models import WorkerConfig


class _NightWindowPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    start: str = Field(min_length=1)
    end: str = Field(min_length=1)
    tz: str = Field(min_length=1)


class _CameraPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    camera_id: str = Field(min_length=1)
    facility_id: str | None = Field(default=None, min_length=1)
    space_id: str | None = Field(default=None, min_length=1)
    label: str | None = Field(default=None, min_length=1)
    rtsp_url: str | None = None
    online: bool = True
    space_name: str | None = None
    floor_name: str | None = None
    created_at: str | None = None
    fps: float | None = Field(default=None, gt=0)
    decode_backend: str | None = None
    domains: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _require_location(self) -> _CameraPayload:
        if self.facility_id is None and self.space_id is None:
            raise ConfigValidationError("camera must include facility_id or space_id")
        return self

    @property
    def resolved_facility_id(self) -> str:
        return self.facility_id or self.space_id or ""

    @property
    def resolved_space_id(self) -> str:
        return self.space_id or self.facility_id or ""


class BackendWorkerConfigPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    registry_version: int | None = Field(default=None, ge=0)
    config_version: int | None = Field(default=None, ge=0)
    restart_epoch: int | None = Field(default=None, ge=0)
    # Deprecated alias for detection_windows["bed_exit"]; kept for old
    # payload producers/LKG files.
    night_window: _NightWindowPayload | None = None
    # A ``None`` value for one domain (e.g. a hand-edited LKG file, or a
    # version-skewed ml-api) means ALWAYS for that domain and is dropped at
    # parse time in ``resolved_detection_windows`` -- it must not fail
    # pydantic validation for the whole payload over one domain's opinion.
    #
    # The value type is left as ``object`` (not ``_NightWindowPayload | None``)
    # so a member with the *wrong shape* (a string, or an object missing
    # start/end/tz) does not fail pydantic's dict validation for the whole
    # payload either -- shape and semantic validation both happen per-domain
    # in ``resolved_detection_windows``, mirroring
    # ``contracts/worker_config.py``'s ``_pulled_detection_windows`` and
    # ``backend/app/lifespan.py``'s ``_pulled_night_window``.
    detection_windows: dict[str, object] | None = None
    # Same reasoning as ``detection_windows`` above: left as ``object`` so one
    # malformed camera entry (bad field type, missing required field) is
    # dropped per-camera in ``resolved_cameras`` rather than rejecting the
    # whole payload.
    cameras: tuple[object, ...]

    @model_validator(mode="after")
    def _require_version(self) -> BackendWorkerConfigPayload:
        if self.registry_version is None and self.config_version is None:
            raise ConfigValidationError("worker config payload must include a version")
        return self

    @property
    def resolved_registry_version(self) -> int:
        return self.registry_version if self.registry_version is not None else 0

    @property
    def directive(self) -> RestartDirective:
        version = self.config_version
        if version is None:
            version = self.resolved_registry_version
        return RestartDirective(
            generation=self.restart_epoch or 0,
            version=version,
        )

    @property
    def resolved_detection_windows(self) -> dict[str, PulledNightWindow]:
        """Per-domain windows, preferring ``detection_windows`` over the
        deprecated single ``night_window`` (mapped to "bed_exit").

        A member with the wrong shape (not an object, or missing
        start/end/tz) and a well-shaped but invalid/degenerate window (bad
        HH:MM, unknown tz, start == end) both fail open to ALWAYS/24-7
        detection for that one domain -- dropped from the map with a loud
        stderr log -- rather than raising and discarding the whole pulled
        payload.
        """
        if self.detection_windows is not None:
            windows: dict[str, PulledNightWindow] = {}
            for domain, window in self.detection_windows.items():
                if window is None:
                    # Explicit null for this domain: ALWAYS, not an error.
                    continue
                if not isinstance(window, dict):
                    _log_invalid_detection_window(domain, window, "must be an object or null")
                    continue
                try:
                    payload = _NightWindowPayload.model_validate(window)
                except ValidationError as exc:
                    _log_invalid_detection_window(domain, window, _validation_error_reason(exc))
                    continue
                validated = _validated_pulled_window(domain, payload)
                if validated is not None:
                    windows[domain] = validated
            return windows
        window = self.night_window
        if window is None:
            return {}
        validated = _validated_pulled_window("bed_exit", window)
        return {} if validated is None else {"bed_exit": validated}

    @property
    def resolved_cameras(self) -> tuple[_CameraPayload, ...]:
        """Camera roster entries, degrading like ``resolved_detection_windows``:
        an entry with the wrong shape (not an object, a bad field type such
        as ``fps``, or missing the required facility_id/space_id location)
        is dropped and logged by camera identity rather than failing the
        whole payload.
        """
        cameras: list[_CameraPayload] = []
        for entry in self.cameras:
            if not isinstance(entry, dict):
                _log_invalid_camera(entry, "must be an object")
                continue
            try:
                camera = _CameraPayload.model_validate(entry)
            except ValidationError as exc:
                _log_invalid_camera(entry, _validation_error_reason(exc))
                continue
            cameras.append(camera)
        return tuple(cameras)

    def to_pulled_config(self) -> PulledWorkerConfig:
        detection_windows = self.resolved_detection_windows
        return PulledWorkerConfig(
            config_version=self.directive.version,
            restart_epoch=self.directive.generation,
            night_window=detection_windows.get("bed_exit"),
            cameras=tuple(
                PulledCameraConfig(
                    camera_id=camera.camera_id,
                    space_id=camera.resolved_space_id,
                    label=camera.label or camera.camera_id,
                    rtsp_url=camera.rtsp_url,
                    online=camera.online,
                    space_name=camera.space_name,
                    floor_name=camera.floor_name,
                    created_at=camera.created_at,
                )
                for camera in self.resolved_cameras
            ),
            detection_windows=detection_windows,
        )

    def to_worker_config(self, relay_url: str, relay_token: str | None) -> WorkerConfig:
        token = "" if relay_token is None else relay_token.strip()
        if not token:
            raise WorkerConfigError("RELAY_TOKEN is required for pulled worker config")
        resolved_cameras = self.resolved_cameras
        cameras = tuple(
            _runtime_camera(camera)
            for camera in resolved_cameras
            if camera.rtsp_url is not None
        )
        if not cameras:
            raise WorkerConfigError("worker config must include at least one camera")
        domains = tuple(sorted({name for camera in resolved_cameras for name in camera.domains}))
        detection_windows = {
            domain: NightWindowConfig(start=window.start, end=window.end, tz=window.tz)
            for domain, window in self.resolved_detection_windows.items()
        }
        return WorkerConfig(
            relay=RelayConfig.model_validate({"url": relay_url, "token": token}),
            domains=DomainsConfig(
                enabled=domains or None,
                detection_windows=detection_windows or None,
            ),
            cameras=cameras,
        )


def _validated_pulled_window(
    domain: str, window: _NightWindowPayload
) -> PulledNightWindow | None:
    reason = detection_window_validation_error(window.start, window.end, window.tz)
    if reason is not None:
        print(
            f"detection window for domain {domain!r} is invalid ({reason}): "
            f"start={window.start!r} end={window.end!r} tz={window.tz!r}; "
            "falling open to ALWAYS/24-7 detection for this domain",
            file=sys.stderr,
        )
        return None
    return PulledNightWindow(start=window.start, end=window.end, tz=window.tz)


def _log_invalid_detection_window(domain: str, value: object, reason: str) -> None:
    print(
        f"detection window for domain {domain!r} is invalid ({reason}): {value!r}; "
        "falling open to ALWAYS/24-7 detection for this domain",
        file=sys.stderr,
    )


def _log_invalid_camera(value: object, reason: str) -> None:
    identifier = value.get("camera_id") if isinstance(value, dict) else None
    label = f"camera_id={identifier!r}" if identifier else f"entry={value!r}"
    print(
        f"camera config entry is invalid ({reason}): {label}; dropping this camera",
        file=sys.stderr,
    )


def _validation_error_reason(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )


def _runtime_camera(payload: _CameraPayload) -> CameraRuntimeConfig:
    if payload.rtsp_url is None:
        raise WorkerConfigError("worker camera is missing an RTSP URL")
    return CameraRuntimeConfig(
        camera_id=payload.camera_id,
        facility_id=payload.resolved_facility_id,
        rtsp_url=payload.rtsp_url,
        fps=payload.fps or 5.0,
        decode_backend=payload.decode_backend,
        label=payload.label,
    )


__all__ = ["BackendWorkerConfigPayload"]
