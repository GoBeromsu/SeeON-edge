from __future__ import annotations

import sys
from pathlib import PurePosixPath
from typing import ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)

from contracts.worker_config import (
    PulledCameraConfig,
    PulledNightWindow,
    PulledWorkerConfig,
    detection_window_validation_error,
)
from worker.runtime.config.camera_models import CameraRuntimeConfig, RelayConfig
from worker.runtime.config.domain_models import (
    KNOWN_DOMAIN_NAMES,
    BedExitDomainConfig,
    DomainsConfig,
    FallDomainConfig,
    NightWindowConfig,
)
from worker.runtime.config.errors import ConfigValidationError, WorkerConfigError
from worker.runtime.config.restart import RestartDirective
from worker.runtime.config.worker_models import (
    ClipRecordingConfig,
    DevMjpegConfig,
    WorkerConfig,
    WorkerModelsConfig,
)


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
    frame_stride: int | None = Field(default=None, gt=0)
    decode_backend: str | None = None
    # ``None`` (the relay payload omitted this camera's ``domains`` key) and
    # ``()`` (the relay explicitly declared this camera monitors zero
    # domains) are different signals and must stay distinguishable -- see
    # issue #191. Defaulting to ``()`` here would make every camera look
    # like an explicit "no domains" opt-out even when the relay said
    # nothing at all, which is exactly the ambiguity ``to_worker_config``
    # needs to resolve to decide between the ambient registry default and a
    # genuine opt-out.
    domains: tuple[str, ...] | None = None
    bed_zone_polygon: tuple[tuple[int, int], ...] | None = None
    bed_zone_image_width: int | None = Field(default=None, gt=0)
    bed_zone_image_height: int | None = Field(default=None, gt=0)

    @property
    def resolved_facility_id(self) -> str:
        """Local wire facility for RelayAlertRequest (min_length=1).

        Not site identity: when worker-config omits facility_id, use the fixed
        placeholder ``"local"``. space_id alone is not treated as facility.
        """
        if self.facility_id is not None:
            return self.facility_id
        return "local"

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
    # ml-api-local per-domain enable/disable override (see
    # ``backend/app/features/detection_settings``), keyed by domain name,
    # e.g. ``{"fall": {"enabled": true}, "bed_exit": {"enabled": false}}``.
    # Only present once an operator has saved detection settings at least
    # once; absent otherwise, preserving the pre-existing ambient-default
    # behavior (all domains enabled, driven by per-camera ``domains``).
    # Left as ``object`` for the same fail-open reason as
    # ``detection_windows``: a malformed per-domain entry is dropped in
    # ``resolved_domain_enabled`` rather than rejecting the whole payload.
    domains: dict[str, object] | None = None
    # ml-api-local clip storage subdirectory selection (see
    # ``backend/app/features/clips/storage_location_store.py``), relative to
    # the worker's fixed ``CLIP_STORE_DIR`` volume. Left as ``object`` (not
    # ``str | None``) so a malformed value (wrong type, absolute path, ``..``
    # traversal) is dropped in ``resolved_clip_store_subdir`` -- falling back
    # to the store root -- rather than rejecting the whole payload.
    clip_store_subdir: object = None
    # Persisted ml-api runtime policy. Missing fields from old payloads and
    # last-known-good rows are deliberately OFF at version zero.
    clip_export_enabled: StrictBool = False
    clip_export_version: StrictInt = Field(default=0, ge=0)

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
    def resolved_domain_enabled(self) -> dict[str, bool]:
        """Per-domain enable/disable overrides, degrading like
        ``resolved_detection_windows``: a malformed entry (wrong shape, bad
        ``enabled`` type, or an unrecognized domain name) is dropped and
        logged rather than failing the whole payload."""
        if self.domains is None:
            return {}
        resolved: dict[str, bool] = {}
        for domain, value in self.domains.items():
            if not isinstance(domain, str) or domain not in KNOWN_DOMAIN_NAMES:
                _log_invalid_domain_config(domain, value, "unknown or non-string domain")
                continue
            if not isinstance(value, dict):
                _log_invalid_domain_config(domain, value, "must be an object")
                continue
            enabled = value.get("enabled")
            if not isinstance(enabled, bool):
                _log_invalid_domain_config(domain, value, "enabled must be a boolean")
                continue
            resolved[domain] = enabled
        return resolved

    @property
    def resolved_clip_store_subdir(self) -> str | None:
        """The pulled clip storage subdirectory, or ``None`` if absent or
        malformed (falls open to the store root, same fail-open shape as
        ``resolved_detection_windows``)."""
        value = self.clip_store_subdir
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            _log_invalid_clip_store_subdir(value, "must be a non-empty string")
            return None
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            _log_invalid_clip_store_subdir(value, "must be a relative path without .. segments")
            return None
        return value

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

    def to_worker_config(
        self,
        relay_url: str,
        relay_token: str | None,
        *,
        models: WorkerModelsConfig | None = None,
        clip: ClipRecordingConfig | None = None,
        dev_mjpeg: DevMjpegConfig | None = None,
    ) -> WorkerConfig:
        """Build the effective ``WorkerConfig`` from a relay pull.

        The backend-pulled payload only ever carries fleet-level state
        (relay/domains/cameras); ``models``/``clip``/``dev_mjpeg`` are
        locally-sourced (env, or local YAML when ``--config`` is passed)
        and must be passed in explicitly by the caller (``config_pull.py``'s
        ``resolve_local_overrides``) rather than silently dropped -- see
        issues #66/#68 (models/clip) and #113 (dev_mjpeg: an explicit local
        ``dev_mjpeg.enabled: true`` used to be silently reset to the pydantic
        default -- disabled -- on every successful pull, so the operator
        diagnostic MJPEG port never bound and never logged why).
        """
        token = "" if relay_token is None else relay_token.strip()
        if not token:
            raise WorkerConfigError("RELAY_TOKEN is required for pulled worker config")
        resolved_cameras = self.resolved_cameras
        cameras = tuple(
            _runtime_camera(camera) for camera in resolved_cameras if camera.rtsp_url is not None
        )
        # Issue #150: an *empty roster* (`self.cameras == ()`) is a legitimate
        # config now -- a fresh install has no cameras until an operator
        # registers one, and the worker must still boot so its probe/MJPEG
        # server is reachable to validate that first camera's RTSP URL. This
        # used to raise, and `config_pull.py` swallowed it as "malformed
        # payload", so the first registration was structurally impossible.
        #
        # But "the payload declared cameras and every one of them failed to
        # parse" is a different thing entirely, and still malformed. `cameras`
        # is typed `tuple[object, ...]` precisely so one bad entry degrades
        # per-camera in `resolved_cameras` instead of rejecting the payload --
        # if that degradation consumed the *whole* roster, the payload is
        # corrupt and we must not hand back an empty config. Doing so would
        # silently discard a good last-known-good roster and stop monitoring
        # every room. Raising here keeps the LKG fallback in charge.
        #
        # Note this checks `resolved_cameras` (parsed OK), not `cameras`
        # (parsed OK *and* carries an RTSP URL). A roster whose entries all
        # parse but have no `rtsp_url` is a real, non-corrupt state -- a
        # camera registered in the cloud that this edge has no local record
        # for yet -- and must boot with an empty usable roster rather than
        # reject the pull.
        if self.cameras and not resolved_cameras:
            raise WorkerConfigError("worker config declared cameras but none of them parsed")
        detection_windows = {
            domain: NightWindowConfig(start=window.start, end=window.end, tz=window.tz)
            for domain, window in self.resolved_detection_windows.items()
        }
        domain_enabled = self.resolved_domain_enabled
        if self.domains is not None:
            # An explicit local override wins outright over the
            # per-camera-domains-derived set below, but is threaded through
            # as a *partial* per-domain overlay -- only the domains this
            # override actually names -- rather than the legacy
            # replace-list. A domain it never mentions is simply absent from
            # both per-domain fields below, which ``DomainsConfig
            # .resolved_overrides`` reads as "no opinion, defer to the
            # registry" rather than forcing it on (the previous
            # ``domain_enabled.get(name, True)`` hardcoded that default
            # here instead of letting the registry decide it -- the
            # config-replaces-registry defect this overlay exists to fix).
            domains_config = DomainsConfig(
                fall=(
                    FallDomainConfig(enabled=domain_enabled["fall"])
                    if "fall" in domain_enabled
                    else None
                ),
                bed_exit=(
                    BedExitDomainConfig(enabled=domain_enabled["bed_exit"])
                    if "bed_exit" in domain_enabled
                    else None
                ),
                detection_windows=detection_windows or None,
            )
        else:
            # Same None-vs-empty distinction as the override above, one
            # level down: a camera whose ``domains`` is ``None`` said
            # nothing, but a camera whose ``domains`` is ``()`` explicitly
            # opted out, and the union must not blur the two (issue #191).
            camera_declared_domains = any(camera.domains is not None for camera in resolved_cameras)
            camera_domains = (
                tuple(
                    sorted({name for camera in resolved_cameras for name in (camera.domains or ())})
                )
                if camera_declared_domains
                else None
            )
            domains_config = DomainsConfig(
                enabled=camera_domains,
                detection_windows=detection_windows or None,
            )
        # Issue #191's fresh-install failure -- a pull with no domains
        # signal at all (no override, no per-camera domains, no detection
        # windows) silently resolving to zero active domains -- no longer
        # needs special-casing here. ``domains_config`` is always passed
        # through unconditionally: ``WorkerConfig.enabled_domains`` resolves
        # every domain against the registry (``DOMAIN_REGISTRY[name]
        # .enabled``) overlaid by ``domains_config.resolved_overrides()``,
        # and a ``DomainsConfig`` carrying no real signal produces an empty
        # overrides map, which reads as "defer to the registry" rather than
        # "nothing is active" -- whether or not the field was ever "set".
        base_clip = clip if clip is not None else ClipRecordingConfig()
        subdir = self.resolved_clip_store_subdir
        resolved_clip = (
            base_clip if subdir is None else base_clip.model_copy(update={"store_subdir": subdir})
        )
        return WorkerConfig(
            relay=RelayConfig.model_validate({"url": relay_url, "token": token}),
            models=models if models is not None else WorkerModelsConfig(),
            domains=domains_config,
            clip=resolved_clip,
            dev_mjpeg=dev_mjpeg if dev_mjpeg is not None else DevMjpegConfig(),
            clip_export_enabled=self.clip_export_enabled,
            clip_export_version=self.clip_export_version,
            cameras=cameras,
        )


def _validated_pulled_window(domain: str, window: _NightWindowPayload) -> PulledNightWindow | None:
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


def _log_invalid_domain_config(domain: object, value: object, reason: str) -> None:
    print(
        f"domain enable/disable override for domain {domain!r} is invalid ({reason}): "
        f"{value!r}; ignoring this domain's override",
        file=sys.stderr,
    )


def _log_invalid_clip_store_subdir(value: object, reason: str) -> None:
    print(
        f"clip_store_subdir is invalid ({reason}): {value!r}; falling back to the clip store root",
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
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()
    )


def _runtime_camera(payload: _CameraPayload) -> CameraRuntimeConfig:
    if payload.rtsp_url is None:
        raise WorkerConfigError("worker camera is missing an RTSP URL")
    return CameraRuntimeConfig(
        camera_id=payload.camera_id,
        facility_id=payload.resolved_facility_id,
        rtsp_url=payload.rtsp_url,
        fps=payload.fps or 5.0,
        frame_stride=payload.frame_stride or 1,
        decode_backend=payload.decode_backend,
        label=payload.label,
        bed_zone_polygon=payload.bed_zone_polygon,
        bed_zone_image_width=payload.bed_zone_image_width,
        bed_zone_image_height=payload.bed_zone_image_height,
    )


__all__ = ["BackendWorkerConfigPayload"]
