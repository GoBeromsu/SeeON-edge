"""Worker runtime-config pull contract (ml-api camera registry SSOT -> worker).

Dependency-light shared shape for the config the worker pulls from ml-api.
The worker roster is authoritative at ``/api/v1/cameras/worker-config`` and is
derived from the ml-api camera registry; ``/api/v1/relay/config`` is only a
backward-compatible alias. Backend-pulled ML settings such as night window and
config version are optional metadata on that same response, so the worker does
not consume a second config authority.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Worker-facing paths (full suffixes mounted under the /api/v1 app prefix).
WORKER_CONFIG_PATH = "/api/v1/cameras/worker-config"
WORKER_RESTART_PATH = "/api/v1/relay/restart"

CONFIG_VERSION_KEY = "config_version"
RESTART_EPOCH_KEY = "restart_epoch"


@dataclass(frozen=True, slots=True)
class PulledCameraConfig:
    camera_id: str
    space_id: str
    label: str
    rtsp_url: str | None
    online: bool
    space_name: str | None = None
    floor_name: str | None = None
    created_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PulledCameraConfig:
        return cls(
            camera_id=_require_str(data, "camera_id"),
            space_id=_require_str(data, "space_id"),
            label=_require_str(data, "label"),
            rtsp_url=_optional_str(data, "rtsp_url"),
            online=bool(data.get("online", False)),
            space_name=_optional_str(data, "space_name"),
            floor_name=_optional_str(data, "floor_name"),
            created_at=_optional_str(data, "created_at"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "space_id": self.space_id,
            "label": self.label,
            "rtsp_url": self.rtsp_url,
            "online": self.online,
            "space_name": self.space_name,
            "floor_name": self.floor_name,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class PulledNightWindow:
    start: str
    end: str
    tz: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PulledNightWindow:
        return cls(
            start=_require_str(data, "start"),
            end=_require_str(data, "end"),
            tz=_require_str(data, "tz"),
        )

    def as_dict(self) -> dict[str, object]:
        return {"start": self.start, "end": self.end, "tz": self.tz}


@dataclass(frozen=True, slots=True)
class PulledWorkerConfig:
    config_version: int
    restart_epoch: int
    night_window: PulledNightWindow | None
    cameras: tuple[PulledCameraConfig, ...]
    # Per-domain detection windows, keyed by domain name (e.g. "bed_exit",
    # "fall"). ``night_window`` above is a deprecated alias kept in sync with
    # ``detection_windows.get("bed_exit")`` for old callers/workers.
    detection_windows: Mapping[str, PulledNightWindow] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PulledWorkerConfig:
        raw_cameras = data.get("cameras") or []
        if not isinstance(raw_cameras, list):
            raise TypeError("cameras must be a list")
        detection_windows = _pulled_detection_windows(data)
        return cls(
            config_version=_require_int(data, "config_version"),
            restart_epoch=_require_int(data, "restart_epoch"),
            night_window=detection_windows.get("bed_exit"),
            cameras=tuple(
                PulledCameraConfig.from_dict(item)
                for item in raw_cameras
                if isinstance(item, dict)
            ),
            detection_windows=detection_windows,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "restart_epoch": self.restart_epoch,
            "night_window": (
                None if self.night_window is None else self.night_window.as_dict()
            ),
            "cameras": [camera.as_dict() for camera in self.cameras],
            "detection_windows": {
                domain: window.as_dict() for domain, window in self.detection_windows.items()
            },
        }


def _pulled_detection_windows(
    data: dict[str, object],
) -> Mapping[str, PulledNightWindow]:
    raw_map = data.get("detection_windows")
    if isinstance(raw_map, dict):
        windows: dict[str, PulledNightWindow] = {}
        for domain, window in raw_map.items():
            if not isinstance(domain, str):
                continue
            if window is None:
                # Explicit null entry: dropped, domain absent from the map
                # means ALWAYS/24-7 for that domain. Not an error, so no log.
                continue
            if not isinstance(window, dict):
                _log_invalid_window(domain, window, "must be an object or null")
                continue
            validated = _validated_night_window(domain, window)
            if validated is not None:
                windows[domain] = validated
        return MappingProxyType(windows)
    raw_window = data.get("night_window")
    if isinstance(raw_window, dict):
        validated = _validated_night_window("bed_exit", raw_window)
        if validated is not None:
            return MappingProxyType({"bed_exit": validated})
    return MappingProxyType({})


def _validated_night_window(domain: str, window: dict[str, object]) -> PulledNightWindow | None:
    """Parse and validate a raw window dict, failing open (dropping the
    entry, i.e. resolving to ALWAYS/24-7 for that domain) rather than
    raising, so one malformed domain never brings down the whole pull."""
    try:
        parsed = PulledNightWindow.from_dict(window)
    except (ValueError, TypeError) as exc:
        _log_invalid_window(domain, window, str(exc))
        return None
    reason = detection_window_validation_error(parsed.start, parsed.end, parsed.tz)
    if reason is not None:
        _log_invalid_window(domain, window, reason)
        return None
    return parsed


def detection_window_validation_error(start: str, end: str, tz: str) -> str | None:
    """Return a reason the ``(start, end, tz)`` triple is invalid or
    degenerate, or ``None`` if it is a well-formed, non-empty window.

    Shared by every boundary that parses a detection window out of raw,
    externally-supplied data (this module's own dict parsing, the backend's
    ``lifespan.py`` pull, and the worker's ``pull_models.py`` ml-api pull) so
    they all fail open onto ALWAYS/24-7 detection under the same rules
    instead of ever silently disabling a domain. ``start == end`` is
    included because ``DetectionWindow.contains`` treats an equal
    start/end pair as an empty window that matches nothing -- the opposite
    of fail-open -- and a UI time picker can produce it trivially.
    """
    try:
        start_time = datetime.strptime(start, "%H:%M").time()
    except ValueError:
        return f"start={start!r} is not a valid HH:MM time"
    try:
        end_time = datetime.strptime(end, "%H:%M").time()
    except ValueError:
        return f"end={end!r} is not a valid HH:MM time"
    if start_time == end_time:
        return f"start and end are both {start!r}; an equal start/end window matches nothing"
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        return f"tz={tz!r} is not a recognized IANA timezone"
    return None


def _log_invalid_window(domain: str, value: object, reason: str) -> None:
    print(
        f"detection window for domain {domain!r} is invalid ({reason}): {value!r}; "
        "falling open to ALWAYS/24-7 detection for this domain",
        file=sys.stderr,
    )


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _require_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    # bool is an int subclass; reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


__all__ = [
    "WORKER_CONFIG_PATH",
    "WORKER_RESTART_PATH",
    "CONFIG_VERSION_KEY",
    "RESTART_EPOCH_KEY",
    "PulledCameraConfig",
    "PulledNightWindow",
    "PulledWorkerConfig",
    "detection_window_validation_error",
]
