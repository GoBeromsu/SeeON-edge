"""Camera registry value parsing and privacy projections."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict
from urllib.parse import parse_qsl, urlsplit, urlunsplit

CameraStatus = Literal["online", "offline", "starting", "unknown"]


class CameraRegistryData(TypedDict):
    registry_version: int
    cameras: list[dict[str, object]]


ProbeErrorClass = Literal["timeout", "decode", "auth", "unsupported"]
FLOOR_MIN = -1
FLOOR_MAX = 10
DEFAULT_FLOOR = 1
FLOOR_VALUES: tuple[int, ...] = (FLOOR_MIN, *range(1, FLOOR_MAX + 1))
_FLOOR_BASEMENT_RE = re.compile(r"^B\s*(\d+)$", re.IGNORECASE)
_FLOOR_LEVEL_RE = re.compile(r"^(-?\d+)\s*층$")
_FLOOR_PLAIN_INT_RE = re.compile(r"^-?\d+$")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    ok: bool
    error_class: ProbeErrorClass | None = None
    width: int | None = None
    height: int | None = None
    probe_unavailable: bool = False


class DuplicateCameraError(Exception):
    def __init__(self, existing_record: dict[str, object]) -> None:
        super().__init__("a camera is already registered for this stream")
        self.existing_record = existing_record


def is_valid_floor(value: int) -> bool:
    return value in FLOOR_VALUES


def floor_label(value: int) -> str:
    return f"B{-value}" if value < 0 else f"{value}층"


def parse_legacy_floor(value: object, *, camera_id: str | None = None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, bool) and isinstance(value, int):
        return value if is_valid_floor(value) else _floor_parse_failed(value, camera_id)
    if isinstance(value, str):
        text = value.strip()
        basement = _FLOOR_BASEMENT_RE.match(text)
        level = _FLOOR_LEVEL_RE.match(text)
        if basement is not None:
            candidate = -int(basement.group(1))
        elif level is not None:
            candidate = int(level.group(1))
        elif _FLOOR_PLAIN_INT_RE.match(text):
            candidate = int(text)
        else:
            return _floor_parse_failed(value, camera_id)
        return candidate if is_valid_floor(candidate) else _floor_parse_failed(value, camera_id)
    return _floor_parse_failed(value, camera_id)


def _floor_parse_failed(value: object, camera_id: str | None) -> int:
    logger.warning(
        "camera floor value could not be parsed, defaulting to %s (camera_id=%s, value=%r)",
        DEFAULT_FLOOR,
        camera_id,
        value,
    )
    return DEFAULT_FLOOR


def public_camera(record: dict[str, object]) -> dict[str, object]:
    response: dict[str, object] = {
        "id": str(record.get("id", "")),
        "label": str(record.get("label", "")),
        "rtsp_url_masked": mask_rtsp_url(str(record.get("rtsp_url", ""))),
        "space_id": _optional_str(record.get("space_id")),
        "backend_camera_id": _optional_str(record.get("backend_camera_id")),
        "mapping_pending": _optional_bool(record.get("mapping_pending")),
        "status": _status(record.get("status")),
        "decode_backend": _optional_str(record.get("decode_backend")),
        "floor": parse_legacy_floor(record.get("floor"), camera_id=_optional_str(record.get("id"))),
        "created_at": str(record.get("created_at", "")),
        "never_connected": _optional_bool(record.get("never_connected")),
        "last_ok_at": _optional_str(record.get("last_ok_at")),
        "last_probed_at": _optional_str(record.get("last_probed_at")),
    }
    for key in ("edge_ref", "room_edge_ref"):
        value = _optional_str(record.get(key))
        if value is not None:
            response[key] = value
    return response


def normalize_stream_identity(rtsp_url: str) -> str:
    try:
        parsed = urlsplit(rtsp_url.strip())
    except ValueError:
        return rtsp_url.strip().lower()
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    port_part = "" if port is None or (scheme == "rtsp" and port == 554) else f":{port}"
    path = parsed.path.rstrip("/") if len(parsed.path) > 1 else parsed.path
    query = "&".join(
        f"{key}={value}" for key, value in sorted(parse_qsl(parsed.query, keep_blank_values=True))
    )
    identity = f"{scheme}://{host}{port_part}{path}"
    return f"{identity}?{query}" if query else identity


def mask_rtsp_url(rtsp_url: str) -> str:
    try:
        parsed = urlsplit(rtsp_url)
        port = parsed.port
    except ValueError:
        return "RTSP URL masked"
    if not parsed.hostname:
        return "RTSP URL masked"
    host = "redacted-camera" if port is None else f"redacted-camera:{port}"
    if parsed.username is None and parsed.password is None:
        return urlunsplit(parsed._replace(netloc=host))
    user = "***" if parsed.username is not None else ""
    password = ":***" if parsed.password is not None else ""
    return urlunsplit(parsed._replace(netloc=f"{user}{password}@{host}"))


def status_from_probe(result: ProbeResult) -> CameraStatus:
    return "online" if result.ok else "offline"


def registry_expected_cameras(
    store: CameraRegistryReader | None,
) -> dict[str, dict[str, str | None]]:
    if store is None:
        return {}
    index: dict[str, dict[str, str | None]] = {}
    for record in store.snapshot()["cameras"]:
        local_id = record.get("id")
        backend_id = record.get("backend_camera_id")
        canonical = backend_id or local_id
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        binding = {"camera_id": canonical, "facility_id": None, "resident_id": None}
        index[canonical] = binding
        if isinstance(local_id, str) and local_id and local_id != canonical:
            index[local_id] = binding
    return index


class CameraRegistryReader(Protocol):
    def snapshot(self) -> CameraRegistryData: ...


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _status(value: object) -> CameraStatus:
    if value == "online":
        return "online"
    if value == "offline":
        return "offline"
    if value == "starting":
        return "starting"
    return "unknown"


__all__ = [
    "CameraRegistryData",
    "CameraStatus",
    "DEFAULT_FLOOR",
    "DuplicateCameraError",
    "FLOOR_MAX",
    "FLOOR_MIN",
    "FLOOR_VALUES",
    "ProbeResult",
    "floor_label",
    "is_valid_floor",
    "mask_rtsp_url",
    "normalize_stream_identity",
    "parse_legacy_floor",
    "public_camera",
    "registry_expected_cameras",
    "status_from_probe",
]
