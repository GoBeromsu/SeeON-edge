"""The HTTP surface the worker's live-view server (``ml-worker:8090``) serves.

The worker is the *provider* of this interface; ml-api is its only consumer
and reaches it server-side (``worker_stream_origin`` / ``worker_probe_origin``).
The two packages never import each other, so each owns its own definition:
this module names the routes the worker matches and the JSON bodies it reads
and writes, ``backend/app/features/cameras/*`` names what the backend sends
and expects, and ``tests/test_backend_worker_runtime_contracts.py`` round-trips
one through the other so drift fails a test instead of a deploy.

Stdlib only -- ``_mjpeg_http.py`` keeps the sockets, the auth gate, and the
frame plumbing; nothing here touches a frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias
from urllib.parse import unquote

# Shared secret header; the token-gated routes fail closed (403) without it.
RELAY_TOKEN_HEADER: Final = "X-Edge-Relay-Token"

# MJPEG multipart framing of ``GET /stream/{camera_id}``.
MJPEG_BOUNDARY: Final = b"frame"
MJPEG_MEDIA_TYPE: Final = f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY.decode()}"

# Fixed routes.
PROBE_PATH: Final = "/probe"
REPLAY_PATH: Final = "/replay"

# Parameterised routes; every identity segment arrives percent-encoded.
STREAM_PREFIX: Final = "/stream/"
SNAPSHOT_PREFIX: Final = "/snapshot/"
OVERLAY_PREFIX: Final = "/overlay/"
POSE_SUFFIX: Final = "/pose"
BED_ZONE_SUFFIX: Final = "/bed-zone/recognize"
CLIPS_SEGMENT: Final = "clips"
CLIP_DELETION_PREFLIGHT_SEGMENT: Final = "deletion-preflight"


def stream_camera_id(path: str) -> str | None:
    """``GET /stream/{camera_id}``; ``""`` for a bare prefix (the handler 404s)."""
    if not path.startswith(STREAM_PREFIX):
        return None
    return unquote(path[len(STREAM_PREFIX) :])


def snapshot_camera_id(path: str) -> str | None:
    """``GET /snapshot/{camera_id}``; ``""`` for a bare prefix (the handler 404s)."""
    if not path.startswith(SNAPSHOT_PREFIX):
        return None
    return unquote(path[len(SNAPSHOT_PREFIX) :])


def pose_camera_id(path: str) -> str | None:
    """``GET|POST /overlay/{camera_id}/pose``; ``None`` for an empty camera id."""
    return _overlay_camera_id(path, POSE_SUFFIX)


def bed_zone_camera_id(path: str) -> str | None:
    """``POST /overlay/{camera_id}/bed-zone/recognize``; ``None`` for an empty id."""
    return _overlay_camera_id(path, BED_ZONE_SUFFIX)


def _overlay_camera_id(path: str, suffix: str) -> str | None:
    if not path.startswith(OVERLAY_PREFIX) or not path.endswith(suffix):
        return None
    camera_id = unquote(path[len(OVERLAY_PREFIX) : -len(suffix)])
    return camera_id or None


def clip_deletion_clip_id(path: str) -> str | None:
    """``DELETE /clips/{clip_id}`` (exactly); ``None`` for an empty clip id."""
    parts = path.split("/")
    if len(parts) != 3 or parts[1] != CLIPS_SEGMENT:
        return None
    return unquote(parts[2]) or None


def clip_deletion_preflight_clip_id(path: str) -> str | None:
    """``GET /clips/{clip_id}/deletion-preflight``; ``None`` for an empty clip id."""
    parts = path.split("/")
    if len(parts) != 4 or parts[1] != CLIPS_SEGMENT or parts[3] != CLIP_DELETION_PREFLIGHT_SEGMENT:
        return None
    return unquote(parts[2]) or None


# --- /overlay/{camera_id}/pose: ``{"mode": ...}`` both ways ----------------

OverlayMode: TypeAlias = Literal["none", "bedexit", "fall"]


def parse_overlay_mode(value: object) -> OverlayMode | None:
    if value == "none":
        return "none"
    if value == "bedexit":
        return "bedexit"
    if value == "fall":
        return "fall"
    return None


def parse_pose_body(payload: object) -> OverlayMode | None:
    """Accept exactly ``{"mode": <OverlayMode>}``; anything else is ``None``."""
    if not isinstance(payload, Mapping) or len(payload) != 1 or "mode" not in payload:
        return None
    return parse_overlay_mode(payload["mode"])


def pose_body(mode: OverlayMode) -> dict[str, str]:
    return {"mode": mode}


# --- POST /probe -----------------------------------------------------------

ProbeErrorClass: TypeAlias = Literal["auth", "timeout", "decode", "unsupported"]


def normalize_probe_error_class(value: object) -> ProbeErrorClass:
    """Collapse any failure category onto the wire vocabulary (``decode`` is the catch-all)."""
    if value == "auth":
        return "auth"
    if value == "timeout":
        return "timeout"
    if value == "unsupported":
        return "unsupported"
    return "decode"


def parse_probe_request(payload: object) -> str | None:
    """The ``rtsp_url`` of ``{"rtsp_url": ...}``; ``None`` when absent or blank."""
    if not isinstance(payload, Mapping):
        return None
    rtsp_url = payload.get("rtsp_url")
    if not isinstance(rtsp_url, str) or rtsp_url.strip() == "":
        return None
    return rtsp_url


@dataclass(frozen=True, slots=True)
class ProbeResponse:
    """What ``/probe`` writes -- never the URL or a free-text message.

    Failure: ``{"ok": false, "error_class": ...}``. Success: ``{"ok": true}``
    plus whichever of ``backend``/``width``/``height`` the probe learned.
    """

    ok: bool
    error_class: ProbeErrorClass | None = None
    backend: str | None = None
    width: int | None = None
    height: int | None = None

    @classmethod
    def sanitized(cls, payload: Mapping[str, object]) -> ProbeResponse:
        """Reduce the runtime probe's raw result to the wire shape."""
        if payload.get("ok") is not True:
            return cls(
                ok=False, error_class=normalize_probe_error_class(payload.get("error_class"))
            )
        backend = payload.get("backend")
        width = payload.get("width")
        height = payload.get("height")
        return cls(
            ok=True,
            backend=backend if isinstance(backend, str) else None,
            width=width if isinstance(width, int) else None,
            height=height if isinstance(height, int) else None,
        )

    def as_dict(self) -> dict[str, bool | str | int]:
        if not self.ok:
            return {"ok": False, "error_class": normalize_probe_error_class(self.error_class)}
        payload: dict[str, bool | str | int] = {"ok": True}
        if self.backend is not None:
            payload["backend"] = self.backend
        if self.width is not None:
            payload["width"] = self.width
        if self.height is not None:
            payload["height"] = self.height
        return payload


# --- POST /overlay/{camera_id}/bed-zone/recognize --------------------------

# Structured 404 body when recognition ran but found no bed.
BED_ZONE_NOT_FOUND_BODY: Final = {"error_class": "bed_not_found"}


@dataclass(frozen=True, slots=True)
class BedZoneRecognizeResponse:
    """The 200 body: pixel polygon of the best bed plus the image it was found in."""

    polygon: tuple[tuple[int, int], ...]
    image_width: int
    image_height: int

    def as_dict(self) -> dict[str, object]:
        return {
            "polygon": [[x, y] for x, y in self.polygon],
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


__all__ = [
    "BED_ZONE_NOT_FOUND_BODY",
    "MJPEG_BOUNDARY",
    "MJPEG_MEDIA_TYPE",
    "PROBE_PATH",
    "RELAY_TOKEN_HEADER",
    "REPLAY_PATH",
    "BedZoneRecognizeResponse",
    "OverlayMode",
    "ProbeErrorClass",
    "ProbeResponse",
    "bed_zone_camera_id",
    "clip_deletion_clip_id",
    "clip_deletion_preflight_clip_id",
    "normalize_probe_error_class",
    "parse_overlay_mode",
    "parse_pose_body",
    "parse_probe_request",
    "pose_body",
    "pose_camera_id",
    "snapshot_camera_id",
    "stream_camera_id",
]
