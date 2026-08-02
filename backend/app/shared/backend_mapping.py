"""Best-effort backend edge-camera mapping client.

G002 note: unlike the Event API ingest clients in ``backend/app/lifespan.py``,
this module's env readers (``backend_status_from_env``,
``_mapping_endpoint_from_env``, ``BackendCameraMapper.from_env``) still read
raw process env, not ``ConnectionSettingsStore``. The mapping endpoint has its
own override chain (``API_BACKEND_EDGE_CAMERAS_URL``, ``API_BACKEND_URL``,
falling back to deriving from ``API_BACKEND_EVENTS_URL``) that
``ConnectionSettingsStore`` does not model (it only tracks a single
``events_url``), so routing this through the store would either drop those
overrides or require inventing new store fields -- left for a follow-up if
runtime-relinking the camera-mapping endpoint is ever needed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime

API_BACKEND_EDGE_CAMERAS_URL_ENV = "API_BACKEND_EDGE_CAMERAS_URL"
API_BACKEND_URL_ENV = "API_BACKEND_URL"
API_BACKEND_EVENTS_URL_ENV = "API_BACKEND_EVENTS_URL"
API_BACKEND_CAMERA_MAPPING_TIMEOUT_SEC_ENV = "API_BACKEND_CAMERA_MAPPING_TIMEOUT_SEC"
EDGE_FACILITY_TOKEN_ENV = "EDGE_FACILITY_TOKEN"
API_FACILITY_TOKEN_ENV = "API_FACILITY_TOKEN"
API_BACKEND_FACILITY_TOKEN_ENV = "API_BACKEND_FACILITY_TOKEN"
API_EDGE_FACILITY_TOKEN_ENV = "API_EDGE_FACILITY_TOKEN"
API_FACILITY_ID_ENV = "API_FACILITY_ID"


@dataclass(frozen=True, slots=True)
class MappingResult:
    backend_camera_id: str | None
    pending: bool
    reachable: bool | None


class BackendCameraMapper:
    def __init__(
        self,
        *,
        endpoint: str | None,
        token: str | None,
        facility_id: str | None = None,
        timeout_sec: float = 0.5,
    ) -> None:
        self.endpoint = _clean(endpoint)
        self.token = _clean(token)
        self.facility_id = _clean(facility_id)
        self.timeout_sec = timeout_sec

    @classmethod
    def from_env(cls) -> BackendCameraMapper:
        return cls(
            endpoint=_mapping_endpoint_from_env(),
            token=_facility_token_from_env(),
            facility_id=os.environ.get(API_FACILITY_ID_ENV),
            timeout_sec=_timeout_sec(),
        )

    @property
    def configured(self) -> bool:
        return self.endpoint is not None and self.token is not None

    def put_mapping(self, *, edge_camera_ref: str, label: str, space_id: str) -> MappingResult:
        if not self.configured:
            return MappingResult(backend_camera_id=None, pending=True, reachable=None)
        payload = {
            "edge_camera_ref": edge_camera_ref,
            "label": label,
            "spaceId": space_id,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                **({"x-facility-id": self.facility_id} if self.facility_id is not None else {}),
            },
            method="PUT",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                parsed = json.loads(response.read().decode("utf-8") or "{}")
        except (TimeoutError, OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError):
            return MappingResult(backend_camera_id=None, pending=True, reachable=False)
        camera_id = _camera_id_from_response(parsed)
        return MappingResult(
            backend_camera_id=camera_id,
            pending=camera_id is None,
            reachable=True,
        )


def backend_status_from_env() -> dict[str, object]:
    configured = bool(_mapping_endpoint_from_env() or os.environ.get(API_BACKEND_EVENTS_URL_ENV))
    return {"configured": configured, "reachable": None, "last_ok_at": None}


def mark_backend_status(state: object, reachable: bool | None) -> None:
    state.backend_reachable = reachable
    if reachable:
        state.backend_last_ok_at = _utc_now()


def _mapping_endpoint_from_env() -> str | None:
    explicit = _clean(os.environ.get(API_BACKEND_EDGE_CAMERAS_URL_ENV))
    if explicit is not None:
        return explicit
    base = _clean(os.environ.get(API_BACKEND_URL_ENV))
    if base is not None:
        return f'{base.rstrip("/")}/api/v1/edge/cameras'
    events_url = _clean(os.environ.get(API_BACKEND_EVENTS_URL_ENV))
    if events_url is None:
        return None
    parsed = urllib.parse.urlsplit(events_url)
    origin = urllib.parse.urlunsplit(parsed._replace(path="", query="", fragment=""))
    return f'{origin.rstrip("/")}/api/v1/edge/cameras'


def _facility_token_from_env() -> str | None:
    for name in (
        # Canonical name shared with the ingest client (api/lifespan.py) and
        # compose.edge.yaml; the same value is copied verbatim across the edge
        # and host env files. Legacy aliases below are kept for compatibility.
        EDGE_FACILITY_TOKEN_ENV,
        API_FACILITY_TOKEN_ENV,
        API_BACKEND_FACILITY_TOKEN_ENV,
        API_EDGE_FACILITY_TOKEN_ENV,
    ):
        token = _clean(os.environ.get(name))
        if token is not None:
            return token
    return None


def _camera_id_from_response(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    for key in ("cameraId", "camera_id", "id"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _timeout_sec() -> float:
    raw = os.environ.get(API_BACKEND_CAMERA_MAPPING_TIMEOUT_SEC_ENV)
    if raw is None:
        return 0.5
    return float(raw)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "BackendCameraMapper",
    "MappingResult",
    "backend_status_from_env",
    "mark_backend_status",
]
