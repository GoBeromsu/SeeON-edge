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

G004 note: ``BackendCameraMapper.put_roster`` (roster sync, used by
``features/cameras/roster_sync.py``) is the one caller that DOES route
endpoint/token through ``ConnectionSettingsStore`` effective settings,
constructing its own ``BackendCameraMapper`` instance via
``derive_edge_cameras_endpoint(store_settings.events_url)`` rather than
``BackendCameraMapper.from_env()`` -- see roster_sync.py's ``_build_mapper``.
That path only recognizes ``EDGE_FACILITY_TOKEN`` (the store's single
fallback), not the three legacy token aliases below; see
``docs/decisions/0004-camera-roster-sync-contract-assumptions.md``.

``PUT /v1/edge/cameras`` (the ``EdgeCamerasController.upsert`` handler on the
fall-ai backend) takes exactly ONE ``EdgeCameraMappingRequestDto`` per call --
``{edge_camera_ref, label, spaceId}``, all three required (see the ADR) -- not
a bulk array. ``put_roster`` therefore issues one PUT per camera (via the same
request-building ``put_mapping`` uses) and aggregates per-camera outcomes,
rather than sending one bulk body.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

RosterErrorClass = Literal["unreachable", "timeout", "auth", "unconfigured"]

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


@dataclass(frozen=True, slots=True)
class CameraPushResult:
    """Outcome of a single camera's ``PUT /v1/edge/cameras`` push."""

    ok: bool
    error_class: RosterErrorClass | None
    status_code: int | None
    backend_camera_id: str | None = None


@dataclass(frozen=True, slots=True)
class RosterPushResult:
    """Aggregate outcome of a roster sync attempt (story G004): one PUT per
    eligible camera (see ``put_roster``), rolled up into a single ok/
    error_class for callers that only need the headline result, plus
    ``cameras`` for callers (roster_sync.py) that need each camera's own
    outcome to report a partial success/failure accurately.

    ``ok`` is True only when every attempted camera succeeded (or none were
    attempted). ``error_class``/``status_code`` mirror the first camera that
    failed, in call order -- a representative reason, not necessarily the
    only one.
    """

    ok: bool
    error_class: RosterErrorClass | None
    status_code: int | None
    cameras: dict[str, CameraPushResult]


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

    def _mapping_request(
        self, *, edge_camera_ref: str, label: str, space_id: str
    ) -> urllib.request.Request:
        """Build the single-object PUT request body ``EdgeCameraMappingRequestDto``
        requires (``edge_camera_ref``/``label``/``spaceId``, all required) --
        shared by ``put_mapping`` and ``push_camera`` so both send an
        identical request shape.
        """
        payload = {
            "edge_camera_ref": edge_camera_ref,
            "label": label,
            "spaceId": space_id,
        }
        return urllib.request.Request(
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

    def put_mapping(self, *, edge_camera_ref: str, label: str, space_id: str) -> MappingResult:
        if not self.configured:
            return MappingResult(backend_camera_id=None, pending=True, reachable=None)
        request = self._mapping_request(
            edge_camera_ref=edge_camera_ref, label=label, space_id=space_id
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

    def push_camera(self, *, edge_camera_ref: str, label: str, space_id: str) -> CameraPushResult:
        """Push a single camera via the same request ``put_mapping`` builds,
        but classify failures into the richer error vocabulary
        (``auth``/``timeout``/``unreachable``) that ``put_mapping``'s own
        try/except collapses into a single ``pending`` bit -- used by
        ``put_roster`` (below) so callers get an actionable per-camera
        reason rather than just "not confirmed yet".
        """
        if not self.configured:
            return CameraPushResult(ok=False, error_class="unconfigured", status_code=None)
        request = self._mapping_request(
            edge_camera_ref=edge_camera_ref, label=label, space_id=space_id
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                parsed = json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return CameraPushResult(ok=False, error_class="auth", status_code=exc.code)
            return CameraPushResult(ok=False, error_class="unreachable", status_code=exc.code)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                return CameraPushResult(ok=False, error_class="timeout", status_code=None)
            return CameraPushResult(ok=False, error_class="unreachable", status_code=None)
        except TimeoutError:
            return CameraPushResult(ok=False, error_class="timeout", status_code=None)
        except (OSError, ValueError):
            return CameraPushResult(ok=False, error_class="unreachable", status_code=None)
        return CameraPushResult(
            ok=True,
            error_class=None,
            status_code=200,
            backend_camera_id=_camera_id_from_response(parsed),
        )

    def put_roster(self, cameras: list[dict[str, object]]) -> RosterPushResult:
        """Push the facility's camera roster to the external backend.

        ``PUT /v1/edge/cameras`` takes one camera per call (see this module's
        docstring), so this issues a sequential ``push_camera`` per entry and
        aggregates the outcomes -- not a single bulk request. ``cameras``
        entries must already carry a non-empty ``spaceId`` (the external DTO
        requires it; callers -- roster_sync.py -- are responsible for
        excluding cameras that don't have one yet).
        """
        if not self.configured:
            return RosterPushResult(
                ok=False, error_class="unconfigured", status_code=None, cameras={}
            )
        per_camera: dict[str, CameraPushResult] = {}
        for entry in cameras:
            ref = str(entry.get("edge_camera_ref"))
            per_camera[ref] = self.push_camera(
                edge_camera_ref=ref,
                label=str(entry.get("label")),
                space_id=str(entry.get("spaceId")),
            )
        failed = [result for result in per_camera.values() if not result.ok]
        if not failed:
            return RosterPushResult(ok=True, error_class=None, status_code=200, cameras=per_camera)
        first_failure = failed[0]
        return RosterPushResult(
            ok=False,
            error_class=first_failure.error_class,
            status_code=first_failure.status_code,
            cameras=per_camera,
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
    return derive_edge_cameras_endpoint(os.environ.get(API_BACKEND_EVENTS_URL_ENV))


def derive_edge_cameras_endpoint(events_url: str | None) -> str | None:
    """Derive the ``/api/v1/edge/cameras`` endpoint from an events URL's origin.

    Factored out of ``_mapping_endpoint_from_env`` so callers with their own
    events-url source (e.g. roster_sync.py, which reads it from
    ``ConnectionSettingsStore`` rather than raw env) can reuse the same
    derivation instead of re-implementing it.
    """
    cleaned = _clean(events_url)
    if cleaned is None:
        return None
    parsed = urllib.parse.urlsplit(cleaned)
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
    "CameraPushResult",
    "MappingResult",
    "RosterErrorClass",
    "RosterPushResult",
    "backend_status_from_env",
    "derive_edge_cameras_endpoint",
    "mark_backend_status",
]
