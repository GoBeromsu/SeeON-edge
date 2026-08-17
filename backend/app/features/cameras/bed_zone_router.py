"""On-demand bed-zone recognition: proxy + persistence.

``POST /api/v1/cameras/{camera_id}/bed-zone/recognize`` proxies to the
worker's one-shot bed segmentation route (``POST
/overlay/{camera_id}/bed-zone/recognize``, see
``worker/pipeline/output/_mjpeg_http.py``) using the same urllib-proxy
pattern as ``streams_router.py``, then persists a successful result in
``BedZoneStore`` so it survives restarts and can be pulled back down to the
worker as the authoritative bed region (see
``cameras.router.worker_config_snapshot``).

This is a dashboard-only action (an operator manually triggering
recognition), so it requires the same server-issued dashboard session as the
camera CRUD routes. Only the server-side proxy forwards the worker relay token
upstream; browser-supplied relay credentials are never operator authority.

Response contract:
- worker success -> 200 ``{"bed_zone": {...}}`` (and the polygon is persisted)
- worker ran but found no usable bed (structured 404,
  ``{"error_class": "bed_not_found"}``) -> 422
  ``{"detail": {"error_class": "bed_not_found"}}``
- anything else (unknown camera, no frame, decode failure, connection
  failure, malformed upstream payload) -> 503
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Protocol

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import get_settings
from backend.app.features.cameras.bed_zone_store import BedZoneStore
from backend.app.features.cameras.store import utc_now_iso
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(prefix="/cameras", tags=["cameras"])


class _ResponseHeaders(Protocol):
    def get(self, name: str, default: str | None = None) -> str | None: ...


class _ReadableResponse(Protocol):
    status: int
    headers: _ResponseHeaders

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...


class BedZonePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    polygon: list[list[int]] = Field(min_length=3)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    recognized_at: str


class BedZoneRecognizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bed_zone: BedZonePayload


@router.post("/{camera_id}/bed-zone/recognize")
def recognize_bed_zone(
    camera_id: str,
    request: Request,
) -> BedZoneRecognizeResponse:
    _authorize(request)
    settings = get_settings()
    upstream_url = _bed_zone_url(settings.worker_stream_origin, camera_id)
    # Worker :8090 gates bed-zone recognition with the same relay token as
    # /probe (security finding #3). Forward server-side only; the browser
    # already proved a dashboard session via `_authorize` above.
    upstream_request = urllib.request.Request(
        upstream_url,
        data=b"",
        method="POST",
        headers=_worker_relay_headers(request),
    )

    try:
        upstream: _ReadableResponse = urllib.request.urlopen(
            upstream_request,
            timeout=settings.worker_bed_zone_timeout_s,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read()
        if exc.code == status.HTTP_404_NOT_FOUND and _is_bed_not_found(body):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"error_class": "bed_not_found"},
            ) from exc
        raise _upstream_unavailable() from exc
    except urllib.error.URLError as exc:
        raise _upstream_unavailable() from exc
    except (TimeoutError, OSError) as exc:
        # A read timeout waiting on getresponse() can surface as a bare
        # TimeoutError instead of being wrapped in URLError (CPython's
        # urllib.request.AbstractHTTPHandler.do_open does not guarantee the
        # wrap), so it must be handled alongside HTTPError/URLError to avoid
        # an unhandled 500.
        raise _upstream_unavailable() from exc

    try:
        upstream_status = int(getattr(upstream, "status", status.HTTP_200_OK))
        raw = upstream.read()
    except (TimeoutError, OSError) as exc:
        raise _upstream_unavailable() from exc
    finally:
        upstream.close()
    if upstream_status != status.HTTP_200_OK:
        raise _upstream_unavailable()

    polygon, image_width, image_height = _parse_worker_payload(raw)
    recognized_at = utc_now_iso()
    saved = _store(request.app).put(
        camera_id,
        polygon=polygon,
        image_width=image_width,
        image_height=image_height,
        recognized_at=recognized_at,
    )
    return BedZoneRecognizeResponse(bed_zone=BedZonePayload(**saved.as_dict()))


def _parse_worker_payload(raw: bytes) -> tuple[list[list[int]], int, int]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _upstream_unavailable() from exc
    if not isinstance(parsed, dict):
        raise _upstream_unavailable()
    polygon = parsed.get("polygon")
    image_width = parsed.get("image_width")
    image_height = parsed.get("image_height")
    if (
        not _is_valid_polygon(polygon)
        or not _is_positive_int(image_width)
        or not _is_positive_int(image_height)
    ):
        raise _upstream_unavailable()
    return polygon, image_width, image_height  # type: ignore[return-value]


def _is_valid_polygon(value: object) -> bool:
    if not isinstance(value, list) or len(value) < 3:
        return False
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            return False
        if not all(isinstance(coordinate, int) for coordinate in point):
            return False
    return True


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_bed_not_found(raw: bytes) -> bool:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("error_class") == "bed_not_found"


def _bed_zone_url(origin: str, camera_id: str) -> str:
    base = origin.strip().rstrip("/")
    if not base:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker stream origin is not configured",
        )
    encoded_camera_id = urllib.parse.quote(camera_id, safe="")
    return f"{base}/overlay/{encoded_camera_id}/bed-zone/recognize"


def _authorize(request: Request) -> None:
    authorize_dashboard(request)


_RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"


def _relay_token(request: Request) -> str | None:
    """Server-side worker relay token (``app.state.edge_relay_token`` only).

    Mirrors ``streams_router._relay_token`` so bed-zone upstream calls carry
    the same header as stream/snapshot/pose without importing that module's
    private helpers.
    """
    expected = getattr(request.app.state, "edge_relay_token", None)
    return expected if isinstance(expected, str) and expected else None


def _worker_relay_headers(request: Request) -> dict[str, str]:
    token = _relay_token(request)
    if token is None:
        return {}
    return {_RELAY_TOKEN_HEADER: token}


def _upstream_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="worker bed-zone recognition unavailable",
    )


def _store(app: FastAPI) -> BedZoneStore:
    store = getattr(app.state, "bed_zone_store", None)
    if not isinstance(store, BedZoneStore):
        store = BedZoneStore.from_env()
        app.state.bed_zone_store = store
    return store


__all__ = ["router", "BedZonePayload", "BedZoneRecognizeResponse"]
