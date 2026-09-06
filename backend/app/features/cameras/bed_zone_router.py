"""Dashboard APIs for proposing and explicitly saving canonical bed zones."""

from __future__ import annotations

import json
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Literal, Protocol

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from backend.app.core.config import get_settings
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.http import append_transactional
from backend.app.features.audit.store import AuditEvent
from backend.app.features.audit.store import utc_now as audit_now
from backend.app.features.cameras.bed_zone_store import (
    BedZoneRegion,
    BedZoneStore,
    validate_bed_zone,
)
from backend.app.features.cameras.camera_repository import record_registry_mutation
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


class BedZoneRegionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    polygon: list[tuple[StrictInt, StrictInt]] = Field(min_length=3, max_length=16)
    origin: Literal["manual", "model"]


class BedZonePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regions: list[BedZoneRegionPayload] = Field(max_length=8)
    image_width: StrictInt = Field(gt=0)
    image_height: StrictInt = Field(gt=0)
    recognized_at: str


class BedZoneSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regions: list[BedZoneRegionPayload] = Field(max_length=8)
    image_width: StrictInt = Field(gt=0)
    image_height: StrictInt = Field(gt=0)


class BedZoneRecognizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    confidence: float = Field(default=0.15, ge=0.05, le=0.95, allow_inf_nan=False)


class BedZoneResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bed_zone: BedZonePayload | None


@router.post("/{camera_id}/bed-zone/recognize", response_model=BedZoneResponse)
def recognize_bed_zone(
    camera_id: str,
    request: Request,
    payload: BedZoneRecognizeRequest | None = None,
) -> BedZoneResponse:
    _authorize(request)
    store = _store(request.app)
    _require_camera(store, camera_id)
    settings = get_settings()
    upstream_request = urllib.request.Request(
        _bed_zone_url(settings.worker_stream_origin, camera_id),
        data=json.dumps(
            {"confidence": 0.15 if payload is None else payload.confidence},
            separators=(",", ":"),
        ).encode("utf-8"),
        method="POST",
        headers={
            **_worker_relay_headers(request),
            "Content-Type": "application/json",
        },
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
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
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

    candidate = _parse_worker_payload(raw)
    return BedZoneResponse(
        bed_zone=BedZonePayload(
            regions=candidate.regions,
            image_width=candidate.image_width,
            image_height=candidate.image_height,
            recognized_at=utc_now_iso(),
        )
    )


@router.put("/{camera_id}/bed-zone", response_model=BedZoneResponse)
def save_bed_zone(
    camera_id: str,
    payload: BedZoneSaveRequest,
    request: Request,
) -> BedZoneResponse:
    actor = _authorize(request)
    store = _store(request.app)
    _require_camera(store, camera_id)
    recognized_at = utc_now_iso()
    regions = _region_values(payload.regions)
    hook = _save_hook(request, actor, camera_id)

    try:
        if not regions:
            store.delete(camera_id, after_write=hook)
            bed_zone = None
        else:
            saved = store.put(
                camera_id,
                regions=regions,
                image_width=payload.image_width,
                image_height=payload.image_height,
                recognized_at=recognized_at,
                after_write=hook,
            )
            bed_zone = BedZonePayload(**saved.as_dict())
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return BedZoneResponse(bed_zone=bed_zone)


def _parse_worker_payload(raw: bytes) -> BedZoneSaveRequest:
    try:
        parsed = BedZoneSaveRequest.model_validate_json(raw)
        # The store owns the full geometric and encoded-size invariant. Run it
        # without writing so malformed provider output remains a clean 503.
        validate_bed_zone(
            _region_values(parsed.regions),
            image_width=parsed.image_width,
            image_height=parsed.image_height,
            recognized_at="candidate",
        )
    except (ValueError, TypeError):
        raise _upstream_unavailable() from None
    return parsed


def _region_values(regions: list[BedZoneRegionPayload]) -> tuple[BedZoneRegion, ...]:
    return tuple(
        BedZoneRegion(
            id=region.id,
            polygon=tuple(region.polygon),
            origin=region.origin,
        )
        for region in regions
    )


def _require_camera(store: BedZoneStore, camera_id: str) -> None:
    if not store.camera_exists(camera_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")


def _save_hook(
    request: Request, actor: str, camera_id: str
) -> Callable[[sqlite3.Connection], None]:
    def after_write(connection: sqlite3.Connection) -> None:
        record_registry_mutation(connection)
        append_transactional(
            request,
            connection,
            AuditEvent(
                occurred_at=audit_now(),
                actor_id=actor,
                action=AuditAction.BED_ZONE_UPDATE,
                target_id=camera_id,
                detail=empty_detail(AuditAction.BED_ZONE_UPDATE),
            ),
        )

    return after_write


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
    return f"{base}/overlay/{urllib.parse.quote(camera_id, safe='')}/bed-zone/recognize"


def _authorize(request: Request) -> str:
    return authorize_dashboard(request)


_RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"


def _worker_relay_headers(request: Request) -> dict[str, str]:
    expected = getattr(request.app.state, "edge_relay_token", None)
    if not isinstance(expected, str) or not expected:
        return {}
    return {_RELAY_TOKEN_HEADER: expected}


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


__all__ = [
    "BedZonePayload",
    "BedZoneRecognizeRequest",
    "BedZoneRegionPayload",
    "BedZoneResponse",
    "BedZoneSaveRequest",
    "router",
]
