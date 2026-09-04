"""Authenticated dashboard API for persisted live runtime settings."""

from __future__ import annotations

from typing import ClassVar

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.http import append_transactional
from backend.app.features.audit.store import AuditEvent, utc_now
from backend.app.features.runtime_settings.store import (
    RuntimeSettingsVersionConflict,
    get_runtime_settings_store,
)
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(prefix="/runtime-settings", tags=["runtime-settings"])


class RuntimeSettingsUpdateRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    clip_export_enabled: bool = Field()
    expected_version: int = Field(ge=0)


class RuntimeSettingsResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    clip_export_enabled: bool = Field()
    version: int = Field(ge=0)


@router.get("", response_model=RuntimeSettingsResponse)
def get_runtime_settings(
    request: Request,
) -> dict[str, object]:
    _authorize(request)
    return get_runtime_settings_store(request.app).get().as_dict()


@router.put("", response_model=RuntimeSettingsResponse)
def put_runtime_settings(
    payload: RuntimeSettingsUpdateRequest,
    request: Request,
) -> dict[str, object]:
    actor = _authorize(request)
    event = AuditEvent(
        occurred_at=utc_now(),
        actor_id=actor,
        action=AuditAction.RUNTIME_SETTINGS_UPDATE,
        target_id="runtime-settings",
        detail=empty_detail(AuditAction.RUNTIME_SETTINGS_UPDATE),
    )
    try:
        setting = get_runtime_settings_store(request.app).set_clip_export_enabled(
            payload.clip_export_enabled,
            expected_version=payload.expected_version,
            after_write=lambda connection: append_transactional(request, connection, event),
        )
    except RuntimeSettingsVersionConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "runtime_settings_version_conflict",
                "current": exc.current.as_dict(),
            },
        ) from exc
    return setting.as_dict()


def _authorize(request: Request) -> str:
    return authorize_dashboard(request)


__all__ = ["RuntimeSettingsResponse", "router"]
