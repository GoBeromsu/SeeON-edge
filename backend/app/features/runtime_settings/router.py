"""Authenticated dashboard API for persisted live runtime settings."""

from __future__ import annotations

from typing import Annotated, ClassVar

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

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
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    return get_runtime_settings_store(request.app).get().as_dict()


@router.put("", response_model=RuntimeSettingsResponse)
def put_runtime_settings(
    payload: RuntimeSettingsUpdateRequest,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    try:
        setting = get_runtime_settings_store(request.app).set_clip_export_enabled(
            payload.clip_export_enabled,
            expected_version=payload.expected_version,
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


def _authorize(request: Request, authorization: str | None) -> None:
    authorize_dashboard(request, legacy_token=_bearer_token(authorization))


def _bearer_token(value: str | None) -> str | None:
    if value is None or not value.startswith("Bearer "):
        return None
    return value.removeprefix("Bearer ").strip() or None


__all__ = ["RuntimeSettingsResponse", "router"]
