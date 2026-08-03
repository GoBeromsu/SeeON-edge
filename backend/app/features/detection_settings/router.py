"""Global per-domain detection on/off + time-window settings routes.

``GET``/``PUT /api/v1/detection-settings`` -- a dashboard-only, facility-wide
toggle (applied to every camera) for whether each detection domain (``fall``,
``bed_exit``) runs at all, and optionally restricts it to a nightly window.
Persisted in ``DetectionSettingsStore`` (its own row per domain in
``catalog.sqlite3``); once saved, these local settings take precedence over
whatever the backend externally pulls, merged in at
``cameras.router.worker_config_snapshot`` response-build time -- this router
never touches ``app.state.pulled_config`` itself (see that function's
``_apply_local_detection_overrides``).

``GET`` with nothing yet persisted for a domain falls back to reflecting the
live externally-pulled detection window for that domain (so an operator who
has never opened this settings page sees the schedule that's actually in
effect, not a fabricated default), and finally to on=true/mode=always if
there is no external window either (matching the worker's own ambient
default: no configured window means 24/7 detection).
"""

from __future__ import annotations

import re
from typing import Annotated, ClassVar, Literal

from fastapi import APIRouter, FastAPI, Header, Request
from pydantic import BaseModel, ConfigDict, model_validator

from backend.app.features.detection_settings.store import (
    DOMAINS,
    DetectionSettingsStore,
    DomainDetectionSetting,
)
from backend.app.shared.dashboard_auth import authorize_dashboard
from contracts.worker_config import PulledWorkerConfig

router = APIRouter(tags=["detection-settings"])

_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class DomainSettingPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    on: bool
    mode: Literal["always", "window"]
    start: str | None = None
    end: str | None = None

    @model_validator(mode="after")
    def _validate_window(self) -> DomainSettingPayload:
        if self.mode != "window":
            return self
        if not self.start or not self.end:
            raise ValueError("start and end are required when mode is window")
        if not _HHMM_RE.fullmatch(self.start) or not _HHMM_RE.fullmatch(self.end):
            raise ValueError("start and end must be HH:MM")
        if self.start == self.end:
            raise ValueError("start and end must not be equal")
        return self


class DetectionSettingsDomains(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    fall: DomainSettingPayload
    bed_exit: DomainSettingPayload


class DetectionSettingsPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    domains: DetectionSettingsDomains


class DomainSettingResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    on: bool
    mode: Literal["always", "window"]
    start: str | None = None
    end: str | None = None


class DetectionSettingsResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    domains: dict[str, DomainSettingResponse]


@router.get("/detection-settings", response_model=DetectionSettingsResponse)
def get_detection_settings(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    return {"domains": current_settings_snapshot(request.app)}


@router.put("/detection-settings", response_model=DetectionSettingsResponse)
def put_detection_settings(
    payload: DetectionSettingsPayload,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    settings = {
        domain: _to_domain_setting(getattr(payload.domains, domain))
        for domain in DOMAINS
    }
    _store(request.app).replace_all(settings)
    return {"domains": {domain: setting.as_dict() for domain, setting in settings.items()}}


def _to_domain_setting(payload: DomainSettingPayload) -> DomainDetectionSetting:
    if payload.mode == "always":
        # Normalize away any stray start/end sent alongside mode=always so
        # persisted state always matches the GET shape for an always-on
        # domain (start/end null), regardless of what the client submitted.
        return DomainDetectionSetting(on=payload.on, mode="always", start=None, end=None)
    return DomainDetectionSetting(
        on=payload.on, mode="window", start=payload.start, end=payload.end
    )


def current_settings_snapshot(app: FastAPI) -> dict[str, dict[str, object]]:
    """The effective per-domain settings dict used by both the GET response
    and (indirectly, via the same defaulting rule) documented for
    ``cameras.router.worker_config_snapshot``'s local-override merge."""
    stored = _store(app).get_all()
    pulled = getattr(app.state, "pulled_config", None)
    result: dict[str, dict[str, object]] = {}
    for domain in DOMAINS:
        setting = stored.get(domain)
        if setting is not None:
            result[domain] = setting.as_dict()
            continue
        result[domain] = _default_setting_dict(pulled, domain)
    return result


def _default_setting_dict(pulled: object, domain: str) -> dict[str, object]:
    window = None
    if isinstance(pulled, PulledWorkerConfig):
        window = pulled.detection_windows.get(domain)
        if window is None and domain == "bed_exit":
            window = pulled.night_window
    if window is not None:
        return {"on": True, "mode": "window", "start": window.start, "end": window.end}
    return {"on": True, "mode": "always", "start": None, "end": None}


def _store(app: FastAPI) -> DetectionSettingsStore:
    store = getattr(app.state, "detection_settings_store", None)
    if not isinstance(store, DetectionSettingsStore):
        store = DetectionSettingsStore.from_env()
        app.state.detection_settings_store = store
    return store


def _authorize(request: Request, authorization: str | None) -> None:
    authorize_dashboard(request, legacy_token=_bearer_token(authorization))


def _bearer_token(value: str | None) -> str | None:
    if value is None or not value.startswith("Bearer "):
        return None
    return value.removeprefix("Bearer ").strip() or None


__all__ = ["router", "DetectionSettingsResponse", "current_settings_snapshot"]
