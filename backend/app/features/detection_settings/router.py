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

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.detection_settings.policy_store import (
    DetectionPolicyStore,
    PolicyActivationRefused,
    PolicyCameraIdentity,
    PolicyRevisionConflict,
    PolicyRollbackUnavailable,
)
from backend.app.features.detection_settings.store import (
    DOMAINS,
    DetectionSettingsStore,
    DomainDetectionSetting,
)
from backend.app.shared.dashboard_auth import authorize_dashboard
from contracts.worker_config import PulledWorkerConfig
from shared.detection_policies import POLICY_DEFINITIONS

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


class DetectionPolicyChangeRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    module_id: str = Field(min_length=1)
    module_version: int = Field(ge=1)
    schema_id: str = Field(min_length=1)
    schema_version: int = Field(ge=1)
    camera_id: str | None = Field(default=None, min_length=1)
    values: dict[str, object] | None
    # Required on apply. Diff ignores this field; token 0 is generation-zero /
    # image-default / inherited camera state. None is never an unchecked write.
    expected_revision_id: int | None = Field(default=None, ge=0)


class DetectionPolicyRollbackRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    module_id: str = Field(min_length=1)
    module_version: int = Field(ge=1)
    camera_id: str | None = Field(default=None, min_length=1)
    expected_revision_id: int = Field(ge=0)


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


@router.get("/detection-policies")
def get_detection_policies(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    facility_id = _require_enrolled_facility(request.app)
    store = _policy_store(request.app)
    registry = getattr(request.app.state, "camera_registry", None)
    if not isinstance(registry, CameraRegistryStore):
        registry = CameraRegistryStore.from_env()
    camera_ids = tuple(
        PolicyCameraIdentity(str(record.get("backend_camera_id") or record["id"]))
        for record in registry.snapshot()["cameras"]
    )
    try:
        effective = store.resolve_bundle(facility_id, camera_ids).as_dict()
        effective_error = None
    except PolicyActivationRefused as error:
        effective = {"schema_version": 1, "defaults": {}, "cameras": {}}
        effective_error = error.reason
    return {
        "activation_generation": store.generation(facility_id),
        "modules": [
            {
                "qualified_id": definition.qualified_module_id,
                "policy_qualified_id": definition.qualified_schema_id,
                "units": dict(definition.units),
            }
            for definition in POLICY_DEFINITIONS.values()
        ],
        "effective": effective,
        "effective_error": effective_error,
        "activations": [activation.as_dict() for activation in store.activations(facility_id)],
    }


@router.post("/detection-policies/diff")
def diff_detection_policy(
    payload: DetectionPolicyChangeRequest,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    facility_id = _require_enrolled_facility(request.app)
    _require_policy_camera(request.app, payload.camera_id)
    if payload.values is None and payload.camera_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="facility policy diff requires numeric values",
        )
    try:
        return (
            _policy_store(request.app)
            .diff(
                facility_id=facility_id,
                module_id=payload.module_id,
                module_version=payload.module_version,
                schema_id=payload.schema_id,
                schema_version=payload.schema_version,
                camera_id=payload.camera_id,
                values=payload.values,
            )
            .as_dict()
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error


@router.post("/detection-policies/apply", status_code=status.HTTP_202_ACCEPTED)
def apply_detection_policy(
    payload: DetectionPolicyChangeRequest,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    facility_id = _require_enrolled_facility(request.app)
    _require_policy_camera(request.app, payload.camera_id)
    if payload.expected_revision_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="expected_revision_id is required for policy apply",
        )
    try:
        activation = _policy_store(request.app).apply(
            facility_id=facility_id,
            module_id=payload.module_id,
            module_version=payload.module_version,
            schema_id=payload.schema_id,
            schema_version=payload.schema_version,
            camera_id=payload.camera_id,
            values=payload.values,
            expected_revision_id=payload.expected_revision_id,
        )
    except PolicyRevisionConflict as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    return activation.as_dict()


@router.post("/detection-policies/rollback", status_code=status.HTTP_202_ACCEPTED)
def rollback_detection_policy(
    payload: DetectionPolicyRollbackRequest,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    facility_id = _require_enrolled_facility(request.app)
    _require_policy_camera(request.app, payload.camera_id)
    try:
        activation = _policy_store(request.app).rollback(
            facility_id=facility_id,
            module_id=payload.module_id,
            module_version=payload.module_version,
            camera_id=payload.camera_id,
            expected_revision_id=payload.expected_revision_id,
        )
    except PolicyRevisionConflict as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except PolicyRollbackUnavailable as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    return activation.as_dict()


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


def _policy_store(app: FastAPI) -> DetectionPolicyStore:
    store = getattr(app.state, "detection_policy_store", None)
    if not isinstance(store, DetectionPolicyStore):
        store = DetectionPolicyStore.from_env()
        app.state.detection_policy_store = store
    return store


def _connection_store(app: FastAPI) -> ConnectionSettingsStore:
    store = getattr(app.state, "connection_settings_store", None)
    if isinstance(store, ConnectionSettingsStore):
        return store
    return ConnectionSettingsStore.from_env()


def _require_enrolled_facility(app: FastAPI) -> str:
    facility_id = _connection_store(app).load().facility_id
    if facility_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="facility enrollment is required before editing detection policy",
        )
    return facility_id


def _require_policy_camera(app: FastAPI, camera_id: str | None) -> None:
    if camera_id is None:
        return
    registry = getattr(app.state, "camera_registry", None)
    if not isinstance(registry, CameraRegistryStore):
        registry = CameraRegistryStore.from_env()
    records = registry.snapshot()["cameras"]
    if any(
        camera_id == (record.get("backend_camera_id") or record.get("id")) for record in records
    ):
        return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")


def _authorize(request: Request, authorization: str | None) -> None:
    authorize_dashboard(request, legacy_token=_bearer_token(authorization))


def _bearer_token(value: str | None) -> str | None:
    if value is None or not value.startswith("Bearer "):
        return None
    return value.removeprefix("Bearer ").strip() or None


__all__ = ["router", "DetectionSettingsResponse", "current_settings_snapshot"]
