from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, status  # used only for require_available decisions at router

from backend.app.features.cameras.bed_zone_store import BedZone
from backend.app.features.cameras.store import CameraRegistryData
from backend.app.features.detection_settings.policy_store import PolicyBundle, PolicyCameraIdentity
from backend.app.features.detection_settings.store import DomainDetectionSetting
from contracts.worker_config import PulledWorkerConfig


@dataclass(frozen=True)
class WorkerConfigInputs:
    registry_snapshot: CameraRegistryData | Mapping[str, Any]
    bed_zones: Mapping[str, BedZone]
    pulled: PulledWorkerConfig | None
    live_config_version: int
    live_restart_epoch: int
    detection_settings: Mapping[str, DomainDetectionSetting]
    clip_store_subdir: str | None
    facility_id: str | None
    policy_generation: int
    policy_bundle: PolicyBundle | None


def assemble_worker_config(inputs: WorkerConfigInputs) -> dict[str, Any]:
    """
    Compose the worker-config response deterministically from typed inputs.
    Preserves today's field order and numeric semantics.
    """
    cameras = _build_camera_entries(inputs.registry_snapshot, inputs.bed_zones)
    response: dict[str, Any] = {
        "registry_version": int(inputs.registry_snapshot.get("registry_version", 0)),  # type: ignore[call-overload]
        "cameras": cameras,
    }
    live_pulled = _resolve_live_pulled(
        inputs.pulled, inputs.live_config_version, inputs.live_restart_epoch
    )
    if live_pulled is not None:
        response["config_version"] = live_pulled.config_version
        response["restart_epoch"] = live_pulled.restart_epoch
        if live_pulled.night_window is not None:
            response["night_window"] = live_pulled.night_window.as_dict()
        if live_pulled.detection_windows:
            response["detection_windows"] = {
                domain: window.as_dict() for domain, window in live_pulled.detection_windows.items()
            }
    # Local overrides
    _apply_local_detection_overrides(
        response=response,
        stored=inputs.detection_settings,
        live_pulled=live_pulled,
    )
    # Clip store override
    _apply_clip_storage_override(response=response, selected=inputs.clip_store_subdir or "")
    # Numeric detection policies
    _apply_numeric_detection_policies(
        response=response,
        facility_id=inputs.facility_id,
        generation=inputs.policy_generation,
        bundle=inputs.policy_bundle,
    )
    # Runtime export was already threaded by caller's runtime settings (the service is pure)
    return response


def _build_camera_entries(
    registry_snapshot: CameraRegistryData | Mapping[str, Any],
    bed_zones: Mapping[str, BedZone],
) -> list[dict[str, Any]]:
    def _mapping_state(record: Mapping[str, Any]) -> str:
        backend_camera_id = record.get("backend_camera_id")
        if isinstance(backend_camera_id, str) and backend_camera_id.strip():
            return "mapped"
        if bool(record.get("mapping_pending", False)):
            return "pending"
        return "unmapped"

    def _hub_canonical_id(record: Mapping[str, Any]) -> str | None:
        backend_camera_id = record.get("backend_camera_id")
        if isinstance(backend_camera_id, str) and backend_camera_id.strip():
            return backend_camera_id
        return None

    def _lookup_bed_zone(local_id: Any) -> BedZone | None:
        return bed_zones.get(local_id) if isinstance(local_id, str) else None

    def _records(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        cameras = snapshot.get("cameras")
        return [record for record in cameras or [] if isinstance(record, Mapping)]

    result: list[dict[str, Any]] = []
    for record in _records(registry_snapshot):
        rtsp_url = record.get("rtsp_url")
        if not isinstance(rtsp_url, str) or not rtsp_url.strip():
            continue
        canonical_id = str(record.get("backend_camera_id") or record.get("id", ""))
        if _hub_canonical_id(record) is None:
            # mirror the router's warning shape; log left to the router caller
            pass
        camera: dict[str, Any] = {
            "camera_id": canonical_id,
            "space_id": record.get("space_id"),
            "rtsp_url": rtsp_url,
        }
        decode_backend = record.get("decode_backend")
        if decode_backend is not None:
            camera["decode_backend"] = decode_backend
        bed_zone = _lookup_bed_zone(record.get("id"))
        if bed_zone is not None:
            camera["bed_zone_regions"] = [region.as_dict() for region in bed_zone.regions]
            camera["bed_zone_image_width"] = bed_zone.image_width
            camera["bed_zone_image_height"] = bed_zone.image_height
        result.append(camera)
    return result


def _resolve_live_pulled(
    pulled: PulledWorkerConfig | None, config_version: int, restart_epoch: int
) -> PulledWorkerConfig | None:
    if pulled is None:
        return None
    return PulledWorkerConfig(
        config_version=int(config_version),
        restart_epoch=int(restart_epoch),
        night_window=pulled.night_window,
        cameras=pulled.cameras,
        detection_windows=pulled.detection_windows,
    )


def _apply_local_detection_overrides(
    *,
    response: dict[str, Any],
    stored: Mapping[str, DomainDetectionSetting],
    live_pulled: PulledWorkerConfig | None,
) -> None:
    if not stored:
        return
    domains: dict[str, dict[str, Any]] = {}
    detection_windows = _as_window_dict_map(response.get("detection_windows"))
    for domain, setting in stored.items():
        domains[domain] = {"enabled": setting.on}
        if not setting.on or setting.mode == "always":
            detection_windows.pop(domain, None)
            continue
        detection_windows[domain] = {
            "start": setting.start,
            "end": setting.end,
            "tz": _resolved_tz(live_pulled, domain),
        }
    response["domains"] = domains
    if detection_windows:
        response["detection_windows"] = detection_windows
    else:
        response.pop("detection_windows", None)
    bed_exit_window = detection_windows.get("bed_exit")
    if bed_exit_window is not None:
        response["night_window"] = bed_exit_window
    elif "bed_exit" in stored:
        response.pop("night_window", None)
    pulled_version = 0 if live_pulled is None else live_pulled.config_version
    response["config_version"] = _local_config_version(
        pulled_version, domains, detection_windows, response.get("night_window")
    )


def _local_config_version(
    pulled_version: int,
    domains: Mapping[str, Mapping[str, Any]],
    detection_windows: Mapping[str, Mapping[str, Any]],
    night_window: Any,
) -> int:
    import hashlib
    import json as _json

    payload = _json.dumps(
        {"domains": domains, "detection_windows": detection_windows, "night_window": night_window},
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    offset = 1 + (int(digest[:8], 16) % 1_000_000)
    return pulled_version + offset


def _as_window_dict_map(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {
        domain: window
        for domain, window in value.items()
        if isinstance(domain, str) and isinstance(window, dict)
    }


def _resolved_tz(live_pulled: PulledWorkerConfig | None, domain: str) -> str:
    if live_pulled is not None:
        window = live_pulled.detection_windows.get(domain)
        if window is None and domain == "bed_exit":
            window = live_pulled.night_window
        if window is not None:
            return window.tz
    return "UTC"


def _apply_clip_storage_override(*, response: dict[str, Any], selected: str) -> None:
    if selected:
        response["clip_store_subdir"] = selected


def _apply_numeric_detection_policies(
    *,
    response: dict[str, Any],
    facility_id: str | None,
    generation: int,
    bundle: PolicyBundle | None,
) -> None:
    if generation == 0:
        return
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="detection policy bundle is unavailable",
        )
    response["detection_policies"] = bundle.as_dict()
    response_cameras = response.get("cameras")
    if facility_id is not None and isinstance(response_cameras, list):
        for camera in response_cameras:
            if isinstance(camera, dict):
                camera["facility_id"] = facility_id
    raw_base_version = response.get("config_version", 0)
    base_version = raw_base_version if isinstance(raw_base_version, int) else 0
    policy_hash_part = int(bundle.content_sha256[:8], 16) % 1_000_000_000
    response["config_version"] = base_version * 1_000_000_000 + policy_hash_part
    raw_restart_epoch = response.get("restart_epoch", 0)
    restart_epoch = raw_restart_epoch if isinstance(raw_restart_epoch, int) else 0
    response["restart_epoch"] = restart_epoch + generation


def compute_policy_camera_identities(
    registry_snapshot: CameraRegistryData | Mapping[str, Any]
) -> tuple[PolicyCameraIdentity, ...]:
    def _records(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        cameras = snapshot.get("cameras")
        return [record for record in cameras or [] if isinstance(record, Mapping)]

    identities: list[PolicyCameraIdentity] = []
    for record in _records(registry_snapshot):
        rtsp_url = record.get("rtsp_url")
        if not isinstance(rtsp_url, str) or not rtsp_url.strip():
            continue
        canonical_id = str(record.get("backend_camera_id") or record.get("id", ""))
        identities.append(PolicyCameraIdentity(camera_id=canonical_id))
    return tuple(identities)

