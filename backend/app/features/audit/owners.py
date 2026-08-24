"""Exhaustive catalog-to-production callable ownership declaration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from backend.app.features.audit.catalog import AuditAction

ActionOwners = Mapping[AuditAction, tuple[Callable[..., Any], ...]]


class AuditOwnerCatalogError(RuntimeError):
    pass


def production_action_owners() -> dict[AuditAction, tuple[Callable[..., Any], ...]]:
    """Return actual handlers/stores that own each shipped audit action."""
    from backend.app.features.audit.router import get_audit, list_audit
    from backend.app.features.audit.sessions import (
        append_with_recovery,
        close_session,
        start_session,
    )
    from backend.app.features.auth.router import login, logout, session, update_credentials
    from backend.app.features.cameras.bed_zone_router import recognize_bed_zone
    from backend.app.features.cameras.router import (
        create_camera,
        create_topology_floor,
        create_topology_room,
        delete_camera,
        delete_topology_floor,
        delete_topology_room,
        test_camera,
        update_camera,
        update_topology_floor,
        update_topology_room,
    )
    from backend.app.features.clips.router import (
        clip_artifacts,
        clip_thumbnail,
        clip_video,
        delete_clip,
        get_clip_metadata,
        list_clips,
    )
    from backend.app.features.clips.storage_router import put_clip_storage_location
    from backend.app.features.connection.router import put_connection, sync_cameras
    from backend.app.features.connection.topology_confirmation_router import (
        confirm_topology_preview,
    )
    from backend.app.features.detection_settings.router import (
        apply_detection_policy,
        put_detection_settings,
        rollback_detection_policy,
    )
    from backend.app.features.evidence.operator_router import (
        get_incident,
        list_incidents,
        review_incident,
    )
    from backend.app.features.evidence.router import export_clip
    from backend.app.features.relay.router import (
        relay_alert,
        relay_snapshot_attachment,
        relay_snapshot_disposition,
    )
    from backend.app.features.runtime_settings.router import put_runtime_settings

    return {
        AuditAction.AUTH_LOGIN: (login,),
        AuditAction.AUTH_SESSION_READ: (session,),
        AuditAction.AUTH_LOGOUT: (logout,),
        AuditAction.CREDENTIAL_ROTATE: (update_credentials,),
        AuditAction.CAMERA_CREATE: (create_camera,),
        AuditAction.CAMERA_UPDATE: (update_camera,),
        AuditAction.CAMERA_DELETE: (delete_camera,),
        AuditAction.CAMERA_PROBE: (test_camera,),
        AuditAction.LOCATION_CREATE: (create_topology_floor, create_topology_room),
        AuditAction.LOCATION_UPDATE: (update_topology_floor, update_topology_room),
        AuditAction.LOCATION_DELETE: (delete_topology_floor, delete_topology_room),
        AuditAction.BED_ZONE_UPDATE: (recognize_bed_zone,),
        AuditAction.CONNECTION_UPDATE: (put_connection,),
        AuditAction.CONNECTION_SYNC: (sync_cameras,),
        AuditAction.TOPOLOGY_CONFIRM: (confirm_topology_preview,),
        AuditAction.CLIP_STORAGE_UPDATE: (put_clip_storage_location,),
        AuditAction.DETECTION_SETTINGS_UPDATE: (put_detection_settings,),
        AuditAction.RUNTIME_SETTINGS_UPDATE: (put_runtime_settings,),
        AuditAction.POLICY_APPLY: (apply_detection_policy,),
        AuditAction.POLICY_ROLLBACK: (rollback_detection_policy,),
        AuditAction.INCIDENT_LIST: (list_incidents,),
        AuditAction.INCIDENT_DETAIL: (get_incident,),
        AuditAction.INCIDENT_REVIEW: (review_incident,),
        AuditAction.CLIP_LIST: (list_clips,),
        AuditAction.CLIP_DETAIL: (get_clip_metadata,),
        AuditAction.CLIP_PLAY: (clip_video,),
        AuditAction.CLIP_THUMBNAIL: (clip_thumbnail,),
        AuditAction.CLIP_ARTIFACT: (clip_artifacts,),
        AuditAction.CLIP_DELETE: (delete_clip,),
        AuditAction.EVIDENCE_RECEIPT: (export_clip,),
        AuditAction.AUDIT_LIST: (list_audit,),
        AuditAction.AUDIT_DETAIL: (get_audit,),
        AuditAction.RELAY_ALERT: (relay_alert,),
        AuditAction.RELAY_SNAPSHOT_ATTACHMENT: (relay_snapshot_attachment,),
        AuditAction.RELAY_SNAPSHOT_DISPOSITION: (relay_snapshot_disposition,),
        AuditAction.AUDIT_SESSION_START: (start_session,),
        AuditAction.AUDIT_SESSION_CLOSE: (close_session,),
        AuditAction.RECOVERY_FENCE: (append_with_recovery,),
    }


def assert_owner_catalog_complete(owners: ActionOwners) -> None:
    if set(owners) != set(AuditAction):
        raise AuditOwnerCatalogError("audit production owner catalog is incomplete")
    invalid = any(
        not values or any(not callable(owner) for owner in values)
        for values in owners.values()
    )
    if invalid:
        raise AuditOwnerCatalogError("audit production owner catalog contains an invalid owner")


__all__ = [
    "ActionOwners", "AuditOwnerCatalogError", "assert_owner_catalog_complete",
    "production_action_owners",
]
