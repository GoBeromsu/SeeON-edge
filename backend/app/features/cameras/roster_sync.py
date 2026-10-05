"""Event-driven publication of complete edge topology snapshots."""

from __future__ import annotations

from typing import TypeAlias

from pydantic import JsonValue

from backend.app.features.audit.postgres_runtime import AuditMutation
from backend.app.features.connection.topology_retry_coordinator import (
    TopologyRetryCoordinator,
    TopologyRetryResult,
    TopologySyncStatus,
)

SyncStatus: TypeAlias = TopologySyncStatus
RosterSyncResult: TypeAlias = TopologyRetryResult


def sync_camera_roster(
    coordinator: TopologyRetryCoordinator,
    *,
    _now: float | None = None,
    _force: bool = False,
    _refresh: bool = False,
    audit: AuditMutation | None = None,
) -> RosterSyncResult:
    """Publish at most one durable snapshot for this explicit event."""
    return coordinator.trigger(
        force=_force,
        refresh=_refresh,
        now_epoch=_now,
        audit=audit,
    )


def recover_camera_roster_on_boot(coordinator: TopologyRetryCoordinator) -> RosterSyncResult:
    """Resume one pending snapshot or recover one dirty registry snapshot."""
    return sync_camera_roster(coordinator, _force=True)


def resume_camera_roster_after_connectivity(
    coordinator: TopologyRetryCoordinator,
) -> RosterSyncResult:
    """Resume pending work after backend state and connectivity refresh."""
    return sync_camera_roster(coordinator, _force=True, _refresh=True)


def camera_sync_view(
    coordinator: TopologyRetryCoordinator, _camera_id: str
) -> dict[str, JsonValue]:
    """Expose the durable complete-topology state through the legacy camera view."""
    result = coordinator.current_result()
    return {
        "status": result.status,
        "error_class": result.error_class,
        "detail": result.detail,
        "last_ok_at": result.last_ok_at,
    }


__all__ = [
    "RosterSyncResult",
    "SyncStatus",
    "camera_sync_view",
    "recover_camera_roster_on_boot",
    "resume_camera_roster_after_connectivity",
    "sync_camera_roster",
]
