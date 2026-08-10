"""Event-driven publication of complete edge topology snapshots."""

from __future__ import annotations

from typing import TypeAlias

from fastapi import FastAPI

from backend.app.features.connection.topology_retry_coordinator import (
    TopologyRetryResult,
    TopologySyncStatus,
    topology_retry_coordinator,
)

SyncStatus: TypeAlias = TopologySyncStatus
RosterSyncResult: TypeAlias = TopologyRetryResult


def sync_camera_roster(
    app: FastAPI,
    *,
    _now: float | None = None,
    _force: bool = False,
    _refresh: bool = False,
) -> RosterSyncResult:
    """Publish at most one durable snapshot for this explicit event."""
    return topology_retry_coordinator(app).trigger(
        force=_force,
        refresh=_refresh,
        now_epoch=_now,
    )


def recover_camera_roster_on_boot(app: FastAPI) -> RosterSyncResult:
    """Resume one pending snapshot or recover one dirty registry snapshot."""
    return sync_camera_roster(app, _force=True)


def resume_camera_roster_after_connectivity(app: FastAPI) -> RosterSyncResult:
    """Resume pending work after backend state and connectivity refresh."""
    return sync_camera_roster(app, _force=True, _refresh=True)


def camera_sync_view(app: FastAPI, _camera_id: str) -> dict[str, object]:
    """Expose the durable complete-topology state through the legacy camera view."""
    result = topology_retry_coordinator(app).current_result()
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
