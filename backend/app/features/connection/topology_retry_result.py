from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypeAlias

from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncState,
    EdgeTopologySyncStateStore,
    TopologyPauseReason,
)
from backend.app.features.cameras.store import CameraRegistryStore

TopologySyncStatus: TypeAlias = Literal["pending", "synced", "failed", "disabled"]
TopologySyncErrorClass: TypeAlias = Literal[
    "unconfigured", "auth", "timeout", "unreachable", "conflict"
]

_UNCONFIGURED = "백엔드 등록이 완료되지 않아 토폴로지를 동기화할 수 없습니다."
_INCOMPLETE = "모든 카메라에 명시적인 층/방/카메라 참조를 배정해야 합니다."
_PAUSED = "백엔드 상태를 새로 고치기 전까지 토폴로지 동기화를 일시 중지했습니다."


@dataclass(frozen=True, slots=True)
class TopologyRetryResult:
    attempted: bool
    status: TopologySyncStatus
    error_class: TopologySyncErrorClass | None
    detail: str | None
    last_ok_at: str | None
    next_retry_at: str | None
    camera_count: int


def current_retry_result(
    registry: CameraRegistryStore,
    state_store: EdgeTopologySyncStateStore,
    *,
    attempted: bool = False,
) -> TopologyRetryResult:
    state = state_store.load()
    if state.pause_reason is not None:
        pause_error: TopologySyncErrorClass = (
            "auth"
            if state.pause_reason in {TopologyPauseReason.AUTH, TopologyPauseReason.FORBIDDEN}
            else "conflict"
        )
        return retry_result(registry, state, attempted, "failed", pause_error, _PAUSED)
    if state.pending is not None:
        pending_error: TopologySyncErrorClass | None = (
            "unreachable" if state.consecutive_failures else None
        )
        status: TopologySyncStatus = "failed" if pending_error else "pending"
        return retry_result(registry, state, attempted, status, pending_error, None)
    topology = registry.topology_snapshot()
    if topology.readiness_error is not None:
        return retry_result(registry, state, attempted, "pending", None, _INCOMPLETE)
    status = (
        "synced"
        if topology.dirty is None and state.last_snapshotted_registry_version > 0
        else "pending"
    )
    return retry_result(registry, state, attempted, status, None, None)


def unconfigured_retry_result(registry: CameraRegistryStore) -> TopologyRetryResult:
    topology = registry.topology_snapshot()
    status: TopologySyncStatus = "pending" if topology.dirty is not None else "disabled"
    return TopologyRetryResult(
        False,
        status,
        "unconfigured",
        _UNCONFIGURED,
        None,
        None,
        _camera_count(registry),
    )


def retry_result(
    registry: CameraRegistryStore,
    state: EdgeTopologySyncState,
    attempted: bool,
    status: TopologySyncStatus,
    error_class: TopologySyncErrorClass | None,
    detail: str | None,
) -> TopologyRetryResult:
    return TopologyRetryResult(
        attempted,
        status,
        error_class,
        detail,
        _iso_timestamp(state.last_accepted_at),
        _iso_timestamp(state.next_retry_at),
        _camera_count(registry),
    )


def _camera_count(registry: CameraRegistryStore) -> int:
    return len(registry.snapshot()["cameras"])


def _iso_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


__all__ = [
    "TopologyRetryResult",
    "TopologySyncErrorClass",
    "TopologySyncStatus",
    "current_retry_result",
    "retry_result",
    "unconfigured_retry_result",
]
