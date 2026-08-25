"""Backend reachability status helpers used by lifespan.

Per-camera ``push_camera``/``put_roster`` mapping is retired. Roster
publication uses ``TopologyClient`` snapshots, not a mapper bundle slot.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


class BackendStatusState(Protocol):
    backend_reachable: bool | None
    backend_last_ok_at: str | None


def mark_backend_status(state: BackendStatusState, reachable: bool | None) -> None:
    state.backend_reachable = reachable
    if reachable:
        state.backend_last_ok_at = _utc_now()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = ["BackendStatusState", "mark_backend_status"]
