"""Durable recovery records for Flow Smart Record clips."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from worker.pipeline.output.evidence.smart_record_actor import (
    ClipContributor,
    ClipSealed,
)
from worker.types import BusinessEvent


class FlowSealedMediaMissingError(RuntimeError):
    """A sealed clip was lost before its durable publication could be retried."""

    def __init__(self, clip_id: str, path: Path) -> None:
        super().__init__(f"sealed Flow clip media is missing clip_id={clip_id} path={path}")
        self.clip_id = clip_id
        self.path = path


@dataclass(frozen=True, slots=True)
class FlowSealedRecovery:
    sealed: ClipSealed
    events: dict[str, BusinessEvent]
    camera_id: str
    sidecar_path: Path


class FlowSealedSidecars:
    """Persist sealed Flow clip attribution in the worker state directory.

    The state directory is used rather than the plane's output directory: Flow
    owns the latter and deployments may clean it independently of worker state.
    The sidecar records the media path, so it remains recoverable across that
    ownership boundary.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def persist(self, sealed: ClipSealed, events: dict[str, BusinessEvent]) -> Path:
        contributors = tuple(sorted(sealed.contributors, key=lambda item: item.detected_at))
        missing = [item.event_ref for item in contributors if item.event_ref not in events]
        if missing:
            raise ValueError(f"sealed Flow clip has unknown contributor {missing[0]}")
        payload = {
            "clip_id": sealed.clip_id,
            "path": sealed.path,
            "duration_ms": sealed.duration_ms,
            "camera_id": events[contributors[0].event_ref].camera_id if contributors else "",
            "boundary": sealed.boundary,
            "contributors": [asdict(item) for item in contributors],
            "events": [_event_payload(events[item.event_ref]) for item in contributors],
        }
        self._directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        target = self._directory / f"{sealed.clip_id}.json"
        temporary = self._directory / f".{sealed.clip_id}.{uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            self._fsync_directory()
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def pending_for_camera(self, camera_id: str) -> tuple[FlowSealedRecovery, ...]:
        if not self._directory.exists():
            return ()
        recoveries: list[FlowSealedRecovery] = []
        for sidecar_path in sorted(self._directory.glob("*.json")):
            recovery = self._read(sidecar_path)
            if recovery.camera_id == camera_id:
                recoveries.append(recovery)
        return tuple(recoveries)

    def discard_missing_media(self, recovery: FlowSealedRecovery) -> FlowSealedMediaMissingError:
        recovery.sidecar_path.unlink()
        self._fsync_directory()
        return FlowSealedMediaMissingError(recovery.sealed.clip_id, Path(recovery.sealed.path))

    def remove(self, recovery: FlowSealedRecovery) -> None:
        recovery.sidecar_path.unlink(missing_ok=True)
        self._fsync_directory()

    def _read(self, sidecar_path: Path) -> FlowSealedRecovery:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        contributors = tuple(ClipContributor(**item) for item in payload["contributors"])
        events = {str(item["identity"]): BusinessEvent(**item) for item in payload["events"]}
        return FlowSealedRecovery(
            sealed=ClipSealed(
                clip_id=payload["clip_id"],
                path=payload["path"],
                duration_ms=payload["duration_ms"],
                contributors=contributors,
                boundary=payload["boundary"],
            ),
            events=events,
            camera_id=payload["camera_id"],
            sidecar_path=sidecar_path,
        )

    def _fsync_directory(self) -> None:
        descriptor = os.open(self._directory, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _event_payload(event: BusinessEvent) -> dict[str, object]:
    """Keep only the immutable event facts publication needs, never pixels."""
    return {
        "domain": event.domain,
        "event_type": event.event_type,
        "identity": str(event.identity),
        "camera_id": event.camera_id,
        "facility_id": event.facility_id,
        "time_sec": event.time_sec,
        "probability": event.probability,
    }


__all__ = [
    "FlowSealedMediaMissingError",
    "FlowSealedRecovery",
    "FlowSealedSidecars",
]
