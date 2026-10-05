"""Durable recovery records for Flow Smart Record clips."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from worker.pipeline.output.evidence.durability import fsync_directory
from worker.pipeline.output.evidence.flow_sealed_observation import (
    ClipSealedUnavailable,
    FlowSealedMalformedError,
    FlowSealedObservation,
    decode_sidecar,
    encode_sidecar,
)
from worker.pipeline.output.evidence.flow_sealed_storage import (
    FlowSealedUnreadableError,
    publish_once,
    read_bounded,
    remove_durable,
    sidecar_names,
)
from worker.types import BusinessEvent

LOGGER: Final = logging.getLogger(__name__)


class FlowSealedMediaMissingError(RuntimeError):
    """A sealed clip was lost before its durable publication could be retried."""

    def __init__(self, clip_id: str, path: Path) -> None:
        super().__init__(f"sealed Flow clip media is missing clip_id={clip_id} path={path}")
        self.clip_id = clip_id
        self.path = path


class FlowSealedNegativeRetirementError(RuntimeError):
    """Unavailable evidence never retires through the missing-media path."""

    def __init__(self, clip_id: str) -> None:
        super().__init__(f"unavailable sealed Flow clip cannot retire as missing clip_id={clip_id}")
        self.clip_id = clip_id


@dataclass(frozen=True, slots=True)
class FlowSealedRecovery:
    sealed: FlowSealedObservation
    events: dict[str, BusinessEvent]
    camera_id: str
    sidecar_path: Path


class FlowSealedSidecars:
    """Persist sealed Flow clip attribution in the worker state directory.

    The state directory is used rather than the plane's output directory: Flow
    owns the latter and deployments may clean it independently of worker state.
    The sidecar records the reported media path, so it remains recoverable
    across that ownership boundary. Each sidecar is published once: an
    identical retry is idempotent and contradictory, malformed, or unreadable
    committed evidence is preserved and refused.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def persist(self, sealed: FlowSealedObservation, events: Mapping[str, BusinessEvent]) -> Path:
        payload = encode_sidecar(sealed, events)
        self._directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        target = self._directory / f"{sealed.clip_id}.json"
        publish_once(target, payload)
        return target

    def pending_for_camera(self, camera_id: str) -> tuple[FlowSealedRecovery, ...]:
        """This camera's sidecars in file-name order.

        Malformed or unreadable sidecars stay on disk and are reported, never
        republished or deleted; other I/O failures propagate.
        """
        recoveries: list[FlowSealedRecovery] = []
        for name in sidecar_names(self._directory):
            sidecar_path = self._directory / name
            try:
                data = read_bounded(sidecar_path)
                if data is None:
                    continue
                sealed, events, recorded_camera = decode_sidecar(data, name)
            except (FlowSealedUnreadableError, FlowSealedMalformedError):
                LOGGER.exception("sealed Flow sidecar retained unreadable file=%s", name)
                continue
            if recorded_camera == camera_id:
                recoveries.append(FlowSealedRecovery(sealed, events, camera_id, sidecar_path))
        return tuple(recoveries)

    def discard_missing_media(self, recovery: FlowSealedRecovery) -> FlowSealedMediaMissingError:
        """Retire a ready sidecar whose terminal and every owned media copy are absent."""
        sealed = recovery.sealed
        if isinstance(sealed, ClipSealedUnavailable):
            raise FlowSealedNegativeRetirementError(sealed.clip_id)
        recovery.sidecar_path.unlink()
        fsync_directory(recovery.sidecar_path.parent)
        return FlowSealedMediaMissingError(sealed.clip_id, Path(sealed.path))

    def remove(self, recovery: FlowSealedRecovery) -> None:
        remove_durable(recovery.sidecar_path)


__all__ = [
    "FlowSealedMediaMissingError",
    "FlowSealedNegativeRetirementError",
    "FlowSealedRecovery",
    "FlowSealedSidecars",
]
