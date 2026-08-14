"""Attach audit metadata and a bounded JPEG snapshot onto admitted events.

Mirrors edge's ``CameraWorker._attach_alert_metadata``
(edge/runtime/camera_worker.py:289-336): for events whose domain has a
registered audit envelope, attach ``audit`` and (when a renderer is
configured) a size-bounded ``snapshot_jpeg``. Any failure degrades to
returning the event unmodified -- audit/snapshot metadata must never block
an alert from reaching the sink, same as edge's broad except.

Split out of the pump (worker/pipeline/camera_pipeline.py) so this business
logic lives in the output layer rather than the wiring stage that calls it
or the composition root that constructs it (worker/runtime/worker.py builds
one ``AlertEvidenceAttacher`` per camera and injects it into that camera's
pump; it contains no audit/snapshot logic itself).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, Protocol

from contracts.observation import FrameObservation
from worker.pipeline.output.evidence.evidence_metadata import (
    RUNTIME_MANIFEST_SHA256_KEY,
    validate_runtime_manifest_sha256,
)
from worker.types import BusinessEvent, FramePacket

LOGGER: Final = logging.getLogger(__name__)


class SnapshotRenderer(Protocol):
    """Structural seam matched by ``worker.pipeline.output.overlay.OverlayRenderer``."""

    def encode_jpeg_bounded(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[Any, ...] = (),
    ) -> bytes | None: ...


@dataclass(frozen=True, slots=True)
class AlertEvidenceAttacher:
    """Per-camera collaborator: attach audit + snapshot onto admitted events.

    ``domain_audit`` is a precomputed, static per-domain envelope (built once
    by the composition root -- ``worker/domains/registry.py``'s
    ``audit_metadata_provider`` is a pure function of a static
    ``AuditContext``), so no domain-registry lookup happens per event here.
    A missing/empty entry for an event's domain means that domain has no
    audit provider registered, so the event passes through untouched --
    matching edge's ``registration is None`` early return.
    """

    domain_audit: Mapping[str, Mapping[str, object]]
    overlay_renderer: SnapshotRenderer | None = None
    debug_snapshots_provider: Callable[[int], tuple[Any, ...]] | None = None
    runtime_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_runtime_manifest_sha256(self.runtime_manifest_sha256)

    def attach(
        self,
        event: BusinessEvent,
        packet: FramePacket,
        observation: FrameObservation,
    ) -> BusinessEvent:
        audit = dict(event.audit or {})
        audit.update(self.domain_audit.get(event.domain, {}))
        if self.runtime_manifest_sha256 is not None:
            audit[RUNTIME_MANIFEST_SHA256_KEY] = self.runtime_manifest_sha256
        if not audit:
            return event
        try:
            snapshot_jpeg = None
            if self.overlay_renderer is not None:
                debug_snapshots = (
                    ()
                    if self.debug_snapshots_provider is None
                    else self.debug_snapshots_provider(packet.frame.index)
                )
                snapshot_jpeg = self.overlay_renderer.encode_jpeg_bounded(
                    packet, observation, debug_snapshots
                )
            return replace(event, audit=audit, snapshot_jpeg=snapshot_jpeg)
        except Exception:  # noqa: BLE001 - audit/snapshot must not block alert emit
            LOGGER.warning(
                "failed to attach audit/snapshot metadata to event",
                extra={"camera_id": event.camera_id, "domain": event.domain},
            )
            return event


__all__ = ["AlertEvidenceAttacher", "SnapshotRenderer"]
