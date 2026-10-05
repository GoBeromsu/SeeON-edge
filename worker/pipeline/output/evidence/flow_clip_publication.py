"""Publication adapter for clips sealed by the Flow Smart Record plane."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from worker.pipeline.output.evidence.clip_identity import (
    ClipIdAllocator,
    ClipIdCollisionError,
    ClipReservation,
)
from worker.pipeline.output.evidence.clip_publication_types import (
    ClipPublicationConflictError,
    ClipPublicationMetadata,
    PublishedClip,
)
from worker.pipeline.output.evidence.evidence_manifest import (
    ReadyClipManifest,
    UnavailableClipManifest,
    parse_manifest,
)
from worker.pipeline.output.evidence.evidence_outbox_types import ClipId, EvidenceReasonCode
from worker.pipeline.output.evidence.flow_sealed_observation import (
    ClipSealedUnavailable,
    FlowSealedObservation,
    unavailable_reason,
)
from worker.pipeline.output.evidence.flow_sealed_sidecar import FlowSealedMediaMissingError
from worker.pipeline.output.evidence.manifest_models import (
    ClipExtension,
    ExtensionContributor,
)
from worker.pipeline.output.evidence.smart_record_actor import ClipContributor, ClipSealed
from worker.types import BusinessEvent


class FlowClipPublicationPort(Protocol):
    def publish_adopted_ready(
        self,
        reservation: ClipReservation,
        source_path: Path,
        metadata: ClipPublicationMetadata,
    ) -> PublishedClip: ...

    def publish_unavailable(
        self,
        reservation: ClipReservation,
        metadata: ClipPublicationMetadata,
        reason_code: EvidenceReasonCode,
    ) -> PublishedClip: ...

    def reconcile_terminal(
        self,
        reservation: ClipReservation,
        metadata: ClipPublicationMetadata,
    ) -> PublishedClip: ...


@dataclass(slots=True)
class FlowClipPublicationStats:
    failures: int = 0


class FlowClipPublicationError(RuntimeError):
    """A sealed Smart Record clip could not become durable evidence."""


@dataclass(slots=True)
class FlowClipPublisher:
    """Adapt plane-owned Smart Record media to the standard clip publisher."""

    allocator: ClipIdAllocator
    publisher: FlowClipPublicationPort
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    stats: FlowClipPublicationStats = field(default_factory=FlowClipPublicationStats, init=False)

    def publish(
        self,
        sealed: FlowSealedObservation,
        events: Mapping[str, BusinessEvent],
    ) -> PublishedClip:
        """Publish one sealed observation, reconciling a committed terminal first.

        The terminal check precedes source media adoption, codec probing, and
        any clock read; only a committed READY terminal's own media is
        verified against its manifest. A ready observation with no terminal and no original,
        staged, or final media raises :class:`FlowSealedMediaMissingError`
        uncounted and unwrapped; an unavailable observation never depends on
        media and is never reported missing.
        """
        try:
            resumed = self._resume_terminal(sealed, events)
            if resumed is not None:
                return resumed
            if isinstance(sealed, ClipSealed):
                self._require_owned_media(sealed)
            return self._publish(sealed, events)
        except FlowSealedMediaMissingError:
            raise
        except Exception as exc:
            self.stats.failures += 1
            raise FlowClipPublicationError(
                f"Flow clip publication failed clip_id={sealed.clip_id}"
            ) from exc

    def _publish(
        self,
        sealed: FlowSealedObservation,
        events: Mapping[str, BusinessEvent],
    ) -> PublishedClip:
        metadata = self._metadata(sealed, events, finalized_at=None)
        try:
            reservation = self.allocator.reserve_existing(metadata.camera_id, sealed.clip_id)
        except ClipIdCollisionError:
            resumed = self._resume_terminal(sealed, events)
            if resumed is not None:
                return resumed
            raise
        if isinstance(sealed, ClipSealed):
            return self.publisher.publish_adopted_ready(reservation, Path(sealed.path), metadata)
        return self.publisher.publish_unavailable(reservation, metadata, unavailable_reason(sealed))

    def _metadata(
        self,
        sealed: FlowSealedObservation,
        events: Mapping[str, BusinessEvent],
        *,
        finalized_at: datetime | None,
    ) -> ClipPublicationMetadata:
        contributors, event, event_refs = _attribution(sealed, events)
        detected_at = _parse_timestamp(contributors[0].detected_at)
        duration_s = sealed.duration_ms / 1000.0
        if duration_s <= 0:
            raise ValueError("sealed Flow clip duration must be positive")
        clip_start_at = detected_at - timedelta(seconds=15)
        clip_end_at = clip_start_at + timedelta(seconds=duration_s)
        if finalized_at is None:
            finalized_at = max(self.now().astimezone(UTC), clip_end_at)
        return ClipPublicationMetadata(
            camera_id=event.camera_id,
            event_refs=event_refs,
            event_type=event.event_type,
            clip_start_at=clip_start_at,
            clip_end_at=clip_end_at,
            finalized_at=finalized_at,
            started_at=clip_start_at,
            detected_at=detected_at,
            duration_s=duration_s,
            encoder="deepstream-smart-record",
            truncation_reasons=(),
            domain=event.domain,
            extension=ClipExtension(
                contributors=tuple(
                    ExtensionContributor(event_ref=item.event_ref, detected_at=item.detected_at)
                    for item in contributors
                ),
                duration_s=duration_s,
                boundary=sealed.boundary,
            ),
            facility_id=event.facility_id,
        )

    def _resume_terminal(
        self, sealed: FlowSealedObservation, events: Mapping[str, BusinessEvent]
    ) -> PublishedClip | None:
        """Resume an already committed terminal for this exact observation.

        A crash after publication replays here on every restart. The committed
        manifest's identity and timing outrank anything this replay would
        computes. A malformed, conflicting, or orphan terminal (a final
        directory without a valid manifest) is never overwritten: it raises,
        so the caller keeps the sidecar. A matching terminal is completed by
        the publisher's reconciliation with the committed finalization time,
        verifying READY media and finishing its outcome and relay entry
        without reading the clock.
        """
        clip_id = sealed.clip_id
        final_dir = self.allocator.final_dir(clip_id)
        if not _lexists(final_dir):
            return None
        _, event, event_refs = _attribution(sealed, events)
        manifest_path = final_dir / "manifest.json"
        manifest = parse_manifest(manifest_path)
        if (
            manifest.clip_id != clip_id
            or manifest.camera_id != event.camera_id
            or manifest.event_refs != event_refs
        ):
            raise ClipPublicationConflictError(ClipId(clip_id), "terminal identity differs")
        if isinstance(sealed, ClipSealedUnavailable):
            expected = isinstance(manifest, UnavailableClipManifest) and (
                manifest.reason_code == unavailable_reason(sealed)
            )
        else:
            expected = isinstance(manifest, ReadyClipManifest) or (
                manifest.reason_code is EvidenceReasonCode.CORRUPT
            )
        if not expected:
            raise ClipPublicationConflictError(ClipId(clip_id), "terminal state differs")
        staging_dir = final_dir.parent / ".staging" / clip_id
        reservation = ClipReservation(ClipId(clip_id), event.camera_id, staging_dir, final_dir)
        metadata = self._metadata(
            sealed, events, finalized_at=_parse_timestamp(manifest.finalized_at)
        )
        return self.publisher.reconcile_terminal(reservation, metadata)

    def _require_owned_media(self, sealed: ClipSealed) -> None:
        """Confirm absence only after original, staged, and final media are all gone."""
        final_dir = self.allocator.final_dir(sealed.clip_id)
        owned = (Path(sealed.path), final_dir.parent / ".staging" / sealed.clip_id, final_dir)
        if not any(_lexists(path) for path in owned):
            raise FlowSealedMediaMissingError(sealed.clip_id, Path(sealed.path))


def _attribution(
    sealed: FlowSealedObservation, events: Mapping[str, BusinessEvent]
) -> tuple[tuple[ClipContributor, ...], BusinessEvent, tuple[str, ...]]:
    contributors = tuple(sorted(sealed.contributors, key=lambda item: item.detected_at))
    if not contributors:
        raise ValueError("sealed Flow clip has no contributors")
    event = events[contributors[0].event_ref]
    if any(events[item.event_ref].camera_id != event.camera_id for item in contributors):
        raise ValueError("sealed Flow clip spans cameras")
    if any(events[item.event_ref].facility_id != event.facility_id for item in contributors):
        raise ValueError("sealed Flow clip spans facilities")
    return contributors, event, tuple(item.event_ref for item in contributors)


def _lexists(path: Path) -> bool:
    """Whether a name exists, without following it; other I/O errors propagate."""
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Flow contributor detected_at must be timezone-aware")
    return parsed.astimezone(UTC)


__all__ = [
    "FlowClipPublicationError",
    "FlowClipPublicationPort",
    "FlowClipPublicationStats",
    "FlowClipPublisher",
]
