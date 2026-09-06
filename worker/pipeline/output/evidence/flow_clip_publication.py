"""Publication adapter for clips sealed by the Flow Smart Record plane."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from worker.pipeline.output.evidence.clip_identity import ClipIdAllocator, ClipReservation
from worker.pipeline.output.evidence.clip_publication_types import (
    ClipPublicationMetadata,
    PublishedClip,
)
from worker.pipeline.output.evidence.manifest_models import (
    ClipExtension,
    ExtensionContributor,
)
from worker.pipeline.output.evidence.smart_record_actor import ClipSealed
from worker.types import BusinessEvent


class FlowClipPublicationPort(Protocol):
    def publish_adopted_ready(
        self,
        reservation: ClipReservation,
        source_path: Path,
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
        sealed: ClipSealed,
        events: Mapping[str, BusinessEvent],
    ) -> PublishedClip:
        try:
            return self._publish(sealed, events)
        except Exception as exc:
            self.stats.failures += 1
            raise FlowClipPublicationError(
                f"Flow clip publication failed clip_id={sealed.clip_id}"
            ) from exc

    def _publish(
        self,
        sealed: ClipSealed,
        events: Mapping[str, BusinessEvent],
    ) -> PublishedClip:
        contributors = tuple(sorted(sealed.contributors, key=lambda item: item.detected_at))
        if not contributors:
            raise ValueError("sealed Flow clip has no contributors")
        primary = contributors[0]
        event = events[primary.event_ref]
        if any(events[item.event_ref].camera_id != event.camera_id for item in contributors):
            raise ValueError("sealed Flow clip spans cameras")
        if any(events[item.event_ref].facility_id != event.facility_id for item in contributors):
            raise ValueError("sealed Flow clip spans facilities")
        detected_at = _parse_timestamp(primary.detected_at)
        duration_s = sealed.duration_ms / 1000.0
        if duration_s <= 0:
            raise ValueError("sealed Flow clip duration must be positive")
        finalized_at = self.now().astimezone(UTC)
        clip_start_at = detected_at - timedelta(seconds=15)
        clip_end_at = clip_start_at + timedelta(seconds=duration_s)
        if finalized_at < clip_end_at:
            finalized_at = clip_end_at
        reservation = self.allocator.reserve_existing(event.camera_id, sealed.clip_id)
        return self.publisher.publish_adopted_ready(
            reservation,
            Path(sealed.path),
            ClipPublicationMetadata(
                camera_id=event.camera_id,
                event_refs=tuple(item.event_ref for item in contributors),
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
            ),
        )


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
