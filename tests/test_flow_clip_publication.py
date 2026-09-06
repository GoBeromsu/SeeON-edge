from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from worker.pipeline.output.evidence.clip_identity import ClipIdAllocator, ClipReservation
from worker.pipeline.output.evidence.clip_manifest_payload import manifest_payload
from worker.pipeline.output.evidence.clip_publication_types import ClipPublicationMetadata
from worker.pipeline.output.evidence.evidence_manifest import ReadyClipManifest
from worker.pipeline.output.evidence.flow_clip_publication import (
    FlowClipPublicationError,
    FlowClipPublisher,
)
from worker.pipeline.output.evidence.smart_record_actor import (
    ClipContributor,
    ClipSealed,
)
from worker.types import BusinessEvent


@dataclass
class _Publisher:
    calls: list[tuple[ClipReservation, Path, ClipPublicationMetadata]]
    fail: bool = False

    def publish_adopted_ready(
        self,
        reservation: ClipReservation,
        source_path: Path,
        metadata: ClipPublicationMetadata,
    ) -> object:
        if self.fail:
            raise OSError("store unavailable")
        self.calls.append((reservation, source_path, metadata))
        return type("_Published", (), {"clip_id": reservation.clip_id})()


def _event(event_ref: str) -> BusinessEvent:
    return BusinessEvent("fall", "fall.detected", event_ref, "camera-a", "facility-a", 12.0, 0.99)


def test_sealed_clip_publishes_ordered_extension_with_covering_duration(tmp_path: Path) -> None:
    publisher = _Publisher([])
    flow = FlowClipPublisher(
        ClipIdAllocator(tmp_path),
        publisher,
        now=lambda: datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
    )
    sealed = ClipSealed(
        "clip-1",
        "/plane/clip-1.mp4",
        60_000,
        (
            ClipContributor("late", "2026-01-01T00:00:20Z"),
            ClipContributor("early", "2026-01-01T00:00:00Z"),
        ),
        "extension_bounded",
    )

    published = flow.publish(sealed, {"early": _event("early"), "late": _event("late")})

    assert str(published.clip_id) == "clip-1"
    reservation, source_path, metadata = publisher.calls[0]
    assert str(reservation.clip_id) == "clip-1"
    assert source_path == Path("/plane/clip-1.mp4")
    assert metadata.event_refs == ("early", "late")
    assert metadata.facility_id == "facility-a"
    assert metadata.detected_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert metadata.duration_s == 60.0
    assert metadata.extension is not None
    assert metadata.extension.duration_s == 60.0
    assert [item.event_ref for item in metadata.extension.contributors] == ["early", "late"]
    manifest = ReadyClipManifest.model_construct(
        clip_id="clip-1",
        camera_id="camera-a",
        event_refs=("early", "late"),
        clip_start_at="2025-12-31T23:59:45Z",
        clip_end_at="2026-01-01T00:00:45Z",
        finalized_at="2026-01-01T00:02:00Z",
        sha256="a" * 64,
        size_bytes=1,
        duration_ms=60_000,
    )
    payload = manifest_payload(
        manifest,
        metadata,
        path="clips/clip-1/clip.mp4",
        video_available=True,
    )
    assert payload["extension"] == {
        "boundary": "extension_bounded",
        "contributors": [
            {"detected_at": "2026-01-01T00:00:00Z", "event_ref": "early"},
            {"detected_at": "2026-01-01T00:00:20Z", "event_ref": "late"},
        ],
        "duration_s": 60.0,
    }


def test_publication_failure_is_typed_and_counted_without_consuming_reservation(
    tmp_path: Path,
) -> None:
    publisher = _Publisher([], fail=True)
    flow = FlowClipPublisher(ClipIdAllocator(tmp_path), publisher)
    sealed = ClipSealed(
        "clip-1",
        "/plane/clip-1.mp4",
        60_000,
        (ClipContributor("one", "2026-01-01T00:00:00Z"),),
        "none",
    )

    with pytest.raises(FlowClipPublicationError, match="clip_id=clip-1"):
        flow.publish(sealed, {"one": _event("one")})

    assert flow.stats.failures == 1
    assert (tmp_path / "clips" / ".staging" / "clip-1").is_dir()
