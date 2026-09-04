from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from worker.interfaces.media_plane import RecordingInfo
from worker.pipeline.output.evidence.flow_clip_publication import FlowClipPublicationError
from worker.pipeline.output.evidence.flow_sealed_sidecar import FlowSealedSidecars
from worker.pipeline.output.evidence.smart_record_actor import (
    ClipContributor,
    ClipSealed,
    SmartRecordActor,
)
from worker.runtime.flow.evidence import FlowEvidenceBinding
from worker.types import BusinessEvent, NativeEvidenceTrigger


@dataclass
class _Plane:
    callback: object | None = None

    def start_recording(
        self, camera_id: str, *, lookback_sec: int, duration_sec: int, on_sealed: object
    ) -> int:
        self.callback = on_sealed
        return 1

    def stop_recording(self, camera_id: str, session_id: int) -> None:
        pass


@dataclass
class _Stager:
    completed: list[tuple[str, str | None]] = field(default_factory=list)

    def stage(self, event: dict[str, object]) -> None:
        pass

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        self.completed.append((edge_event_id, clip_id))


@dataclass
class _Publisher:
    fail: bool = False
    calls: list[str] = field(default_factory=list)

    def publish(self, sealed: ClipSealed, events: object) -> object:
        del events
        self.calls.append(sealed.clip_id)
        if self.fail:
            raise FlowClipPublicationError("publication failed")
        return type("_Published", (), {"clip_id": sealed.clip_id})()


def _event(identity: str) -> BusinessEvent:
    return BusinessEvent("fall", "fall.detected", identity, "camera-a", "facility-a", 12.0, 0.99)


def _trigger() -> NativeEvidenceTrigger:
    return NativeEvidenceTrigger("camera-a", "boot", 1, 1, 1, 12_000_000_000, 12.0)


def _binding(
    plane: _Plane, stager: _Stager, publisher: _Publisher, sidecars: FlowSealedSidecars
) -> FlowEvidenceBinding:
    binding: list[FlowEvidenceBinding] = []
    actor = SmartRecordActor(
        camera_id="camera-a",
        media_plane=plane,
        clock=lambda: 0.0,
        sink=lambda sealed: binding[0].on_sealed(sealed),
        lookback_sec=15,
        clip_id_factory=lambda: "clip-1",
    )
    result = FlowEvidenceBinding(
        actor=actor,
        stager=stager,
        publisher=publisher,
        sidecars=sidecars,
        camera_id="camera-a",
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    binding.append(result)
    return result


def test_failed_seal_replays_contributors_and_discards_missing_media(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"clip")
    sidecars = FlowSealedSidecars(tmp_path / "state")
    stager = _Stager()
    publisher = _Publisher(fail=True)
    plane = _Plane()
    binding = _binding(plane, stager, publisher, sidecars)
    binding.emit_for_frame(_event("one"), _trigger())
    binding.emit_for_frame(_event("two"), _trigger())

    assert callable(plane.callback)
    with pytest.raises(FlowClipPublicationError, match="publication failed"):
        plane.callback(RecordingInfo(1, "camera-a", str(media), 12_000, 640, 360))

    pending = sidecars.pending_for_camera("camera-a")
    assert len(pending) == 1
    assert pending[0].sealed.clip_id == "clip-1"
    assert [item.event_ref for item in pending[0].sealed.contributors] == ["one", "two"]
    assert stager.completed == []

    missing = ClipSealed(
        "a-missing",
        str(tmp_path / "gone.mp4"),
        12_000,
        (ClipContributor("gone", "2026-01-01T00:00:00Z"),),
        "none",
    )
    sidecars.persist(missing, {"gone": _event("gone")})

    restarted = _binding(_Plane(), stager, _Publisher(), sidecars)
    restarted.replay_sealed()

    assert stager.completed == [("one", "clip-1"), ("two", "clip-1")]
    assert restarted.sealed_recovery_missing_media_total == 1
    assert sidecars.pending_for_camera("camera-a") == ()

    restarted.replay_sealed()
    assert stager.completed == [("one", "clip-1"), ("two", "clip-1")]
