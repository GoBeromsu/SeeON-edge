from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.events.delivery_queue import AdmissionResult
from tests_support.clip_analysis import no_op_ready_hook
from tests_support.thumbnail import DeterministicThumbnailGenerator
from worker.pipeline.output.evidence.clip_identity import ClipIdAllocator
from worker.pipeline.output.evidence.clip_publication import ClipPublisher
from worker.pipeline.output.evidence.clip_publication_types import PublicationStage
from worker.pipeline.output.evidence.evidence_manifest import (
    UnavailableClipManifest,
    parse_manifest,
)
from worker.pipeline.output.evidence.evidence_media import MediaFacts
from worker.pipeline.output.evidence.evidence_outbox_types import EvidenceReasonCode
from worker.pipeline.output.evidence.flow_clip_publication import (
    FlowClipPublicationError,
    FlowClipPublisher,
)
from worker.pipeline.output.evidence.flow_sealed_observation import ClipSealedUnavailable
from worker.pipeline.output.evidence.flow_sealed_sidecar import (
    FlowSealedNegativeRetirementError,
    FlowSealedSidecars,
)
from worker.pipeline.output.evidence.flow_sealed_storage import FlowSealedConflictError
from worker.pipeline.output.evidence.smart_record_actor import (
    ClipContributor,
    ClipSealed,
    SmartRecordActor,
)
from worker.runtime.flow.evidence import FlowEvidenceBinding
from worker.types import BusinessEvent, NativeEvidenceTrigger

EVENT = "00000000-0000-4000-8000-000000000001"
DETECTED = "2026-01-01T00:00:00Z"
NOW = datetime(2026, 1, 1, 0, 2, tzinfo=UTC)


@dataclass
class _Plane:
    def start_recording(
        self, camera_id: str, *, lookback_sec: int, duration_sec: int, on_sealed: object
    ) -> int:
        return 1

    def stop_recording(self, camera_id: str, session_id: int) -> None:
        pass


@dataclass
class _Stager:
    completed: list[tuple[str, str | None]] = field(default_factory=list)

    def stage(self, event: dict[str, object]) -> AdmissionResult:
        return AdmissionResult(True)

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        self.completed.append((edge_event_id, clip_id))


def _event() -> BusinessEvent:
    return BusinessEvent("fall", "fall.detected", EVENT, "camera-a", "facility-a", 12.0, 0.99)


def _trigger() -> NativeEvidenceTrigger:
    return NativeEvidenceTrigger("camera-a", "boot", 1, 1, 1, 12_000_000_000, 12.0)


def _unavailable(
    *, native_result: int, contains_video: bool, duration_ms: int = 4_000, path: str | None = None
) -> ClipSealedUnavailable:
    return ClipSealedUnavailable(
        clip_id="clip-1",
        duration_ms=duration_ms,
        contributors=(ClipContributor(EVENT, DETECTED),),
        boundary="none",
        native_result=native_result,
        contains_video=contains_video,
        path=path,
    )


def _no_clock() -> datetime:
    raise AssertionError("terminal replay must not read the clock")


def _binding(
    tmp_path: Path,
    *,
    barrier: Callable[[PublicationStage, Path], None] | None = None,
    now: Callable[[], datetime] = lambda: NOW,
) -> tuple[FlowEvidenceBinding, _Stager, FlowSealedSidecars]:
    store = tmp_path / "store"
    options = {} if barrier is None else {"barrier": barrier}
    clip_publisher = ClipPublisher(
        store,
        thumbnail_generator=DeterministicThumbnailGenerator(),
        on_ready=no_op_ready_hook,
        **options,
    )
    stager = _Stager()
    sidecars = FlowSealedSidecars(tmp_path / "state")
    actor = SmartRecordActor(
        camera_id="camera-a",
        media_plane=_Plane(),
        clock=lambda: 0.0,
        sink=lambda _item: None,
        lookback_sec=15,
        clip_id_factory=lambda: "clip-1",
    )
    binding = FlowEvidenceBinding(
        actor=actor,
        stager=stager,
        publisher=FlowClipPublisher(ClipIdAllocator(store), clip_publisher, now=now),
        sidecars=sidecars,
        camera_id="camera-a",
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    return binding, stager, sidecars


def test_unavailable_sidecar_round_trips_publishes_once_and_is_never_discarded(
    tmp_path: Path,
) -> None:
    sidecars = FlowSealedSidecars(tmp_path / "state")
    observation = _unavailable(native_result=0, contains_video=False, duration_ms=2**64 - 1)
    events = {EVENT: _event()}

    path = sidecars.persist(observation, events)
    committed = path.read_bytes()
    assert sidecars.persist(observation, events) == path
    [recovery] = sidecars.pending_for_camera("camera-a")
    assert recovery.sealed == observation
    assert recovery.events == events

    with pytest.raises(FlowSealedConflictError):
        sidecars.persist(replace(observation, native_result=7), events)
    with pytest.raises(FlowSealedNegativeRetirementError):
        sidecars.discard_missing_media(recovery)
    assert path.read_bytes() == committed
    with pytest.raises(ValueError, match="successful native result"):
        _unavailable(native_result=0, contains_video=True)


@pytest.mark.parametrize(
    ("native_result", "contains_video", "reason"),
    [
        (-3, True, EvidenceReasonCode.ENCODER_FAILED),
        (0, False, EvidenceReasonCode.NO_FRAMES),
    ],
)
def test_unavailable_observation_publishes_existing_reason_without_media(
    tmp_path: Path, native_result: int, contains_video: bool, reason: EvidenceReasonCode
) -> None:
    binding, stager, sidecars = _binding(tmp_path)
    binding.emit_for_frame(_event(), _trigger())

    binding.on_sealed(
        _unavailable(
            native_result=native_result,
            contains_video=contains_video,
            path=str(tmp_path / "never-written.mp4"),
        )
    )

    manifest = parse_manifest(tmp_path / "store" / "clips" / "clip-1" / "manifest.json")
    assert isinstance(manifest, UnavailableClipManifest)
    assert manifest.reason_code is reason
    assert stager.completed == [(EVENT, "clip-1")]
    assert sidecars.pending_for_camera("camera-a") == ()


def test_zero_duration_unavailable_stays_durable_and_unpublished(tmp_path: Path) -> None:
    binding, stager, sidecars = _binding(tmp_path)
    binding.emit_for_frame(_event(), _trigger())

    with pytest.raises(FlowClipPublicationError):
        binding.on_sealed(_unavailable(native_result=0, contains_video=False, duration_ms=0))
    binding.replay_sealed()

    [recovery] = sidecars.pending_for_camera("camera-a")
    assert recovery.sealed.duration_ms == 0
    assert stager.completed == []
    assert binding.sealed_recovery_missing_media_total == 0
    assert not (tmp_path / "store" / "clips" / "clip-1").exists()


def test_ready_replay_completes_a_terminal_interrupted_after_manifest_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts("a" * 64, len(b"clip-bytes"), 1000),
    )

    def interrupt(stage: PublicationStage, _path: Path) -> None:
        if stage is PublicationStage.MANIFEST_RENAMED:
            raise RuntimeError("crashed after manifest rename")

    binding, _, sidecars = _binding(tmp_path, barrier=interrupt)
    media = tmp_path / "plane" / "clip-1.mp4"
    media.parent.mkdir()
    media.write_bytes(b"clip-bytes")
    binding.emit_for_frame(_event(), _trigger())
    with pytest.raises(FlowClipPublicationError):
        binding.on_sealed(
            ClipSealed("clip-1", str(media), 60_000, (ClipContributor(EVENT, DETECTED),), "none")
        )
    final_dir = tmp_path / "store" / "clips" / "clip-1"
    manifest_bytes = (final_dir / "manifest.json").read_bytes()
    assert not (final_dir / "terminal-outcome.json").exists()

    restarted, restarted_stager, _ = _binding(tmp_path, now=_no_clock)
    restarted.replay_sealed()

    assert (final_dir / "manifest.json").read_bytes() == manifest_bytes
    assert (final_dir / "terminal-outcome.json").is_file()
    assert restarted_stager.completed == [(EVENT, "clip-1")]
    assert sidecars.pending_for_camera("camera-a") == ()
