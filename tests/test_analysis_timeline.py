from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contracts.frame import Frame
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.pipeline.trace import (
    BoundedTraceWriter,
    TraceCapture,
    TraceIdentity,
    TraceRetentionPolicy,
)
from worker.types import DecisionInput, FramePacket

RUNTIME_MANIFEST_SHA256 = "a" * 64
COMPONENT_SHA256 = "b" * 64
POLICY_SHA256 = "c" * 64


def _packet(seq: int, *, pts: float | None = None) -> FramePacket:
    source_time = float(seq) if pts is None else pts
    return FramePacket(
        camera_id="opaque/camera:alpha",
        frame=Frame(seq, source_time, np.full((4, 6, 3), seq, dtype=np.uint8)),
        pts=pts,
        seq=seq,
        width=6,
        height=4,
        decode_time_ms=0.25,
        worker_boot_id="boot-a",
        stream_epoch=7,
    )


@dataclass(frozen=True)
class _TraceResult:
    module_results: tuple[object, ...]
    observation: FrameObservation
    decision_input: DecisionInput


def _result(seq: int) -> _TraceResult:
    person = BoundingBox(1, 0, 4, 4, 0.875)
    bed = BoundingBox(0, 0, 5, 4, 0.75, ((0, 1), (5, 0), (5, 4), (0, 4)))
    observation = FrameObservation(
        detections=((person,), ()),
        poses=(((2, 1, 0.9), (3, 2, 0.1)),),
        regions=((bed,), ()),
        track_ids=(42,),
    )
    decision_input = DecisionInput(
        observation=observation,
        frame_width=6,
        frame_height=4,
        live_track_ids=(42,),
        time_sec=float(seq),
        frame_index=seq,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )
    return _TraceResult((), observation, decision_input)


def _capture() -> TraceCapture:
    return TraceCapture(
        identities=(
            TraceIdentity(
                module_qualified_id="bed_exit.v1",
                component_qualified_ids=(f"pose.sha256.{COMPONENT_SHA256}",),
                policy_qualified_id="bed_exit.policy.v1",
                effective_policy_id=POLICY_SHA256,
                runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
            ),
        )
    )


def test_analysis_timeline_is_image_free_typed_and_explicit_about_missing_pts(
    tmp_path: Path,
) -> None:
    writer = BoundedTraceWriter(
        tmp_path / "runtime-analysis",
        TraceRetentionPolicy(
            max_frames_per_camera=4,
            max_age_seconds=60.0,
            max_pending_frames=4,
            max_batch_size=2,
            max_numeric_values_per_decision=16,
        ),
    )
    writer.start()
    try:
        assert _capture().capture(writer, _packet(3), _result(3), (), require_persisted=True)
    finally:
        writer.stop()

    recovered = writer.recover_camera("opaque/camera:alpha")
    assert len(recovered.frames) == 1
    frame = recovered.frames[0]
    assert frame.frame_key == ("boot-a", "opaque/camera:alpha", 7, 3)
    assert frame.pts.value is None
    assert frame.pts.missing_reason == "source-not-provided"
    assert frame.source_time.value == 3.0
    assert frame.persons[0].track_id.value == 42
    assert frame.persons[0].box == (1, 0, 4, 4)
    assert tuple((point.x, point.y, point.confidence) for point in frame.persons[0].keypoints) == (
        (2, 1, 0.9),
        (3, 2, 0.1),
    )
    assert frame.beds[0].provenance == "fresh"
    assert frame.beds[0].polygon == ((0, 1), (5, 0), (5, 4), (0, 4))

    recovered_again = writer.recover_camera("opaque/camera:alpha")
    assert recovered_again.frames[0].persons[0].box == (1, 0, 4, 4)
    assert "rtsp://" not in repr(recovered_again)


def test_restart_recovery_preserves_only_the_bounded_camera_ring_and_reports_truncation(
    tmp_path: Path,
) -> None:
    policy = TraceRetentionPolicy(
        max_frames_per_camera=2,
        max_age_seconds=100.0,
        max_pending_frames=8,
        max_batch_size=2,
        max_numeric_values_per_decision=16,
    )
    writer = BoundedTraceWriter(tmp_path / "runtime-analysis", policy)
    writer.start()
    try:
        for seq in range(4):
            assert _capture().capture(
                writer,
                _packet(seq, pts=float(seq)),
                _result(seq),
                (),
                require_persisted=True,
            )
    finally:
        writer.stop()

    restarted = BoundedTraceWriter(tmp_path / "runtime-analysis", policy)
    recovered = restarted.recover_camera("opaque/camera:alpha")
    assert [frame.frame_key[-1] for frame in recovered.frames] == [2, 3]
    assert recovered.truncation.pruned_frames == 2
    assert recovered.truncation.oldest_retained_seq == 2
    assert recovered.truncation.newest_retained_seq == 3


def test_bounded_handoff_drops_without_waiting_and_exposes_the_drop() -> None:
    writer = BoundedTraceWriter(
        Path("/not-opened/edge.sqlite3"),
        TraceRetentionPolicy(
            max_frames_per_camera=2,
            max_age_seconds=10.0,
            max_pending_frames=1,
            max_batch_size=1,
            max_numeric_values_per_decision=4,
        ),
    )
    first = _capture().build(_packet(0, pts=0.0), _result(0), ())
    second = _capture().build(_packet(1, pts=1.0), _result(1), ())

    assert writer.submit(first) is True
    assert writer.submit(second) is False
    assert writer.stats().handoff_dropped_frames == 1


def test_trace_identity_rejects_urls_paths_and_noncanonical_hashes() -> None:
    import pytest

    with pytest.raises(ValueError, match="component_qualified_ids"):
        TraceIdentity(
            module_qualified_id="fall.v1",
            component_qualified_ids=("rtsp://user:secret@camera/model",),
            policy_qualified_id="fall.policy.v1",
            effective_policy_id=POLICY_SHA256,
            runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
        )
    with pytest.raises(ValueError, match="runtime_manifest_sha256"):
        TraceIdentity(
            module_qualified_id="fall.v1",
            component_qualified_ids=(f"pose.sha256.{COMPONENT_SHA256}",),
            policy_qualified_id="fall.policy.v1",
            effective_policy_id=POLICY_SHA256,
            runtime_manifest_sha256="A" * 64,
        )
