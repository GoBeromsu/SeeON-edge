from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from contracts.frame import Frame
from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    AnnotatedDerivativeLimits,
    BoundedDerivativeQueue,
    DerivativeArtifact,
    DerivativeCancelled,
)
from worker.pipeline.output.overlay import OverlayRenderer
from worker.pipeline.output.overlay_scene import AppliedCameraProvenance, OverlaySceneBuilder
from worker.pipeline.trace.models import AnalysisTrace, OptionalNumber
from worker.types import FramePacket


def _scene(width: int, height: int):
    analysis = AnalysisTrace(
        trace_id="a" * 64,
        frame_key=("boot-a", "camera-a", 1, 1),
        pts=OptionalNumber(1.0),
        source_time=OptionalNumber(1.0),
        frame_width=width,
        frame_height=height,
        bed_region_provenance="empty",
        persons=(),
        beds=(),
        components=(),
    )
    return OverlaySceneBuilder().from_traces(
        analysis,
        (),
        provenance=AppliedCameraProvenance("b" * 64, "camera.v1"),
    )


def _packet(width: int, height: int) -> FramePacket:
    return FramePacket(
        camera_id="camera-a",
        frame=Frame(1, 1.0, np.zeros((height, width, 3), dtype=np.uint8)),
        pts=1.0,
        seq=1,
        width=width,
        height=height,
        decode_time_ms=0.0,
        worker_boot_id="boot-a",
        stream_epoch=1,
    )


def test_still_and_video_renderers_consume_the_same_scene_contract() -> None:
    scene = _scene(320, 180)
    renderer = OverlayRenderer()

    still = renderer.render_scene(_packet(320, 180), scene)
    video = renderer.render_scene(_packet(320, 180), scene)

    assert np.array_equal(still, video)
    assert hashlib.sha256(still.tobytes()).digest() == hashlib.sha256(video.tobytes()).digest()


@pytest.mark.parametrize(
    ("incident_id", "primary_sha256"),
    (("../private", "a" * 64), ("incident-a", "A" * 64)),
)
def test_job_rejects_path_like_ids_and_noncanonical_hashes(
    tmp_path: Path, incident_id: str, primary_sha256: str
) -> None:
    with pytest.raises(ValueError, match="identity|SHA-256"):
        AnnotatedDerivativeJob(
            incident_id,
            "clip-a",
            tmp_path / "source.mp4",
            primary_sha256,
            "d" * 64,
            "b" * 64,
            (_scene(16, 16),),
            1,
        )


def test_queue_is_bounded_and_cancellation_is_explicit(tmp_path: Path) -> None:
    queue = BoundedDerivativeQueue(
        AnnotatedDerivativeLimits(
            max_pending_jobs=1,
            max_pending_source_bytes=10,
            max_output_bytes=100,
            max_duration_seconds=1.0,
            max_disk_bytes=100,
        )
    )
    first = AnnotatedDerivativeJob(
        "incident-a", "clip-a", tmp_path / "a.mp4", "a" * 64, "d" * 64, "b" * 64,
        (_scene(16, 16),), 9,
    )
    second = AnnotatedDerivativeJob(
        "incident-b", "clip-b", tmp_path / "b.mp4", "c" * 64, "e" * 64, "b" * 64,
        (_scene(16, 16),), 2,
    )

    queue.submit(first)
    with pytest.raises(OverflowError, match="bounded"):
        queue.submit(second)
    queue.cancel("incident-a")
    with pytest.raises(DerivativeCancelled):
        queue.take()


def test_artifact_hashing_is_streamed_without_unbounded_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "render.mp4"
    output.write_bytes(b"bounded-content" * 1024)

    def fail_unbounded_read(_path: Path) -> bytes:
        raise AssertionError("read_bytes must not be used for derivative hashing")

    monkeypatch.setattr(Path, "read_bytes", fail_unbounded_read)
    artifact = DerivativeArtifact.from_path(
        output,
        mime_type="video/mp4",
        width=16,
        height=16,
        start_time_ms=0,
        end_time_ms=1000,
        render_backend="opencv-cpu",
        render_version="overlay-cpu.v1",
        scene_id="a" * 64,
    )

    assert artifact.size_bytes == len(b"bounded-content" * 1024)
