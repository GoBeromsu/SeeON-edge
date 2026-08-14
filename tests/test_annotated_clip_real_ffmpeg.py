from __future__ import annotations

import hashlib
import shutil
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    AnnotatedDerivativeLimits,
    CpuAnnotatedVideoRenderer,
    DerivativeCancelled,
    DerivativeRenderError,
)
from worker.pipeline.output.overlay_scene import AppliedCameraProvenance, OverlaySceneBuilder
from worker.pipeline.trace.models import AnalysisTrace, OptionalNumber, TracePerson
from worker.types.overlay_scene import SceneLabel

pytestmark = pytest.mark.real_stack


def _require(binary: str) -> str:
    value = shutil.which(binary)
    if value is None:
        pytest.fail(f"{binary} is required for derivative QA", pytrace=False)
    return value


def _source(path: Path, width: int, height: int, *, rate: str = "5", audio: bool = False) -> None:
    command = [
        _require("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r={rate}:d=1",
    ]
    if audio:
        command.extend(("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:d=1"))
    command.extend(("-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "1"))
    if audio:
        command.extend(("-c:a", "aac", "-shortest"))
    else:
        command.append("-an")
    command.extend(("-y", str(path)))
    subprocess.run(tuple(command), check=True)


def _scene(width: int, height: int):
    analysis = AnalysisTrace(
        "a" * 64,
        ("boot-a", "camera-a", 1, 1),
        OptionalNumber(0.0),
        OptionalNumber(0.0),
        width,
        height,
        "fresh",
        (TracePerson(0, OptionalNumber(7), (10, 10, width // 2, height - 10), 0.9),),
        (),
        (),
    )
    scene = OverlaySceneBuilder().from_traces(
        analysis,
        (),
        provenance=AppliedCameraProvenance("b" * 64, "camera.v1"),
    )
    return replace(
        scene,
        labels=scene.labels
        + (SceneLabel("낙상 감지", (10.0, float(height - 5)), (0, 0, 255), 50),),
    )


@pytest.mark.parametrize("geometry", ((320, 180), (240, 320), (320, 240)))
def test_real_ffmpeg_derivative_is_deterministic_and_decodable(
    tmp_path: Path, geometry: tuple[int, int]
) -> None:
    width, height = geometry
    source = tmp_path / "source.mp4"
    _source(source, width, height)
    scene = _scene(width, height)
    job = AnnotatedDerivativeJob(
        "incident-a",
        "clip-a",
        source,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "d" * 64,
        "b" * 64,
        (scene,),
        source.stat().st_size,
    )
    renderer = CpuAnnotatedVideoRenderer(ffmpeg_bin=_require("ffmpeg"))

    first = renderer.render(job, tmp_path / "first.mp4")
    second = renderer.render(job, tmp_path / "second.mp4")
    probe = subprocess.run(
        (
            _require("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "csv=p=0",
            str(first.path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    assert f"h264,{width},{height}" in probe.stdout.strip()


def test_real_ffmpeg_30000_over_1001_keeps_frame_selection_and_audio_timing(tmp_path: Path) -> None:
    source = tmp_path / "fractional-source.mp4"
    _source(source, 320, 180, rate="30000/1001", audio=True)
    first = _scene(320, 180)
    second = replace(
        first,
        frame=replace(first.frame, pts=replace(first.frame.pts, value=1001 / 30000)),
        labels=first.labels + (SceneLabel("FRAME-ONE", (160.0, 20.0), (0, 0, 255), 60),),
    )
    job = AnnotatedDerivativeJob(
        "incident-a",
        "clip-a",
        source,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "d" * 64,
        "b" * 64,
        (first, second),
        source.stat().st_size,
    )

    artifact = CpuAnnotatedVideoRenderer(ffmpeg_bin=_require("ffmpeg")).render(
        job, tmp_path / "fractional-output.mp4"
    )

    def streams(path: Path) -> str:
        return subprocess.run(
            (
                _require("ffprobe"),
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,r_frame_rate,nb_frames",
                "-of",
                "csv=p=0",
                str(path),
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    source_streams = streams(source)
    output_streams = streams(artifact.path)
    capture = cv2.VideoCapture(str(artifact.path))
    _, frame_zero = capture.read()
    _, frame_one = capture.read()
    capture.release()

    assert "video,30000/1001" in output_streams
    assert "audio,0/0" in output_streams
    assert (
        output_streams.splitlines()[0].rsplit(",", 1)[-1]
        == source_streams.splitlines()[0].rsplit(",", 1)[-1]
    )
    assert frame_zero is not None and frame_one is not None
    assert not np.array_equal(frame_zero[:50, 150:300], frame_one[:50, 150:300])


def test_real_renderer_enforces_duration_memory_and_cancellation_bounds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    _source(source, 320, 180)
    scene = _scene(320, 180)
    job = AnnotatedDerivativeJob(
        "incident-a",
        "clip-a",
        source,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "d" * 64,
        "b" * 64,
        (scene,),
        source.stat().st_size,
    )

    with pytest.raises(DerivativeRenderError, match="duration bound"):
        CpuAnnotatedVideoRenderer(
            ffmpeg_bin=_require("ffmpeg"),
            limits=AnnotatedDerivativeLimits(max_duration_seconds=0.1),
        ).render(job, tmp_path / "duration.mp4")
    with pytest.raises(DerivativeRenderError, match="memory bound"):
        CpuAnnotatedVideoRenderer(
            ffmpeg_bin=_require("ffmpeg"),
            limits=AnnotatedDerivativeLimits(max_frame_bytes=100),
        ).render(job, tmp_path / "memory.mp4")
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(DerivativeCancelled):
        CpuAnnotatedVideoRenderer(ffmpeg_bin=_require("ffmpeg")).render(
            job, tmp_path / "cancelled.mp4", cancelled=cancelled
        )
    assert not tuple(tmp_path.glob("*.rendering.mp4"))


def test_real_ffmpeg_bad_source_has_typed_failure(tmp_path: Path) -> None:
    source = tmp_path / "bad.mp4"
    source.write_bytes(b"not-video")
    scene = _scene(320, 180)
    job = AnnotatedDerivativeJob(
        "incident-a",
        "clip-a",
        source,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "d" * 64,
        "b" * 64,
        (scene,),
        source.stat().st_size,
    )

    with pytest.raises(DerivativeRenderError, match="unavailable"):
        CpuAnnotatedVideoRenderer(ffmpeg_bin=_require("ffmpeg")).render(
            job, tmp_path / "output.mp4"
        )
