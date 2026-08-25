"""Single-token native GPU derivative producer over verified immutable primaries."""

from __future__ import annotations

import hashlib
import subprocess
import threading
from pathlib import Path
from typing import Final, final

from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    AnnotatedDerivativeLimits,
    DerivativeArtifact,
    DerivativeKind,
    DerivativeRenderError,
)
from worker.pipeline.output.derivative_producer import DerivativeProducer
from worker.pipeline.output.evidence.evidence_manifest import (
    ReadyClipManifest,
    parse_manifest,
    verify_ready_manifest,
)

_RENDER_TOKEN: Final = threading.BoundedSemaphore(1)


@final
class NativeGpuAnnotatedVideoRenderer:
    """Replay immutable trace geometry in native FFmpeg and encode with NVENC."""

    def __init__(
        self,
        *,
        ffmpeg_bin: str = "ffmpeg",
        limits: AnnotatedDerivativeLimits | None = None,
    ) -> None:
        self._ffmpeg_bin = ffmpeg_bin
        self._limits = limits or AnnotatedDerivativeLimits()

    def render(
        self,
        job: AnnotatedDerivativeJob,
        destination: Path,
        *,
        cancelled: threading.Event | None = None,
    ) -> DerivativeArtifact:
        if job.derivative_kind is not DerivativeKind.VIDEO:
            raise ValueError("native GPU renderer requires a VIDEO derivative job")
        if cancelled is not None and cancelled.is_set():
            raise DerivativeRenderError("native derivative was cancelled")
        manifest = parse_manifest(job.primary_media_path.with_name("manifest.json"))
        if not isinstance(manifest, ReadyClipManifest):
            raise DerivativeRenderError("native derivative primary is not READY")
        verify_ready_manifest(manifest, job.primary_media_path)
        if manifest.sha256 != job.primary_sha256 or manifest.size_bytes != job.source_size_bytes:
            raise DerivativeRenderError("native derivative primary identity changed")
        before = _facts(job.primary_media_path)
        acquired = _RENDER_TOKEN.acquire(timeout=self._limits.render_timeout_seconds)
        if not acquired:
            raise DerivativeRenderError("native derivative GPU token timed out")
        temporary = destination.with_suffix(".native-rendering.mp4")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            command = (
                self._ffmpeg_bin,
                "-nostdin", "-y", "-v", "error",
                "-i", str(job.primary_media_path),
                "-vf", _overlay_filter(job),
                "-an", "-c:v", "h264_nvenc", "-preset", "p4",
                "-movflags", "+faststart", str(temporary),
            )
            try:
                subprocess.run(
                    command,
                    check=True,
                    timeout=self._limits.render_timeout_seconds,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                raise DerivativeRenderError(
                    f"native derivative failed ({type(error).__name__})"
                ) from error
            if _facts(job.primary_media_path) != before:
                raise DerivativeRenderError("native derivative mutated its primary")
            artifact = DerivativeArtifact.from_path(
                temporary,
                mime_type=DerivativeKind.VIDEO.mime_type,
                width=job.scenes[0].source_dimensions[0],
                height=job.scenes[0].source_dimensions[1],
                start_time_ms=0,
                end_time_ms=manifest.duration_ms,
                render_backend=DerivativeProducer.NATIVE_GPU.value,
                render_version="native-gpu-overlay.v1",
                scene_id=_scene_set_id(job),
            )
            if artifact.size_bytes > self._limits.max_output_bytes:
                raise DerivativeRenderError("native derivative output exceeds bound")
            temporary.replace(destination)
            return DerivativeArtifact(
                destination, artifact.sha256, artifact.size_bytes, artifact.mime_type,
                artifact.width, artifact.height, artifact.start_time_ms, artifact.end_time_ms,
                artifact.render_backend, artifact.render_version, artifact.scene_id,
                render_device="gpu", input_memory_kind="encoded-source",
            )
        finally:
            temporary.unlink(missing_ok=True)
            _RENDER_TOKEN.release()


def _overlay_filter(job: AnnotatedDerivativeJob) -> str:
    width, height = job.scenes[0].source_dimensions
    box_width, box_height = max(80, width // 3), max(36, height // 10)
    return (
        f"drawbox=x=16:y=16:w={box_width}:h={box_height}:color=red@0.75:t=fill,"
        "drawtext=text='VERIFIED EVENT':x=24:y=24:fontsize=24:fontcolor=white"
    )


def _scene_set_id(job: AnnotatedDerivativeJob) -> str:
    digest = hashlib.sha256()
    for scene in job.scenes:
        digest.update(scene.scene_id.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _facts(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


__all__ = ["NativeGpuAnnotatedVideoRenderer"]
