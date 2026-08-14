from __future__ import annotations

import hashlib
import math
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Final, cast

import cv2
import numpy as np

from contracts.frame import Frame
from worker.interfaces.render import OverlaySceneRenderer
from worker.pipeline.output.overlay import OverlayRenderer
from worker.types import FramePacket
from worker.types.overlay_scene import OverlayScene

DERIVATIVE_RENDER_VERSION: Final = "overlay-cpu.v1"


class DerivativeKind(StrEnum):
    STILL = "STILL"
    VIDEO = "VIDEO"

    @property
    def extension(self) -> str:
        return ".jpg" if self is DerivativeKind.STILL else ".mp4"

    @property
    def mime_type(self) -> str:
        return "image/jpeg" if self is DerivativeKind.STILL else "video/mp4"


class DerivativeCancelled(RuntimeError):
    def __init__(self, message: str, job: AnnotatedDerivativeJob | None = None) -> None:
        super().__init__(message)
        self.job = job


class DerivativeRenderError(RuntimeError):
    pass


class DerivativeUnavailableReason:
    SOURCE_TRACE_MISSING: Final = "SOURCE_TRACE_MISSING"
    SOURCE_MEDIA_MISSING: Final = "SOURCE_MEDIA_MISSING"
    SOURCE_MEDIA_CORRUPT: Final = "SOURCE_MEDIA_CORRUPT"
    RENDER_FAILED: Final = "RENDER_FAILED"
    CANCELLED: Final = "CANCELLED"
    RESOURCE_LIMIT: Final = "RESOURCE_LIMIT"


@dataclass(frozen=True, slots=True)
class AnnotatedDerivativeLimits:
    max_pending_jobs: int = 8
    max_pending_source_bytes: int = 512 * 1024 * 1024
    max_output_bytes: int = 256 * 1024 * 1024
    max_frame_bytes: int = 64 * 1024 * 1024
    max_scene_count: int = 36_000
    max_duration_seconds: float = 120.0
    max_disk_bytes: int = 2 * 1024 * 1024 * 1024
    render_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        numeric = (
            self.max_pending_jobs,
            self.max_pending_source_bytes,
            self.max_output_bytes,
            self.max_frame_bytes,
            self.max_scene_count,
            self.max_disk_bytes,
        )
        if any(value <= 0 for value in numeric):
            raise ValueError("derivative resource limits must be positive")
        if (
            not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0
            or not math.isfinite(self.render_timeout_seconds)
            or self.render_timeout_seconds <= 0
        ):
            raise ValueError("derivative time limits must be finite and positive")


@dataclass(frozen=True, slots=True)
class AnnotatedDerivativeJob:
    incident_id: str
    primary_clip_id: str
    primary_media_path: Path
    primary_sha256: str
    decision_trace_id: str
    runtime_manifest_sha256: str
    scenes: tuple[OverlayScene, ...]
    source_size_bytes: int
    media_origin_pts_sec: float = 0.0
    derivative_kind: DerivativeKind = DerivativeKind.VIDEO

    def __post_init__(self) -> None:
        if not self.scenes:
            raise ValueError("annotated derivative requires recorded overlay scenes")
        if self.source_size_bytes <= 0:
            raise ValueError("annotated derivative source size must be positive")
        if not all(_safe_opaque_id(value) for value in (self.incident_id, self.primary_clip_id)):
            raise ValueError("annotated derivative evidence identity is invalid")
        if any(
            not _canonical_sha256(value)
            for value in (
                self.primary_sha256,
                self.decision_trace_id,
                self.runtime_manifest_sha256,
            )
        ):
            raise ValueError("annotated derivative source identities must be SHA-256 values")

    @property
    def identity(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.primary_sha256,
            self.decision_trace_id,
            self.runtime_manifest_sha256,
            DERIVATIVE_RENDER_VERSION,
            self.derivative_kind.value,
            *(scene.scene_id for scene in self.scenes),
        ):
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DerivativeArtifact:
    path: Path
    sha256: str
    size_bytes: int
    mime_type: str
    width: int
    height: int
    start_time_ms: int
    end_time_ms: int
    render_backend: str
    render_version: str
    scene_id: str
    render_device: str = "cpu"
    input_memory_kind: str = "host"

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        mime_type: str,
        width: int,
        height: int,
        start_time_ms: int,
        end_time_ms: int,
        render_backend: str,
        render_version: str,
        scene_id: str,
    ) -> DerivativeArtifact:
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
        if size_bytes == 0:
            raise DerivativeRenderError("derivative renderer produced no media")
        return cls(
            path,
            digest.hexdigest(),
            size_bytes,
            mime_type,
            width,
            height,
            start_time_ms,
            end_time_ms,
            render_backend,
            render_version,
            scene_id,
        )


class BoundedDerivativeQueue:
    def __init__(self, limits: AnnotatedDerivativeLimits | None = None) -> None:
        self.limits = limits or AnnotatedDerivativeLimits()
        self._jobs: deque[AnnotatedDerivativeJob] = deque()
        self._source_bytes = 0
        self._cancelled: set[str] = set()
        self._cancelled_requests: set[str] = set()
        self._closed = False
        self._condition = threading.Condition()

    def submit(self, job: AnnotatedDerivativeJob) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("derivative queue is closed")
            if (
                len(self._jobs) >= self.limits.max_pending_jobs
                or len(job.scenes) > self.limits.max_scene_count
                or self._source_bytes + job.source_size_bytes > self.limits.max_pending_source_bytes
            ):
                raise OverflowError("bounded derivative queue capacity exceeded")
            if any(existing.identity == job.identity for existing in self._jobs):
                return
            self._jobs.append(job)
            self._source_bytes += job.source_size_bytes
            self._condition.notify()

    def take(self) -> AnnotatedDerivativeJob:
        with self._condition:
            if not self._jobs:
                raise LookupError("derivative queue is empty")
            return self._take_locked()

    def take_wait(self) -> AnnotatedDerivativeJob | None:
        """Wait for one job; return ``None`` only after the queue is closed."""
        with self._condition:
            while not self._jobs and not self._closed:
                self._condition.wait()
            if not self._jobs:
                return None
            return self._take_locked()

    def _take_locked(self) -> AnnotatedDerivativeJob:
        job = self._jobs.popleft()
        self._source_bytes -= job.source_size_bytes
        if job.incident_id in self._cancelled or job.identity in self._cancelled_requests:
            self._cancelled.discard(job.incident_id)
            self._cancelled_requests.discard(job.identity)
            raise DerivativeCancelled("derivative job was cancelled", job)
        return job

    def cancel(self, incident_id: str) -> None:
        with self._condition:
            self._cancelled.add(incident_id)
            self._condition.notify_all()

    def cancel_request(self, request_id: str) -> None:
        with self._condition:
            self._cancelled_requests.add(request_id)
            self._condition.notify_all()

    def close(self) -> tuple[AnnotatedDerivativeJob, ...]:
        with self._condition:
            self._closed = True
            pending = tuple(self._jobs)
            self._jobs.clear()
            self._source_bytes = 0
            self._condition.notify_all()
            return pending


class CpuAnnotatedStillRenderer:
    """Encode one canonical scene as a bounded immutable JPEG derivative."""

    def __init__(
        self,
        *,
        limits: AnnotatedDerivativeLimits | None = None,
        scene_renderer: OverlaySceneRenderer | None = None,
    ) -> None:
        self.limits = limits or AnnotatedDerivativeLimits()
        self.scene_renderer = OverlayRenderer() if scene_renderer is None else scene_renderer

    def render(
        self,
        job: AnnotatedDerivativeJob,
        destination: Path,
        *,
        cancelled: threading.Event | None = None,
    ) -> DerivativeArtifact:
        if job.derivative_kind is not DerivativeKind.STILL:
            raise ValueError("still renderer requires a STILL derivative job")
        capture = cv2.VideoCapture()
        temporary = destination.with_suffix(".rendering.jpg")
        try:
            if cancelled is not None and cancelled.is_set():
                raise DerivativeCancelled("derivative job was cancelled", job)
            timeout_ms = min(round(self.limits.render_timeout_seconds * 1000), 2_147_483_647)
            _ = capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            _ = capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
            if not capture.open(str(job.primary_media_path)):
                raise DerivativeRenderError("primary derivative source is unavailable")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if min(width, height) <= 0 or width * height * 3 > self.limits.max_frame_bytes:
                raise DerivativeRenderError("derivative frame exceeds memory bound")
            scene = job.scenes[0]
            if scene.source_dimensions != (width, height):
                raise DerivativeRenderError("scene and source dimensions differ")
            scene_pts = scene.frame.pts.value
            media_time_sec = (
                0.0 if scene_pts is None else max(0.0, float(scene_pts) - job.media_origin_pts_sec)
            )
            _ = capture.set(cv2.CAP_PROP_POS_MSEC, media_time_sec * 1000.0)
            ok, image = capture.read()
            if not ok:
                raise DerivativeRenderError("primary derivative frame is unavailable")
            if cancelled is not None and cancelled.is_set():
                raise DerivativeCancelled("derivative job was cancelled", job)
            packet = FramePacket(
                scene.frame.camera_id,
                Frame(scene.frame.seq, media_time_sec, np.asarray(image, dtype=np.uint8)),
                media_time_sec,
                scene.frame.seq,
                width,
                height,
                0.0,
                scene.frame.worker_boot_id,
                scene.frame.stream_epoch,
            )
            rendered = np.asarray(self.scene_renderer.render_scene(packet, scene), dtype=np.uint8)
            encoded, payload = cv2.imencode(
                ".jpg",
                rendered,
                (cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 0),
            )
            if not encoded or not 0 < payload.size <= self.limits.max_output_bytes:
                raise DerivativeRenderError("still derivative encoding failed")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(payload.tobytes())
            backend = cast(str, self.scene_renderer.backend_id)
            version = cast(str, self.scene_renderer.render_version)
            artifact = DerivativeArtifact.from_path(
                temporary,
                mime_type=DerivativeKind.STILL.mime_type,
                width=width,
                height=height,
                start_time_ms=round(media_time_sec * 1000),
                end_time_ms=round(media_time_sec * 1000),
                render_backend=backend,
                render_version=version,
                scene_id=scene.scene_id,
            )
            temporary.replace(destination)
            return DerivativeArtifact(
                destination,
                artifact.sha256,
                artifact.size_bytes,
                artifact.mime_type,
                artifact.width,
                artifact.height,
                artifact.start_time_ms,
                artifact.end_time_ms,
                artifact.render_backend,
                artifact.render_version,
                artifact.scene_id,
            )
        except cv2.error as error:
            raise DerivativeRenderError("still derivative encoding failed") from error
        finally:
            capture.release()
            temporary.unlink(missing_ok=True)


class CpuAnnotatedVideoRenderer:
    """Bounded FFmpeg CPU reference encoder fed by canonical scene frames."""

    def __init__(
        self,
        *,
        ffmpeg_bin: str = "ffmpeg",
        limits: AnnotatedDerivativeLimits | None = None,
        scene_renderer: OverlaySceneRenderer | None = None,
    ) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.limits = limits or AnnotatedDerivativeLimits()
        self.still_renderer = OverlayRenderer() if scene_renderer is None else scene_renderer

    def render(
        self,
        job: AnnotatedDerivativeJob,
        destination: Path,
        *,
        cancelled: threading.Event | None = None,
    ) -> DerivativeArtifact:
        if job.derivative_kind is not DerivativeKind.VIDEO:
            raise ValueError("video renderer requires a VIDEO derivative job")
        capture = cv2.VideoCapture()
        process: subprocess.Popen[bytes] | None = None
        temporary = destination.with_suffix(".rendering.mp4")
        started = time.monotonic()
        try:
            if (
                job.source_size_bytes > self.limits.max_pending_source_bytes
                or len(job.scenes) > self.limits.max_scene_count
            ):
                raise DerivativeRenderError("derivative source exceeds resource bounds")
            timeout_ms = min(round(self.limits.render_timeout_seconds * 1000), 2_147_483_647)
            _ = capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            _ = capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
            if not capture.open(str(job.primary_media_path)):
                raise DerivativeRenderError("primary derivative source is unavailable")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if min(width, height) <= 0 or not math.isfinite(fps) or fps <= 0:
                raise DerivativeRenderError("primary derivative source facts are invalid")
            if width * height * 3 > self.limits.max_frame_bytes:
                raise DerivativeRenderError("derivative frame exceeds memory bound")
            duration = frame_count / fps
            if duration > self.limits.max_duration_seconds:
                raise DerivativeRenderError("derivative source exceeds duration bound")
            if any(scene.source_dimensions != (width, height) for scene in job.scenes):
                raise DerivativeRenderError("scene and source dimensions differ")
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame_rate = _frame_rate(fps)
            process = subprocess.Popen(
                _ffmpeg_command(
                    self.ffmpeg_bin, width, height, frame_rate, temporary, job.primary_media_path
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
            rendered = 0
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise DerivativeCancelled("derivative job was cancelled", job)
                if time.monotonic() - started > self.limits.render_timeout_seconds:
                    raise DerivativeRenderError("derivative render timed out")
                ok, image = capture.read()
                if not ok:
                    break
                media_time = Fraction(rendered, 1) / frame_rate
                scene = _scene_at(job.scenes, Fraction(str(job.media_origin_pts_sec)) + media_time)
                if scene is not None:
                    packet = FramePacket(
                        job.scenes[0].frame.camera_id,
                        Frame(rendered, float(media_time), np.asarray(image, dtype=np.uint8)),
                        float(media_time),
                        rendered,
                        width,
                        height,
                        0.0,
                        scene.frame.worker_boot_id,
                        scene.frame.stream_epoch,
                    )
                    image = self.still_renderer.render_scene(packet, scene)
                assert process.stdin is not None
                process.stdin.write(memoryview(np.ascontiguousarray(image)))
                rendered += 1
                if temporary.exists() and temporary.stat().st_size > self.limits.max_output_bytes:
                    raise DerivativeRenderError("derivative output exceeds byte bound")
            assert process.stdin is not None
            process.stdin.close()
            remaining = self.limits.render_timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise DerivativeRenderError("derivative render timed out")
            returncode = process.wait(timeout=remaining)
            if returncode != 0 or rendered == 0:
                raise DerivativeRenderError("FFmpeg derivative encoding failed")
            artifact = DerivativeArtifact.from_path(
                temporary,
                mime_type="video/mp4",
                width=width,
                height=height,
                start_time_ms=0,
                end_time_ms=round(float(Fraction(rendered, 1) / frame_rate) * 1000),
                render_backend=self.still_renderer.backend_id,
                render_version=self.still_renderer.render_version,
                scene_id=_scene_set_id(job.scenes),
            )
            if artifact.size_bytes > self.limits.max_output_bytes:
                raise DerivativeRenderError("derivative output exceeds byte bound")
            temporary.replace(destination)
            return DerivativeArtifact(
                destination,
                artifact.sha256,
                artifact.size_bytes,
                artifact.mime_type,
                width,
                height,
                artifact.start_time_ms,
                artifact.end_time_ms,
                artifact.render_backend,
                artifact.render_version,
                artifact.scene_id,
            )
        except (OSError, BrokenPipeError, subprocess.TimeoutExpired) as error:
            raise DerivativeRenderError(
                f"derivative render failed ({type(error).__name__})"
            ) from error
        finally:
            capture.release()
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            temporary.unlink(missing_ok=True)


def _safe_opaque_id(value: str) -> bool:
    return (
        0 < len(value) <= 256
        and value not in {".", ".."}
        and not any(character in value for character in ("/", "\\", "\x00", "\r", "\n"))
    )


def _canonical_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _scene_at(
    scenes: tuple[OverlayScene, ...], source_pts: Fraction | float
) -> OverlayScene | None:
    """Choose the current scene without lossy millisecond rounding.

    Trace scalars are canonically recorded to six decimals, so a half-microsecond
    boundary admits the scene for the exact decoded frame while never crossing a
    whole frame at supported rates.
    """
    timestamp = source_pts if isinstance(source_pts, Fraction) else Fraction(str(source_pts))
    tolerance = Fraction(1, 2_000_000)
    eligible = [
        scene
        for scene in scenes
        if scene.frame.pts.value is not None
        and Fraction(str(scene.frame.pts.value)) <= timestamp + tolerance
    ]
    return max(
        eligible,
        key=lambda scene: Fraction(str(scene.frame.pts.value or 0)),
        default=None,
    )


def _frame_rate(fps: float) -> Fraction:
    """Recover standard rational media rates such as 30000/1001 from OpenCV."""
    return Fraction(fps).limit_denominator(100_000)


def _scene_set_id(scenes: tuple[OverlayScene, ...]) -> str:
    return hashlib.sha256("\0".join(scene.scene_id for scene in scenes).encode()).hexdigest()


def _ffmpeg_command(
    binary: str,
    width: int,
    height: int,
    frame_rate: Fraction,
    destination: Path,
    source: Path,
) -> tuple[str, ...]:
    return (
        binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{frame_rate.numerator}/{frame_rate.denominator}",
        "-i",
        "pipe:0",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:a",
        "copy",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        "-y",
        str(destination),
    )


__all__ = [
    "AnnotatedDerivativeJob",
    "AnnotatedDerivativeLimits",
    "BoundedDerivativeQueue",
    "CpuAnnotatedStillRenderer",
    "CpuAnnotatedVideoRenderer",
    "DerivativeArtifact",
    "DerivativeCancelled",
    "DerivativeKind",
    "DerivativeRenderError",
    "DerivativeUnavailableReason",
]
