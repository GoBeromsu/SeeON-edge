from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Final, TypeAlias, cast

from shared.rtsp_url_policy import assert_rtsp_endpoint_allowed
from worker.adapters.decode.cpu_av.adapter import CpuAvAdapter
from worker.adapters.decode.cpu_av.models import CpuAvConfig
from worker.adapters.decode.nvdec_cuvid.adapter import NvdecCuvidAdapter
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig
from worker.adapters.decode.pyav_preserving import PyAvPreservingAdapter
from worker.adapters.decode.vaapi.adapter import VaapiAdapter
from worker.adapters.decode.vaapi.models import VaapiConfig
from worker.interfaces.decode import DecodeAdapter
from worker.interfaces.source_packet import SourcePacketSink
from worker.pipeline.bus.frame_bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import (
    CameraIngestLoop,
    CameraIngestPorts,
    CameraIngestSpec,
    CapturePolicy,
    IngestReporter,
)
from worker.pipeline.ingest.registry import ResolvedSource, SourceRecord, SourceRegistry
from worker.runtime.config import CameraRuntimeConfig, WorkerRuntimeConfig
from worker.runtime.profile.registry import DecodePolicy

LOGGER: Final = logging.getLogger(__name__)

DecodeConfig: TypeAlias = CpuAvConfig | NvdecCuvidConfig | VaapiConfig


def build_camera_source_registry(cameras: tuple[CameraRuntimeConfig, ...]) -> SourceRegistry:
    """Allowlist exactly this worker's configured cameras as trusted live sources.

    ``SourceRegistry.resolve`` rejects raw ``rtsp://`` descriptors outright; a
    ``trusted_live`` record keyed by ``camera_id`` is the sanctioned way through
    that gate. The record's ``path`` is a placeholder -- the real, already
    Pydantic-validated RTSP URL comes from ``CameraRuntimeConfig`` directly (see
    ``compose_camera_ingest_loop``), never from the registry.
    """
    records = {
        camera.camera_id: SourceRecord(
            source_id=camera.camera_id,
            path=Path(camera.camera_id),
            duration_sec=0.0,
            mime_type="",
            kind="live",
            trusted_live=True,
        )
        for camera in cameras
    }
    return SourceRegistry(records=records)


def resolve_decode_backend(decode: DecodePolicy, override: str | None) -> str:
    """Resolve the effective decode backend for one camera.

    ``override`` is a camera's ``CameraRuntimeConfig.decode_backend``.
    ``None``/``"auto"`` defer entirely to the boot-resolved profile token; any
    other recognized value (``"opencv"``, ``"cpu"``, ``"nvdec"``) wins over
    the profile.

    Fail-fast per ADR-0002: requesting ``"nvdec"`` when the boot profile did
    not itself resolve to NVDEC (no verified NVDEC device for this host)
    raises immediately rather than silently falling back to CPU decode --
    CPU decode always works regardless of profile, so the reverse direction
    (an ``"opencv"``/``"cpu"`` override on an ``"nvdec"`` profile) is not a
    conflict.
    """
    resolved = decode if override in (None, "auto") else override
    if resolved == "nvdec" and decode != "nvdec":
        raise RuntimeError(
            "camera decode_backend='nvdec' requires the nvdec boot profile; "
            f"resolved boot profile decode is {decode!r}"
        )
    return resolved


def decoder_for(
    decode: DecodePolicy,
    override: str | None = None,
    *,
    packet_sink: SourcePacketSink | None = None,
) -> DecodeAdapter[DecodeConfig]:
    """Build the real decode adapter for a boot-resolved decode token.

    Fail-fast per ADR-0002: an unrecognized token raises immediately rather
    than silently falling back to a default backend.
    """
    resolved = resolve_decode_backend(decode, override)
    if resolved not in ("opencv", "cpu", "nvdec", "vaapi"):
        raise RuntimeError(f"unsupported decode policy: {resolved!r}")
    if packet_sink is not None:
        return cast(
            "DecodeAdapter[DecodeConfig]",
            PyAvPreservingAdapter(packet_sink, decode_backend=resolved),
        )
    if resolved in ("opencv", "cpu"):
        return cast("DecodeAdapter[DecodeConfig]", CpuAvAdapter())
    if resolved == "nvdec":
        return cast("DecodeAdapter[DecodeConfig]", NvdecCuvidAdapter())
    if resolved == "vaapi":
        return cast("DecodeAdapter[DecodeConfig]", VaapiAdapter())
    raise AssertionError(f"validated decode policy was not selected: {resolved!r}")


def _decode_config_factory(
    decode: DecodePolicy, camera: CameraRuntimeConfig, runtime: WorkerRuntimeConfig
) -> Callable[[str, ResolvedSource], DecodeConfig]:
    resolved = resolve_decode_backend(decode, camera.decode_backend)

    def make(camera_id: str, resolved_source: ResolvedSource) -> DecodeConfig:
        del resolved_source  # the registry only gates which cameras may ingest
        # Resolve every A/AAAA answer and pin an IP literal so the decoder
        # cannot DNS-rebind past policy between check and connect.
        try:
            endpoint = assert_rtsp_endpoint_allowed(camera.inference_rtsp_url)
        except ValueError as exc:
            raise RuntimeError(f"RTSP destination rejected for camera {camera_id}: {exc}") from exc
        pinned_url = endpoint.pinned_url
        if resolved in ("opencv", "cpu"):
            return CpuAvConfig(
                camera_id=camera_id,
                url=pinned_url,
                open_timeout_ms=runtime.open_timeout_ms,
                read_timeout_ms=runtime.read_timeout_ms,
            )
        if resolved == "nvdec":
            return NvdecCuvidConfig(
                camera_id=camera_id,
                url=pinned_url,
                open_timeout_ms=runtime.open_timeout_ms,
                read_timeout_ms=runtime.read_timeout_ms,
            )
        if resolved == "vaapi":
            return VaapiConfig(
                camera_id=camera_id,
                url=pinned_url,
                open_timeout_ms=runtime.open_timeout_ms,
                read_timeout_ms=runtime.read_timeout_ms,
            )
        raise RuntimeError(f"unsupported decode policy: {resolved!r}")

    return make


def compose_camera_ingest_loop(
    camera: CameraRuntimeConfig,
    bus: BoundedFrameBus,
    reporter: IngestReporter,
    *,
    decode: DecodePolicy,
    registry: SourceRegistry,
    runtime: WorkerRuntimeConfig | None = None,
    packet_sink: SourcePacketSink | None = None,
) -> CameraIngestLoop[DecodeConfig]:
    """Compose the real per-camera ingest loop for the boot-resolved decode profile.

    ``camera.decode_backend`` (when not ``None``/``"auto"``) overrides the
    profile-global ``decode`` token for this camera only; see
    ``resolve_decode_backend``. ``runtime`` supplies the effective
    ``max_failures``/``open_timeout_ms``/``read_timeout_ms`` -- callers that
    don't have a ``WorkerConfig.runtime`` in scope (e.g. tests) get the
    adapter/policy dataclass defaults via a fresh ``WorkerRuntimeConfig``.
    """
    effective_runtime = runtime if runtime is not None else WorkerRuntimeConfig()
    resolved_backend = resolve_decode_backend(decode, camera.decode_backend)
    decoder = decoder_for(
        decode,
        camera.decode_backend,
        packet_sink=packet_sink,
    )
    ports = CameraIngestPorts(
        registry=registry,
        decoder=decoder,
        bus=bus,
        reporter=reporter,
    )
    spec = CameraIngestSpec(
        camera_id=camera.camera_id,
        source_id=camera.camera_id,
        make_decode_config=_decode_config_factory(decode, camera, effective_runtime),
        policy=CapturePolicy(target_fps=camera.fps, max_failures=effective_runtime.max_failures),
    )
    # The profile token, per-camera resolution, and concrete adapter must be
    # visible in the rendered boot log. `extra` is retained for structured
    # consumers, but the entrypoint formatter only renders `%(message)s`.
    LOGGER.info(
        "camera ingest decode selected: camera_id=%s requested_profile_decode=%s "
        "resolved_backend=%s actual_adapter_class=%s",
        camera.camera_id,
        decode,
        resolved_backend,
        type(decoder).__name__,
        extra={
            "camera_id": camera.camera_id,
            "requested_profile_decode": decode,
            "resolved_backend": resolved_backend,
            "actual_adapter_class": type(decoder).__name__,
        },
    )
    # `camera.fps` paces both decode and the live-view tap (RTSPSource yields
    # at this rate; CameraPipelinePump publishes every yielded packet to the
    # live view unconditionally). It silently falls back to a hardcoded 5.0
    # whenever nothing sets it (no per-camera value from the backend registry,
    # no ML_DEFAULT_CAMERA_FPS), which previously had zero log trace -- a
    # camera pacing at 5fps against a 15fps source looked externally
    # indistinguishable from a frame_stride bug (5 == 15/3), which is exactly
    # the false lead this line exists to close off. `frame_stride` never
    # reaches this policy -- it only gates the extractor Scheduler.
    LOGGER.info(
        "camera ingest paced: camera_id=%s target_fps=%s frame_stride=%s "
        "(frame_stride does not affect this rate)",
        camera.camera_id,
        camera.fps,
        camera.frame_stride,
        extra={
            "camera_id": camera.camera_id,
            "target_fps": camera.fps,
            "frame_stride": camera.frame_stride,
        },
    )
    return CameraIngestLoop(spec, ports)


__all__ = [
    "DecodeConfig",
    "build_camera_source_registry",
    "compose_camera_ingest_loop",
    "decoder_for",
    "resolve_decode_backend",
]
