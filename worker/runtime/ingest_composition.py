from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from worker.adapters.decode.cpu_av.adapter import CpuAvAdapter
from worker.adapters.decode.cpu_av.models import CpuAvConfig
from worker.adapters.decode.nvdec_cuvid.adapter import NvdecCuvidAdapter
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig
from worker.interfaces.decode import DecodeAdapter
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

DecodeConfig: TypeAlias = CpuAvConfig | NvdecCuvidConfig


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


def _resolve_decode_backend(decode: DecodePolicy, override: str | None) -> str:
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
    decode: DecodePolicy, override: str | None = None
) -> DecodeAdapter[DecodeConfig]:
    """Build the real decode adapter for a boot-resolved decode token.

    Fail-fast per ADR-0002: an unrecognized token raises immediately rather
    than silently falling back to a default backend.
    """
    resolved = _resolve_decode_backend(decode, override)
    if resolved in ("opencv", "cpu"):
        return CpuAvAdapter()
    if resolved == "nvdec":
        return NvdecCuvidAdapter()
    raise RuntimeError(f"unsupported decode policy: {resolved!r}")


def _decode_config_factory(
    decode: DecodePolicy, camera: CameraRuntimeConfig, runtime: WorkerRuntimeConfig
) -> Callable[[str, ResolvedSource], DecodeConfig]:
    resolved = _resolve_decode_backend(decode, camera.decode_backend)

    def make(camera_id: str, resolved_source: ResolvedSource) -> DecodeConfig:
        del resolved_source  # the registry only gates which cameras may ingest
        if resolved in ("opencv", "cpu"):
            return CpuAvConfig(
                camera_id=camera_id,
                url=camera.inference_rtsp_url,
                open_timeout_ms=runtime.open_timeout_ms,
                read_timeout_ms=runtime.read_timeout_ms,
            )
        if resolved == "nvdec":
            return NvdecCuvidConfig(
                camera_id=camera_id,
                url=camera.inference_rtsp_url,
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
) -> CameraIngestLoop[DecodeConfig]:
    """Compose the real per-camera ingest loop for the boot-resolved decode profile.

    ``camera.decode_backend`` (when not ``None``/``"auto"``) overrides the
    profile-global ``decode`` token for this camera only; see
    ``_resolve_decode_backend``. ``runtime`` supplies the effective
    ``max_failures``/``open_timeout_ms``/``read_timeout_ms`` -- callers that
    don't have a ``WorkerConfig.runtime`` in scope (e.g. tests) get the
    adapter/policy dataclass defaults via a fresh ``WorkerRuntimeConfig``.
    """
    effective_runtime = runtime if runtime is not None else WorkerRuntimeConfig()
    ports = CameraIngestPorts(
        registry=registry,
        decoder=decoder_for(decode, camera.decode_backend),
        bus=bus,
        reporter=reporter,
    )
    spec = CameraIngestSpec(
        camera_id=camera.camera_id,
        source_id=camera.camera_id,
        make_decode_config=_decode_config_factory(decode, camera, effective_runtime),
        policy=CapturePolicy(target_fps=camera.fps, max_failures=effective_runtime.max_failures),
    )
    return CameraIngestLoop(spec, ports)


__all__ = [
    "DecodeConfig",
    "build_camera_source_registry",
    "compose_camera_ingest_loop",
    "decoder_for",
]
