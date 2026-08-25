"""CUDA overlay-scene renderer: the same canonical scene, on retained device surfaces.

Implements ``worker.interfaces.render.OverlaySceneRenderer`` -- the identical
port ``worker.pipeline.output.overlay.OverlayRenderer`` (CPU/OpenCV)
implements -- so both renderers draw the exact same
``worker.types.overlay_scene.OverlayScene`` produced once by
``worker.pipeline.output.overlay_scene.OverlaySceneBuilder``. This module
never rebuilds scene semantics (persons/beds/decisions/labels/z-order): it
only draws them onto a different backing store.

Like ``worker.adapters.decode.nvdec_device.pool``, this never imports
``torch``/``cupy`` directly. The actual pixel-drawing primitive is injected as
a ``DeviceSceneDrawer`` callable, so the same lifecycle/accounting logic here
is exercised both by a real CUDA-kernel-backed drawer (constructed only on a
capability-probed NVIDIA host, never built in this repo yet -- see the class
docstring) and by the deterministic
``worker.adapters.encode.nvenc_device.fake.FakeDeviceSceneDrawer`` double used
by every test on this repo's non-NVIDIA CI/dev hosts.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, final

from worker.types import FrameLease, MemoryKind
from worker.types.overlay_scene import OverlayScene

RENDER_VERSION: str = "overlay-cuda.v1"
BACKEND_ID: str = "cuda-device"


class DeviceSceneDrawer(Protocol):
    """Draw one canonical scene onto a device-resident surface in place.

    Receives the same opaque device handle a
    ``worker.adapters.decode.nvdec_device.pool.DeviceResidentFramePool`` lease
    carries -- never a host array -- and returns a (possibly new) device
    handle plus its descriptor for the annotated result. A real
    implementation runs CUDA drawing kernels against the surface directly;
    the deterministic fake mutates a host-backed ``numpy`` array tagged with
    a non-host ``MemoryKind`` (see ``worker.adapters.decode.nvdec_device.fake``
    for why that is still a faithful lifecycle/ownership double).
    """

    def __call__(self, handle: object, scene: OverlayScene) -> object: ...


class DeviceSceneRenderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DeviceRenderMetrics:
    frames_rendered: int
    host_materializations: int
    render_time_ms_total: float


@final
class CudaOverlaySceneRenderer:
    """Render the canonical overlay scene onto a retained CUDA-resident surface.

    ``backend_id``/``render_version``/``input_memory_kind`` satisfy the
    ``OverlaySceneRenderer`` protocol exactly as
    ``worker.pipeline.output.overlay.OverlayRenderer`` does for CPU, so a
    derivative artifact's ``render_backend``/``render_device``/
    ``input_memory_kind`` fields
    (a published media artifact)
    truthfully distinguish which renderer actually produced a given
    annotated derivative.
    """

    backend_id: str = BACKEND_ID
    render_version: str = RENDER_VERSION
    input_memory_kind: str = MemoryKind.CUDA_DEVICE.value

    def __init__(
        self,
        *,
        draw: DeviceSceneDrawer,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._draw = draw
        self._clock = clock
        self._frames_rendered = 0
        self._render_time_ms_total = 0.0

    @property
    def metrics(self) -> DeviceRenderMetrics:
        return DeviceRenderMetrics(
            frames_rendered=self._frames_rendered,
            host_materializations=0,
            render_time_ms_total=self._render_time_ms_total,
        )

    def render_scene(self, lease: FrameLease, scene: OverlayScene) -> FrameLease:
        """Draw ``scene`` onto ``lease``'s device surface; never touches host memory.

        Unlike ``OverlaySceneRenderer.render_scene``'s declared
        ``(packet, scene) -> object`` signature (which the CPU renderer
        satisfies with a host ``FramePacket``/``NDArray``), this device-input
        seam takes and returns a ``FrameLease`` directly: the whole point of
        this renderer is that no host-resident ``FramePacket`` ever exists on
        this path. Callers that need strict ``OverlaySceneRenderer`` duck
        typing use ``render_scene`` positionally the same way; this widened
        signature is why the CUDA renderer is wired explicitly by the
        device-resident derivative pipeline rather than through the same
        generic dispatch as the CPU renderer.
        """
        if lease.descriptor.memory_kind is MemoryKind.HOST:
            raise DeviceSceneRenderError(
                "CUDA overlay renderer requires a device-resident lease, got host memory"
            )
        if (lease.descriptor.width, lease.descriptor.height) != scene.source_dimensions:
            raise DeviceSceneRenderError("overlay scene and device surface dimensions differ")
        started = self._clock()
        try:
            annotated_handle = self._draw(lease.device_handle, scene)
        except Exception as error:  # noqa: BLE001 - drawer failures are reported, not swallowed
            raise DeviceSceneRenderError(
                f"CUDA scene draw failed: {type(error).__name__}"
            ) from error
        elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
        self._frames_rendered += 1
        self._render_time_ms_total += elapsed_ms
        annotated = FrameLease.from_device(annotated_handle, lease.descriptor)
        return annotated


__all__ = [
    "BACKEND_ID",
    "RENDER_VERSION",
    "CudaOverlaySceneRenderer",
    "DeviceRenderMetrics",
    "DeviceSceneDrawer",
    "DeviceSceneRenderError",
]
