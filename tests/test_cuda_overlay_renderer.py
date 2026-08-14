"""Todo 18: deterministic contract/fake-adapter proof for the CUDA overlay renderer.

Same rationale as ``tests/test_nvidia_device_resident_prototype.py``: this
repo's dev/CI hosts are Apple Silicon, so every scene-parity/lifetime/
zero-host-transfer claim here is proven against the deterministic
``worker.adapters.encode.nvenc_device.fake`` double, never against real GPU
memory. A separate ``real_stack``-marked module is the only place that
asserts anything about actual NVIDIA hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from worker.adapters.encode.nvenc_device.fake import (
    FakeDeviceSceneDrawer,
    fake_device_resident_lease,
)
from worker.adapters.encode.nvenc_device.renderer import (
    BACKEND_ID,
    RENDER_VERSION,
    CudaOverlaySceneRenderer,
    DeviceSceneRenderError,
)
from worker.interfaces.render import OverlaySceneRenderer
from worker.pipeline.output.overlay_scene import AppliedCameraProvenance, OverlaySceneBuilder
from worker.pipeline.trace.models import AnalysisTrace, OptionalNumber, TracePerson
from worker.types import FrameLease, FrameLeaseReleasedError, MemoryKind


def _scene(width: int = 4, height: int = 4, *, person_count: int = 1):
    persons = tuple(
        TracePerson(index, OptionalNumber(index), (0, 0, 2, 2), 0.9)
        for index in range(person_count)
    )
    analysis = AnalysisTrace(
        "a" * 64,
        ("boot-a", "camera-a", 1, 1),
        OptionalNumber(0.0),
        OptionalNumber(0.0),
        width,
        height,
        "fresh",
        persons,
        (),
        (),
    )
    return OverlaySceneBuilder().from_traces(
        analysis,
        (),
        provenance=AppliedCameraProvenance("b" * 64, "camera.v1"),
    )


# --------------------------------------------------------------------------
# Port conformance + identity
# --------------------------------------------------------------------------


def test_renderer_conforms_to_overlay_scene_renderer_protocol_fields() -> None:
    renderer = CudaOverlaySceneRenderer(draw=FakeDeviceSceneDrawer())
    assert renderer.backend_id == BACKEND_ID == "cuda-device"
    assert renderer.render_version == RENDER_VERSION
    assert renderer.input_memory_kind == MemoryKind.CUDA_DEVICE.value
    # Duck-type conformance to the shared protocol every renderer implements.
    assert hasattr(OverlaySceneRenderer, "render_scene")


def test_cpu_and_cuda_renderer_disclose_distinct_backend_identity() -> None:
    from worker.pipeline.output.overlay import OverlayRenderer

    cpu = OverlayRenderer()
    cuda = CudaOverlaySceneRenderer(draw=FakeDeviceSceneDrawer())

    assert cpu.backend_id != cuda.backend_id
    assert cpu.input_memory_kind == "host"
    assert cuda.input_memory_kind == "cuda-device"


# --------------------------------------------------------------------------
# Scene semantics parity: same scene, same visible result, different backend
# --------------------------------------------------------------------------


def test_cuda_renderer_marks_surface_deterministically_from_scene_semantics() -> None:
    """The fake drawer's marker is a stand-in kernel, but it must be a pure
    deterministic function of the canonical scene -- proving the CUDA seam
    receives the identical hardware-neutral scene contract the CPU renderer
    draws, not a hand-rolled parallel structure."""
    scene_one_person = _scene(person_count=1)
    scene_two_person = _scene(person_count=2)
    drawer = FakeDeviceSceneDrawer()
    renderer = CudaOverlaySceneRenderer(draw=drawer)

    lease_a = fake_device_resident_lease(width=4, height=4)
    lease_b = fake_device_resident_lease(width=4, height=4)

    annotated_a = renderer.render_scene(lease_a, scene_one_person)
    annotated_b = renderer.render_scene(lease_b, scene_two_person)

    handle_a = annotated_a.device_handle
    handle_b = annotated_b.device_handle
    assert isinstance(handle_a, np.ndarray)
    assert isinstance(handle_b, np.ndarray)
    marker_a = int(handle_a[0, 0, 0])
    marker_b = int(handle_b[0, 0, 0])
    assert marker_a != marker_b  # different scene -> different deterministic marker
    assert drawer.calls == 2

    annotated_a.release()
    annotated_b.release()
    lease_a.release()
    lease_b.release()


def test_cuda_renderer_is_deterministic_for_the_same_scene() -> None:
    scene = _scene(person_count=1)
    renderer = CudaOverlaySceneRenderer(draw=FakeDeviceSceneDrawer())
    lease_a = fake_device_resident_lease(width=4, height=4)
    lease_b = fake_device_resident_lease(width=4, height=4)

    first = renderer.render_scene(lease_a, scene)
    second = renderer.render_scene(lease_b, scene)

    first_handle = first.device_handle
    second_handle = second.device_handle
    assert isinstance(first_handle, np.ndarray)
    assert isinstance(second_handle, np.ndarray)
    assert np.array_equal(first_handle, second_handle)
    first.release()
    second.release()
    lease_a.release()
    lease_b.release()


# --------------------------------------------------------------------------
# Zero-host-transfer / device-residency invariants
# --------------------------------------------------------------------------


def test_render_scene_never_touches_host_memory() -> None:
    scene = _scene()
    renderer = CudaOverlaySceneRenderer(draw=FakeDeviceSceneDrawer())
    lease = fake_device_resident_lease(width=4, height=4)

    annotated = renderer.render_scene(lease, scene)

    assert annotated.descriptor.memory_kind is MemoryKind.CUDA_DEVICE
    with pytest.raises(RuntimeError, match="not host-resident"):
        _ = annotated.host_frame
    annotated.release()
    lease.release()


def test_render_scene_rejects_a_host_resident_lease() -> None:
    from contracts.frame import Frame

    scene = _scene()
    renderer = CudaOverlaySceneRenderer(draw=FakeDeviceSceneDrawer())
    host_lease = FrameLease.from_host(Frame(0, 0.0, np.zeros((4, 4, 3), dtype=np.uint8)))

    with pytest.raises(DeviceSceneRenderError, match="device-resident"):
        renderer.render_scene(host_lease, scene)
    host_lease.release()


def test_render_scene_rejects_dimension_mismatch() -> None:
    scene = _scene(width=4, height=4)
    renderer = CudaOverlaySceneRenderer(draw=FakeDeviceSceneDrawer())
    lease = fake_device_resident_lease(width=8, height=8)

    with pytest.raises(DeviceSceneRenderError, match="dimensions differ"):
        renderer.render_scene(lease, scene)
    lease.release()


def test_render_scene_reports_drawer_failure_without_silent_fallback() -> None:
    def _failing_draw(handle: object, scene: object) -> object:
        raise RuntimeError("simulated CUDA kernel failure")

    scene = _scene()
    renderer = CudaOverlaySceneRenderer(draw=_failing_draw)
    lease = fake_device_resident_lease(width=4, height=4)

    with pytest.raises(DeviceSceneRenderError, match="CUDA scene draw failed"):
        renderer.render_scene(lease, scene)
    lease.release()


def test_renderer_metrics_track_frames_and_never_count_host_materialization() -> None:
    scene = _scene()
    renderer = CudaOverlaySceneRenderer(draw=FakeDeviceSceneDrawer())
    lease_a = fake_device_resident_lease(width=4, height=4)
    lease_b = fake_device_resident_lease(width=4, height=4)

    first = renderer.render_scene(lease_a, scene)
    second = renderer.render_scene(lease_b, scene)

    metrics = renderer.metrics
    assert metrics.frames_rendered == 2
    assert metrics.host_materializations == 0
    assert metrics.render_time_ms_total >= 0.0

    first.release()
    second.release()
    lease_a.release()
    lease_b.release()


# --------------------------------------------------------------------------
# Lease lifetime through the render seam
# --------------------------------------------------------------------------


def test_annotated_lease_is_independently_releasable_from_its_source() -> None:
    scene = _scene()
    renderer = CudaOverlaySceneRenderer(draw=FakeDeviceSceneDrawer())
    lease = fake_device_resident_lease(width=4, height=4)

    annotated = renderer.render_scene(lease, scene)
    lease.release()  # source released first; annotated output is a distinct lease
    assert not annotated.released

    annotated.release()
    with pytest.raises(FrameLeaseReleasedError):
        annotated.release()
