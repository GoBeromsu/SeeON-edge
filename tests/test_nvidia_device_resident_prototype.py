"""Todo 17: deterministic contract/fake-adapter proof for the NVIDIA
device-resident analysis prototype.

This repo's dev/CI hosts are Apple Silicon (no NVIDIA hardware) -- every test
here proves lifecycle, bounded backpressure, transfer accounting, and
CPU/"device" numeric parity against a deterministic fake storage backend
(``worker.adapters.decode.nvdec_device.fake``), never against real GPU
memory. A separate, explicitly ``real_stack``-marked module
(``tests/test_nvidia_device_resident_real_stack.py``) is the only place that
asserts anything about actual hardware, and it is deselected from the default
suite exactly like every other ``real_stack`` test in this repo.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from worker.adapters.decode.nvdec_device.capability import (
    DeviceResidentCapability,
    probe_device_resident_capability,
)
from worker.adapters.decode.nvdec_device.errors import DevicePoolExhaustedError
from worker.adapters.decode.nvdec_device.fake import (
    FakeDeviceResidentBatcher,
    fake_device_resident_pool,
)
from worker.adapters.decode.nvdec_device.models import DeviceResidentPoolConfig
from worker.adapters.decode.nvdec_device.pool import DeviceResidentFramePool
from worker.adapters.device.cuda.probe import CudaCapability
from worker.adapters.device.nvml.probe import NvmlGpuStatus
from worker.interfaces.device_batch import DeviceResidentBatcher, DeviceResidentPool
from worker.runtime.telemetry.device_residency import device_residency_diagnostics
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.types import (
    FrameDescriptor,
    FrameLease,
    FrameLeaseReleasedError,
    MemoryKind,
    PixelFormat,
)

# --------------------------------------------------------------------------
# Pool config validation
# --------------------------------------------------------------------------


def test_pool_config_rejects_non_positive_capacity_and_dimensions() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        DeviceResidentPoolConfig(camera_id="camera-a", capacity=0, width=4, height=3)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        DeviceResidentPoolConfig(camera_id="camera-a", capacity=1, width=0, height=3)
    with pytest.raises(ValueError, match="requires a camera id"):
        DeviceResidentPoolConfig(camera_id="", capacity=1, width=4, height=3)


# --------------------------------------------------------------------------
# Lifecycle, ownership, backpressure
# --------------------------------------------------------------------------


def test_acquire_mints_bounded_slots_and_refuses_past_capacity() -> None:
    pool, allocator = fake_device_resident_pool(camera_id="camera-a", capacity=2, width=2, height=2)

    lease_a = pool.acquire()
    lease_b = pool.acquire()
    assert allocator.allocations == 2
    assert pool.outstanding == 2

    with pytest.raises(DevicePoolExhaustedError) as excinfo:
        pool.acquire()
    assert excinfo.value.capacity == 2
    assert excinfo.value.outstanding == 2
    assert pool.telemetry.snapshot().pool_exhaustion_events == 1

    lease_a.release()
    lease_b.release()


def test_released_slot_is_reused_without_reallocating() -> None:
    pool, allocator = fake_device_resident_pool(camera_id="camera-a", capacity=1, width=2, height=2)

    first = pool.acquire()
    first.release()
    assert pool.outstanding == 0

    second = pool.acquire()
    assert allocator.allocations == 1  # reused the recycled slot, no second allocation
    second.release()


def test_pool_conforms_to_device_resident_pool_protocol() -> None:
    pool, _allocator = fake_device_resident_pool(
        camera_id="camera-a", capacity=1, width=2, height=2
    )
    assert isinstance(pool, DeviceResidentPool)
    lease = pool.acquire()
    assert isinstance(lease, FrameLease)
    lease.release()


def test_acquired_lease_is_device_resident_never_host() -> None:
    pool, _allocator = fake_device_resident_pool(
        camera_id="camera-a", capacity=1, width=2, height=2
    )
    lease = pool.acquire()

    assert lease.descriptor.memory_kind is MemoryKind.CUDA_DEVICE
    with pytest.raises(RuntimeError, match="not host-resident"):
        _ = lease.host_frame
    assert isinstance(lease.device_handle, np.ndarray)
    lease.release()


def test_double_release_and_use_after_release_fail_closed() -> None:
    pool, _allocator = fake_device_resident_pool(
        camera_id="camera-a", capacity=1, width=2, height=2
    )
    lease = pool.acquire()
    lease.release()

    with pytest.raises(FrameLeaseReleasedError):
        lease.release()
    with pytest.raises(FrameLeaseReleasedError):
        _ = lease.descriptor


def test_retained_lease_keeps_slot_outstanding_until_every_retain_releases() -> None:
    pool, _allocator = fake_device_resident_pool(
        camera_id="camera-a", capacity=1, width=2, height=2
    )
    lease = pool.acquire()
    child = lease.retain()
    assert pool.outstanding == 1

    lease.release()
    assert pool.outstanding == 1  # child still holds the slot outstanding

    child.release()
    assert pool.outstanding == 0


# --------------------------------------------------------------------------
# Transfer accounting (H2D/D2H counts + bytes)
# --------------------------------------------------------------------------


def test_h2d_and_d2h_transfers_are_counted_with_bytes() -> None:
    pool, allocator = fake_device_resident_pool(camera_id="camera-a", capacity=1, width=2, height=2)
    lease = pool.acquire()
    host_image = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)

    allocator.upload(lease, host_image)
    _ = allocator.download(lease)

    snapshot = pool.telemetry.snapshot()
    assert snapshot.h2d_transfers == 1
    assert snapshot.h2d_bytes == host_image.nbytes
    assert snapshot.d2h_transfers == 1
    assert snapshot.d2h_bytes == host_image.nbytes
    lease.release()


def test_pool_pressure_watermark_and_capacity_are_reported() -> None:
    pool, _allocator = fake_device_resident_pool(
        camera_id="camera-a", capacity=3, width=2, height=2
    )
    leases = [pool.acquire() for _ in range(3)]
    status = pool.status()
    assert status.capacity == 3
    assert status.outstanding == 3
    assert status.high_watermark == 3
    assert status.exhaustion_events == 0

    for lease in leases:
        lease.release()
    assert pool.status().outstanding == 0
    assert pool.status().high_watermark == 3  # watermark never decreases


# --------------------------------------------------------------------------
# Batching + numeric CPU/"device" parity
# --------------------------------------------------------------------------


def test_batcher_rejects_batches_larger_than_declared_max() -> None:
    pool, allocator = fake_device_resident_pool(camera_id="camera-a", capacity=2, width=2, height=2)
    batcher = FakeDeviceResidentBatcher(max_batch_size=1, allocator=allocator)
    assert isinstance(batcher, DeviceResidentBatcher)

    lease_a = pool.acquire()
    lease_b = pool.acquire()
    with pytest.raises(ValueError, match="exceeds max_batch_size"):
        batcher.form_batch([lease_a, lease_b])
    lease_a.release()
    lease_b.release()


def test_batched_inference_matches_cpu_reference_within_declared_tolerance() -> None:
    pool, allocator = fake_device_resident_pool(camera_id="camera-a", capacity=2, width=3, height=3)
    batcher = FakeDeviceResidentBatcher(max_batch_size=2, allocator=allocator)

    host_a = np.full((3, 3, 3), 10, dtype=np.uint8)
    host_b = np.full((3, 3, 3), 200, dtype=np.uint8)
    lease_a = pool.acquire()
    lease_b = pool.acquire()
    allocator.upload(lease_a, host_a)
    allocator.upload(lease_b, host_b)

    batch = batcher.form_batch([lease_a, lease_b])
    device_means = batcher.infer_mean_rgb(batch)

    cpu_reference = (
        tuple(float(v) for v in host_a.reshape(-1, 3).mean(axis=0)),
        tuple(float(v) for v in host_b.reshape(-1, 3).mean(axis=0)),
    )
    tolerance = 1e-9
    for device_mean, cpu_mean in zip(device_means, cpu_reference, strict=True):
        for device_channel, cpu_channel in zip(device_mean, cpu_mean, strict=True):
            assert abs(device_channel - cpu_channel) <= tolerance

    lease_a.release()
    lease_b.release()


def test_batch_formation_never_reads_full_frame_back_to_host() -> None:
    """Only the reduced numeric result crosses back to host memory -- the
    batch itself stays a sequence of device-resident lease handles."""
    pool, allocator = fake_device_resident_pool(camera_id="camera-a", capacity=1, width=2, height=2)
    batcher = FakeDeviceResidentBatcher(max_batch_size=1, allocator=allocator)
    lease = pool.acquire()
    allocator.upload(lease, np.zeros((2, 2, 3), dtype=np.uint8))

    batch = batcher.form_batch([lease])
    assert all(isinstance(item.device_handle, np.ndarray) for item in batch)

    before = pool.telemetry.snapshot().d2h_transfers
    _ = batcher.infer_mean_rgb(batch)
    after = pool.telemetry.snapshot().d2h_transfers
    assert after == before  # the reduction never called `download`/recorded a D2H

    lease.release()


# --------------------------------------------------------------------------
# Real, no-guess capability probe -- honest on this Apple Silicon host
# --------------------------------------------------------------------------


class _FakeCuda:
    def __init__(self, *, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def get_arch_list(self) -> tuple[str, ...]:
        return ("sm_90",) if self._available else ()

    def device_count(self) -> int:
        return 1 if self._available else 0

    def Stream(self) -> object:  # noqa: N802 - mirrors torch.cuda.Stream's real name
        return object()

    def Event(self) -> object:  # noqa: N802 - mirrors torch.cuda.Event's real name
        return object()


class _FakeTensor:
    def __dlpack__(self) -> object:  # pragma: no cover - presence-only check
        return object()


class _FakeTorchModule:
    def __init__(self, *, cuda_available: bool, stream_event_ok: bool = True) -> None:
        self.cuda = _FakeCuda(available=cuda_available)
        if not stream_event_ok:

            def _raise() -> object:
                raise RuntimeError("no CUDA context")

            self.cuda.Stream = _raise  # type: ignore[method-assign]
            self.cuda.Event = _raise  # type: ignore[method-assign]
        self.Tensor = _FakeTensor
        self.from_dlpack: Callable[[Any], Any] = lambda handle: handle


def test_probe_against_real_host_always_states_a_reason() -> None:
    """실제 하드웨어 프로브를 이 호스트에 대고 그대로 실행한다.

    호스트에 GPU 가 있는지는 단언하지 않는다. 그건 코드가 아니라 실행 머신의
    성질이고, 그렇게 쓴 테스트가 nvidia 프로파일 전환만으로 무더기로 뒤집혔다
    (tests/AGENTS.md 의 Local Hero 항목 참조). 검증하는 것은 어느 호스트에서든
    성립하는 계약 하나다: 프로브는 자기 판단의 사유를 반드시 댄다.

    available 값과 metadata 필드는 단언하지 않는다. available=True 여도
    metadata 가 None 일 수 있고(같은 파일의 가짜 주입 테스트가 그 계약을 고정한다),
    device_count/arch_list 는 게이트가 아니라 프로브가 보고하려는 진단값이다.
    음성 경로는 같은 파일의 가짜 주입 테스트가 덮는다.
    """
    capability = probe_device_resident_capability()
    assert capability.reason


def test_probe_fails_closed_when_plain_cuda_capability_is_unavailable() -> None:
    capability = probe_device_resident_capability(
        cuda_probe=lambda: CudaCapability(available=False, reason="no device"),
        nvml_probe=lambda: NvmlGpuStatus(nvml_available=True, reason="ok"),
    )
    assert capability.available is False
    assert "cuda capability unavailable" in capability.reason


def test_probe_fails_closed_when_nvml_identity_is_unavailable() -> None:
    capability = probe_device_resident_capability(
        cuda_probe=lambda: CudaCapability(
            available=True, reason="ok", device_count=1, arch_list=("sm_90",)
        ),
        nvml_probe=lambda: NvmlGpuStatus(nvml_available=False, reason="no nvml"),
    )
    assert capability.available is False
    assert "nvml device identity unavailable" in capability.reason


def test_probe_fails_closed_when_stream_event_construction_fails() -> None:
    capability = probe_device_resident_capability(
        cuda_probe=lambda: CudaCapability(
            available=True, reason="ok", device_count=1, arch_list=("sm_90",)
        ),
        nvml_probe=lambda: NvmlGpuStatus(nvml_available=True, reason="ok"),
        torch_importer=lambda: _FakeTorchModule(cuda_available=True, stream_event_ok=False),
    )
    assert capability.available is False
    assert "Stream/Event" in capability.reason


def test_probe_reports_available_when_every_gate_passes() -> None:
    capability = probe_device_resident_capability(
        cuda_probe=lambda: CudaCapability(
            available=True, reason="ok", device_count=1, arch_list=("sm_90",)
        ),
        nvml_probe=lambda: NvmlGpuStatus(
            nvml_available=True, reason="ok", driver_version="580.1", device_name="RTX 5070 Ti"
        ),
        torch_importer=lambda: _FakeTorchModule(cuda_available=True),
    )
    assert capability.available is True
    assert capability.stream_event_supported is True
    assert capability.dlpack_supported is True
    assert isinstance(capability, DeviceResidentCapability)


# --------------------------------------------------------------------------
# Telemetry/provenance projection
# --------------------------------------------------------------------------


def test_device_residency_diagnostics_projects_pool_telemetry() -> None:
    pool, allocator = fake_device_resident_pool(camera_id="camera-a", capacity=2, width=2, height=2)
    lease = pool.acquire()
    allocator.upload(lease, np.zeros((2, 2, 3), dtype=np.uint8))
    lease.release()

    diagnostics = device_residency_diagnostics(
        pool.telemetry.snapshot(),
        residency_path="decode->preprocess->inference",
        wall_clock=lambda: 123.0,
    )
    assert diagnostics.residency_path == "decode->preprocess->inference"
    assert diagnostics.h2d_transfers == 1
    assert diagnostics.pool_capacity == 2
    assert diagnostics.pool_outstanding == 0
    assert diagnostics.unavailable_reason is None
    assert diagnostics.updated_at_sec == 123.0


def test_device_residency_diagnostics_rejects_blank_residency_path() -> None:
    pool, _allocator = fake_device_resident_pool(
        camera_id="camera-a", capacity=1, width=2, height=2
    )
    with pytest.raises(ValueError, match="residency_path"):
        device_residency_diagnostics(pool.telemetry.snapshot(), residency_path="")


def test_worker_diagnostics_carries_device_residency_per_camera() -> None:
    pool, _allocator = fake_device_resident_pool(
        camera_id="camera-a", capacity=1, width=2, height=2
    )
    diagnostics = WorkerDiagnostics()
    payload = device_residency_diagnostics(
        pool.telemetry.snapshot(),
        residency_path="decode->preprocess->inference",
    )
    diagnostics.record_device_residency("camera-a", payload)

    snapshot = diagnostics.snapshot()
    camera = next(camera for camera in snapshot.cameras if camera.camera_id == "camera-a")
    assert camera.device_residency == payload
    assert diagnostics.device_residency_selection("camera-a") == payload
    assert diagnostics.device_residency_snapshot() == {"camera-a": payload}


# --------------------------------------------------------------------------
# Allocator invariants (fail-closed, no silent host fallback)
# --------------------------------------------------------------------------


def test_pool_rejects_an_allocator_that_returns_host_memory() -> None:
    def _bad_allocate() -> tuple[object, FrameDescriptor]:
        return object(), FrameDescriptor(
            width=2,
            height=2,
            memory_kind=MemoryKind.HOST,
            pixel_format=PixelFormat.RGB24,
            plane_strides=(6,),
            size_bytes=12,
        )

    config = DeviceResidentPoolConfig(camera_id="camera-a", capacity=1, width=2, height=2)
    pool = DeviceResidentFramePool(config, allocate=_bad_allocate)
    with pytest.raises(ValueError, match="host-memory descriptor"):
        pool.acquire()
