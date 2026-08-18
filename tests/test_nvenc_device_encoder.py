"""Todo 18: deterministic contract/fake-adapter proof for the device-input NVENC seam.

Same rationale as ``tests/test_nvidia_device_resident_prototype.py`` and
``tests/test_cuda_overlay_renderer.py``: every lifecycle/backpressure/
cancellation/failure/zero-host-transfer claim here is proven against
``worker.adapters.encode.nvenc_device.fake``, never against real GPU memory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from contracts.frame import Frame
from worker.adapters.decode.nvdec_device.capability import DeviceResidentCapability
from worker.adapters.device.cuda.probe import CudaCapability, NvencCapability
from worker.adapters.device.nvml.probe import NvmlGpuStatus
from worker.adapters.encode.nvenc_device.capability import (
    DeviceInputNvencCapability,
    probe_device_input_nvenc_capability,
)
from worker.adapters.encode.nvenc_device.errors import (
    DeviceEncoderPoolExhaustedError,
    DeviceEncoderRejectedInputError,
)
from worker.adapters.encode.nvenc_device.fake import (
    fake_device_input_nvenc_encoder,
    fake_device_resident_lease,
)
from worker.adapters.encode.nvenc_device.models import (
    DeviceEncoderCodec,
    DeviceEncoderContainer,
    DeviceEncoderPoolConfig,
    DeviceEncoderProfile,
)
from worker.interfaces.encode import DeviceInputEncoder
from worker.types import FrameLease

# --------------------------------------------------------------------------
# Pool config validation
# --------------------------------------------------------------------------


def test_pool_config_rejects_non_positive_capacity_and_dimensions() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        DeviceEncoderPoolConfig(camera_id="camera-a", capacity=0, width=4, height=3)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        DeviceEncoderPoolConfig(camera_id="camera-a", capacity=1, width=0, height=3)
    with pytest.raises(ValueError, match="requires a camera id"):
        DeviceEncoderPoolConfig(camera_id="", capacity=1, width=4, height=3)


def test_pool_config_requires_at_least_one_of_each_candidate() -> None:
    with pytest.raises(ValueError, match="codec candidate"):
        DeviceEncoderPoolConfig(
            camera_id="camera-a", capacity=1, width=4, height=3, codec_candidates=()
        )
    with pytest.raises(ValueError, match="container candidate"):
        DeviceEncoderPoolConfig(
            camera_id="camera-a", capacity=1, width=4, height=3, container_candidates=()
        )
    with pytest.raises(ValueError, match="profile candidate"):
        DeviceEncoderPoolConfig(
            camera_id="camera-a", capacity=1, width=4, height=3, profile_candidates=()
        )


def test_pool_config_default_candidates_are_declared_not_guessed() -> None:
    config = DeviceEncoderPoolConfig(camera_id="camera-a", capacity=1, width=4, height=3)
    assert config.codec_candidates == (DeviceEncoderCodec.H264,)
    assert config.container_candidates == (DeviceEncoderContainer.MP4,)
    assert config.profile_candidates == (
        DeviceEncoderProfile.HIGH,
        DeviceEncoderProfile.MAIN,
        DeviceEncoderProfile.BASELINE,
    )


# --------------------------------------------------------------------------
# Truthful encoder choice / session open
# --------------------------------------------------------------------------


def test_open_selects_from_declared_candidates_and_reports_device_residency() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=4, height=4)

    selection = encoder.open()

    assert selection.requested_codec == DeviceEncoderCodec.H264
    assert selection.selected_codec == DeviceEncoderCodec.H264
    assert selection.selected_container == DeviceEncoderContainer.MP4
    assert selection.device_resident is True
    assert selection.reason
    assert encoder.telemetry.snapshot().sessions_opened == 1


def test_open_is_idempotent_and_does_not_recount_sessions() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=4, height=4)
    first = encoder.open()
    second = encoder.open()
    assert first == second
    assert encoder.telemetry.snapshot().sessions_opened == 1


def test_encoder_conforms_to_device_input_encoder_protocol() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=2, height=2)
    assert isinstance(encoder, DeviceInputEncoder)


# --------------------------------------------------------------------------
# No host readback: rejected host-memory submissions
# --------------------------------------------------------------------------


def test_submit_rejects_host_memory_lease_without_reading_it() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=2, height=2)
    _ = encoder.open()
    host_lease = FrameLease.from_host(Frame(0, 0.0, np.zeros((2, 2, 3), dtype=np.uint8)))

    with pytest.raises(DeviceEncoderRejectedInputError, match="host-memory"):
        encoder.submit(host_lease)

    snapshot = encoder.telemetry.snapshot()
    assert snapshot.submissions_rejected_host_input == 1
    assert snapshot.submissions_accepted == 0
    assert snapshot.d2h_transfers == 0  # never touched -- not even to inspect it
    host_lease.release()


def test_submit_rejects_after_close() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=2, height=2)
    _ = encoder.open()
    encoder.close()
    lease = fake_device_resident_lease(width=2, height=2)

    with pytest.raises(DeviceEncoderRejectedInputError, match="closed"):
        encoder.submit(lease)
    lease.release()


# --------------------------------------------------------------------------
# Bounded pool / backpressure (in-flight submission queue)
# --------------------------------------------------------------------------


def test_submit_past_capacity_raises_pool_exhausted_with_counts() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=2, width=2, height=2)
    _ = encoder.open()
    lease_a = fake_device_resident_lease(width=2, height=2)
    lease_b = fake_device_resident_lease(width=2, height=2)
    lease_c = fake_device_resident_lease(width=2, height=2)

    encoder.submit(lease_a)
    encoder.submit(lease_b)
    assert encoder.outstanding == 2

    with pytest.raises(DeviceEncoderPoolExhaustedError) as excinfo:
        encoder.submit(lease_c)
    assert excinfo.value.capacity == 2
    assert excinfo.value.outstanding == 2
    assert encoder.telemetry.snapshot().pool_exhaustion_events == 1

    lease_c.release()
    encoder.retire_all()


def test_retiring_a_submission_frees_its_pool_slot_for_reuse() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=2, height=2)
    _ = encoder.open()
    lease_a = fake_device_resident_lease(width=2, height=2)
    lease_b = fake_device_resident_lease(width=2, height=2)

    encoder.submit(lease_a)
    assert encoder.outstanding == 1
    encoder.retire_one()
    assert encoder.outstanding == 0

    encoder.submit(lease_b)  # would have raised DeviceEncoderPoolExhaustedError if not freed
    encoder.retire_one()


def test_pool_pressure_watermark_never_decreases() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=3, width=2, height=2)
    _ = encoder.open()
    leases = [fake_device_resident_lease(width=2, height=2) for _ in range(3)]
    for lease in leases:
        encoder.submit(lease)

    snapshot = encoder.telemetry.snapshot()
    assert snapshot.pool_capacity == 3
    assert snapshot.pool_outstanding == 3
    assert snapshot.pool_high_watermark == 3

    encoder.retire_all()
    assert encoder.telemetry.snapshot().pool_outstanding == 0
    assert encoder.telemetry.snapshot().pool_high_watermark == 3


# --------------------------------------------------------------------------
# Ownership: submitted lease is retained until retirement, not just borrowed
# --------------------------------------------------------------------------


def test_submitted_lease_stays_valid_after_caller_releases_its_own_reference() -> None:
    """The encoder takes real ownership on submit (retains its own reference),
    so a caller that releases its handle immediately after submit does not
    invalidate the in-flight encode -- proving explicit ownership transfer,
    not a borrowed/aliased pointer."""
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=2, height=2)
    _ = encoder.open()
    lease = fake_device_resident_lease(width=2, height=2, fill=42)

    encoder.submit(lease)
    lease.release()  # caller's own reference is gone; encoder's retained one remains

    encoder.retire_one()  # must not raise FrameLeaseReleasedError
    assert encoder.outstanding == 0


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------


def test_cancel_pending_drops_in_flight_submissions_and_frees_slots() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=2, width=2, height=2)
    _ = encoder.open()
    lease_a = fake_device_resident_lease(width=2, height=2)
    lease_b = fake_device_resident_lease(width=2, height=2)
    encoder.submit(lease_a)
    encoder.submit(lease_b)

    cancelled_count = encoder.cancel_pending()

    assert cancelled_count == 2
    assert encoder.outstanding == 0
    # A cancelled frame must never appear in the final artifact.
    with pytest.raises(DeviceEncoderRejectedInputError, match="zero retired frames"):
        _ = encoder.finalize(_tmp_path())


def test_close_cancels_any_still_pending_submissions() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=2, height=2)
    _ = encoder.open()
    lease = fake_device_resident_lease(width=2, height=2)
    encoder.submit(lease)

    encoder.close()

    assert encoder.outstanding == 0


# --------------------------------------------------------------------------
# Failure: finalize before retirement, or with zero frames
# --------------------------------------------------------------------------


def test_finalize_before_open_fails_closed() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=2, height=2)
    with pytest.raises(DeviceEncoderRejectedInputError, match="before a session was opened"):
        _ = encoder.finalize(_tmp_path())


def test_finalize_with_in_flight_submissions_fails_closed() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=2, height=2)
    _ = encoder.open()
    lease = fake_device_resident_lease(width=2, height=2)
    encoder.submit(lease)

    with pytest.raises(DeviceEncoderRejectedInputError, match="pending retirement"):
        _ = encoder.finalize(_tmp_path())

    encoder.retire_all()


def test_finalize_with_zero_retired_frames_fails_closed() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=2, height=2)
    _ = encoder.open()
    with pytest.raises(DeviceEncoderRejectedInputError, match="zero retired frames"):
        _ = encoder.finalize(_tmp_path())


def test_retire_one_with_nothing_pending_fails_closed() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=2, height=2)
    _ = encoder.open()
    with pytest.raises(DeviceEncoderRejectedInputError, match="no in-flight submission"):
        encoder.retire_one()


# --------------------------------------------------------------------------
# Zero-host-transfer accounting + artifact provenance
# --------------------------------------------------------------------------


def test_submission_and_retirement_never_record_a_host_transfer() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=1, width=2, height=2)
    _ = encoder.open()
    lease = fake_device_resident_lease(width=2, height=2)

    encoder.submit(lease)
    before = encoder.telemetry.snapshot().d2h_transfers
    encoder.retire_one()
    after = encoder.telemetry.snapshot().d2h_transfers

    assert before == 0
    assert after == 0  # only the final bitstream materialization counts as D2H


def test_finalize_records_exactly_one_host_transfer_for_the_artifact() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=2, width=2, height=2)
    _ = encoder.open()
    lease_a = fake_device_resident_lease(width=2, height=2, fill=1)
    lease_b = fake_device_resident_lease(width=2, height=2, fill=2)
    encoder.submit(lease_a)
    encoder.submit(lease_b)
    encoder.retire_all()

    result = encoder.finalize(_tmp_path())

    snapshot = encoder.telemetry.snapshot()
    assert snapshot.d2h_transfers == 1
    assert snapshot.d2h_bytes == result.size_bytes
    assert snapshot.artifacts_finalized == 1
    assert snapshot.artifact_bytes_total == result.size_bytes
    assert result.sha256
    assert result.selection.device_resident is True
    assert result.path.read_bytes()


def test_finalize_artifact_reflects_only_retired_frames_not_cancelled_ones() -> None:
    encoder = fake_device_input_nvenc_encoder(camera_id="camera-a", capacity=2, width=2, height=2)
    _ = encoder.open()
    kept = fake_device_resident_lease(width=2, height=2, fill=9)
    dropped = fake_device_resident_lease(width=2, height=2, fill=250)
    encoder.submit(kept)
    encoder.retire_one()
    encoder.submit(dropped)
    _ = encoder.cancel_pending()

    result = encoder.finalize(_tmp_path())

    only_kept = fake_device_input_nvenc_encoder(camera_id="camera-b", capacity=1, width=2, height=2)
    _ = only_kept.open()
    reference_lease = fake_device_resident_lease(width=2, height=2, fill=9)
    only_kept.submit(reference_lease)
    only_kept.retire_one()
    reference_result = only_kept.finalize(_tmp_path())

    assert result.sha256 == reference_result.sha256


# --------------------------------------------------------------------------
# Combined capability probe: honest on this non-NVIDIA host, fails closed
# --------------------------------------------------------------------------


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
    capability = probe_device_input_nvenc_capability()
    assert capability.reason


def test_probe_fails_closed_when_device_resident_capability_is_unavailable() -> None:
    unavailable = DeviceResidentCapability(
        available=False,
        reason="cuda unavailable",
        cuda=CudaCapability(available=False, reason="no device"),
        nvml=NvmlGpuStatus(nvml_available=False, reason="no nvml"),
        stream_event_supported=False,
        dlpack_supported=False,
    )
    capability = probe_device_input_nvenc_capability(
        device_resident_probe=lambda: unavailable,
        nvenc_probe=lambda: NvencCapability(True, "should not be reached"),
    )
    assert capability.available is False
    assert "device-resident capability unavailable" in capability.reason
    assert capability.nvenc.available is False  # never probed after the first gate failed


def test_probe_fails_closed_when_nvenc_ffmpeg_build_is_unavailable() -> None:
    available_device_resident = DeviceResidentCapability(
        available=True,
        reason="ok",
        cuda=CudaCapability(available=True, reason="ok", device_count=1, arch_list=("sm_90",)),
        nvml=NvmlGpuStatus(nvml_available=True, reason="ok"),
        stream_event_supported=True,
        dlpack_supported=True,
    )
    capability = probe_device_input_nvenc_capability(
        device_resident_probe=lambda: available_device_resident,
        nvenc_probe=lambda: NvencCapability(False, "ffmpeg has no h264_nvenc encoder"),
    )
    assert capability.available is False
    assert "h264_nvenc encoder unavailable" in capability.reason


def test_probe_reports_available_when_every_gate_passes() -> None:
    available_device_resident = DeviceResidentCapability(
        available=True,
        reason="ok",
        cuda=CudaCapability(available=True, reason="ok", device_count=1, arch_list=("sm_90",)),
        nvml=NvmlGpuStatus(nvml_available=True, reason="ok"),
        stream_event_supported=True,
        dlpack_supported=True,
    )
    capability = probe_device_input_nvenc_capability(
        device_resident_probe=lambda: available_device_resident,
        nvenc_probe=lambda: NvencCapability(True, "ffmpeg h264_nvenc encoder is available"),
    )
    assert capability.available is True
    assert isinstance(capability, DeviceInputNvencCapability)


def _tmp_path() -> Path:
    return Path(tempfile.mkdtemp()) / "artifact.bin"
