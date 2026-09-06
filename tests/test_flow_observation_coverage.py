from __future__ import annotations

import logging
import threading
import uuid
from types import MappingProxyType, SimpleNamespace

import pytest

from worker.runtime.flow.metadata_slot import AcceptanceToken, LatestMetadataSlot
from worker.runtime.flow.observation_coverage import ObservationCoverage
from worker.runtime.flow.policy_pump import NativePolicyPump
from worker.types.metadata import MetadataFrame, SourceBinding
from worker.types.perception_frame import (
    BedRegionChannel,
    ChannelState,
    HumanPoseChannel,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBoxChannel,
)
from worker.types.preview import FallPreviewState


def _binding(
    *,
    boot: str = "boot-a",
    camera: str = "camera-a",
    child: str = "child-a",
    generation: int = 1,
    epoch: int = 2,
) -> SourceBinding:
    return SourceBinding(
        worker_boot_id=boot,
        child_instance_id=child,
        camera_id=camera,
        source_generation=generation,
        stream_epoch=epoch,
        transform_id="transform-a",
    )


def _metadata(
    *,
    boot: str = "boot-a",
    camera: str = "camera-a",
    child: uuid.UUID | None = None,
    generation: int = 1,
    epoch: int = 2,
    seq: int = 3,
    pts_ns: int | None = 100,
    publish_sequence: int = 4,
) -> MetadataFrame:
    identity = PerceptionFrameIdentity(
        worker_boot_id=boot,
        camera_id=camera,
        stream_epoch=epoch,
        seq=seq,
        source_pts=pts_ns,
    )
    return MetadataFrame(
        frame=PerceptionFrameV1(
            identity=identity,
            person_box=PersonBoxChannel(ChannelState.INFERRED_EMPTY),
            human_pose=HumanPoseChannel(ChannelState.INFERRED_EMPTY),
            bed_region=BedRegionChannel(ChannelState.SKIPPED),
        ),
        source_generation=generation,
        child_instance_id=uuid.uuid4() if child is None else child,
        native_publish_sequence=publish_sequence,
        transform_id="transform-a",
    )


def _logged_message(
    caplog: pytest.LogCaptureFixture,
    starts_with: str,
) -> str:
    return next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(starts_with)
    )


def test_gap_before_first_observation_preserves_unknown_start_identity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coverage = ObservationCoverage(_binding())

    with caplog.at_level(logging.INFO):
        opened = coverage.detect_gap(host_time=9.0)
        recovery = coverage.observe(_metadata(), host_time=9.5)

    assert opened is not None
    assert opened.last_actual is None
    assert recovery is not None
    assert recovery.host_observation_duration is None
    assert recovery.source_duration_ns is None
    assert recovery.native_publish_sequence_gap is None
    message = _logged_message(caplog, "policy input coverage gap opened")
    assert "expected_worker_boot_id=boot-a" in message
    assert "expected_source_generation=1" in message
    assert "last_actual_worker_boot_id=unknown" in message
    assert "last_actual_camera_id=unknown" in message
    assert "last_actual_source_generation=unknown" in message
    assert "last_actual_stream_epoch=unknown" in message
    assert "last_actual_frame_seq=unknown" in message
    assert "last_actual_source_pts_ns=unknown" in message
    closed_message = _logged_message(caplog, "policy input coverage gap closed")
    assert "last_actual_worker_boot_id=unknown" in closed_message
    assert "last_actual_frame_host_time=unknown" in closed_message
    assert "host_observation_duration=unknown" in closed_message


def test_gap_opens_once_and_preserves_delayed_last_actual_time(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coverage = ObservationCoverage(_binding())
    coverage.observe(_metadata(), host_time=10.0)

    with caplog.at_level(logging.WARNING):
        opened = coverage.detect_gap(host_time=10.5)
        repeated = coverage.detect_gap(host_time=11.0)

    assert opened is not None
    assert repeated is None
    assert opened.last_actual is not None
    assert opened.last_actual.host_time == 10.0
    assert opened.loss_detected_host_time == 10.5
    assert caplog.text.count("policy input coverage gap opened") == 1
    assert "gap_state=open" in caplog.text
    assert "gap_cause=unknown" in caplog.text
    assert "capture_loss=unknown" in caplog.text
    assert "person_count_claim=unknown" in caplog.text


def test_rebind_before_gap_labels_expected_last_and_next_identities_separately(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coverage = ObservationCoverage(_binding(boot="boot-old", generation=3, epoch=4))
    coverage.observe(
        _metadata(
            boot="boot-old",
            generation=3,
            epoch=4,
            seq=17,
            pts_ns=700,
            publish_sequence=20,
        ),
        host_time=10.0,
    )
    coverage.rebind(_binding(boot="boot-new", generation=8, epoch=9))

    with caplog.at_level(logging.INFO):
        opened = coverage.detect_gap(host_time=10.5)
        recovery = coverage.observe(
            _metadata(
                boot="boot-new",
                generation=8,
                epoch=9,
                seq=1,
                pts_ns=50,
                publish_sequence=1,
            ),
            host_time=11.0,
        )

    assert opened is not None
    assert recovery is not None
    opened_message = _logged_message(caplog, "policy input coverage gap opened")
    assert "expected_worker_boot_id=boot-new" in opened_message
    assert "expected_source_generation=8" in opened_message
    assert "expected_stream_epoch=9" in opened_message
    assert "last_actual_worker_boot_id=boot-old" in opened_message
    assert "last_actual_source_generation=3" in opened_message
    assert "last_actual_stream_epoch=4" in opened_message
    assert "last_actual_frame_seq=17" in opened_message
    closed_message = _logged_message(caplog, "policy input coverage gap closed")
    assert "gap_expected_worker_boot_id=boot-new" in closed_message
    assert "last_actual_worker_boot_id=boot-old" in closed_message
    assert "last_actual_source_generation=3" in closed_message
    assert "next_actual_worker_boot_id=boot-new" in closed_message
    assert "next_actual_source_generation=8" in closed_message
    assert "host_observation_duration=unknown" in closed_message
    assert "source_duration_ns=unknown" in closed_message
    assert "native_publish_sequence_gap=unknown" in closed_message
    assert "gap_cause=unknown" in closed_message
    assert "capture_loss=unknown" in closed_message


def test_recovery_pairs_last_and_next_actual_host_times() -> None:
    coverage = ObservationCoverage(_binding())
    coverage.observe(_metadata(pts_ns=100, publish_sequence=5), host_time=20.0)
    assert coverage.detect_gap(host_time=20.6) is not None

    recovery = coverage.observe(
        _metadata(seq=4, pts_ns=900, publish_sequence=9),
        host_time=21.25,
    )

    assert recovery is not None
    assert recovery.gap.last_actual is not None
    assert recovery.gap.last_actual.host_time == 20.0
    assert recovery.next_actual.host_time == 21.25
    assert recovery.host_observation_duration == 1.25
    assert recovery.source_duration_ns == 800
    assert recovery.native_publish_sequence_gap == 3
    assert coverage.open_gap is None


def test_epoch_reset_keeps_host_elapsed_but_source_duration_unknown() -> None:
    coverage = ObservationCoverage(_binding(epoch=2))
    coverage.observe(_metadata(epoch=2, pts_ns=900, publish_sequence=8), host_time=30.0)
    assert coverage.detect_gap(host_time=30.5) is not None
    coverage.rebind(_binding(generation=2, epoch=3))

    recovery = coverage.observe(
        _metadata(generation=2, epoch=3, pts_ns=10, publish_sequence=1),
        host_time=32.0,
    )

    assert recovery is not None
    assert recovery.host_observation_duration == 2.0
    assert recovery.source_duration_ns is None
    assert recovery.native_publish_sequence_gap is None


def test_boot_reset_does_not_backfill_elapsed_or_source_health() -> None:
    coverage = ObservationCoverage(_binding(boot="boot-a"))
    coverage.observe(_metadata(boot="boot-a", pts_ns=100), host_time=40.0)
    assert coverage.detect_gap(host_time=40.5) is not None
    coverage.rebind(_binding(boot="boot-b", generation=1, epoch=1))

    recovery = coverage.observe(
        _metadata(boot="boot-b", generation=1, epoch=1, pts_ns=200),
        host_time=41.0,
    )

    assert recovery is not None
    assert recovery.host_observation_duration is None
    assert recovery.source_duration_ns is None
    assert recovery.native_publish_sequence_gap is None


@pytest.mark.parametrize(
    "unexpected",
    [
        _metadata(camera="camera-b", pts_ns=200, publish_sequence=9),
        _metadata(generation=2, pts_ns=200, publish_sequence=9),
    ],
)
def test_unexpected_camera_or_source_cannot_close_an_open_coverage_gap(
    unexpected: MetadataFrame,
) -> None:
    coverage = ObservationCoverage(_binding(camera="camera-a"))
    coverage.observe(
        _metadata(camera="camera-a", pts_ns=100, publish_sequence=5),
        host_time=40.0,
    )
    assert coverage.detect_gap(host_time=40.5) is not None

    with pytest.raises(ValueError, match="expected source binding"):
        coverage.observe(unexpected, host_time=41.0)

    assert coverage.open_gap is not None
    assert coverage.last_actual is not None
    assert coverage.last_actual.identity.camera_id == "camera-a"
    assert coverage.last_actual.identity.source_generation == 1


def test_misordered_same_boot_monotonic_times_do_not_report_negative_elapsed() -> None:
    coverage = ObservationCoverage(_binding())
    coverage.observe(_metadata(pts_ns=100), host_time=42.0)
    assert coverage.detect_gap(host_time=42.5) is not None

    recovery = coverage.observe(
        _metadata(seq=4, pts_ns=200, publish_sequence=5),
        host_time=41.0,
    )

    assert recovery is not None
    assert recovery.host_observation_duration is None


@pytest.mark.parametrize("intermediate_outcome", ["rejected", "coalesced"])
def test_publication_sequence_gap_does_not_assert_rejection_or_coalescing_cause(
    intermediate_outcome: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    child = uuid.uuid4()
    binding = _binding(child=str(child))
    slot = LatestMetadataSlot()
    slot.register_source(binding)
    first = _metadata(
        child=child,
        seq=1,
        pts_ns=100,
        publish_sequence=10,
    )
    assert slot.publish(first)
    token = slot.subscribe(binding)
    coverage = ObservationCoverage(binding)
    coverage.observe(first, host_time=50.0)

    if intermediate_outcome == "rejected":
        intermediate = _metadata(
            child=child,
            seq=1,
            pts_ns=100,
            publish_sequence=11,
        )
        assert not slot.publish(intermediate)
        final = _metadata(
            child=child,
            seq=2,
            pts_ns=200,
            publish_sequence=12,
        )
    else:
        intermediate = _metadata(
            child=child,
            seq=2,
            pts_ns=200,
            publish_sequence=11,
        )
        assert slot.publish(intermediate)
        final = _metadata(
            child=child,
            seq=3,
            pts_ns=300,
            publish_sequence=12,
        )
    assert slot.publish(final)
    counters = slot.counters()
    if intermediate_outcome == "rejected":
        assert counters.late == 1
        assert counters.overwritten == 1
    else:
        assert counters.late == 0
        assert counters.overwritten == 2

    with caplog.at_level(logging.DEBUG):
        recovery = coverage.observe(
            slot.wait_accepted(token, timeout_sec=0.01),
            host_time=51.0,
        )

    assert recovery is None
    assert coverage.open_gap is None
    message = _logged_message(
        caplog,
        "policy input native publication sequence gap observed",
    )
    assert "coverage_scope=policy_input" in message
    assert "native_publish_sequence_gap=1" in message
    assert "gap_cause=unknown" in message
    assert "capture_loss=unknown" in message


def test_policy_pump_timeout_clears_stale_preview_state_without_reopening_gap() -> None:
    pump = object.__new__(NativePolicyPump)
    pump._binding = _binding()  # noqa: SLF001
    pump._stop = threading.Event()  # noqa: SLF001
    pump._observation_coverage = ObservationCoverage(_binding())  # noqa: SLF001
    pump._preview_states_lock = threading.Lock()  # noqa: SLF001
    pump._preview_states = MappingProxyType(  # noqa: SLF001
        {7: FallPreviewState(track_id=7, status="suspected", probability=0.93)}
    )
    pump._replay_trace = None  # noqa: SLF001
    pump._trace_epoch = None  # noqa: SLF001
    pump._trace_source_lost = False  # noqa: SLF001
    pump._recreate_decision = None  # noqa: SLF001

    class TimeoutSlot:
        def subscribe(self, binding: SourceBinding) -> AcceptanceToken:
            return AcceptanceToken(binding, 0)

        def wait_accepted(
            self,
            token: AcceptanceToken,
            *,
            timeout_sec: float,
        ) -> MetadataFrame:
            assert timeout_sec == 0.5
            pump._stop.set()  # noqa: SLF001
            raise TimeoutError

        def expected_binding(self, camera_id: str) -> SourceBinding:
            assert camera_id == "camera-a"
            return _binding()

    pump._metadata = TimeoutSlot()  # type: ignore[assignment]  # noqa: SLF001
    pump.run()

    first_gap = pump._observation_coverage.open_gap  # noqa: SLF001
    pump._record_observation_timeout()  # noqa: SLF001

    assert pump.preview_states() == {}
    assert first_gap is not None
    assert pump._observation_coverage.open_gap is first_gap  # noqa: SLF001


def test_policy_pump_records_observation_before_processing_failure() -> None:
    metadata = _metadata(seq=31, pts_ns=1234, publish_sequence=12)
    detection_attempts: list[str] = []
    pump = object.__new__(NativePolicyPump)
    pump._binding = _binding()  # noqa: SLF001
    pump._stop = threading.Event()  # noqa: SLF001
    pump._observation_coverage = ObservationCoverage(_binding())  # noqa: SLF001
    pump._diagnostics = SimpleNamespace(  # noqa: SLF001
        record_native_detection_attempt=detection_attempts.append
    )
    pump.failure_count = 0
    pump.processed_count = 0

    class OneFrameSlot:
        def subscribe(self, binding: SourceBinding) -> AcceptanceToken:
            return AcceptanceToken(binding, 0)

        def wait_accepted(
            self,
            token: AcceptanceToken,
            *,
            timeout_sec: float,
        ) -> MetadataFrame:
            assert timeout_sec == 0.5
            pump._stop.set()  # noqa: SLF001
            return metadata

    def fail_processing(frame: MetadataFrame) -> None:
        assert frame is metadata
        raise ValueError("scripted processing failure")

    pump._metadata = OneFrameSlot()  # type: ignore[assignment]  # noqa: SLF001
    pump._process = fail_processing  # type: ignore[method-assign]  # noqa: SLF001
    pump.run()

    actual = pump._observation_coverage.last_actual  # noqa: SLF001
    assert actual is not None
    assert actual.seq == 31
    assert actual.source_pts_ns == 1234
    assert detection_attempts == ["camera-a"]
    assert pump.failure_count == 1
    assert pump.processed_count == 1
