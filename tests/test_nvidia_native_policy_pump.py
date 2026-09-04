from __future__ import annotations

import socket
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from worker.domains.fall import FallPolicyDeciderV2, FallPolicyV2, FallV2Probabilities
from worker.native.deepstream.control import ControlIdentity, DeepStreamControlClient
from worker.native.deepstream.ipc import (
    ControlMessage,
    MessageKind,
    MetadataFrame,
    decode_control_message,
    encode_message,
)
from worker.native.deepstream.metadata import LatestMetadataSlot, SourceBinding
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.perception import SceneState
from worker.runtime.deepstream.native_policy_pump import (
    NativePolicyContext,
    NativePolicyPump,
)
from worker.types import (
    AssociationResult,
    BedRegionChannel,
    BusinessEvent,
    ChannelState,
    DecisionInput,
    HumanPoseChannel,
    Keypoint,
    NativeEvidenceTrigger,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBox,
    PersonBoxChannel,
)

_BOOT = uuid.UUID("12345678-1234-5678-1234-567812345678")
_CHILD = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


@dataclass(slots=True)
class _Decider:
    received: DecisionInput | None = None

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        self.received = input_value
        return (
            BusinessEvent(
                "fall",
                "fall",
                "source-event",
                "camera-a",
                "facility-a",
                input_value.time_sec or 0.0,
                0.9,
                person_id=7,
            ),
        )


@dataclass(slots=True)
class _Sink:
    emitted: tuple[BusinessEvent, NativeEvidenceTrigger] | None = None

    def emit_for_frame(
        self,
        event: BusinessEvent,
        trigger: NativeEvidenceTrigger,
    ) -> None:
        self.emitted = (event, trigger)


@dataclass(slots=True)
class _EpisodeDecider:
    policy: FallPolicyDeciderV2

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        return self.policy.update(
            {7: FallV2Probabilities(0.0, 0.9, 0.0)},
            (7,),
            frame_index=input_value.frame_index,
            time_sec=input_value.time_sec or 0.0,
        )

    def release_onset(self, event: BusinessEvent) -> None:
        self.policy.release_onset(event)


@dataclass(slots=True)
class _FlakySink:
    calls: int = 0
    emitted: list[BusinessEvent] | None = None

    def emit_for_frame(
        self,
        event: BusinessEvent,
        trigger: NativeEvidenceTrigger,
    ) -> None:
        del trigger
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("durable staging failed")
        if self.emitted is None:
            self.emitted = []
        self.emitted.append(event)


class _SnapshotUnavailable:
    def snapshot(self, camera_id: str) -> None:
        del camera_id


@dataclass(slots=True)
class _Diagnostics:
    completed: int = 0
    polygon_sources: list[str] | None = None
    replay_trace_write_failures: int = 0
    absorbed_switch_totals: list[int] | None = None

    def update_measured_fps(self, camera_id: str, measured_fps: float | None) -> None:
        del camera_id, measured_fps

    def record_detection_completed(self, camera_id: str) -> None:
        assert camera_id == "camera-a"
        self.completed += 1

    def record_track_id_switch(self, camera_id: str) -> None:
        assert camera_id == "camera-a"

    def record_track_id_switch_absorbed_total(self, camera_id: str, total: int) -> None:
        assert camera_id == "camera-a"
        if self.absorbed_switch_totals is None:
            self.absorbed_switch_totals = []
        self.absorbed_switch_totals.append(total)

    def record_bed_polygon_source(self, camera_id: str, source: str) -> None:
        assert camera_id == "camera-a"
        if self.polygon_sources is None:
            self.polygon_sources = []
        self.polygon_sources.append(source)

    def record_replay_trace_write_failure(self, camera_id: str) -> None:
        assert camera_id == "camera-a"
        self.replay_trace_write_failures += 1


def _metadata() -> MetadataFrame:
    identity = PerceptionFrameIdentity(str(_BOOT), "camera-a", 4, 11, 2_000_000_000)
    association = AssociationResult(
        "legacy-greedy-bbox-iou.v1",
        (7,),
        (0,),
        identity,
        live_track_ids=(7,),
    )
    return MetadataFrame(
        PerceptionFrameV1(
            identity,
            PersonBoxChannel(ChannelState.INFERRED, (PersonBox(10, 20, 40, 80, 0.8),)),
            HumanPoseChannel(
                ChannelState.INFERRED,
                (tuple(Keypoint(20, 30, 0.9) for _ in range(17)),),
            ),
            BedRegionChannel(ChannelState.INFERRED_EMPTY),
            association,
        ),
        3,
        _CHILD,
        12,
        "seeon-perception-v1",
        640,
        360,
        2_100_000_000,
    )


def test_native_policy_uses_child_association_and_image_free_evidence_trigger(
    tmp_path: Path,
) -> None:
    # Given
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    control = DeepStreamControlClient(
        parent,
        ControlIdentity(_BOOT, _CHILD, "seeon-perception-v1"),
    )
    control.connect()
    request_received = threading.Event()

    def serve_snapshot() -> None:
        request = decode_control_message(child.recv(65_535))
        assert request.kind is MessageKind.SNAPSHOT
        request_received.set()
        reply = ControlMessage(
            MessageKind.ACK,
            request.worker_boot_id,
            request.child_instance_id,
            request.camera_id,
            request.source_generation,
            request.stream_epoch,
            0,
            0,
            0,
            request.request_id,
            request.transform_id,
            b"\xff\xd8native\xff\xd9",
        )
        child.sendall(encode_message(reply))

    responder = threading.Thread(target=serve_snapshot)
    responder.start()
    decider = _Decider()
    sink = _Sink()
    diagnostics = _Diagnostics()
    binding = SourceBinding(
        str(_BOOT),
        str(_CHILD),
        "camera-a",
        3,
        4,
        "seeon-perception-v1",
    )
    pump = NativePolicyPump(
        binding,
        NativePolicyContext(
            LatestMetadataSlot(),
            control,
            SceneState("camera-a"),
            EventAggregator((decider,), IncidentManager(0.0, tmp_path / "events.jsonl")),
            sink,
            AlertEvidenceAttacher({}),
            diagnostics,
            90,
            track_id_switch_absorbed_total=lambda _: 0,
        ),
    )

    # When
    pump._process(_metadata())  # noqa: SLF001

    # Then
    assert request_received.wait(timeout=1.0)
    responder.join(timeout=1.0)
    assert decider.received is not None
    assert decider.received.live_track_ids == (7,)
    assert decider.received.observation.track_ids == (7,)
    assert sink.emitted is not None
    event, trigger = sink.emitted
    assert event.snapshot_jpeg == b"\xff\xd8native\xff\xd9"
    assert trigger.source_generation == 3
    assert trigger.frame_key.stream_epoch == 4
    assert diagnostics.completed == 1
    assert diagnostics.polygon_sources == ["none"]
    control.close()
    child.close()


def test_native_policy_releases_a_failed_durable_admission_for_one_retry(
    tmp_path: Path,
) -> None:
    decider = _EpisodeDecider(
        FallPolicyDeciderV2(
            camera_id="camera-a",
            facility_id="facility-a",
            boot_id="boot",
            stream_epoch="epoch",
            source_generation=3,
            policy=FallPolicyV2(transition_votes=1, transition_window=1),
        )
    )
    sink = _FlakySink()
    pump = NativePolicyPump(
        _binding(generation=3, epoch=4),
        NativePolicyContext(
            LatestMetadataSlot(),
            _SnapshotUnavailable(),
            SceneState("camera-a"),
            EventAggregator((decider,), IncidentManager(300.0, tmp_path / "events.jsonl")),
            sink,
            AlertEvidenceAttacher({}),
            _Diagnostics(),
            90,
            track_id_switch_absorbed_total=lambda _: 0,
        ),
    )

    with pytest.raises(RuntimeError, match="durable staging failed"):
        pump._process(_metadata())  # noqa: SLF001
    pump._process(_metadata())  # noqa: SLF001

    assert sink.calls == 2
    assert sink.emitted is not None
    assert len(sink.emitted) == 1


def _binding(*, generation: int, epoch: int) -> SourceBinding:
    return SourceBinding(
        str(_BOOT),
        str(_CHILD),
        "camera-a",
        generation,
        epoch,
        "seeon-perception-v1",
    )


class _UnusedControl:
    """Rebinding never touches the child control channel."""

    def snapshot(self, camera_id: str) -> None:
        raise AssertionError(f"control must not be used while rebinding {camera_id}")


class _BrokenTraceWriter:
    def append(self, row: object) -> bool:
        del row
        raise OSError("read-only trace volume")


class _NoSnapshotControl:
    def snapshot(self, camera_id: str) -> None:
        assert camera_id == "camera-a"


def test_trace_write_failure_does_not_block_decision_or_sink(tmp_path: Path) -> None:
    decider = _Decider()
    sink = _Sink()
    diagnostics = _Diagnostics()
    pump = NativePolicyPump(
        _binding(generation=3, epoch=4),
        NativePolicyContext(
            LatestMetadataSlot(),
            _NoSnapshotControl(),  # pyright: ignore[reportArgumentType]
            SceneState("camera-a"),
            EventAggregator((decider,), IncidentManager(0.0, tmp_path / "events.jsonl")),
            sink,
            AlertEvidenceAttacher({}),
            diagnostics,
            90,
            replay_trace=_BrokenTraceWriter(),  # pyright: ignore[reportArgumentType]
            track_id_switch_absorbed_total=lambda _: 0,
        ),
    )

    pump._process(_metadata())  # noqa: SLF001

    assert decider.received is not None
    assert sink.emitted is not None
    assert diagnostics.completed == 1
    assert diagnostics.replay_trace_write_failures == 1


def _pump_for(
    slot: LatestMetadataSlot,
    binding: SourceBinding,
    tmp_path: Path,
) -> NativePolicyPump:
    return NativePolicyPump(
        binding,
        NativePolicyContext(
            slot,
            _UnusedControl(),  # pyright: ignore[reportArgumentType]
            SceneState("camera-a"),
            EventAggregator((_Decider(),), IncidentManager(0.0, tmp_path / "events.jsonl")),
            _Sink(),
            AlertEvidenceAttacher({}),
            _Diagnostics(),
            90,
            track_id_switch_absorbed_total=lambda _: 0,
        ),
    )


def test_pump_adopts_the_new_binding_after_a_source_rebuild(tmp_path: Path) -> None:
    """Regression: a rebuilt source must not starve the pump forever.

    ``SourceLifecycle.rebuild`` re-registers the slot with a fresh binding
    whose ``source_generation``/``stream_epoch`` have advanced. The slot then
    accepts frames against that new binding while the pump still matches
    against the binding it was constructed with, so ``wait_accepted`` never
    fires again. Nothing counts that pump-side refusal, so the symptom is a
    clean accept tally with every frame overwritten unread and zero decisions.
    """
    # Given
    slot = LatestMetadataSlot()
    original = _binding(generation=3, epoch=4)
    _ = slot.register_source(original)
    pump = _pump_for(slot, original, tmp_path)
    rebuilt = _binding(generation=4, epoch=5)
    _ = slot.register_source(rebuilt)
    # When
    pump._rebind_if_source_was_rebuilt()  # noqa: SLF001
    # Then
    assert pump._binding == rebuilt  # noqa: SLF001
    assert pump.camera_id == "camera-a"


def test_pump_keeps_its_binding_when_the_source_was_not_rebuilt(tmp_path: Path) -> None:
    # Given
    slot = LatestMetadataSlot()
    original = _binding(generation=3, epoch=4)
    _ = slot.register_source(original)
    pump = _pump_for(slot, original, tmp_path)
    # When
    pump._rebind_if_source_was_rebuilt()  # noqa: SLF001
    # Then
    assert pump._binding == original  # noqa: SLF001


def test_pump_keeps_its_binding_when_the_source_is_gone(tmp_path: Path) -> None:
    """A removed source must not clear the binding out from under the pump."""
    # Given
    slot = LatestMetadataSlot()
    original = _binding(generation=3, epoch=4)
    _ = slot.register_source(original)
    pump = _pump_for(slot, original, tmp_path)
    slot.remove_source("camera-a")
    # When
    pump._rebind_if_source_was_rebuilt()  # noqa: SLF001
    # Then
    assert pump._binding == original  # noqa: SLF001


class _FailingSnapshotControl:
    """The Flow plane raises a plain RuntimeError when no OSD encoder is wired."""

    def snapshot(self, camera_id: str) -> bytes:
        raise RuntimeError(f"OSD snapshot encoder is not configured for {camera_id}")


def test_snapshot_failure_stages_the_alert_without_one_instead_of_dropping_it(
    tmp_path: Path,
) -> None:
    """An admitted safety event outranks its thumbnail.

    Admission has already consumed the onset and the cooldown, so a snapshot
    failure must degrade to "no snapshot" and still reach the sink. Propagating
    it would lose the alert entirely, and the event would not even be released
    back to the decision for a retry.
    """
    decider = _Decider()
    sink = _Sink()
    diagnostics = _Diagnostics()
    pump = NativePolicyPump(
        _binding(generation=3, epoch=4),
        NativePolicyContext(
            LatestMetadataSlot(),
            _FailingSnapshotControl(),  # pyright: ignore[reportArgumentType]
            SceneState("camera-a"),
            EventAggregator((decider,), IncidentManager(0.0, tmp_path / "events.jsonl")),
            sink,
            AlertEvidenceAttacher({}),
            diagnostics,
            90,
            track_id_switch_absorbed_total=lambda _: 0,
        ),
    )

    pump._process(_metadata())  # noqa: SLF001

    assert sink.emitted is not None
    event, _trigger = sink.emitted
    assert event.snapshot_jpeg is None
    assert diagnostics.completed == 1
