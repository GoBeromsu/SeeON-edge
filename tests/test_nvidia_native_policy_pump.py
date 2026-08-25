from __future__ import annotations

import socket
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

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
class _Diagnostics:
    completed: int = 0

    def update_measured_fps(self, camera_id: str, measured_fps: float | None) -> None:
        del camera_id, measured_fps

    def record_detection_completed(self, camera_id: str) -> None:
        assert camera_id == "camera-a"
        self.completed += 1


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
                ((Keypoint(20, 30, 0.9),),),
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
    control.close()
    child.close()
