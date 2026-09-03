from __future__ import annotations

import uuid
from pathlib import Path

from contracts.replay_trace import ReplayTraceHeader, decode_jsonl, encode_jsonl
from worker.native.deepstream.ipc import MetadataFrame
from worker.native.deepstream.metadata import LatestMetadataSlot, SourceBinding
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.perception import SceneState
from worker.pipeline.trace.replay_trace_writer import ReplayTraceWriter
from worker.runtime.deepstream.native_policy_pump import NativePolicyContext, NativePolicyPump
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.types import (
    AssociationResult,
    BedRegionChannel,
    BusinessEvent,
    ChannelState,
    HumanPoseChannel,
    Keypoint,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBox,
    PersonBoxChannel,
)

_BOOT = uuid.UUID("12345678-1234-5678-1234-567812345678")
_CHILD = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


class _Control:
    def snapshot(self, camera_id: str) -> None:
        raise AssertionError(f"unexpected snapshot for {camera_id}")


class _Sink:
    def emit_for_frame(self, event: object, trigger: object) -> None:
        raise AssertionError(f"unexpected event {event!r} {trigger!r}")


def _metadata(*, epoch: int, track_id: int, generation: int) -> MetadataFrame:
    identity = PerceptionFrameIdentity(str(_BOOT), "camera-a", epoch, 1, 2_000_000_000)
    association = AssociationResult(
        "legacy-greedy-bbox-iou.v1", (track_id,), (0,), identity, live_track_ids=(track_id,)
    )
    pose = tuple(Keypoint(20, 30, 0.9) for _ in range(17))
    return MetadataFrame(
        PerceptionFrameV1(
            identity,
            PersonBoxChannel(ChannelState.INFERRED, (PersonBox(10, 20, 40, 80, 0.8),)),
            HumanPoseChannel(ChannelState.INFERRED, (pose,)),
            BedRegionChannel(ChannelState.INFERRED_EMPTY),
            association,
        ),
        generation,
        _CHILD,
        1,
        "seeon-perception-v1",
        640,
        360,
        2_100_000_000,
    )


def test_native_pump_captures_epoch_rows_and_writer_rotates(tmp_path: Path) -> None:
    binding = SourceBinding(str(_BOOT), str(_CHILD), "camera-a", 3, 4, "seeon-perception-v1")
    diagnostics = WorkerDiagnostics()
    writer = ReplayTraceWriter(tmp_path, "camera-a")
    incidents = IncidentManager(30.0)
    pump = NativePolicyPump(
        binding,
        NativePolicyContext(
            LatestMetadataSlot(),
            _Control(),  # pyright: ignore[reportArgumentType]
            SceneState("camera-a"),
            EventAggregator((), incidents),
            _Sink(),  # pyright: ignore[reportArgumentType]
            AlertEvidenceAttacher({}),
            diagnostics,
            90,
            replay_trace=writer,
        ),
    )

    pump._process(_metadata(epoch=4, track_id=7, generation=3))  # noqa: SLF001
    pump._process(_metadata(epoch=5, track_id=8, generation=4))  # noqa: SLF001

    _, rows = decode_jsonl((tmp_path / "camera-a.jsonl").read_text())
    assert [row.source_event for row in rows] == ["open", "reconnect"]
    assert [track.lifecycle for track in rows[0].tracks] == ["new"]
    assert rows[1].tracks[0].lifecycle == "new"
    diagnostics.update_measured_fps("camera-a", 17.0)
    diagnostics.register_incident_manager("camera-a", incidents)
    event = BusinessEvent("fall", "fall", "source", "camera-a", "facility-a", 1.0, 0.9)
    assert incidents.admit(event, now_sec=1.0) is not None
    assert incidents.admit(event, now_sec=2.0) is None
    snapshot = diagnostics.snapshot().cameras[0]
    assert snapshot.track_id_switch_total == 1
    assert snapshot.bed_polygon_source == "none"
    assert snapshot.inference_fps == 17.0
    assert snapshot.camera_fps_unpinned
    assert snapshot.incident_cooldown_suppressed_total == 1

    row = rows[0]
    header_size = len(encode_jsonl(ReplayTraceHeader(), []))
    row_size = len(encode_jsonl(ReplayTraceHeader(), [row])) - header_size
    bounded = ReplayTraceWriter(tmp_path / "bounded", "camera-a", max_bytes=header_size + row_size)
    assert bounded.append(row)
    assert bounded.append(row)
    assert (tmp_path / "bounded" / "camera-a.jsonl.1").exists()
    assert bounded.written_rows_total == 2
    assert bounded.dropped_rows_total == 0
    dropping = ReplayTraceWriter(tmp_path / "dropping", "camera-a", max_bytes=1)
    assert not dropping.append(row)
    assert dropping.dropped_rows_total == 1
