from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from contracts.observation import BoundingBox
from contracts.replay_trace import ReplayRow, ReplayTraceHeader, decode_jsonl, encode_jsonl
from shared.detection_policies import BedExitPolicyV1, make_effective_policy
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.perception import SceneState
from worker.pipeline.trace.replay_trace_writer import ReplayTraceWriter
from worker.replay.engine import replay, replay_trace_frames
from worker.runtime.deepstream.native_policy_pump import NativePolicyContext, NativePolicyPump
from worker.runtime.flow.metadata_slot import LatestMetadataSlot
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.types import (
    AssociationResult,
    BedRegionChannel,
    BusinessEvent,
    ChannelState,
    HumanPoseChannel,
    Keypoint,
    NativeEvidenceTrigger,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBox,
    PersonBoxChannel,
)
from worker.types.metadata import MetadataFrame, SourceBinding

_BOOT = uuid.UUID("12345678-1234-5678-1234-567812345678")
_CHILD = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


class _Control:
    def snapshot(self, camera_id: str) -> bytes:
        raise AssertionError(f"unexpected snapshot for {camera_id}")


class _Sink:
    def emit_for_frame(self, event: BusinessEvent, trigger: NativeEvidenceTrigger) -> None:
        raise AssertionError(f"unexpected event {event!r} {trigger!r}")


def _metadata(
    *,
    epoch: int,
    track_id: int,
    generation: int,
    strategy: str = "legacy-greedy-bbox-iou.v1",
    live_track_ids: tuple[int, ...] | None = None,
) -> MetadataFrame:
    identity = PerceptionFrameIdentity(str(_BOOT), "camera-a", epoch, 1, 2_000_000_000)
    association = AssociationResult(
        strategy,
        (track_id,),
        (0,),
        identity,
        live_track_ids=(track_id,) if live_track_ids is None else live_track_ids,
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
            _Control(),
            SceneState("camera-a"),
            EventAggregator((), incidents),
            _Sink(),
            AlertEvidenceAttacher({}),
            diagnostics,
            90,
            replay_trace=writer,
            track_id_switch_absorbed_total=lambda _decision: 0,
        ),
    )

    pump._process(_metadata(epoch=4, track_id=7, generation=3))  # noqa: SLF001
    pump._process(_metadata(epoch=5, track_id=8, generation=4))  # noqa: SLF001

    trace_path = tmp_path / f"{hashlib.sha256(b'camera-a').hexdigest()[:16]}.jsonl"
    _, rows = decode_jsonl(trace_path.read_text())
    assert [row.source_event for row in rows] == ["open", "frame", "reconnect", "frame"]
    assert {row.source for row in rows} == {"legacy-association"}
    assert [row.seq for row in rows] == [0, 1, 2, 3]
    assert [track.lifecycle for track in rows[1].tracks] == ["new"]
    assert rows[3].tracks[0].lifecycle == "new"
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
    bounded_path = tmp_path / "bounded" / f"{hashlib.sha256(b'camera-a').hexdigest()[:16]}.jsonl"
    assert bounded_path.with_suffix(".jsonl.1").exists()
    assert bounded.written_rows_total == 2
    assert bounded.dropped_rows_total == 0
    dropping = ReplayTraceWriter(tmp_path / "dropping", "camera-a", max_bytes=1)
    assert not dropping.append(row)
    assert dropping.dropped_rows_total == 1


def test_flow_pump_captures_nvdcf_trace_source(tmp_path: Path) -> None:
    binding = SourceBinding(str(_BOOT), str(_CHILD), "camera-a", 3, 4, "seeon-perception-v1")
    writer = ReplayTraceWriter(tmp_path, "camera-a")
    pump = NativePolicyPump(
        binding,
        NativePolicyContext(
            LatestMetadataSlot(),
            _Control(),
            SceneState("camera-a"),
            EventAggregator((), IncidentManager(30.0)),
            _Sink(),
            AlertEvidenceAttacher({}),
            WorkerDiagnostics(),
            90,
            replay_trace=writer,
            track_id_switch_absorbed_total=lambda _decision: 0,
        ),
    )

    pump._process(_metadata(epoch=4, track_id=7, generation=3, strategy="nvdcf"))  # noqa: SLF001

    trace_path = tmp_path / f"{hashlib.sha256(b'camera-a').hexdigest()[:16]}.jsonl"
    _, rows = decode_jsonl(trace_path.read_text())
    assert {row.source for row in rows} == {"nvdcf"}


def test_writer_hashes_untrusted_camera_ids_beneath_root(tmp_path: Path) -> None:
    row = ReplayRow(
        "camera", 0, 0, 0, "frame", "legacy-association", (), None, None, None, False, 640, 360
    )
    for camera_id in ("a/b", "../outside", "/absolute", "카메라/../유니코드"):
        writer = ReplayTraceWriter(tmp_path, camera_id)
        assert writer.append(row)
        expected = tmp_path / f"{hashlib.sha256(camera_id.encode()).hexdigest()[:16]}.jsonl"
        assert expected.exists()
        assert expected.resolve().is_relative_to(tmp_path.resolve())


def test_capture_normalizes_persisted_polygon_using_its_source_size(tmp_path: Path) -> None:
    binding = SourceBinding(str(_BOOT), str(_CHILD), "camera-a", 3, 4, "seeon-perception-v1")
    writer = ReplayTraceWriter(tmp_path, "camera-a")
    scene = SceneState(
        "camera-a",
        persisted_bed_regions=(
            BoundingBox(0, 0, 1920, 1080, 1.0, ((0, 0), (1920, 0), (1920, 1080))),
        ),
        bed_zone_image_width=1920,
        bed_zone_image_height=1080,
    )
    pump = NativePolicyPump(
        binding,
        NativePolicyContext(
            LatestMetadataSlot(),
            _Control(),
            scene,
            EventAggregator((), IncidentManager(30.0)),
            _Sink(),
            AlertEvidenceAttacher({}),
            WorkerDiagnostics(),
            90,
            replay_trace=writer,
            track_id_switch_absorbed_total=lambda _decision: 0,
        ),
    )
    pump._process(_metadata(epoch=4, track_id=7, generation=3))  # noqa: SLF001
    _, rows = decode_jsonl(
        (tmp_path / f"{hashlib.sha256(b'camera-a').hexdigest()[:16]}.jsonl").read_text()
    )
    assert rows[1].bed_polygon == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


def test_rotation_chain_accepts_seq_restart_at_new_boot_segment(tmp_path: Path) -> None:
    binding = SourceBinding(str(_BOOT), str(_CHILD), "camera-a", 3, 4, "seeon-perception-v1")

    def pump() -> NativePolicyPump:
        return NativePolicyPump(
            binding,
            NativePolicyContext(
                LatestMetadataSlot(),
                _Control(),
                SceneState("camera-a"),
                EventAggregator((), IncidentManager(30.0)),
                _Sink(),
                AlertEvidenceAttacher({}),
                WorkerDiagnostics(),
                90,
                replay_trace=ReplayTraceWriter(tmp_path, "camera-a"),
                track_id_switch_absorbed_total=lambda _decision: 0,
            ),
        )

    first = pump()
    first._process(_metadata(epoch=4, track_id=7, generation=3))  # noqa: SLF001
    first._process(_metadata(epoch=4, track_id=7, generation=3))  # noqa: SLF001
    first._replay_trace._rotate()  # noqa: SLF001 - forced retained-chain boundary
    second = pump()
    second._process(_metadata(epoch=4, track_id=8, generation=3))  # noqa: SLF001
    second._process(_metadata(epoch=4, track_id=8, generation=3))  # noqa: SLF001

    trace_name = f"{hashlib.sha256(b'camera-a').hexdigest()[:16]}.jsonl"
    _, first_rows = decode_jsonl((tmp_path / f"{trace_name}.1").read_text())
    _, second_rows = decode_jsonl((tmp_path / trace_name).read_text())
    rows = first_rows + second_rows
    assert [(row.source_event, row.seq) for row in rows] == [
        ("open", 0),
        ("frame", 1),
        ("frame", 2),
        ("open", 0),
        ("frame", 1),
        ("frame", 2),
    ]
    # Replay no longer resamples here: it forwards each captured source row to
    # the production decider, which owns the single 15 fps cadence. Both frames
    # of each boot therefore survive, grouped and ordered by boot segment.
    frames = replay_trace_frames(rows)
    assert [frame.boot_segment for frame in frames] == [0, 0, 1, 1]
    assert [frame.seq for frame in frames] == [1, 2, 1, 2]
    policy = make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BedExitPolicyV1(min_containment=0.5, hold_frames=1, grace_frames=1),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    run = replay(camera_id="camera-a", rows=rows, module_id="bed_exit", policy=policy)
    # frame_key = (boot-scoped kind, camera, stream_epoch, resampled seq): unique per boot.
    assert [frame.frame_key[0] for frame in run.frames] == [
        "replay-trace-v2:boot-0",
        "replay-trace-v2:boot-0",
        "replay-trace-v2:boot-1",
        "replay-trace-v2:boot-1",
    ]
    assert run.boot_ids == ("boot-0", "boot-1")


def test_trace_lifecycle_uses_association_live_ids_for_shadow_and_lost(tmp_path: Path) -> None:
    binding = SourceBinding(str(_BOOT), str(_CHILD), "camera-a", 3, 4, "seeon-perception-v1")
    writer = ReplayTraceWriter(tmp_path, "camera-a")
    pump = NativePolicyPump(
        binding,
        NativePolicyContext(
            LatestMetadataSlot(),
            _Control(),  # pyright: ignore[reportArgumentType]
            SceneState("camera-a"),
            EventAggregator((), IncidentManager(30.0)),
            _Sink(),  # pyright: ignore[reportArgumentType]
            AlertEvidenceAttacher({}),
            WorkerDiagnostics(),
            90,
            replay_trace=writer,
            track_id_switch_absorbed_total=lambda _decision: 0,
        ),
    )
    pump._process(_metadata(epoch=4, track_id=7, generation=3))  # noqa: SLF001
    pump._process(  # noqa: SLF001
        _metadata(epoch=4, track_id=8, generation=3, live_track_ids=(7, 8))
    )
    pump._process(_metadata(epoch=4, track_id=8, generation=3, live_track_ids=(8,)))  # noqa: SLF001

    path = tmp_path / f"{hashlib.sha256(b'camera-a').hexdigest()[:16]}.jsonl"
    _, rows = decode_jsonl(path.read_text())
    assert [track.lifecycle for track in rows[2].tracks] == ["new", "shadow"]
    assert [track.lifecycle for track in rows[3].tracks] == ["tracked", "lost"]
