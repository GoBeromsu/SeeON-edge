from dataclasses import replace
from pathlib import Path

from contracts.replay_trace import decode_document
from shared.detection_policies import BedExitPolicyV1, FallPolicyV2, make_effective_policy
from worker.domains.fall import FallPolicyDeciderV2, FallV2DomainDecider, FallV2Probabilities
from worker.domains.fall.classifier_v2 import FallWindowClassifierV2
from worker.replay.engine import replay, replay_trace_frames
from worker.replay.inputs import replay_trace_to_decision_input

FIXTURES = Path("tests/fixtures/replay")


def _rows(name: str):
    return decode_document((FIXTURES / f"{name}.json").read_text())[1]


def _policy():
    return make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BedExitPolicyV1(min_containment=0.5, hold_frames=1, grace_frames=1),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


def _run(name: str):
    return replay(camera_id="fixture", rows=_rows(name), module_id="bed_exit", policy=_policy())


def _alerts(run):
    return [(frame.pts_ns, event.event_type) for frame in run.frames for event in frame.events]


def test_pts_gap_changes_the_full_replay_episode_outcome() -> None:
    control = _run("gap-control-v2")
    gap = _run("gap-axis-v2")
    assert _alerts(control) == [(200_000_001, "bed-exit")]
    assert _alerts(gap) == []
    assert all(frame.valid for frame in gap.frames)
    assert gap.resample_gap_rows_total == 0
    assert [frame.valid for frame in replay_trace_frames(_rows("gap-axis-v2"))] == [1, 1, 1]


def test_reconnect_recreates_camera_local_decider_state_within_one_boot() -> None:
    control = _run("reconnect-control-v2")
    reconnected = _run("reconnect-axis-v2")
    assert _alerts(control) == [(200_000_001, "bed-exit")]
    assert _alerts(reconnected) == []
    assert [frame.stream_epoch for frame in reconnected.frames] == [0, 0, 1, 1, 1]


def test_control_rows_are_not_replay_frames() -> None:
    _, rows = decode_document(
        Path("tests/fixtures/replay/reconnect-axis-v2.json").read_text(encoding="utf-8")
    )
    frames = replay_trace_frames(rows)
    assert all(frame.row is None or frame.row.source_event == "frame" for frame in frames)


def test_id_switch_reports_churn_and_changes_declared_episode_outcome() -> None:
    control = _run("id-switch-control-v2")
    switched = _run("id-switch-axis-v2")
    assert _alerts(control) == [(200_000_001, "bed-exit")]
    assert _alerts(switched) == [(266_666_668, "bed-exit")]
    assert switched.track_id_switch_total == 1


def _open_row(template, *, seq: int, epoch: int):
    from dataclasses import replace

    return replace(
        template,
        seq=seq,
        epoch=epoch,
        source_event="open",
        tracks=(),
        bed_polygon_id=None,
        bed_polygon=None,
        bed_polygon_image_size=None,
    )


def _with_open(rows):
    """Prefix a producer-shaped open control row (fixtures start truncated)."""
    from dataclasses import replace

    first = rows[0]
    return (
        _open_row(first, seq=0, epoch=first.epoch),
        *(replace(row, seq=row.seq + 1) for row in rows),
    )


def _rebooted(rows, *, epoch_offset: int = 0):
    """Append the same frames again as a second boot: open(seq 0) then re-sequenced frames."""
    from dataclasses import replace

    frames = [row for row in rows if row.source_event == "frame"]
    reboot = [_open_row(frames[0], seq=0, epoch=frames[0].epoch + epoch_offset)]
    reboot.extend(
        replace(row, seq=index, epoch=row.epoch + epoch_offset)
        for index, row in enumerate(frames, start=1)
    )
    return tuple(rows) + tuple(reboot)


def test_second_boot_starts_with_fresh_cooldown_and_distinct_boot_identity() -> None:
    """Two boots that each contain the same exit episode both alert: cooldown never leaks across
    a worker boot, boot_ids come from open rows (not epochs), and frame keys stay unique."""
    rows = _rebooted(_with_open(_rows("reconnect-control-v2")))
    run = replay(camera_id="fixture", rows=rows, module_id="bed_exit", policy=_policy())
    assert run.boot_ids == ("boot-0", "boot-1")
    assert [event.event_type for frame in run.frames for event in frame.events] == [
        "bed-exit",
        "bed-exit",
    ]
    assert run.incident_cooldown_suppressed_total == 0
    keys = [frame.frame_key for frame in run.frames]
    assert len(keys) == len(set(keys))


def test_reconnect_epochs_within_one_boot_keep_unique_frame_keys_and_one_boot_id() -> None:
    run = _run("reconnect-axis-v2")
    keys = [frame.frame_key for frame in run.frames]
    assert len(keys) == len(set(keys))
    assert {key[0] for key in keys} == {"replay-trace-v2:boot-0"}
    assert run.boot_ids == ("boot-0",)
    assert {frame.stream_epoch for frame in run.frames} == {0, 1}


def test_truncated_prefix_forms_its_own_segment_before_the_first_open() -> None:
    from worker.replay.engine import boot_segments

    rows = _rows("reconnect-control-v2")
    frames_only = tuple(row for row in rows if row.source_event == "frame")
    truncated = _rebooted(frames_only)  # no open before the retained tail
    assert boot_segments(truncated)[: len(frames_only)] == (0,) * len(frames_only)
    assert set(boot_segments(truncated)[len(frames_only) :]) == {1}
    run = replay(camera_id="fixture", rows=truncated, module_id="bed_exit", policy=_policy())
    assert run.boot_ids == ("boot-0", "boot-1")


class _RecordingFallModel:
    def __init__(self) -> None:
        self.windows: list[tuple[tuple[float, ...], ...]] = []

    def predict(self, window: tuple[tuple[float, ...], ...]) -> FallV2Probabilities:
        self.windows.append(window)
        return FallV2Probabilities(0.1, 0.1, 0.8)


def _fall_policy():
    return make_effective_policy(
        module_id="fall",
        module_version=2,
        values=FallPolicyV2(transition_threshold=0.7),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


def test_fall_replay_feeds_production_byte_identical_windows_including_gaps() -> None:
    template = _rows("gap-axis-v2")[0]
    rows = tuple(
        replace(
            template,
            pts_ns=index * 66_666_667 + (2 * 66_666_667 if index >= 10 else 0),
            seq=index,
        )
        for index in range(32)
    )
    production_model = _RecordingFallModel()
    production = FallV2DomainDecider(
        classifier=FallWindowClassifierV2(production_model),
        policy=FallPolicyDeciderV2(
            camera_id="fixture",
            facility_id="replay",
            boot_id="boot-0",
            stream_epoch="0",
            source_generation=0,
        ),
    )
    for row in rows:
        production.update(replay_trace_to_decision_input(row, pts_ns=row.pts_ns, seq=row.seq))

    replay_model = _RecordingFallModel()
    run = replay(
        camera_id="fixture",
        rows=rows,
        module_id="fall",
        policy=_fall_policy(),
        fall_model=replay_model,
    )

    assert replay_model.windows == production_model.windows
    assert run.resample_gap_rows_total == 2


def test_fall_pts_rollback_resets_the_partial_classifier_window() -> None:
    template = _rows("gap-control-v2")[0]
    model = _RecordingFallModel()
    decider = FallV2DomainDecider(
        classifier=FallWindowClassifierV2(model),
        policy=FallPolicyDeciderV2(
            camera_id="fixture",
            facility_id="replay",
            boot_id="boot-0",
            stream_epoch="0",
            source_generation=0,
        ),
    )
    for sequence in range(30):
        row = replace(template, pts_ns=sequence * 66_666_667, seq=sequence)
        decider.update(replay_trace_to_decision_input(row, pts_ns=row.pts_ns, seq=row.seq))
    assert len(model.windows) == 1

    for sequence in range(29):
        row = replace(template, pts_ns=sequence * 66_666_667, seq=sequence)
        decider.update(replay_trace_to_decision_input(row, pts_ns=row.pts_ns, seq=row.seq))
    assert len(model.windows) == 1

    row = replace(template, pts_ns=29 * 66_666_667, seq=29)
    decider.update(replay_trace_to_decision_input(row, pts_ns=row.pts_ns, seq=row.seq))
    assert len(model.windows) == 2
