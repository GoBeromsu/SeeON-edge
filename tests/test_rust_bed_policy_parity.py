"""Scored CPU differential against the real BedExitMonitor/EpisodeAuthority.

Parent builds bed_policy_probe; SEEON_TEST_BED_POLICY_PROBE opts in. Absent env
skips, configured failure fails. No build, model, GPU or geometry parity claim.
The Rust seam consumes scores from REAL containment_ratio, including polygons.
Prior-box rows are a complete set from observation history, NOT reference
assignments, decisions, recipients or traces. Reference internals are read ONLY
for output comparison (Python has no public assignment/scoring snapshot).

Admission: finite f64, u64 IDs/counters, i64 frames/coordinates, 64 collection
entries/retained identities/episodes, 128 calls, 128 KiB request, 4 MiB response,
1024-byte text. Rust-only rejection/fatal conservation is tested separately and
is not equivalence for Python's wider/unbounded domain. All event fields, trace
fields/types/bits/order, assignment fields, debug fields, scoring callbacks and
counters, recovery/lost lists and watched episode states are compared. Private
unwatched episode rows and Rust-only error/complete flags are not Python APIs.
"""

from __future__ import annotations

import os
import selectors
import struct
import subprocess
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from time import monotonic

import pytest

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains import detection_window as canonical
from worker.domains.bed_exit.detector import BedExitMonitor
from worker.domains.bed_exit.geometry import containment_ratio
from worker.domains.bed_exit.schema import BedExitConfig
from worker.domains.episode import EpisodeAuthority
from worker.types import BusinessEvent, DecisionInput
from worker.types.bed_pose_features import BedPoseFeatures, FrameBedPoseFeatures

_HEADER = "BEDPROBE\t1"
_IDS = ("synthetic-camera\t가", "synthetic-facility\né", "synthetic-boot:β", "synthetic-epoch\0끝")
_MAX_INPUT, _MAX_OUTPUT = 128 * 1024, 4 * 1024 * 1024
_BED = BoundingBox(0, 0, 100, 100, 0.99)
_OTHER = BoundingBox(200, 0, 300, 100, 0.98)
_IN = BoundingBox(10, 10, 30, 30, 0.75)
_OUT = BoundingBox(110, 10, 130, 30, 0.5)
_FAR = BoundingBox(400, 10, 420, 30, 0.25)
_WATCH = tuple((track, bed) for track in (0, 1, 2, 4, 7, 8, 9, 99) for bed in (0, 1))


@pytest.fixture
def bed_probe() -> str:
    configured = os.environ.get("SEEON_TEST_BED_POLICY_PROBE")
    if configured is None:
        pytest.skip("bed probe not configured; skip is not differential qualification")
    if not configured:
        pytest.fail("configured bed probe must name an executable", pytrace=False)
    try:
        path = Path(configured).resolve()
        usable = path.is_file() and os.access(path, os.X_OK)
    except (OSError, RuntimeError):
        pytest.fail("configured bed probe path is unusable", pytrace=False)
    if not usable:
        pytest.fail("configured bed probe is missing or unexecutable", pytrace=False)
    return str(path)


def _invoke(probe, payload, *, env=None, args=()):
    assert len(payload) <= _MAX_INPUT + 1
    try:
        process = subprocess.Popen(
            [probe, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        with process, selectors.DefaultSelector() as poller:
            try:
                assert (
                    process.stdin is not None
                    and process.stdout is not None
                    and process.stderr is not None
                )
                streams = (process.stdin, process.stdout, process.stderr)
                buffers = (bytearray(), bytearray())
                limits = (_MAX_OUTPUT, 1024)
                for index, stream in enumerate(streams):
                    os.set_blocking(stream.fileno(), False)
                    poller.register(
                        stream, selectors.EVENT_WRITE if index == 0 else selectors.EVENT_READ, index
                    )
                offset, deadline = 0, monotonic() + 5.0
                while poller.get_map():
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(probe, 5.0)
                    for key, _ in poller.select(remaining):
                        stream, index = key.fileobj, key.data
                        if index == 0:
                            try:
                                offset += os.write(stream.fileno(), payload[offset : offset + 4096])
                            except BrokenPipeError:
                                offset = len(payload)
                            if offset == len(payload):
                                poller.unregister(stream)
                                stream.close()
                        else:
                            buffer, limit = buffers[index - 1], limits[index - 1]
                            chunk = os.read(stream.fileno(), min(65536, limit - len(buffer) + 1))
                            if not chunk:
                                poller.unregister(stream)
                                stream.close()
                            else:
                                buffer.extend(chunk)
                                if len(buffer) > limit:
                                    pytest.fail(
                                        "configured bed probe exceeded output bounds", pytrace=False
                                    )
                code = process.wait(timeout=max(0.001, deadline - monotonic()))
                return subprocess.CompletedProcess(
                    [probe, *args], code, bytes(buffers[0]), bytes(buffers[1])
                )
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
    except (OSError, subprocess.TimeoutExpired):
        pytest.fail("configured bed probe failed to execute within its bound", pytrace=False)


def _hex(text):
    return str(text).encode("utf-8").hex()


def _num(value):
    assert type(value) in (int, float)
    return struct.pack(">d", value).hex() if type(value) is float else str(value)


def _opt(value):
    return "-" if value is None else _num(value)


def _line(*parts):
    return "\t".join(map(str, parts))


def _payload(rows):
    return ("\n".join(rows) + "\n").encode("ascii")


def _box(box):
    polygon = (
        "-"
        if box.polygon is None
        else _line(len(box.polygon), *(coordinate for point in box.polygon for coordinate in point))
    )
    return _line(box.x1, box.y1, box.x2, box.y2, _num(box.confidence), polygon)


def _collection(values, encode=str):
    values = tuple(values)
    return _line(len(values), *(encode(value) for value in values))


def _target_tzinfo(window):
    # Resolve through the canonical module's fixture-injected real TZif lookup.
    # Never infer identity from a zone key, offset, membership or monitor state.
    return None if window is None else canonical.ZoneInfo(window.tz)


def _wall(now, *, actual_target_tzinfo):
    if now is None:
        return "-"
    offset = now.utcoffset()
    seconds = None if offset is None else offset.total_seconds()
    assert seconds is None or seconds.is_integer()
    return _line(
        "S" if actual_target_tzinfo is not None and now.tzinfo is actual_target_tzinfo else "E",
        _hex(now.replace(tzinfo=None).isoformat(timespec="microseconds")),
        "-" if seconds is None else int(seconds),
    )


def _window(window):
    return "-" if window is None else _line(*map(_hex, (window.start, window.end, window.tz)))


def _event(event: BusinessEvent):
    return _line(
        *map(
            _hex,
            (event.domain, event.event_type, event.identity, event.camera_id, event.facility_id),
        ),
        _num(event.time_sec),
        _opt(event.probability),
        _opt(event.person_id),
        _opt(event.bed_id),
    )


def _config(**changes):
    return BedExitConfig(
        camera_id=_IDS[0],
        facility_id=_IDS[1],
        min_containment=0.5,
        hold_frames=1,
        in_bed_dwell_sec=2.0,
        outside_dwell_sec=2.0,
        **changes,
    )


def _new(config, *, capacity=64, watch=_WATCH):
    return _line(
        "N",
        _hex(config.camera_id),
        _hex(config.facility_id),
        _hex(_IDS[2]),
        _hex(_IDS[3]),
        7,
        _num(config.min_containment),
        config.hold_frames,
        config.grace_frames,
        _num(config.in_bed_dwell_sec),
        _num(config.outside_dwell_sec),
        _num(3.0),
        capacity,
        _window(config.night_window),
        _collection(watch, lambda pair: _line(*pair)),
    )


def _pose(track=7, **changes):
    return replace(
        BedPoseFeatures(track, 0, 0.7, 0.6, 0.8, 0.1, 1.5, 0.0, 0.4, 0.3, 0.35, True), **changes
    )


@dataclass(frozen=True)
class _Update:
    input: DecisionInput
    observed: float
    snapshot: float
    wall: datetime | None = None


def _u(
    time,
    persons=((7, _IN),),
    *,
    frame=None,
    beds=(_BED,),
    live=None,
    poses=None,
    region="fresh",
    cycles=0,
    observed=None,
    snapshot=None,
    wall=None,
    positional=False,
):
    ids = tuple(track for track, _ in persons)
    effective = tuple(range(len(persons))) if positional else ids
    clock = 0.0 if time is None else float(time)
    observed = clock if observed is None else observed
    return _Update(
        DecisionInput(
            observation=FrameObservation(
                detections=(tuple(box for _, box in persons), ()),
                regions=(beds, ()),
                track_ids=() if positional else ids,
            ),
            frame_width=640,
            frame_height=480,
            live_track_ids=tuple(track for track in effective if track is not None)
            if live is None
            else live,
            time_sec=None if time is None else float(time),
            frame_index=int(clock) if frame is None else frame,
            bed_region=BedRegionDebugSnapshot(BedRegionCacheState(region), cycles),
            bed_pose_features=FrameBedPoseFeatures(
                tuple(_pose(track) for track in effective if track is not None)
                if poses is None
                else poses
            ),
        ),
        observed,
        observed if snapshot is None else snapshot,
        wall,
    )


class _History:
    """Observation-only geometry ledger. Never receives a monitor/reference result."""

    def __init__(self):
        self.boxes = {}

    def line(self, step, *, actual_target_tzinfo, geometry=True, overlaps=True):
        i, o = step.input, step.input.observation
        ids = o.track_ids or tuple(range(len(o.boxes)))
        live = set(i.live_track_ids) if o.track_ids else set(ids)

        def pose(p):
            return _line(
                p.track_id,
                _opt(p.bed_id),
                *(_num(getattr(p, field.name)) for field in fields(p)[2:-1]),
                int(p.bed_polygon_valid),
            )

        def ratios(boxes):
            return _collection(boxes, lambda b: _num(containment_ratio(b[0], b[1])))

        # All stale observation-history rows against ALL current boxes. Liveness
        # comes from the request, not assignments, posture or handoff decisions.
        prior = (
            tuple((track, box) for track, box in sorted(self.boxes.items()) if track not in live)
            if overlaps
            else ()
        )
        row = _line(
            "U",
            i.frame_index,
            _opt(i.time_sec),
            _num(step.observed),
            _num(step.snapshot),
            _wall(step.wall, actual_target_tzinfo=actual_target_tzinfo),
            _hex(i.bed_region.source),
            i.bed_region.empty_cycles,
            _collection(o.boxes, _box),
            _collection(o.bed_boxes, _box),
            _collection(o.track_ids, _opt),
            _collection(i.live_track_ids),
            _collection(i.bed_pose_features.items, pose),
            _collection(
                o.boxes if geometry else (), lambda b: ratios((b, bed) for bed in o.bed_boxes)
            ),
            _collection(
                prior, lambda p: _line(p[0], _box(p[1]), ratios((b, p[1]) for b in o.boxes))
            ),
        )
        if i.bed_region.source in ("fresh", "cached") and o.bed_boxes:
            self.boxes = {track: box for track, box in self.boxes.items() if track in live}
            for track, box in zip(ids, o.boxes, strict=True):
                if track is not None and track in live:
                    self.boxes[track] = box
        return row


@dataclass(frozen=True)
class _Coast:
    snapshot: float
    frame: int | None = None


@dataclass(frozen=True)
class _Release:
    index: int
    mode: str = "exact"


class _Clock:
    def __init__(self):
        self.samples = iter(())
        self.wall = None

    def monotonic(self):
        return next(self.samples)

    def aware(self):
        assert self.wall is not None, "window must use an explicitly injected clock"
        return self.wall


class _Recorder:
    def __init__(self):
        self.rows = []

    def record_bed_exit_scoring(self, *values):
        self.rows.append(values)


def _render(index, m, events, scores, complete, watch):
    # Reading private Python fields here is observation only, never Rust input.
    d = m.last_debug_snapshot
    disposition = m._episodes.last_disposition
    rows = [
        _line(
            "O",
            index,
            m.track_id_switch_absorbed_total,
            m.last_shadow_trace_count,
            int(complete),
            int(bool(scores)),
            len(events),
            len(m.last_trace_snapshots),
            len(m._assignments),
            len(m._recovery_events),
            len(m._lost_track_ids),
            int(d is not None),
            "-" if disposition is None else _hex(disposition),
        )
    ]
    rows.extend(_line("E", _event(event)) for event in events)
    for t in m.last_trace_snapshots:
        rows.append(
            _line(
                "T",
                _hex(t.reason),
                _hex(t.previous_state),
                _hex(t.current_state),
                int(t.triggered),
                _opt(t.track_id),
                _opt(t.bed_id),
                len(t.values),
                len(t.missing_values),
            )
        )
        rows.extend(
            _line("V", _hex(name), "I" if type(value) is int else "F", _num(value))
            for name, value in sorted(t.values.items())
        )
        rows.extend(
            _line("M", _hex(name), _hex(reason))
            for name, reason in sorted(t.missing_values.items())
        )
    rows.append(
        _line(
            "G",
            _num(m._max_containment_observed),
            m._grace_positive_transitions,
            m._assignments_made,
        )
    )
    for camera, maximum, transitions, assignments in scores:
        assert camera == m.config.camera_id
        rows.append(_line("Q", _num(maximum), transitions, assignments))
    for track, a in sorted(m._assignments.items()):
        rows.append(
            _line(
                "A",
                track,
                _opt(a.bed_id),
                _opt(a.candidate_bed_id),
                a.candidate_frames,
                _num(a.in_bed_dwell_sec),
                _num(a.outside_dwell_sec),
                int(a.armed),
                _opt(a.last_time_sec),
                "-" if a.last_box is None else _line(1, _box(a.last_box)),
            )
        )
    rows.extend(_line("H", e.person_id, e.bed_id) for e in m._recovery_events)
    rows.extend(_line("L", track) for track in m._lost_track_ids)
    if d is not None:
        region = (
            "-"
            if d.bed_region is None
            else _line(_hex(d.bed_region.source), d.bed_region.empty_cycles)
        )
        rows.append(
            _line(
                "D",
                _opt(d.frame_index),
                int(d.stale),
                _opt(d.observation_age_sec),
                region,
                len(d.person_boxes),
                len(d.bed_boxes),
                len(d.statuses),
                len(d.events),
            )
        )
        rows.extend(_line("P", _box(box)) for box in d.person_boxes)
        rows.extend(_line("B", _box(box)) for box in d.bed_boxes)
        rows.extend(
            _line("S", s.bed_id, _hex(s.occupancy), _opt(s.person_id), _box(s.box))
            for s in d.statuses
        )
        rows.extend(_line("J", e.person_id, e.bed_id) for e in d.events)
    rows.extend(
        _line(
            "Z",
            track,
            bed,
            _hex(
                m._episodes.state_for(
                    camera_id=m.config.camera_id, event_type="bed-exit", bed_id=bed, track_id=track
                )
            ),
        )
        for track, bed in watch
    )
    return rows


def _exercise(probe, steps, config=None, *, watch=_WATCH):
    assert len(steps) < 128
    config = _config() if config is None else config
    clock, recorder, history = _Clock(), _Recorder(), _History()
    m = BedExitMonitor(
        config=config,
        clock=clock.aware,
        staleness_clock=clock.monotonic,
        scoring_recorder=recorder,
        boot_id=_IDS[2],
        stream_epoch=_IDS[3],
        source_generation=7,
    )
    assert isinstance(m._episodes, EpisodeAuthority)
    requests = [_HEADER, _new(config, watch=watch)]
    expected = [_HEADER, *_render(0, m, (), (), False, watch)]
    active_window = config.night_window  # Input-only config ledger, not m.config.
    emitted, observations, complete = [], [], False
    for index, step in enumerate(steps, 1):
        before = len(recorder.rows)
        if isinstance(step, _Update):
            requests.append(
                history.line(step, actual_target_tzinfo=_target_tzinfo(active_window))
            )  # Built BEFORE evaluating Python.
            clock.samples, clock.wall = iter((step.observed, step.snapshot)), step.wall
            events = m.update(step.input)
            complete = True
        elif isinstance(step, _Coast):
            requests.append(_line("C", _opt(step.frame), _num(step.snapshot)))
            clock.samples = iter((step.snapshot,))
            events = m.coast(frame_index=step.frame)
            complete = True
        elif isinstance(step, _Release):
            event = emitted[step.index]
            if step.mode == "foreign":
                event = replace(event, identity=f"{event.identity}:foreign")
            elif step.mode == "metadata":
                event = replace(
                    event,
                    domain="different",
                    event_type="different",
                    camera_id="different",
                    facility_id="different",
                    time_sec=77.0,
                    probability=0.2,
                    person_id=99,
                    bed_id=42,
                )
            else:
                assert step.mode == "exact"
            requests.append(_line("R", _event(event)))
            m.release_onset(event)
            events = ()
        else:
            assert step is None or isinstance(step, canonical.DetectionWindow)
            requests.append(_line("W", _window(step)))
            active_window = step
            m.update_night_window(step)
            events = ()
        emitted.extend(events)
        rows = _render(index, m, events, recorder.rows[before:], complete, watch)
        expected.extend(rows)
        observations.append((events, m.last_trace_snapshots, m.last_debug_snapshot, rows))
    expected.append(_line("END", len(steps) + 1))
    result = _invoke(probe, _payload(requests))
    assert result.returncode == 0 and result.stderr == b"", result.stderr
    assert result.stdout.splitlines(keepends=True) == _payload(expected).splitlines(keepends=True)
    return observations


@pytest.mark.parametrize(
    "hip,observability,valid",
    [
        (0.1, 0.35, True),
        (0.09999999999999999, 0.35, True),
        (0.1, 0.3499999999999999, True),
        (0.1, 0.35, False),
        (0.11, 0.36, True),
    ],
)
def test_inclusive_posture_thresholds(bed_probe, hip, observability, valid):
    pose = (_pose(hip_depth=hip, observability=observability, bed_polygon_valid=valid),)
    observed = _exercise(
        bed_probe, [_u(0), _u(2, poses=pose), _u(4, ((7, _OUT),)), _u(7, ((7, _OUT),))]
    )
    assert sum(len(row[0]) for row in observed) == int(
        valid and hip >= 0.1 and observability >= 0.35
    )


@pytest.mark.parametrize("x", [89, 90, 91])
def test_inclusive_geometry_gate(bed_probe, x):
    box = BoundingBox(x, 10, x + 20, 30, 0.9)
    _exercise(bed_probe, [_u(0, ((7, box),)), _u(2, ((7, box),)), _u(4, ((7, _OUT),))])


def test_hold_candidate_dwell_reset_armed_retention_and_other_bed(bed_probe):
    config = replace(_config(), hold_frames=2)
    other_person = BoundingBox(210, 10, 230, 30, 0.9)
    steps = [
        _u(t, ((7, box),), beds=(_BED, _OTHER), poses=poses)
        for t, box, poses in [
            (0, _IN, None),
            (1, _OUT, None),
            (2, _IN, None),
            (3, _IN, None),
            (4, _IN, None),
            (5, _IN, ()),
            (6, _IN, None),
            (7, _IN, None),
            (8, _IN, ()),
            (9, other_person, None),
            (10, _OUT, None),
            (11, _IN, ()),
            (12, _OUT, None),
            (13, _OUT, None),
            (14, _OUT, None),
        ]
    ]
    rows = _exercise(bed_probe, steps, config)
    assert [index for index, row in enumerate(rows) if row[0]] == [13]
    assert any(t.reason == "contained-in-other-bed" for row in rows for t in row[1])


@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_observations_features_and_tie_order(bed_probe, reverse):
    persons = ((9, _IN), (2, _IN), (9, _OUT), (None, _FAR), (99, _IN))
    if reverse:
        persons = tuple(reversed(persons))
    features = (_pose(9, hip_depth=-1.0), _pose(9), _pose(2), _pose(99))
    _exercise(
        bed_probe,
        [
            _u(t, persons, beds=(_BED, _BED), poses=features, live=(9, 2, 9))
            for t in (0, 1, 2, 4, 6)
        ],
    )


def test_positional_ids_ignore_explicit_live_list_and_missing_pose(bed_probe):
    _exercise(
        bed_probe,
        [
            _u(0, ((None, _IN), (None, _OUT)), positional=True, live=(99,)),
            _u(2, ((None, _IN),), positional=True, poses=()),
            _u(4, ((None, _IN),), positional=True),
            _u(6, ((None, _OUT),), positional=True),
        ],
    )


def test_cache_early_returns_coast_freshness_and_reversed_clock(bed_probe):
    _exercise(
        bed_probe,
        [
            _Coast(0.0),
            _u(0, observed=10.0, snapshot=9.0),
            _u(2, region="cached", cycles=1, observed=10.0, snapshot=13.0),
            _Coast(12.999999, -1),
            _Coast(13.0),
            _Coast(9.0),
            _u(4, ((7, _OUT),), region="expired", cycles=2),
            _Coast(20.0),
            _u(5, region="empty", cycles=3),
            _u(6, beds=()),
            _u(7, ((7, _OUT),)),
            _u(8, (), live=(7,)),
            _u(9, (), live=()),
        ],
    )


def test_missing_signed_reversed_pts_and_coast_do_not_move_anchor(bed_probe):
    _exercise(
        bed_probe,
        [
            _u(None),
            _u(-2),
            _u(-1),
            _u(None, poses=()),
            _u(-5),
            _u(-3),
            _u(None, ((7, _OUT),)),
            _Coast(10.0),
            _u(-2, ((7, _OUT),)),
            _u(None, ((7, _OUT),)),
            _u(-1, ((7, _OUT),)),
        ],
    )


def test_live_unobserved_track_and_coast_retain_the_actual_pts_anchor(bed_probe):
    rows = _exercise(
        bed_probe,
        [
            _u(0),
            _u(2),
            _u(3, ((7, _OUT),)),
            _u(100, ((None, _FAR),), live=(7,), poses=()),
            _Coast(103.0),
            _u(4, ((7, _OUT),)),
        ],
    )
    assert rows[3][1][0].reason == "person-observation-missing"
    assert [index for index, row in enumerate(rows) if row[0]] == [5]
    assert rows[5][0][0].time_sec == 4.0


@pytest.mark.parametrize("edge", [1.9999999999999998, 2.0, 2.0000000000000004])
def test_unrounded_dwell_edges_and_outside_interruption(bed_probe, edge):
    _exercise(
        bed_probe,
        [
            _u(0),
            _u(edge),
            _u(edge + 1.0, ((7, _OUT),)),
            _u(edge + 2.0, poses=()),
            _u(edge + 3.0, ((7, _OUT),)),
            _u(edge + 4.0, ((7, _OUT),)),
        ],
    )
    # The reference retains in-bed progress across an outside observation;
    # the harness follows the implementation, not a copied continuous-dwell rule.
    _exercise(bed_probe, [_u(0), _u(1), _u(2, ((7, _OUT),)), _u(3), _u(3.0 + edge, ((7, _OUT),))])


@pytest.mark.parametrize("frame", [-(2**63), 2**63 - 5])
def test_integer_boundaries_and_unused_pose_scalars_are_preserved(bed_probe, frame):
    track = 2**64 - 1
    pose = _pose(
        track,
        bed_id=None,
        torso_in_frac=-1.0,
        lower_in_frac=2.0,
        torso_angle=-4.0,
        hip_x_rel=9.0,
        centroid_displacement=12.0,
    )
    _exercise(
        bed_probe,
        [
            _u(-0.0, ((track, _IN),), frame=frame, poses=(pose,)),
            _u(2, ((track, _IN),), frame=frame + 2, poses=(pose,)),
            _u(4, ((track, _OUT),), frame=frame + 4, poses=(pose,)),
        ],
        watch=((track, 0),),
    )


@pytest.mark.parametrize(
    "recipients",
    [
        ((8, _IN), (4, _IN)),
        ((4, _IN), (8, _IN)),
        ((8, _IN), (8, _IN)),
    ],
)
def test_inside_handoff_ties_choose_last_observation(bed_probe, recipients):
    rows = _exercise(
        bed_probe,
        [_u(0), _u(1), _u(2, recipients), _u(4, tuple((track, _OUT) for track, _ in recipients))],
    )
    assert rows[2][1][0].reason == "identity-handoff"
    assert rows[2][1][0].track_id == recipients[-1][0]


@pytest.mark.parametrize(
    "recipients",
    [
        ((8, _OUT),),
        ((8, _FAR),),
        ((8, _OUT), (9, _OUT)),
        ((8, _OUT), (8, _OUT)),
        ((8, _OUT), (9, _IN)),
        ((8, BoundingBox(210, 10, 230, 30, 0.5)),),
    ],
)
def test_outside_handoff_requires_unique_overlap_and_no_reoccupancy(bed_probe, recipients):
    rows = _exercise(
        bed_probe,
        [
            _u(0, beds=(_BED, _OTHER)),
            _u(2, beds=(_BED, _OTHER)),
            _u(3, ((7, _OUT),), beds=(_BED, _OTHER)),
            _u(4, recipients, beds=(_BED, _OTHER), poses=()),
            _u(6, recipients, beds=(_BED, _OTHER), poses=()),
        ],
    )
    assert bool(rows[3][0]) == (recipients == ((8, _OUT),))


def test_recovery_reassociation_release_identity_and_sequence(bed_probe):
    rows = _exercise(
        bed_probe,
        [
            _u(0),
            _u(2),
            _u(4, ((7, _OUT),)),
            _Release(0, "foreign"),
            _u(5, ()),
            _u(6, ((8, _IN),)),
            _Release(0, "metadata"),
            _u(8, ((8, _IN),)),
            _u(10, ((8, _OUT),)),
            _Release(0),
            _Release(1),
            _u(12, ((8, _OUT),)),
            _u(13, ((8, _IN),)),
            _u(15, ((8, _IN),)),
            _u(17, ((8, _OUT),)),
        ],
    )
    events = [e for row in rows for e in row[0]]
    assert [e.identity for e in events] == [
        f"{_IDS[2]}:{_IDS[3]}:bed-exit:0:{track}:7:0:{sequence}"
        for track, sequence in ((7, 1), (8, 2), (8, 3))
    ]
    assert [e.time_sec for e in events] == [4.0, 10.0, 17.0]


@pytest.mark.parametrize("track", [7, 8])
@pytest.mark.parametrize("frame_gap,time_gap", [(75, 5.0), (76, 5.0), (75, 5.000001), (-1, -1.0)])
def test_reassociation_frame_pts_boundaries_and_recovery(bed_probe, track, frame_gap, time_gap):
    time, frame = 5.0 + time_gap, 5 + frame_gap
    _exercise(
        bed_probe,
        [
            _u(0),
            _u(2),
            _u(4, ((7, _OUT),)),
            _u(5, ()),
            _u(time, ((track, _IN),), frame=frame),
            _u(time + 2, ((track, _IN),), frame=frame + 2),
            _u(time + 4, ((track, _OUT),), frame=frame + 4),
        ],
    )


def test_stale_retirement_order_and_removed_own_bed(bed_probe):
    persons = ((9, _IN), (2, BoundingBox(210, 10, 230, 30, 0.9)))
    _exercise(
        bed_probe,
        [
            _u(0, persons, beds=(_BED, _OTHER)),
            _u(2, persons, beds=(_BED, _OTHER)),
            _u(3, ((2, _OUT),), beds=(_BED,)),
            _u(5, ((2, _OUT),), beds=(_BED,)),
            _u(6, ()),
        ],
    )


@pytest.mark.parametrize(
    "polygon",
    [
        None,
        (),
        ((0, 0), (100, 0), (100, 30), (30, 30), (30, 100), (0, 100)),
        ((0, 0), (100, 100), (0, 100), (100, 0)),
    ],
)
def test_real_polygon_scores_and_full_box_metadata_transport(bed_probe, polygon):
    bed = replace(_BED, polygon=polygon)
    _exercise(bed_probe, [_u(0, beds=(bed,)), _u(2, beds=(bed,)), _u(4, ((7, _OUT),), beds=(bed,))])


def test_prior_box_polygon_uses_real_geometry_not_an_aabb_substitute(bed_probe):
    previous = replace(_OUT, polygon=((110, 10), (130, 10), (120, 30)))
    rows = _exercise(
        bed_probe, [_u(0), _u(2), _u(3, ((7, previous),)), _u(4, ((8, _OUT),), poses=())]
    )
    assert rows[-1][1][0].reason == "identity-handoff"
    assert rows[-1][0][0].person_id == 8


def test_sixty_four_events_conserved_and_ordered_at_collection_bound(bed_probe):
    people = tuple((track, _IN) for track in reversed(range(64)))
    rows = _exercise(
        bed_probe,
        [_u(0, people), _u(2, people), _u(4, tuple((track, _OUT) for track, _ in people))],
        watch=tuple((track, 0) for track in range(64)),
    )
    assert [e.person_id for e in rows[-1][0]] == list(reversed(range(64)))
    assert len({e.identity for e in rows[-1][0]}) == 64


def _request(steps, *, capacity=64, missing_geometry=False, missing_overlap=False):
    history = _History()
    return _payload(
        [
            _HEADER,
            _new(_config(), capacity=capacity),
            *(
                history.line(
                    step,
                    actual_target_tzinfo=None,
                    geometry=not (missing_geometry and index == len(steps) - 1),
                    overlaps=not (missing_overlap and index == len(steps) - 1),
                )
                for index, step in enumerate(steps)
            ),
        ]
    )


@pytest.mark.parametrize("missing_overlap", [False, True])
def test_missing_geometry_is_explicit_rejection_not_empty_success(bed_probe, missing_overlap):
    steps = [_u(0), _u(2), _u(3, ((7, _OUT),)), _u(4, ((8, _OUT),))]
    result = _invoke(
        bed_probe,
        _request(steps, missing_geometry=not missing_overlap, missing_overlap=missing_overlap),
    )
    assert result.returncode == 2 and result.stderr == b"bed-policy-probe: rejected\n"
    assert b"X\t4\trejected\t" in result.stdout
    assert (
        _hex(
            "MissingPriorOverlap(7)" if missing_overlap else 'InvalidShape("containments")'
        ).encode()
        in result.stdout
    )
    assert result.stdout.endswith(b"END\t5\n")
    blocks = {}
    current = None
    for row in result.stdout.splitlines():
        tag = row.split(b"\t", 1)[0]
        if tag == b"O":
            current = int(row.split(b"\t")[1])
            blocks[current] = []
        elif current is not None and tag not in (b"Q", b"X", b"END"):
            blocks[current].append(row)
    assert blocks[3] == blocks[4], "admission rejection must conserve the entire previous snapshot"


def test_fatal_recovery_capacity_poison_never_retries(bed_probe):
    # Outside the shared admitted domain: two episodes, explicit capacity ONE.
    # Recovery itself hits the bound here. The timezone suite separately checks
    # a nonempty accepted-onset prefix, followed by release on the poisoned owner.
    people = ((9, _IN), (2, _IN))
    result = _invoke(bed_probe, _request([_u(0, people), _u(2, people), _u(4, people)], capacity=1))
    assert result.returncode == 2
    assert b"X\t2\tfatal\t" in result.stdout and b"X\t3\tpoisoned\t" in result.stdout
    assert b"\nE\t" not in result.stdout  # No accepted prefix at recovery, not fake events.
    assert result.stdout.endswith(b"END\t4\n")


def test_output_budget_rejects_instead_of_truncating_to_success(bed_probe):
    polygon = tuple((x, 0) for x in range(16)) + tuple((16, y) for y in range(16))
    polygon += tuple((x, 16) for x in range(16, 0, -1)) + tuple((0, y) for y in range(16, 0, -1))
    box = replace(_BED, polygon=polygon)
    update = _u(0, ((None, box),) * 64, beds=(box,) * 64, poses=())
    payload = _payload(
        [
            _HEADER,
            _new(_config()),
            _History().line(update, actual_target_tzinfo=None),
            *[_line("C", "-", _num(1.0))] * 126,
        ]
    )
    assert len(payload) <= _MAX_INPUT  # Output budget, not input/call rejection.
    result = _invoke(bed_probe, payload)
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == b"bed-policy-probe: rejected\n"


@pytest.mark.parametrize(
    "defect",
    [
        "header",
        "truncated",
        "extra",
        "unknown",
        "hex",
        "utf8",
        "float",
        "nonfinite",
        "count",
        "calls",
        "input",
        "cli",
    ],
)
def test_transport_rejection_is_not_python_domain_parity(bed_probe, defect):
    payload = _request([_u(0)])
    if defect == "header":
        payload = payload.replace(b"BEDPROBE\t1", b"BEDPROBE\t99", 1)
    elif defect == "truncated":
        payload = payload[:-1]
    elif defect in ("extra", "unknown", "count", "float", "nonfinite"):
        tail = {
            "extra": "C\t-\t0000000000000000\textra",
            "unknown": "?",
            "count": "U\t0\t-\t0000000000000000\t0000000000000000\t-\t6672657368\t0\t65",
            "float": "C\t-\txyz",
            "nonfinite": "C\t-\t7ff0000000000000",
        }[defect]
        payload = _payload([_HEADER, _new(_config()), tail])
    elif defect in ("hex", "utf8"):
        payload = payload.replace(_hex(_IDS[0]).encode(), b"f" if defect == "hex" else b"ff", 1)
    elif defect == "calls":
        payload = _payload([_HEADER, _new(_config()), *["C\t-\t0000000000000000"] * 128])
    elif defect == "input":
        payload += b"x" * (_MAX_INPUT + 1 - len(payload))
    result = _invoke(bed_probe, payload, args=("unexpected",) if defect == "cli" else ())
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == b"bed-policy-probe: rejected\n"
