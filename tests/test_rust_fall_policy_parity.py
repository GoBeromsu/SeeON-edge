"""Executed scored-policy parity, not Rust-only expectations or inference evidence.

Parent-owned build: cargo build -p seeon-worker --example fall_policy_probe --offline --locked
Gate: SEEON_TEST_FALL_POLICY_PROBE="$PWD/target/debug/examples/fall_policy_probe" \
    uv run pytest -q tests/test_rust_fall_policy_parity.py
An absent executable setting skips, never qualifies; a configured failure fails.
The v1 TSV grammar is declared in worker/policy/examples/fall_policy_probe.rs.

Only the shared admitted domain is compared: finite f64 values, u64 IDs/source
and track generations, i64 frames, <=64 collection entries/retained identities,
<=64 authority vote window, <=128 calls including construction, <=128 KiB input.
Python intentionally accepts some inputs beyond those Rust admission bounds.
Transport rejection tests below do NOT claim parity for those wider domains.
The Python authority is worker.domains.episode.authority (singular episode),
reached through the real FallPolicyDecider, never a test implementation.
"""

from __future__ import annotations

import os
import struct
import subprocess
from dataclasses import dataclass, fields, replace
from pathlib import Path

import pytest

from shared.detection_policies import FallPolicyV2
from worker.domains.fall.policy import FallPolicyDecider
from worker.interfaces.fall_model import FallProbabilities
from worker.types import BusinessEvent, DecisionTraceSnapshot
from worker.types.trace import DecisionTraceMissingReason

_HEADER = "FALLPROBE\t1"
_IDS = ("synthetic-camera", "synthetic-facility", "synthetic-boot", "synthetic-epoch")
_WATCH = (2, 4, 7, 8, 9, 99)
_MAX_INPUT = 128 * 1024
_MAX_OUTPUT = 4 * 1024 * 1024


@pytest.fixture
def fall_probe() -> str:
    configured = os.environ.get("SEEON_TEST_FALL_POLICY_PROBE")
    if configured is None:
        pytest.skip("fall probe is not configured; skip is not differential qualification")
    if not configured:
        pytest.fail("configured fall probe must name an executable", pytrace=False)
    try:
        executable = Path(configured).resolve()
        usable = executable.is_file() and os.access(executable, os.X_OK)
    except (OSError, RuntimeError):
        pytest.fail("configured fall probe path is unusable", pytrace=False)
    if not usable:
        pytest.fail("configured fall probe is missing or unexecutable", pytrace=False)
    return str(executable)


def _invoke(probe: str, payload: bytes) -> subprocess.CompletedProcess[bytes]:
    # One extra byte is permitted solely for the input-size rejection test.
    assert len(payload) <= _MAX_INPUT + 1
    try:
        result = subprocess.run(
            [probe], input=payload, capture_output=True, timeout=5.0, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.fail("configured fall probe failed to execute within its bound", pytrace=False)
    if len(result.stdout) > _MAX_OUTPUT or len(result.stderr) > 1024:
        pytest.fail("configured fall probe exceeded output bounds", pytrace=False)
    return result


def _hex(text: str) -> str:
    return text.encode("utf-8").hex()


def _number(value: int | float) -> str:
    assert type(value) in (int, float)
    return struct.pack(">d", value).hex() if type(value) is float else str(value)


def _optional(value: int | None) -> str:
    return "-" if value is None else str(value)


def _line(*parts: object) -> str:
    return "\t".join(str(part) for part in parts)


def _new_line(policy: FallPolicyV2, ids: tuple[str, ...] = _IDS) -> str:
    return _line(
        "N",
        *map(_hex, ids),
        7,
        *(_number(getattr(policy, item.name)) for item in fields(policy)),
        len(_WATCH),
        *_WATCH,
    )


def _decider(policy: FallPolicyV2, ids: tuple[str, ...] = _IDS) -> FallPolicyDecider:
    return FallPolicyDecider(
        camera_id=ids[0],
        facility_id=ids[1],
        boot_id=ids[2],
        stream_epoch=ids[3],
        source_generation=7,
        policy=policy,
    )


def _event_fields(event: BusinessEvent) -> tuple[str, ...]:
    assert type(event.identity) is str
    return (
        _hex(event.domain),
        _hex(event.event_type),
        _hex(event.identity),
        _hex(event.camera_id),
        _hex(event.facility_id),
        _number(event.time_sec),
        "-" if event.probability is None else _number(event.probability),
        _optional(event.person_id),
        _optional(event.bed_id),
    )


@dataclass(frozen=True)
class _Update:
    frame: int
    scores: dict[int, tuple[float, float]]
    live: tuple[int, ...]
    missing: dict[int, str] | None
    time: float


def _u(
    frame: int,
    scores: dict[int, tuple[float, float]] | None = None,
    live: tuple[int, ...] | None = None,
    missing: dict[int, str] | None = None,
    time: float | None = None,
) -> _Update:
    scores = {} if scores is None else scores
    return _Update(
        frame,
        scores,
        tuple(scores) if live is None else live,
        missing,
        float(frame) if time is None else time,
    )


def _update_line(step: _Update) -> str:
    scores = [
        value
        for track, (transition, fallen) in step.scores.items()
        for value in (str(track), _number(0.0), _number(transition), _number(fallen))
    ]
    missing = [
        value
        for track, reason in (step.missing or {}).items()
        for value in (str(track), _hex(reason))
    ]
    return _line(
        "U",
        step.frame,
        _number(step.time),
        len(step.live),
        *step.live,
        len(step.scores),
        *scores,
        "-" if step.missing is None else len(step.missing),
        *missing,
    )


@dataclass(frozen=True)
class _Release:
    event_index: int
    mode: str = "exact"


@dataclass(frozen=True)
class _Observation:
    events: tuple[BusinessEvent, ...]
    traces: tuple[DecisionTraceSnapshot, ...]
    fresh: bool
    switches: int
    tracks: tuple[tuple[int, int | None, bool], ...]


def _observe(d: FallPolicyDecider, events: tuple[BusinessEvent, ...]) -> _Observation:
    return _Observation(
        events,
        d.last_trace_snapshots,
        d.last_update_evaluated,
        d.track_id_switch_absorbed_total,
        tuple((track, d.generation_for(track), d.is_fallen(track)) for track in _WATCH),
    )


def _render(index: int, observation: _Observation) -> list[str]:
    rows = [
        _line(
            "O",
            index,
            int(observation.fresh),
            observation.switches,
            len(observation.events),
            len(observation.traces),
        )
    ]
    rows.extend(_line("E", *_event_fields(event)) for event in observation.events)
    for trace in observation.traces:
        rows.append(
            _line(
                "T",
                _hex(trace.reason),
                _hex(trace.previous_state),
                _hex(trace.current_state),
                int(trace.triggered),
                _optional(trace.track_id),
                _optional(trace.bed_id),
                len(trace.values),
                len(trace.missing_values),
            )
        )
        for name, value in sorted(trace.values.items()):
            rows.append(_line("V", _hex(name), "I" if type(value) is int else "F", _number(value)))
        rows.extend(
            _line("M", _hex(name), _hex(reason))
            for name, reason in sorted(trace.missing_values.items())
        )
    rows.extend(
        _line("S", track, _optional(generation), int(fallen))
        for track, generation, fallen in observation.tracks
    )
    return rows


def _exercise(
    probe: str,
    steps: list[_Update | _Release | str],
    policy: FallPolicyV2 | None = None,
    ids: tuple[str, ...] = _IDS,
) -> list[_Observation]:
    assert len(steps) < 128
    policy = FallPolicyV2() if policy is None else policy
    d = _decider(policy, ids)
    requests = [_HEADER, _new_line(policy, ids)]
    expected = [_HEADER, *_render(0, _observe(d, ()))]
    observations: list[_Observation] = []
    emitted: list[BusinessEvent] = []
    for index, step in enumerate(steps, 1):
        if isinstance(step, _Update):
            requests.append(_update_line(step))
            events = d.update(
                {track: FallProbabilities(0.0, *score) for track, score in step.scores.items()},
                step.live,
                frame_index=step.frame,
                time_sec=step.time,
                missing_score_reasons=None
                if step.missing is None
                else {
                    track: DecisionTraceMissingReason(reason)
                    for track, reason in step.missing.items()
                },
            )
        elif isinstance(step, _Release):
            event = emitted[step.event_index]
            if step.mode == "foreign":
                event = replace(event, identity=f"{event.identity}:not-emitted")
            elif step.mode == "metadata":
                event = replace(
                    event,
                    domain="synthetic-other-domain",
                    event_type="synthetic-other-event",
                    camera_id="synthetic-other-camera",
                    facility_id="synthetic-other-facility",
                    time_sec=77.0,
                    probability=0.123,
                    person_id=99,
                    bed_id=42,
                )
            else:
                assert step.mode == "exact"
            requests.append(_line("R", *_event_fields(event)))
            d.release_onset(event)
            events = ()
        else:
            assert step == "coast"
            requests.append("C")
            events = d.coast()
        emitted.extend(events)
        observation = _observe(d, events)
        observations.append(observation)
        expected.extend(_render(index, observation))
    expected.append(_line("END", len(steps) + 1))
    result = _invoke(probe, ("\n".join(requests) + "\n").encode("ascii"))
    if result.returncode != 0 or result.stderr:
        pytest.fail("configured fall probe rejected a differential sequence", pytrace=False)
    # A canonical typed transcript compares EVERY row, including event/trace order,
    # all numeric types/bits, missing maps, stale snapshots and each watched state.
    # Sorting map keys changes no map semantics; event and trace order is untouched.
    reference = ("\n".join(expected) + "\n").encode("ascii")
    assert result.stdout.splitlines(keepends=True) == reference.splitlines(keepends=True)
    return observations


def _track(observation: _Observation, track: int) -> tuple[int | None, bool]:
    return next(
        (generation, fallen) for key, generation, fallen in observation.tracks if key == track
    )


def _identity(track: int, generation: int, sequence: int) -> str:
    return f"{_IDS[2]}:{_IDS[3]}:fall:none:{track}:7:{generation}:{sequence}"


@pytest.mark.parametrize("reason", tuple(DecisionTraceMissingReason))
def test_d1_every_missing_reason_on_unknown_track(fall_probe: str, reason: str) -> None:
    observed = _exercise(fall_probe, [_u(0, live=(7,), missing={7: reason})])[0]
    assert observed.events == ()
    assert _track(observed, 7) == (None, False)
    (trace,) = observed.traces
    assert (trace.reason, trace.previous_state, trace.current_state) == (
        "score-missing",
        "unknown",
        "unknown",
    )
    assert not trace.triggered and trace.bed_id is None and trace.track_id == 7
    assert dict(trace.values) == {}
    assert dict(trace.missing_values) == {"fall_transition_probability": reason}


def test_d1_unknown_reason_is_rejected_by_both_real_owners(fall_probe: str) -> None:
    reason = "synthetic-unknown-reason"
    d = _decider(FallPolicyV2())
    with pytest.raises(ValueError, match="missing reason must use compiled vocabulary"):
        d.update({}, (7,), frame_index=0, time_sec=0.0, missing_score_reasons={7: reason})
    payload = "\n".join(
        (
            _HEADER,
            _new_line(FallPolicyV2()),
            _update_line(_u(0, live=(7,), missing={7: reason})),
            "",
        )
    ).encode("ascii")
    result = _invoke(fall_probe, payload)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == b"fall-policy-probe: rejected\n"


@pytest.mark.parametrize(
    "score,event_count",
    [
        (0.49999999999999994, 0),
        (0.5, 1),
        (0.5000000000000001, 1),
    ],
)
def test_d3_threshold_uses_unrounded_score(fall_probe: str, score: float, event_count: int) -> None:
    observed = _exercise(
        fall_probe,
        [_u(0, {7: (score, 0.0)})],
        FallPolicyV2(transition_votes=1),
    )[0]
    assert len(observed.events) == event_count
    assert observed.traces[0].values["fall_transition_probability"] == 0.5
    if event_count:
        assert _number(observed.events[0].probability) == _number(score)


@pytest.mark.parametrize(
    "score,canonical",
    [
        (0.0078125, 0.007812),
        (0.0234375, 0.023438),
        (0.0000005, 0.0),
        (0.0000015, 0.000002),
        (-0.0, 0.0),
    ],
)
def test_d3_canonical_trace_numbers_keep_unrounded_event_bits(
    fall_probe: str,
    score: float,
    canonical: float,
) -> None:
    observed = _exercise(
        fall_probe,
        [_u(0, {7: (score, score)}, time=-0.0)],
        FallPolicyV2(transition_votes=1, transition_threshold=0.0),
    )[0]
    assert len(observed.events) == 1
    assert _number(observed.events[0].probability) == _number(score)
    for key in ("fall_transition_probability", "fallen_probability"):
        assert _number(observed.traces[0].values[key]) == _number(canonical)
    for key, value in (("transition_votes", 1), ("transition_window", 5)):
        assert type(observed.traces[0].values[key]) is int
        assert observed.traces[0].values[key] == value


def test_d4_missing_coast_initial_fallen_reassociation_and_exact_release(fall_probe: str) -> None:
    observed = _exercise(
        fall_probe,
        [
            _u(0, {7: (0.7, 0.0)}),
            _u(1, live=(7, 8), missing={7: "classifier-warmup", 8: "pose-unavailable"}),
            "coast",
            _u(2, {8: (0.7, 0.8)}),
            _Release(0),
            _u(3, {8: (0.7, 0.0)}),
        ],
        FallPolicyV2(transition_votes=1),
    )
    assert [len(item.events) for item in observed] == [1, 0, 0, 0, 0, 1]
    assert observed[2].traces == observed[1].traces and not observed[2].fresh
    assert observed[3].switches == 1 and _track(observed[3], 8) == (0, True)
    assert observed[0].events[0].identity == _identity(7, 0, 1)
    assert observed[5].events[0].identity == _identity(8, 0, 2)


def test_release_is_identity_exact_metadata_independent_and_never_rewinds(fall_probe: str) -> None:
    observed = _exercise(
        fall_probe,
        [
            _u(0, {7: (0.7, 0.0)}),
            _Release(0, "foreign"),
            _u(1, {7: (0.7, 0.0)}),
            _Release(0, "metadata"),
            _Release(0),
            _u(2, {7: (0.7, 0.0)}),
            _Release(0),
            _u(3, {7: (0.7, 0.0)}),
            _Release(1),
            _u(4, {7: (0.7, 0.0)}),
        ],
        FallPolicyV2(transition_votes=1),
    )
    assert [event.identity for item in observed for event in item.events] == [
        _identity(7, 0, 1),
        _identity(7, 0, 2),
        _identity(7, 0, 3),
    ]
    assert [index for index, item in enumerate(observed) if item.events] == [0, 5, 9]


def test_d5_strict_recovery_boundaries_rearm_without_cooldown(fall_probe: str) -> None:
    scores = [
        (0.7, 0.0),
        (0.0, 0.0),
        (0.4, 0.0),
        (0.0, 0.0),
        (0.39, 0.5),
        (0.39, 0.49),
        (0.39, 0.49),
        (0.7, 0.0),
    ]
    observed = _exercise(
        fall_probe,
        [_u(frame, {7: score}) for frame, score in enumerate(scores)],
        FallPolicyV2(transition_votes=1, recovery_consecutive=2),
    )
    assert [index for index, item in enumerate(observed) if item.events] == [0, 7]
    assert observed[7].events[0].identity == _identity(7, 0, 2)
    assert all(item.traces[0].current_state == "transition-confirmed" for item in observed[:6])
    assert observed[6].traces[0].current_state == "clear"


def test_d5_initial_fallen_requires_two_recovery_scores(fall_probe: str) -> None:
    observed = _exercise(
        fall_probe,
        [
            _u(0, {7: (0.7, 0.8)}),
            _u(1, {7: (0.0, 0.0)}),
            _u(2, {7: (0.0, 0.0)}),
            _u(3, {7: (0.7, 0.0)}),
        ],
        FallPolicyV2(transition_votes=1, recovery_consecutive=2),
    )
    assert [_track(item, 7)[1] for item in observed] == [True, True, False, False]
    assert observed[2].traces[0].reason == "fall-recovered"
    assert [index for index, item in enumerate(observed) if item.events] == [3]


def test_d6_loss_insertion_order_reassociation_and_release_identity(fall_probe: str) -> None:
    # Retained Python states are inserted as 9,2, unlike Rust's numeric map order.
    # Both real authorities must still choose the numeric-lowest eligible episode.
    observed = _exercise(
        fall_probe,
        [
            _u(0, {9: (0.7, 0.0)}),
            _u(1, {2: (0.7, 0.0)}, live=(9, 2)),
            _u(2, {8: (0.7, 0.0)}),
            _Release(1),
            _u(3, {8: (0.7, 0.0)}),
            _Release(0),
            _u(4, {8: (0.7, 0.0), 9: (0.7, 0.0)}, live=(9, 8)),
        ],
        FallPolicyV2(transition_votes=1),
    )
    assert observed[2].switches == 1 and not observed[2].events
    assert [event.identity for item in observed for event in item.events] == [
        _identity(9, 0, 1),
        _identity(2, 0, 2),
        _identity(8, 0, 3),
        _identity(9, 0, 4),
    ]


def test_d6_ttl_44_retained_45_evicted_46_generation_one(fall_probe: str) -> None:
    observed = _exercise(
        fall_probe,
        [
            _u(0, {7: (0.0, 0.0)}),
            _u(44),
            _u(45),
            _u(46, {7: (0.7, 0.0)}),
        ],
        FallPolicyV2(transition_votes=1),
    )
    assert [_track(item, 7)[0] for item in observed] == [0, 0, None, 1]
    assert observed[3].events[0].identity == _identity(7, 1, 1)


def test_d6_scoreless_live_refreshes_liveness_without_advancing_streaks(fall_probe: str) -> None:
    observed = _exercise(
        fall_probe,
        [
            _u(0, {7: (0.0, 0.6)}),
            _u(1, {7: (0.0, 0.8)}),
            _u(1000, live=(7,), missing={7: "pose-unavailable"}),
            "coast",
            _u(1001, {7: (0.0, 0.8)}),
            _u(1002, {7: (0.0, 0.0)}),
            _u(1003, live=(7,), missing={7: "classifier-stride-not-due"}),
            "coast",
            _u(1004, {7: (0.0, 0.0)}),
            _u(1048),
            _u(1049),
            _u(1050, {7: (0.7, 0.0)}),
        ],
        FallPolicyV2(transition_votes=1, fallen_consecutive=2, recovery_consecutive=2),
    )
    assert [_track(item, 7)[1] for item in observed[:9]] == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert [_track(item, 7)[0] for item in observed[9:]] == [0, None, 1]
    assert observed[8].traces[0].reason == "fall-recovered"


def test_d7_six_votes_use_authority_window_not_five_entry_trace_history(fall_probe: str) -> None:
    observed = _exercise(
        fall_probe,
        [_u(frame, {7: (0.7, 0.0)}) for frame in range(6)],
        FallPolicyV2(transition_votes=6, transition_window=6),
    )
    assert [len(item.events) for item in observed] == [0, 0, 0, 0, 0, 1]


def test_d7_trace_history_expires_after_five_scores_not_policy_window(fall_probe: str) -> None:
    observed = _exercise(
        fall_probe,
        [_u(0, {7: (0.7, 0.6)})] + [_u(frame, {7: (0.49, 0.6)}) for frame in range(1, 6)],
        FallPolicyV2(transition_votes=2, transition_window=2),
    )
    assert not any(item.events for item in observed)
    assert [item.traces[0].reason for item in observed[1:]] == [
        "transition-candidate",
        "transition-candidate",
        "transition-candidate",
        "transition-candidate",
        "below-threshold",
    ]


def test_d7_sorted_events_traces_unknown_live_and_ignored_nonlive_score(fall_probe: str) -> None:
    observed = _exercise(
        fall_probe,
        [
            _u(0, {2: (0.7, 0.0), 9: (0.8, 0.0), 99: (1.0, 0.0)}, live=(9, 2, 9, 4)),
        ],
        FallPolicyV2(transition_votes=1),
    )[0]
    assert [event.person_id for event in observed.events] == [2, 9]
    assert [trace.track_id for trace in observed.traces] == [2, 4, 9]
    assert _track(observed, 4) == _track(observed, 99) == (None, False)
    assert observed.traces[1].current_state == "unknown"


@pytest.mark.parametrize("empty_map", [None, {}], ids=["none", "empty"])
def test_per_call_missing_reasons_do_not_inherit_or_reset_votes(
    fall_probe: str,
    empty_map: dict[int, str] | None,
) -> None:
    observed = _exercise(
        fall_probe,
        [
            _u(0, {7: (0.7, 0.0)}),
            _u(1, live=(7, 8), missing={7: "classifier-warmup", 8: "pose-unavailable"}),
            "coast",
            _u(2, live=(7, 8), missing={7: "classifier-stride-not-due"}),
            _u(3, live=(7, 8), missing={7: "resample-gap"}),
            _u(4, live=(7, 8), missing=empty_map),
            _u(5, {7: (0.7, 0.0)}, missing={7: "classifier-warmup"}),
            _u(6, {7: (0.7, 0.0)}),
        ],
    )
    assert observed[2].traces == observed[1].traces and not observed[2].fresh
    assert dict(observed[3].traces[1].missing_values) == {
        "fall_transition_probability": "no-live-classified-track",
    }
    assert dict(observed[5].traces[0].missing_values) == {
        "fall_transition_probability": "no-live-classified-track",
    }
    assert not observed[6].traces[0].missing_values
    assert [index for index, item in enumerate(observed) if item.events] == [7]


def test_utf8_hex_transport_preserves_identity_and_all_event_strings(fall_probe: str) -> None:
    ids = (
        "synthetic-camera\t가",
        "synthetic-facility\né",
        "synthetic-boot:β",
        "synthetic-epoch\0끝",
    )
    observed = _exercise(
        fall_probe,
        [_u(0, {7: (0.7, 0.0)}), _Release(0), _u(1, {7: (0.7, 0.0)})],
        FallPolicyV2(transition_votes=1),
        ids,
    )
    assert observed[0].events[0].camera_id == ids[0]
    assert observed[2].events[0].identity == f"{ids[2]}:{ids[3]}:fall:none:7:7:0:2"


@pytest.mark.parametrize(
    "defect",
    [
        "header",
        "truncated",
        "extra-field",
        "unknown-command",
        "invalid-hex",
        "invalid-utf8",
        "invalid-float-bits",
        "duplicate-score",
        "duplicate-missing",
        "entry-bound",
        "call-bound",
        "input-bound",
    ],
)
def test_transport_rejects_malformed_or_over_budget_input_not_domain_parity(
    fall_probe: str,
    defect: str,
) -> None:
    prefix = f"{_HEADER}\n{_new_line(FallPolicyV2())}\n"
    update = _update_line(_u(0, {7: (0.7, 0.0)}))
    payload = prefix + update + "\n"
    if defect == "header":
        payload = payload.replace(_HEADER, "FALLPROBE\t99", 1)
    elif defect == "truncated":
        payload = payload[:-1]
    elif defect == "extra-field":
        payload = prefix + "C\textra\n"
    elif defect == "unknown-command":
        payload = prefix + "synthetic-unknown-command\n"
    elif defect in ("invalid-hex", "invalid-utf8"):
        payload = payload.replace(_hex(_IDS[0]), "f" if defect == "invalid-hex" else "ff", 1)
    elif defect == "invalid-float-bits":
        payload = prefix + "U\t0\txyz\t0\t0\t0\n"
    elif defect == "duplicate-score":
        score = _line(7, _number(0.0), _number(0.7), _number(0.0))
        payload = prefix + _line("U", 0, _number(0.0), 1, 7, 2, score, score, 0) + "\n"
    elif defect == "duplicate-missing":
        missing = _line(7, _hex("pose-unavailable"))
        payload = prefix + _line("U", 0, _number(0.0), 1, 7, 0, 2, missing, missing) + "\n"
    elif defect == "entry-bound":
        payload = prefix + _line("U", 0, _number(0.0), 65, *range(65), 0, 0) + "\n"
    elif defect == "call-bound":
        payload = prefix + "C\n" * 128
    else:
        assert defect == "input-bound"
        payload = prefix + "x" * (_MAX_INPUT + 1 - len(prefix))
    result = _invoke(fall_probe, payload.encode("ascii"))
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == b"fall-policy-probe: rejected\n"
