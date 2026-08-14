from __future__ import annotations

from pathlib import Path

from backend.app.features.qa.store import QaStore
from shared.edge_db.migrator import migrate_database
from worker.replay.comparison import MismatchReason, compare_runs
from worker.replay.engine import ReplayFrameResult, ReplayRun
from worker.types import BusinessEvent, DecisionTraceSnapshot

_FRAME_KEY = ("boot-a", "camera-a", 7, 3)


def _snapshot(
    *,
    state: str = "clear",
    reason: str = "below-threshold",
    previous_state: str = "clear",
    triggered: bool = False,
    track_id: int | None = 11,
    bed_id: int | None = None,
    threshold: float = 0.9,
    probability: float = 0.82,
    values: dict[str, float | int] | None = None,
    missing_values: dict[str, str] | None = None,
) -> DecisionTraceSnapshot:
    resolved_values = values
    if resolved_values is None:
        resolved_values = {
            "operating_threshold": threshold,
            "window_frames": 1,
            "fall_probability": probability,
        }
    return DecisionTraceSnapshot(
        reason=reason,
        previous_state=previous_state,
        current_state=state,
        triggered=triggered,
        track_id=track_id,
        bed_id=bed_id,
        values=resolved_values,
        missing_values=missing_values or {},
    )


def _event(
    *,
    domain: str = "fall",
    event_type: str = "fall",
    identity: str | int = 1,
    camera_id: str = "camera-a",
    facility_id: str = "facility-a",
    time_sec: float = 7.0,
    probability: float = 0.82,
    person_id: int | None = 11,
    bed_id: int | None = None,
    audit: dict[str, object] | None = None,
) -> BusinessEvent:
    return BusinessEvent(
        domain=domain,
        event_type=event_type,
        identity=identity,
        camera_id=camera_id,
        facility_id=facility_id,
        time_sec=time_sec,
        probability=probability,
        person_id=person_id,
        bed_id=bed_id,
        audit=audit,
    )


def _run(
    *,
    policy_id: str,
    events: tuple[BusinessEvent, ...] = (),
    snapshots: tuple[DecisionTraceSnapshot, ...] = (),
    frames: tuple[ReplayFrameResult, ...] | None = None,
    reproducible: bool = True,
    non_reproducible_reason: str | None = None,
) -> ReplayRun:
    resolved_frames = frames
    if resolved_frames is None:
        resolved_frames = (ReplayFrameResult(_FRAME_KEY, "analysis-a", events, snapshots),)
    return ReplayRun(
        camera_id="camera-a",
        module_qualified_id="fall.v1",
        policy_qualified_id="fall.policy.v1",
        effective_policy_id=policy_id,
        frames=resolved_frames,
        reproducible=reproducible,
        non_reproducible_reason=non_reproducible_reason,
        boot_ids=("boot-a",),
    )


def _run_payload(run: ReplayRun) -> dict[str, object]:
    return {
        "boot_ids": list(run.boot_ids),
        "camera_id": run.camera_id,
        "module_qualified_id": run.module_qualified_id,
        "policy_qualified_id": run.policy_qualified_id,
        "effective_policy_id": run.effective_policy_id,
        "reproducible": run.reproducible,
        "non_reproducible_reason": run.non_reproducible_reason,
        "frames": [
            {
                "frame_key": list(frame.frame_key),
                "analysis_trace_id": frame.analysis_trace_id,
                "event_count": len(frame.events),
                "snapshots": [
                    {
                        "reason": snapshot.reason,
                        "previous_state": snapshot.previous_state,
                        "current_state": snapshot.current_state,
                        "triggered": snapshot.triggered,
                        "track_id": snapshot.track_id,
                        "bed_id": snapshot.bed_id,
                        "values": {str(name): value for name, value in snapshot.values.items()},
                        "missing_values": {
                            str(name): str(reason)
                            for name, reason in snapshot.missing_values.items()
                        },
                    }
                    for snapshot in frame.snapshots
                ],
            }
            for frame in run.frames
        ],
    }


def test_ab_mismatches_persist_with_run_and_policy_provenance(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    store = QaStore(database)
    baseline_run = _run(
        policy_id="a" * 64,
        events=(),
        snapshots=(_snapshot(state="clear", reason="below-threshold", threshold=0.9),),
    )
    candidate_run = _run(
        policy_id="b" * 64,
        events=(_event(),),
        snapshots=(
            _snapshot(
                state="fall",
                reason="fall-onset",
                triggered=True,
                threshold=0.7,
            ),
        ),
    )
    comparison = compare_runs(baseline_run, candidate_run)

    reasons = {mismatch.reason for mismatch in comparison.mismatches}
    assert MismatchReason.EVENT_COUNT_DIFFERS in reasons
    assert MismatchReason.STATE_DIFFERS in reasons
    assert MismatchReason.REASON_DIFFERS in reasons
    assert MismatchReason.TRIGGERED_DIFFERS in reasons
    assert all(
        mismatch.frame_key == _FRAME_KEY
        for mismatch in comparison.mismatches
        if mismatch.reason is not MismatchReason.REPRODUCIBILITY_DIFFERS
    )

    baseline = store.record_run(
        camera_id=baseline_run.camera_id,
        module_qualified_id=baseline_run.module_qualified_id,
        policy_qualified_id=baseline_run.policy_qualified_id,
        effective_policy_id=baseline_run.effective_policy_id,
        frame_count=len(baseline_run.frames),
        event_count=baseline_run.event_count,
        source_kind="captured",
        source_run_id=None,
        requested_by="qa-operator",
        requested_at="2026-08-14T00:00:00Z",
        result=_run_payload(baseline_run),
    )
    candidate = store.record_run(
        camera_id=candidate_run.camera_id,
        module_qualified_id=candidate_run.module_qualified_id,
        policy_qualified_id=candidate_run.policy_qualified_id,
        effective_policy_id=candidate_run.effective_policy_id,
        frame_count=len(candidate_run.frames),
        event_count=candidate_run.event_count,
        source_kind="replay",
        source_run_id=baseline.run_id,
        requested_by="qa-operator",
        requested_at="2026-08-14T00:01:00Z",
        result=_run_payload(candidate_run),
    )
    persisted = store.record_comparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        created_at="2026-08-14T00:02:00Z",
        comparison=comparison.as_dict(),
    )

    reloaded_candidate = store.get_run(candidate.run_id)
    reloaded = store.get_comparison(persisted.comparison_id)
    assert reloaded_candidate is not None
    assert reloaded_candidate.source_kind == "replay"
    assert reloaded_candidate.source_run_id == baseline.run_id
    assert reloaded_candidate.effective_policy_id == "b" * 64
    assert reloaded is not None
    assert reloaded.baseline_run_id == baseline.run_id
    assert reloaded.candidate_run_id == candidate.run_id
    assert reloaded.mismatch_count == len(comparison.mismatches)
    assert reloaded.comparison == comparison.as_dict()
    assert reloaded.comparison["baseline_effective_policy_id"] == baseline.effective_policy_id
    assert reloaded.comparison["candidate_effective_policy_id"] == candidate.effective_policy_id


def test_identical_runs_compare_clean() -> None:
    run = _run(
        policy_id="a" * 64,
        events=(_event(audit={"score": 0.82}),),
        snapshots=(_snapshot(state="fall", reason="fall-onset", triggered=True, threshold=0.7),),
    )
    comparison = compare_runs(run, run)
    assert comparison.identical is True
    assert comparison.mismatches == ()


def test_event_field_mismatches_are_one_field_at_a_time() -> None:
    baseline_event = _event()
    cases: list[tuple[BusinessEvent, MismatchReason]] = [
        (_event(domain="bed_exit"), MismatchReason.EVENT_DOMAIN_DIFFERS),
        (_event(event_type="near_fall"), MismatchReason.EVENT_TYPE_DIFFERS),
        (_event(identity=99), MismatchReason.EVENT_IDENTITY_DIFFERS),
        (_event(camera_id="camera-b"), MismatchReason.EVENT_CAMERA_DIFFERS),
        (_event(facility_id="facility-b"), MismatchReason.EVENT_FACILITY_DIFFERS),
        (_event(time_sec=8.5), MismatchReason.EVENT_ONSET_TIME_DIFFERS),
        (_event(probability=0.11), MismatchReason.EVENT_PROBABILITY_DIFFERS),
        (_event(person_id=22), MismatchReason.EVENT_TRACK_DIFFERS),
        (_event(bed_id=3), MismatchReason.EVENT_BED_DIFFERS),
        (_event(audit={"k": "v"}), MismatchReason.EVENT_AUDIT_DIFFERS),
    ]
    baseline = _run(policy_id="a" * 64, events=(baseline_event,), snapshots=())
    for candidate_event, expected_reason in cases:
        candidate = _run(policy_id="b" * 64, events=(candidate_event,), snapshots=())
        comparison = compare_runs(baseline, candidate)
        assert comparison.identical is False
        reasons = [mismatch.reason for mismatch in comparison.mismatches]
        assert reasons == [expected_reason], (expected_reason, reasons)


def test_snapshot_field_mismatches_are_one_field_at_a_time() -> None:
    baseline_snap = _snapshot()
    cases: list[tuple[DecisionTraceSnapshot, MismatchReason]] = [
        (_snapshot(previous_state="fall"), MismatchReason.PREVIOUS_STATE_DIFFERS),
        (_snapshot(state="fall", triggered=False), MismatchReason.STATE_DIFFERS),
        (_snapshot(reason="fall-onset"), MismatchReason.REASON_DIFFERS),
        (_snapshot(triggered=True), MismatchReason.TRIGGERED_DIFFERS),
        (_snapshot(track_id=99), MismatchReason.SNAPSHOT_TRACK_DIFFERS),
        (_snapshot(bed_id=2), MismatchReason.SNAPSHOT_BED_DIFFERS),
        (_snapshot(threshold=0.55), MismatchReason.VALUE_DIFFERS),
        (
            _snapshot(
                values={},
                missing_values={"fall_probability": "no-live-classified-track"},
            ),
            MismatchReason.MISSING_VALUE_DIFFERS,
        ),
    ]
    # STATE_DIFFERS case also changes triggered default only when state fall - handle carefully.
    # For state change we keep triggered False so only STATE_DIFFERS fires.
    baseline = _run(policy_id="a" * 64, events=(), snapshots=(baseline_snap,))
    for candidate_snap, expected_reason in cases:
        candidate = _run(policy_id="b" * 64, events=(), snapshots=(candidate_snap,))
        comparison = compare_runs(baseline, candidate)
        assert comparison.identical is False
        reasons = {mismatch.reason for mismatch in comparison.mismatches}
        assert expected_reason in reasons, (expected_reason, reasons)
        if expected_reason is MismatchReason.MISSING_VALUE_DIFFERS:
            # Replacing values with missing also yields VALUE_DIFFERS for absent keys.
            assert MismatchReason.VALUE_DIFFERS in reasons


def test_snapshot_cardinality_mismatch_is_explicit_not_zipped_away() -> None:
    baseline = _run(
        policy_id="a" * 64,
        events=(),
        snapshots=(_snapshot(), _snapshot(track_id=12)),
    )
    candidate = _run(
        policy_id="b" * 64,
        events=(),
        snapshots=(_snapshot(),),
    )
    comparison = compare_runs(baseline, candidate)
    assert comparison.identical is False
    reasons = [mismatch.reason for mismatch in comparison.mismatches]
    assert reasons == [MismatchReason.SNAPSHOT_COUNT_DIFFERS]
    assert "baseline=2 candidate=1" in comparison.mismatches[0].detail


def test_event_cardinality_mismatch_is_explicit() -> None:
    baseline = _run(policy_id="a" * 64, events=(_event(), _event(identity=2)), snapshots=())
    candidate = _run(policy_id="b" * 64, events=(_event(),), snapshots=())
    comparison = compare_runs(baseline, candidate)
    assert [m.reason for m in comparison.mismatches] == [MismatchReason.EVENT_COUNT_DIFFERS]


def test_frame_missing_in_other_is_reported() -> None:
    other_key = ("boot-a", "camera-a", 7, 4)
    baseline = _run(
        policy_id="a" * 64,
        frames=(
            ReplayFrameResult(_FRAME_KEY, "analysis-a", (), ()),
            ReplayFrameResult(other_key, "analysis-b", (), ()),
        ),
    )
    candidate = _run(
        policy_id="b" * 64,
        frames=(ReplayFrameResult(_FRAME_KEY, "analysis-a", (), ()),),
    )
    comparison = compare_runs(baseline, candidate)
    assert any(
        mismatch.reason is MismatchReason.FRAME_MISSING_IN_OTHER and mismatch.frame_key == other_key
        for mismatch in comparison.mismatches
    )


def test_reproducibility_flag_mismatch_is_structured() -> None:
    baseline = _run(policy_id="a" * 64, reproducible=True)
    candidate = _run(
        policy_id="b" * 64,
        reproducible=False,
        non_reproducible_reason="truncated-or-incomplete-initial-state: pruned_frames=2",
    )
    comparison = compare_runs(baseline, candidate)
    assert any(
        mismatch.reason is MismatchReason.REPRODUCIBILITY_DIFFERS
        for mismatch in comparison.mismatches
    )


def test_comparison_as_dict_is_stable_json_shape() -> None:
    baseline = _run(policy_id="a" * 64, events=(_event(),), snapshots=(_snapshot(),))
    candidate = _run(
        policy_id="b" * 64,
        events=(_event(probability=0.1),),
        snapshots=(_snapshot(probability=0.1),),
    )
    payload = compare_runs(baseline, candidate).as_dict()
    assert set(payload) == {
        "baseline_effective_policy_id",
        "candidate_effective_policy_id",
        "baseline_event_count",
        "candidate_event_count",
        "identical",
        "mismatches",
    }
    assert isinstance(payload["mismatches"], list)
    assert payload["mismatches"]
    first = payload["mismatches"][0]
    assert set(first) == {"frame_key", "reason", "detail"}
