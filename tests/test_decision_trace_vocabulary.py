"""Closed decision-trace vocabulary: baseline membership plus additive bed-exit tokens."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest

from backend.app.edge_db.migrator import migrate_database
from contracts.frame import Frame
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.pipeline.analytics.composite import CompositeResult
from worker.pipeline.trace import (
    BoundedTraceWriter,
    TraceCapture,
    TraceIdentity,
    TraceRetentionPolicy,
)
from worker.runtime.provenance.models import AppliedRuntimeManifest
from worker.runtime.provenance.store import AppliedRuntimeManifestStore
from worker.types import DecisionInput, DecisionTraceSnapshot, FramePacket
from worker.types.trace import (
    DecisionTraceMissingReason,
    DecisionTraceReason,
    DecisionTraceState,
    DecisionTraceValueName,
    canonical_trace_number,
)

_RUNTIME_SHA256 = "a" * 64
_COMPONENT_SHA256 = "b" * 64
_POLICY_SHA256 = "c" * 64

# Frozen membership of the four closed vocabularies at the commit this test
# was introduced against. Persisted traces reference these tokens; they must
# remain a subset of whatever the enums grow into.
BASELINE_REASONS: frozenset[str] = frozenset(
    {
        "trace-unavailable",
        "outside-detection-window",
        "score-missing",
        "fall-onset",
        "fall-active",
        "below-threshold",
        "bed-region-unavailable",
        "bed-observation-missing",
        "stale-track-exit",
        "stale-track-clear",
        "assigned",
        "assignment-hold",
        "below-containment",
        "contained",
        "contained-in-other-bed",
        "live-grace-exit",
        "live-grace",
        "person-observation-missing",
    }
)
BASELINE_STATES: frozenset[str] = frozenset(
    {
        "unknown",
        "not-evaluated",
        "no-decision",
        "clear",
        "fall",
        "live-grace",
        "contained",
        "triggered",
        "retired",
        "unassigned",
        "other-bed",
    }
)
BASELINE_VALUE_NAMES: frozenset[str] = frozenset(
    {
        "operating_threshold",
        "window_frames",
        "fall_probability",
        "containment_ratio",
        "max_other_containment_ratio",
        "min_containment",
        "candidate_frames",
        "hold_frames_threshold",
        "grace_frames_before",
        "grace_frames_after",
        "grace_threshold",
        "bed_id",
        "decision_state",
    }
)
BASELINE_MISSING_REASONS: frozenset[str] = frozenset(
    {
        "adapter-not-provided",
        "adapter-returned-no-data",
        "outside-detection-window",
        "no-live-classified-track",
        "bed-region-unavailable",
        "bed-observation-missing",
        "track-no-longer-live",
        "no-observed-person",
    }
)

BED_EXIT_STATES: frozenset[str] = frozenset(
    {
        "in-bed",
        "sitting-up",
        "edge-sitting",
        "out-of-bed",
        "uncertain",
        "absent",
    }
)
BED_EXIT_VALUE_NAMES: frozenset[str] = frozenset(
    {
        "torso_in_frac",
        "lower_in_frac",
        "keypoint_in_frac",
        "hip_depth",
        "torso_angle",
        "centroid_displacement",
        "hip_x_rel",
        "hip_y_rel",
        "observability",
        "dwell_frames",
        "dwell_threshold",
    }
)
BED_EXIT_MISSING_REASONS: frozenset[str] = frozenset(
    {
        "pose-unavailable",
        "bed-polygon-invalid",
    }
)
BED_EXIT_REASONS: frozenset[str] = frozenset(
    {
        "in-bed-hold",
        "sitting-up-hold",
        "edge-sitting-hold",
        "out-of-bed-hold",
        "uncertain-hold",
        "absent-hold",
        "entered-in-bed",
        "entered-sitting-up",
        "entered-edge-sitting",
        "entered-out-of-bed",
        "entered-uncertain",
        "entered-absent",
        "pose-unavailable",
        "bed-polygon-invalid",
    }
)

_HOLD_REASON_BY_STATE: dict[str, str] = {
    "in-bed": "in-bed-hold",
    "sitting-up": "sitting-up-hold",
    "edge-sitting": "edge-sitting-hold",
    "out-of-bed": "out-of-bed-hold",
    "uncertain": "uncertain-hold",
    "absent": "absent-hold",
}
_ENTRY_REASON_BY_STATE: dict[str, str] = {
    "in-bed": "entered-in-bed",
    "sitting-up": "entered-sitting-up",
    "edge-sitting": "entered-edge-sitting",
    "out-of-bed": "entered-out-of-bed",
    "uncertain": "entered-uncertain",
    "absent": "entered-absent",
}


def _values(enum_type: type[StrEnum]) -> frozenset[str]:
    return frozenset(member.value for member in enum_type)


def _packet() -> FramePacket:
    return FramePacket(
        camera_id="camera-a",
        frame=Frame(7, 1.0, np.zeros((4, 4, 3), dtype=np.uint8)),
        pts=1.0,
        seq=11,
        width=4,
        height=4,
        decode_time_ms=0.0,
        worker_boot_id="boot-a",
        stream_epoch=3,
    )


def _result() -> CompositeResult:
    person = BoundingBox(0, 0, 2, 3, 0.9)
    bed = BoundingBox(0, 0, 4, 4, 0.8)
    observation = FrameObservation(
        detections=((person,), ()),
        regions=((bed,), ()),
        track_ids=(5,),
    )
    decision_input = DecisionInput(
        observation=observation,
        frame_width=4,
        frame_height=4,
        live_track_ids=(5,),
        time_sec=1.0,
        frame_index=7,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH),
    )
    return CompositeResult((), observation, decision_input)


def _capture(snapshots: tuple[DecisionTraceSnapshot, ...]) -> TraceCapture:
    return TraceCapture(
        identities=(
            TraceIdentity(
                module_qualified_id="bed_exit.v1",
                component_qualified_ids=(f"bed-exit.sha256.{_COMPONENT_SHA256}",),
                policy_qualified_id="bed_exit.policy.v1",
                effective_policy_id=_POLICY_SHA256,
                runtime_manifest_sha256=_RUNTIME_SHA256,
                snapshot_provider=lambda: snapshots,
            ),
        )
    )


def _seed(database: Path) -> None:
    migrate_database(database)
    AppliedRuntimeManifestStore(database).persist(
        AppliedRuntimeManifest(
            1,
            json.dumps(
                {
                    "cameras": [{"camera_id": "camera-a"}],
                    "manifest_schema_version": 1,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            _RUNTIME_SHA256,
        ),
        boot_instance_id="boot-a",
        applied_at="2026-08-13T00:00:00Z",
    )


def _new_token_snapshots() -> tuple[DecisionTraceSnapshot, ...]:
    snapshots: list[DecisionTraceSnapshot] = []
    for state, reason in _HOLD_REASON_BY_STATE.items():
        snapshots.append(
            DecisionTraceSnapshot(
                reason=reason,
                previous_state=state,
                current_state=state,
                triggered=False,
                track_id=5,
                bed_id=0,
                values={
                    "torso_in_frac": 0.81,
                    "lower_in_frac": 0.74,
                    "keypoint_in_frac": 0.66,
                    "hip_depth": 0.05,
                    "torso_angle": 0.41,
                    "centroid_displacement": 0.01,
                    "hip_x_rel": 0.48,
                    "hip_y_rel": 0.52,
                    "observability": 0.92,
                    "dwell_frames": 3,
                    "dwell_threshold": 5,
                },
            )
        )
    for state, reason in _ENTRY_REASON_BY_STATE.items():
        snapshots.append(
            DecisionTraceSnapshot(
                reason=reason,
                previous_state="absent" if state != "absent" else "uncertain",
                current_state=state,
                triggered=state == "out-of-bed",
                track_id=5,
                bed_id=0,
                values={"dwell_frames": 1, "dwell_threshold": 3},
            )
        )
    snapshots.append(
        DecisionTraceSnapshot(
            reason="pose-unavailable",
            previous_state="unknown",
            current_state="uncertain",
            triggered=False,
            track_id=5,
            bed_id=0,
            missing_values={
                "torso_in_frac": "pose-unavailable",
                "hip_depth": "pose-unavailable",
            },
        )
    )
    snapshots.append(
        DecisionTraceSnapshot(
            reason="bed-polygon-invalid",
            previous_state="unknown",
            current_state="no-decision",
            triggered=False,
            track_id=5,
            bed_id=None,
            missing_values={
                "torso_in_frac": "bed-polygon-invalid",
                "hip_depth": "bed-polygon-invalid",
            },
        )
    )
    return tuple(snapshots)


def test_baseline_closed_vocabularies_remain_intact() -> None:
    """Pin the pre-extension membership so persisted tokens cannot silently vanish."""
    assert _values(DecisionTraceReason) >= BASELINE_REASONS
    assert _values(DecisionTraceState) >= BASELINE_STATES
    assert _values(DecisionTraceValueName) >= BASELINE_VALUE_NAMES
    assert _values(DecisionTraceMissingReason) >= BASELINE_MISSING_REASONS


def test_baseline_vocabularies_are_exactly_the_pre_extension_sets() -> None:
    """Fail if a later change silently drops or renames a persisted token.

    Before the bed-exit extension this is an exact-set pin. After the
    extension the same names stay a subset (asserted above) and the
    additive tokens are asserted separately.
    """
    current_reasons = _values(DecisionTraceReason)
    current_states = _values(DecisionTraceState)
    current_value_names = _values(DecisionTraceValueName)
    current_missing = _values(DecisionTraceMissingReason)

    extra_reasons = current_reasons - BASELINE_REASONS
    extra_states = current_states - BASELINE_STATES
    extra_value_names = current_value_names - BASELINE_VALUE_NAMES
    extra_missing = current_missing - BASELINE_MISSING_REASONS

    if extra_reasons | extra_states | extra_value_names | extra_missing:
        assert extra_reasons == BED_EXIT_REASONS
        assert extra_states == BED_EXIT_STATES
        assert extra_value_names == BED_EXIT_VALUE_NAMES
        assert extra_missing == BED_EXIT_MISSING_REASONS
    else:
        assert current_reasons == BASELINE_REASONS
        assert current_states == BASELINE_STATES
        assert current_value_names == BASELINE_VALUE_NAMES
        assert current_missing == BASELINE_MISSING_REASONS


def test_bed_exit_tokens_are_additive_and_closed() -> None:
    assert _values(DecisionTraceReason) == BASELINE_REASONS | BED_EXIT_REASONS
    assert _values(DecisionTraceState) == BASELINE_STATES | BED_EXIT_STATES
    assert _values(DecisionTraceValueName) == BASELINE_VALUE_NAMES | BED_EXIT_VALUE_NAMES
    assert (
        _values(DecisionTraceMissingReason) == BASELINE_MISSING_REASONS | BED_EXIT_MISSING_REASONS
    )


def test_unknown_reason_still_raises() -> None:
    with pytest.raises(ValueError, match="compiled vocabulary"):
        DecisionTraceSnapshot(
            reason="not-a-real-token",
            previous_state="unknown",
            current_state="unknown",
            triggered=False,
            track_id=None,
            bed_id=None,
        )


def test_misspelled_new_token_fails_closed() -> None:
    with pytest.raises(ValueError, match="compiled vocabulary"):
        DecisionTraceSnapshot(
            reason="entered-inbed",
            previous_state="in-bed",
            current_state="in-bed",
            triggered=False,
            track_id=5,
            bed_id=0,
        )
    with pytest.raises(ValueError, match="compiled vocabulary"):
        DecisionTraceSnapshot(
            reason="in-bed-hold",
            previous_state="inbed",
            current_state="in-bed",
            triggered=False,
            track_id=5,
            bed_id=0,
        )
    with pytest.raises(ValueError, match="compiled vocabulary"):
        DecisionTraceSnapshot(
            reason="in-bed-hold",
            previous_state="in-bed",
            current_state="in-bed",
            triggered=False,
            track_id=5,
            bed_id=0,
            values={"torso_in_frac_": 0.8},
        )
    with pytest.raises(ValueError, match="compiled vocabulary"):
        DecisionTraceSnapshot(
            reason="pose-unavailable",
            previous_state="unknown",
            current_state="uncertain",
            triggered=False,
            track_id=5,
            bed_id=0,
            missing_values={"torso_in_frac": "pose_unavailable"},
        )


def test_canonical_trace_number_is_unchanged() -> None:
    assert canonical_trace_number(1) == 1
    assert type(canonical_trace_number(1)) is int
    assert canonical_trace_number(1.23456789) == 1.234568
    assert canonical_trace_number(-0.0) == 0.0
    assert str(canonical_trace_number(-0.0)) == "0.0"
    with pytest.raises(ValueError, match="finite"):
        canonical_trace_number(float("nan"))


def test_new_tokens_round_trip_through_trace_adapter_and_writer(tmp_path: Path) -> None:
    snapshots = _new_token_snapshots()
    seen_reasons = {snapshot.reason for snapshot in snapshots}
    seen_states = {snapshot.current_state for snapshot in snapshots} | {
        snapshot.previous_state for snapshot in snapshots
    }
    seen_value_names: set[str] = set()
    seen_missing_reasons: set[str] = set()
    for snapshot in snapshots:
        seen_value_names.update(snapshot.values)
        seen_value_names.update(snapshot.missing_values)
        seen_missing_reasons.update(snapshot.missing_values.values())

    assert seen_reasons >= BED_EXIT_REASONS
    assert seen_states >= BED_EXIT_STATES
    assert seen_value_names >= BED_EXIT_VALUE_NAMES
    assert seen_missing_reasons >= BED_EXIT_MISSING_REASONS

    database = tmp_path / "edge.sqlite3"
    _seed(database)
    writer = BoundedTraceWriter(database, TraceRetentionPolicy.testing())
    writer.start()
    try:
        persisted = _capture(snapshots).capture(
            writer, _packet(), _result(), (), require_persisted=True
        )
    finally:
        writer.stop()
    assert persisted is True

    recovered = writer.recover_camera("camera-a")
    assert len(recovered.decisions) == len(snapshots)
    recovered_reasons = {decision.snapshot.reason for decision in recovered.decisions}
    recovered_states = {decision.snapshot.current_state for decision in recovered.decisions} | {
        decision.snapshot.previous_state for decision in recovered.decisions
    }
    recovered_value_names: set[str] = set()
    recovered_missing_reasons: set[str] = set()
    for decision in recovered.decisions:
        recovered_value_names.update(str(name) for name in decision.snapshot.values)
        recovered_value_names.update(str(name) for name in decision.snapshot.missing_values)
        recovered_missing_reasons.update(
            str(reason) for reason in decision.snapshot.missing_values.values()
        )

    assert recovered_reasons == seen_reasons
    assert recovered_states == seen_states
    assert recovered_value_names >= BED_EXIT_VALUE_NAMES
    assert recovered_missing_reasons == seen_missing_reasons
    recovered_by_key = {
        (
            str(decision.snapshot.reason),
            str(decision.snapshot.previous_state),
            str(decision.snapshot.current_state),
            decision.snapshot.triggered,
        ): decision.snapshot
        for decision in recovered.decisions
    }
    for original in snapshots:
        recovered_snapshot = recovered_by_key[
            (
                str(original.reason),
                str(original.previous_state),
                str(original.current_state),
                original.triggered,
            )
        ]
        for name, value in original.values.items():
            assert recovered_snapshot.values[name] == pytest.approx(float(value))
        for name, reason in original.missing_values.items():
            assert str(recovered_snapshot.missing_values[name]) == str(reason)
