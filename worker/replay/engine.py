"""Deterministic camera-local decider replay over persisted analysis traces.

Replay re-runs exactly one compiled ``DetectionModuleDefinition`` (fall.v1 or
bed_exit.v1) against a pinned module graph, a chosen numeric policy revision,
and a fixed time origin, driving it with ``DecisionInput`` values reconstructed
frame-by-frame from ``AnalysisTrace`` rows already captured by the real
pipeline (see ``worker.replay.inputs``). No extractor, model runner, GPU, or
network call happens here -- only the same pure numeric decider code path
production already runs, executed again against the frozen inputs.

Camera-local decider / live-track state is recreated at every worker boot
boundary exactly as production does after a process restart. Stream-epoch
changes within one boot do **not** reset that state (production keeps the same
camera module across RTSP reconnects). Truncated or mid-window recoveries are
never silently presented as deterministic: the run is explicitly marked
non-reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from shared.detection_policies import EffectivePolicy
from worker.domains.fall import FallModelProtocol
from worker.domains.module_definition import (
    CameraModuleContext,
    DetectionModuleDefinition,
)
from worker.domains.registry import DETECTION_MODULE_REGISTRY
from worker.interfaces.decision import Decider
from worker.pipeline.trace.models import (
    AnalysisTrace,
    DecisionTrace,
    RecoveredCameraTrace,
    TraceTruncation,
)
from worker.replay.inputs import (
    _LiveTrackWindow,
    analysis_trace_to_decision_input,
    replayed_track_id,
)
from worker.types import BusinessEvent, DecisionTraceSnapshot

_STATIC_CLOCK = lambda: datetime(1970, 1, 1, tzinfo=UTC)  # noqa: E731


class ReplayConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReplayFrameResult:
    """One replayed frame: the reconstructed input identity and decider output."""

    frame_key: tuple[str, str, int, int]
    analysis_trace_id: str
    events: tuple[BusinessEvent, ...]
    snapshots: tuple[DecisionTraceSnapshot, ...]


@dataclass(frozen=True, slots=True)
class ReplayRun:
    """The full deterministic output of replaying one camera trace once."""

    camera_id: str
    module_qualified_id: str
    policy_qualified_id: str
    effective_policy_id: str
    frames: tuple[ReplayFrameResult, ...]
    reproducible: bool = True
    non_reproducible_reason: str | None = None
    boot_ids: tuple[str, ...] = ()

    @property
    def event_count(self) -> int:
        return sum(len(frame.events) for frame in self.frames)


def assess_reproducibility(
    analyses: Sequence[AnalysisTrace],
    truncation: TraceTruncation | None,
) -> tuple[bool, str | None]:
    """Return whether a recovered camera timeline can be deterministically replayed.

    Production camera-local state starts cold only at process boot. A recovered
    window that lost leading frames (pruned / dropped / failed) therefore lacks
    the initial latch/track state those frames would have produced. Replay still
    runs for inspection, but the result must not claim bit-level reproducibility.
    """
    reasons: list[str] = []
    if truncation is not None:
        if truncation.handoff_dropped_frames > 0:
            reasons.append(f"handoff_dropped_frames={truncation.handoff_dropped_frames}")
        if truncation.pruned_frames > 0:
            reasons.append(f"pruned_frames={truncation.pruned_frames}")
        if truncation.persistence_failed_frames > 0:
            reasons.append(f"persistence_failed_frames={truncation.persistence_failed_frames}")
        if truncation.retention_blocked_frames > 0:
            reasons.append(f"retention_blocked_frames={truncation.retention_blocked_frames}")
    if analyses:
        first_boot = analyses[0].frame_key[0]
        first_epoch = analyses[0].frame_key[2]
        first_seq = analyses[0].frame_key[3]
        if truncation is not None and truncation.oldest_retained_key is not None:
            oldest = truncation.oldest_retained_key
            if oldest[0] == first_boot and (oldest[2] != first_epoch or oldest[3] != first_seq):
                reasons.append("oldest_retained_key_does_not_match_first_recovered_frame")
    if not reasons:
        return True, None
    return False, "truncated-or-incomplete-initial-state: " + ", ".join(reasons)


def replay_recovered(
    *,
    camera_id: str,
    recovered: RecoveredCameraTrace,
    module_id: str,
    policy: EffectivePolicy,
    facility_id: str = "replay",
    fall_model: FallModelProtocol | None = None,
    clock: Callable[[], datetime] = _STATIC_CLOCK,
) -> ReplayRun:
    """Replay a full ``RecoveredCameraTrace``, honoring truncation markers."""
    return replay_camera(
        camera_id=camera_id,
        analyses=recovered.frames,
        module_id=module_id,
        policy=policy,
        facility_id=facility_id,
        fall_model=fall_model,
        clock=clock,
        truncation=recovered.truncation,
    )


def replay_camera(
    *,
    camera_id: str,
    analyses: Sequence[AnalysisTrace],
    module_id: str,
    policy: EffectivePolicy,
    facility_id: str = "replay",
    fall_model: FallModelProtocol | None = None,
    clock: Callable[[], datetime] = _STATIC_CLOCK,
    truncation: TraceTruncation | None = None,
) -> ReplayRun:
    """Replay one camera's persisted, boot-partitioned analysis frames.

    ``analyses`` must already be ordered exactly as
    ``TraceStore.recover_camera`` returns them (boot chronology ascending, then
    stream epoch/seq within each boot). Replay never reorders or interpolates
    frames -- a gap in ``frame_seq`` is replayed as a gap, not filled in.

    At every change of ``worker_boot_id`` the camera-local decider and live-track
    window are recreated, matching production process restart. Stream-epoch
    changes inside one boot keep the same state, matching production reconnect.
    """
    definition = DETECTION_MODULE_REGISTRY.get(module_id)
    expected_schema = f"{policy.schema_id}.v{policy.schema_version}"
    if definition.policy_schema.qualified_id != expected_schema:
        message = (
            f"policy schema {expected_schema} does not match "
            f"module {definition.qualified_id!r} schema "
            f"{definition.policy_schema.qualified_id!r}"
        )
        raise ReplayConfigurationError(message)
    shared_components: dict[str, object] = {}
    if "fall-classifier" in definition.camera_component_ids | {
        binding.component_id for binding in definition.shared_bindings
    }:
        if fall_model is None:
            raise ReplayConfigurationError(
                f"module {definition.qualified_id!r} requires a fall_model for replay"
            )
        shared_components["fall-classifier"] = fall_model

    reproducible, non_reproducible_reason = assess_reproducibility(analyses, truncation)
    frames: list[ReplayFrameResult] = []
    boot_ids: list[str] = []
    current_boot: str | None = None
    decider: Decider | None = None
    live_window: _LiveTrackWindow | None = None

    for analysis in analyses:
        boot_id = analysis.frame_key[0]
        if boot_id != current_boot:
            current_boot = boot_id
            boot_ids.append(boot_id)
            context = CameraModuleContext(
                camera_id=camera_id,
                facility_id=facility_id,
                shared_components=shared_components,
                camera_components={},
                detection_window=None,
                clock=clock,
                diagnostics=None,
                policy=policy,
            )
            camera_module = definition.create_camera_module(context)
            decider = camera_module.decider
            live_window = _LiveTrackWindow()
        assert decider is not None
        assert live_window is not None
        seen_ids = frozenset(
            resolved
            for person in analysis.persons
            if (resolved := replayed_track_id(person.track_id.value)) is not None
        )
        live_ids = live_window.update(seen_ids)
        decision_input = analysis_trace_to_decision_input(analysis, live_track_ids=live_ids)
        events = decider.update(decision_input)
        snapshots = _decider_trace_snapshots(decider, definition)
        frames.append(
            ReplayFrameResult(
                frame_key=analysis.frame_key,
                analysis_trace_id=analysis.trace_id,
                events=events,
                snapshots=snapshots,
            )
        )
    return ReplayRun(
        camera_id=camera_id,
        module_qualified_id=definition.qualified_id,
        policy_qualified_id=definition.policy_schema.qualified_id,
        effective_policy_id=policy.effective_policy_id,
        frames=tuple(frames),
        reproducible=reproducible,
        non_reproducible_reason=non_reproducible_reason,
        boot_ids=tuple(boot_ids),
    )


def _decider_trace_snapshots(
    decider: Decider, definition: DetectionModuleDefinition
) -> tuple[DecisionTraceSnapshot, ...]:
    if definition.trace_adapter is None:
        return ()
    value = definition.trace_adapter(decider)
    if isinstance(value, DecisionTraceSnapshot):
        return (value,)
    if isinstance(value, tuple) and all(isinstance(item, DecisionTraceSnapshot) for item in value):
        return value
    return ()


def decision_traces_for_analysis(
    decisions: Sequence[DecisionTrace], analysis_trace_id: str, module_qualified_id: str
) -> tuple[DecisionTrace, ...]:
    """Select the originally persisted decisions for one frame and module."""
    return tuple(
        decision
        for decision in decisions
        if decision.analysis_trace_id == analysis_trace_id
        and decision.module_qualified_id == module_qualified_id
    )


__all__ = [
    "ReplayConfigurationError",
    "ReplayFrameResult",
    "ReplayRun",
    "assess_reproducibility",
    "decision_traces_for_analysis",
    "replay_camera",
    "replay_recovered",
]
