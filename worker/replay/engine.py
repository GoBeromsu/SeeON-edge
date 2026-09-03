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

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from contracts.replay_trace import ReplayRow
from shared.detection_policies import EffectivePolicy
from worker.domains.fall import FallModelProtocol
from worker.domains.module_definition import (
    CameraModuleContext,
    DetectionModuleDefinition,
)
from worker.domains.registry import DETECTION_MODULE_REGISTRY
from worker.interfaces.decision import Decider
from worker.pipeline.decision.incident_manager import IncidentManager
from worker.pipeline.perception.pts_resample import resample_pts
from worker.pipeline.trace.models import (
    AnalysisTrace,
    RecoveredCameraTrace,
    TraceTruncation,
)
from worker.replay.inputs import (
    analysis_trace_to_decision_input,
    replay_trace_to_decision_input,
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
    stream_epoch: int | None = None
    seq: int | None = None
    pts_ns: int | None = None
    valid: int = 1


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
    incident_cooldown_suppressed_total: int = 0
    track_id_switch_total: int = 0
    track_id_switch_absorbed_total: int = 0

    @property
    def event_count(self) -> int:
        return sum(len(frame.events) for frame in self.frames)


def assess_reproducibility(
    analyses: Sequence[AnalysisTrace], truncation: TraceTruncation | None
) -> tuple[bool, str | None]:
    """Mark incomplete legacy recoveries without pretending they are exact."""
    if truncation is None:
        return True, None
    counts = (
        ("handoff_dropped_frames", truncation.handoff_dropped_frames),
        ("pruned_frames", truncation.pruned_frames),
        ("persistence_failed_frames", truncation.persistence_failed_frames),
        ("retention_blocked_frames", truncation.retention_blocked_frames),
    )
    reasons = [f"{name}={count}" for name, count in counts if count]
    return (
        (True, None)
        if not reasons
        else (False, "truncated-or-incomplete-initial-state: " + ", ".join(reasons))
    )


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
    """Legacy HTTP replay retained until that production endpoint is retired."""
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
    """Legacy AnalysisTrace replay for the surviving production HTTP endpoint."""
    definition, shared_components = _replay_components(module_id, policy, fall_model)
    # AnalysisTrace omits association liveness, so it can never reproduce the
    # camera-local tracker state used by production.  Keep the endpoint, but
    # make its limitation explicit rather than presenting a deterministic run.
    _, truncation_reason = assess_reproducibility(analyses, truncation)
    reproducible = False
    reason = "legacy-trace-liveness"
    if truncation_reason is not None:
        reason = f"{reason}; {truncation_reason}"
    frames: list[ReplayFrameResult] = []
    incident_manager = IncidentManager()
    suppressed_total = 0
    current_boot: str | None = None
    boot_ids: list[str] = []
    decider: Decider | None = None
    for analysis in analyses:
        if analysis.frame_key[0] != current_boot:
            current_boot = analysis.frame_key[0]
            boot_ids.append(current_boot)
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
            decider = definition.create_camera_module(context).decider
        assert decider is not None
        live_ids = tuple(
            sorted(
                track_id
                for person in analysis.persons
                if (track_id := replayed_track_id(person.track_id.value)) is not None
            )
        )
        events, suppressed = _admit_events(
            incident_manager,
            decider.update(analysis_trace_to_decision_input(analysis, live_track_ids=live_ids)),
        )
        suppressed_total += suppressed
        frames.append(
            ReplayFrameResult(
                frame_key=analysis.frame_key,
                analysis_trace_id=analysis.trace_id,
                events=events,
                snapshots=_decider_trace_snapshots(decider, definition),
            )
        )
    return ReplayRun(
        camera_id=camera_id,
        module_qualified_id=definition.qualified_id,
        policy_qualified_id=definition.policy_schema.qualified_id,
        effective_policy_id=policy.effective_policy_id,
        frames=tuple(frames),
        reproducible=reproducible,
        non_reproducible_reason=reason,
        boot_ids=tuple(boot_ids),
        incident_cooldown_suppressed_total=suppressed_total,
    )


class _ReplayWindow:
    """Mutable per-frame window gate matching the runtime's evaluated window."""

    def __init__(self) -> None:
        self.active = False

    def contains(self, _: datetime) -> bool:
        return self.active


@dataclass(frozen=True, slots=True)
class ReplayTraceFrame:
    """One resampled V2 replay frame; invalid rows represent a PTS gap."""

    stream_epoch: int
    boot_segment: int
    seq: int
    pts_ns: int
    row: ReplayRow | None
    valid: int


def boot_segments(rows: Sequence[ReplayRow]) -> tuple[int, ...]:
    """Boot segment ordinal for every row, in file order.

    Every ``open`` control starts a new segment. Rows that precede the first
    ``open`` (an allowed truncated start) form their own implicit segment 0,
    so the first explicit boot never merges with a retained tail.
    """
    segments: list[int] = []
    segment = -1
    for row in rows:
        if row.source_event == "open" or segment < 0:
            segment += 1
        segments.append(segment)
    return tuple(segments)


def replay_trace_frames(rows: Sequence[ReplayRow]) -> tuple[ReplayTraceFrame, ...]:
    """Resample only frame rows, grouped by (boot segment, stream epoch) in file order."""
    grouped: dict[tuple[int, int], list[ReplayRow]] = {}
    for row, segment in zip(rows, boot_segments(rows), strict=True):
        if row.source_event != "frame":
            continue
        grouped.setdefault((segment, row.epoch), []).append(row)
    output: list[ReplayTraceFrame] = []
    for (segment, epoch), group in grouped.items():
        samples = [(row.pts_ns, row) for row in sorted(group, key=lambda row: row.seq)]
        for seq, sampled in enumerate(resample_pts(samples)):
            if sampled.valid:
                assert sampled.value is not None
                output.append(
                    ReplayTraceFrame(epoch, segment, seq, sampled.pts_ns, sampled.value, 1)
                )
            else:
                output.append(ReplayTraceFrame(epoch, segment, seq, sampled.pts_ns, None, 0))
    return tuple(output)


def replay(
    *,
    camera_id: str,
    rows: Sequence[ReplayRow],
    module_id: str,
    policy: EffectivePolicy,
    facility_id: str = "replay",
    fall_model: FallModelProtocol | None = None,
    clock: Callable[[], datetime] = _STATIC_CLOCK,
) -> ReplayRun:
    """Replay frame-level rows through production resampling and admission."""
    if any(row.camera_id != camera_id for row in rows):
        raise ReplayConfigurationError("all replay rows must belong to camera_id")
    definition, shared_components = _replay_components(module_id, policy, fall_model)
    frames: list[ReplayFrameResult] = []
    incident_manager = IncidentManager()
    suppressed_total = 0
    switch_total = 0
    previous_live_ids: set[int] = set()
    decider: Decider | None = None
    window = _ReplayWindow()

    def new_decider() -> Decider:
        context = CameraModuleContext(
            camera_id=camera_id,
            facility_id=facility_id,
            shared_components=shared_components,
            camera_components={},
            detection_window=window if module_id == "bed_exit" else None,
            clock=clock,
            diagnostics=None,
            policy=policy,
        )
        return definition.create_camera_module(context).decider

    current_segment: int | None = None
    for frame in replay_trace_frames(rows):
        if current_segment != frame.boot_segment:
            # A worker boot starts with fresh in-memory cooldown state in
            # production (one IncidentManager per camera per composition).
            decider = new_decider()
            incident_manager.reset()
            previous_live_ids = set()
            current_segment = frame.boot_segment
        assert decider is not None
        if frame.row is not None:
            window.active = frame.row.night_window_active
        decision_input = replay_trace_to_decision_input(
            frame.row, pts_ns=frame.pts_ns, seq=frame.seq
        )
        raw_events = decider.update(decision_input)
        events, suppressed = _admit_events(incident_manager, raw_events)
        suppressed_total += suppressed
        if frame.row is not None:
            live_ids = {
                track.track_id
                for track in frame.row.tracks
                if track.lifecycle in ("new", "tracked", "shadow")
            }
            switch_total += sum(
                track.lifecycle == "new" and bool(previous_live_ids - live_ids)
                for track in frame.row.tracks
            )
            previous_live_ids = live_ids
        frames.append(
            ReplayFrameResult(
                frame_key=(
                    f"replay-trace-v2:boot-{frame.boot_segment}",
                    camera_id,
                    frame.stream_epoch,
                    frame.seq,
                ),
                analysis_trace_id=f"v2:{frame.boot_segment}:{frame.stream_epoch}:{frame.seq}",
                events=events,
                snapshots=_decider_trace_snapshots(decider, definition),
                stream_epoch=frame.stream_epoch,
                seq=frame.seq,
                pts_ns=frame.pts_ns,
                valid=frame.valid,
            )
        )
    return ReplayRun(
        camera_id=camera_id,
        module_qualified_id=definition.qualified_id,
        policy_qualified_id=definition.policy_schema.qualified_id,
        effective_policy_id=policy.effective_policy_id,
        frames=tuple(frames),
        boot_ids=tuple(f"boot-{segment}" for segment in sorted(set(boot_segments(rows)))),
        incident_cooldown_suppressed_total=suppressed_total,
        track_id_switch_total=switch_total,
    )


def _admit_events(
    incident_manager: IncidentManager, events: tuple[BusinessEvent, ...]
) -> tuple[tuple[BusinessEvent, ...], int]:
    """Apply production cooldown while retaining deterministic source identities."""
    admitted: list[BusinessEvent] = []
    suppressed = 0
    for event in events:
        if incident_manager.admit(event) is None:
            suppressed += 1
        else:
            admitted.append(event)
    return tuple(admitted), suppressed


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


def _replay_components(
    module_id: str, policy: EffectivePolicy, fall_model: FallModelProtocol | None
) -> tuple[DetectionModuleDefinition, dict[str, object]]:
    definition = DETECTION_MODULE_REGISTRY.get(module_id)
    expected_schema = f"{policy.schema_id}.v{policy.schema_version}"
    if definition.policy_schema.qualified_id != expected_schema:
        raise ReplayConfigurationError("policy schema does not match replay module")
    shared_components: dict[str, object] = {}
    required = definition.camera_component_ids | {
        binding.component_id for binding in definition.shared_bindings
    }
    if "fall-classifier" in required:
        if fall_model is None:
            raise ReplayConfigurationError(
                f"module {definition.qualified_id!r} requires a fall_model for replay"
            )
        shared_components["fall-classifier"] = fall_model
    return definition, shared_components


def replay_run_json(run: ReplayRun) -> str:
    """Encode admitted replay alerts and cooldown accounting for metric CLI input."""
    return json.dumps(
        {
            "incident_cooldown_suppressed_total": run.incident_cooldown_suppressed_total,
            "frames": [
                {
                    "pts_ns": (
                        frame.pts_ns
                        if frame.pts_ns is not None
                        else int(frame.events[0].time_sec * 1_000_000_000)
                        if frame.events
                        else 0
                    ),
                    "events": [
                        {"camera_id": event.camera_id, "event_type": event.event_type}
                        for event in frame.events
                    ],
                }
                for frame in run.frames
            ],
        },
        separators=(",", ":"),
    )


__all__ = [
    "ReplayConfigurationError",
    "ReplayFrameResult",
    "ReplayRun",
    "ReplayTraceFrame",
    "assess_reproducibility",
    "replay",
    "replay_camera",
    "replay_recovered",
    "replay_run_json",
    "replay_trace_frames",
]
