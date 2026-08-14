"""Internal ML QA CLI: deterministic replay of captured analysis traces.

``python -m worker.replay`` reads persisted, image-free ``AnalysisTrace``
rows for one camera out of an already-migrated ``edge.sqlite3``, replays the
fall or bed-exit decider against an explicit numeric policy (never a
silently-selected default), and prints one stable, machine-consumed JSON
result to stdout. Optionally replays a second ("candidate") policy and prints
a structured comparison instead.

No training, no GPU, no network call, and no wall-clock sleep happens here:
the fall model (when required) loads a local, already-trained CPU artifact
directory; bed-exit needs no model at all. Every numeric input is exactly
what a real camera pipeline already captured and persisted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from shared.detection_policies import (
    BedExitPolicyV1,
    EffectivePolicy,
    FallPolicyV1,
    PolicyDocumentError,
    make_effective_policy,
)
from shared.edge_db.compatibility import EdgeDatabaseError
from worker.domains.fall import FallModelProtocol
from worker.pipeline.trace.models import TraceContractError, TracePersistenceError
from worker.pipeline.trace.store import TraceStore
from worker.replay.comparison import compare_runs
from worker.replay.engine import (
    ReplayConfigurationError,
    ReplayRun,
    replay_recovered,
)

LOGGER = logging.getLogger(__name__)

CLEAN_EXIT_CODE = 0
GENERIC_RUNTIME_ERROR_EXIT_CODE = 1
CONFIG_ERROR_EXIT_CODE = 2
NO_FRAMES_EXIT_CODE = 3
MISMATCH_EXIT_CODE = 4
NON_REPRODUCIBLE_EXIT_CODE = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m worker.replay",
        description=(
            "Deterministic replay of captured analysis traces against a "
            "pinned module and numeric policy for internal ML QA. Never "
            "trains, never uses GPU or network, never sleeps."
        ),
    )
    parser.add_argument(
        "--edge-db",
        type=Path,
        required=True,
        help="Path to an already-migrated edge.sqlite3",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        required=True,
        help="Camera identity whose persisted analysis traces are replayed",
    )
    parser.add_argument(
        "--module",
        choices=("fall", "bed_exit"),
        required=True,
        help="Detection module to replay (must match the traces' original module)",
    )
    parser.add_argument(
        "--operating-threshold",
        type=float,
        default=None,
        help="fall.policy.v1 operating_threshold (required iff --module fall)",
    )
    parser.add_argument(
        "--min-containment",
        type=float,
        default=None,
        help="bed_exit.policy.v1 min_containment (required iff --module bed_exit)",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=None,
        help="bed_exit.policy.v1 hold_frames (required iff --module bed_exit)",
    )
    parser.add_argument(
        "--grace-frames",
        type=int,
        default=None,
        help="bed_exit.policy.v1 grace_frames (required iff --module bed_exit)",
    )
    parser.add_argument(
        "--fall-model-artifact-dir",
        type=Path,
        default=None,
        help=(
            "Local LSTM fall-model artifact directory (CPU only, no network). "
            "Required iff --module fall."
        ),
    )
    parser.add_argument(
        "--candidate-operating-threshold",
        type=float,
        default=None,
        help="Optional second fall threshold: prints an A/B comparison instead of one run",
    )
    parser.add_argument(
        "--candidate-min-containment",
        type=float,
        default=None,
        help="Optional second bed_exit min_containment for an A/B comparison",
    )
    parser.add_argument(
        "--candidate-hold-frames",
        type=int,
        default=None,
        help="Optional second bed_exit hold_frames for an A/B comparison",
    )
    parser.add_argument(
        "--candidate-grace-frames",
        type=int,
        default=None,
        help="Optional second bed_exit grace_frames for an A/B comparison",
    )
    return parser


def _fall_policy(threshold: float) -> EffectivePolicy:
    return make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=threshold),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


def _bed_exit_policy(
    min_containment: float, hold_frames: int, grace_frames: int
) -> EffectivePolicy:
    return make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BedExitPolicyV1(
            min_containment=min_containment, hold_frames=hold_frames, grace_frames=grace_frames
        ),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


def _resolve_policy(args: argparse.Namespace, *, candidate: bool) -> EffectivePolicy | None:
    prefix = "candidate_" if candidate else ""
    if args.module == "fall":
        threshold = getattr(args, f"{prefix}operating_threshold")
        if threshold is None:
            return None
        return _fall_policy(threshold)
    min_containment = getattr(args, f"{prefix}min_containment")
    hold_frames = getattr(args, f"{prefix}hold_frames")
    grace_frames = getattr(args, f"{prefix}grace_frames")
    provided = (min_containment, hold_frames, grace_frames)
    if all(value is None for value in provided):
        return None
    if any(value is None for value in provided):
        raise SystemExit(
            "--min-containment, --hold-frames, and --grace-frames must be given together"
        )
    return _bed_exit_policy(min_containment, hold_frames, grace_frames)


def _load_fall_model(
    artifact_dir: Path | None, operating_threshold: float
) -> FallModelProtocol | None:
    if artifact_dir is None:
        return None
    from worker.adapters.model.torch_lstm_fall import LstmFallRunner

    return LstmFallRunner.from_artifact_dir(
        artifact_dir, device="cpu", operating_threshold=operating_threshold
    )


def _fall_model_for_policy(
    args: argparse.Namespace, policy: EffectivePolicy
) -> FallModelProtocol | None:
    if args.module != "fall" or not isinstance(policy.values, FallPolicyV1):
        return None
    return _load_fall_model(args.fall_model_artifact_dir, policy.values.operating_threshold)


def _run_to_dict(run: ReplayRun) -> dict[str, object]:
    return {
        "boot_ids": list(run.boot_ids),
        "camera_id": run.camera_id,
        "effective_policy_id": run.effective_policy_id,
        "event_count": run.event_count,
        "frame_count": len(run.frames),
        "frames": [
            {
                "analysis_trace_id": frame.analysis_trace_id,
                "events": [
                    {
                        "audit": None if event.audit is None else dict(event.audit),
                        "bed_id": event.bed_id,
                        "camera_id": event.camera_id,
                        "domain": event.domain,
                        "event_type": event.event_type,
                        "facility_id": event.facility_id,
                        "identity": event.identity,
                        "person_id": event.person_id,
                        "probability": event.probability,
                        "time_sec": event.time_sec,
                    }
                    for event in frame.events
                ],
                "frame_key": list(frame.frame_key),
                "snapshots": [
                    {
                        "bed_id": snapshot.bed_id,
                        "current_state": snapshot.current_state,
                        "missing_values": {
                            str(name): str(reason)
                            for name, reason in snapshot.missing_values.items()
                        },
                        "previous_state": snapshot.previous_state,
                        "reason": snapshot.reason,
                        "track_id": snapshot.track_id,
                        "triggered": snapshot.triggered,
                        "values": {str(name): value for name, value in snapshot.values.items()},
                    }
                    for snapshot in frame.snapshots
                ],
            }
            for frame in run.frames
        ],
        "module_qualified_id": run.module_qualified_id,
        "non_reproducible_reason": run.non_reproducible_reason,
        "policy_qualified_id": run.policy_qualified_id,
        "reproducible": run.reproducible,
    }


def _print_stable_json(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    sys.stdout.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")

    try:
        baseline_policy = _resolve_policy(args, candidate=False)
    except SystemExit as exc:
        LOGGER.error(str(exc))  # noqa: TRY400
        return CONFIG_ERROR_EXIT_CODE
    if baseline_policy is None:
        LOGGER.error(
            "no baseline policy provided for module %r; see --help for the required numeric flags",
            args.module,
        )
        return CONFIG_ERROR_EXIT_CODE

    try:
        candidate_policy = _resolve_policy(args, candidate=True)
    except SystemExit as exc:
        LOGGER.error(str(exc))  # noqa: TRY400
        return CONFIG_ERROR_EXIT_CODE

    if args.module == "fall" and args.fall_model_artifact_dir is None:
        LOGGER.error("--fall-model-artifact-dir is required for --module fall")
        return CONFIG_ERROR_EXIT_CODE

    try:
        recovered = TraceStore(args.edge_db).recover_camera(args.camera_id)
    except (
        EdgeDatabaseError,
        TraceContractError,
        TracePersistenceError,
        sqlite3.DatabaseError,
        OSError,
    ) as exc:
        LOGGER.error("could not read edge database: %s", exc)  # noqa: TRY400
        return CONFIG_ERROR_EXIT_CODE

    if not recovered.frames:
        LOGGER.error(
            "no persisted analysis traces for camera_id=%r in %s", args.camera_id, args.edge_db
        )
        return NO_FRAMES_EXIT_CODE

    try:
        baseline_model = _fall_model_for_policy(args, baseline_policy)
        baseline_run = replay_recovered(
            camera_id=args.camera_id,
            recovered=recovered,
            module_id=args.module,
            policy=baseline_policy,
            fall_model=baseline_model,
        )
    except (ReplayConfigurationError, PolicyDocumentError) as exc:
        LOGGER.error("replay configuration refused: %s", exc)  # noqa: TRY400
        return CONFIG_ERROR_EXIT_CODE

    if candidate_policy is None:
        _print_stable_json(_run_to_dict(baseline_run))
        if not baseline_run.reproducible:
            LOGGER.error(
                "recovered timeline is not deterministically reproducible: %s",
                baseline_run.non_reproducible_reason,
            )
            return NON_REPRODUCIBLE_EXIT_CODE
        return CLEAN_EXIT_CODE

    try:
        candidate_model = _fall_model_for_policy(args, candidate_policy)
        candidate_run = replay_recovered(
            camera_id=args.camera_id,
            recovered=recovered,
            module_id=args.module,
            policy=candidate_policy,
            fall_model=candidate_model,
        )
    except (ReplayConfigurationError, PolicyDocumentError) as exc:
        LOGGER.error("replay configuration refused: %s", exc)  # noqa: TRY400
        return CONFIG_ERROR_EXIT_CODE

    comparison = compare_runs(baseline_run, candidate_run)
    _print_stable_json(
        {
            "baseline": _run_to_dict(baseline_run),
            "candidate": _run_to_dict(candidate_run),
            "comparison": comparison.as_dict(),
        }
    )
    if not baseline_run.reproducible or not candidate_run.reproducible:
        LOGGER.error(
            "recovered timeline is not deterministically reproducible: baseline=%s candidate=%s",
            baseline_run.non_reproducible_reason,
            candidate_run.non_reproducible_reason,
        )
        return NON_REPRODUCIBLE_EXIT_CODE
    return MISMATCH_EXIT_CODE if not comparison.identical else CLEAN_EXIT_CODE


__all__ = [
    "CLEAN_EXIT_CODE",
    "CONFIG_ERROR_EXIT_CODE",
    "GENERIC_RUNTIME_ERROR_EXIT_CODE",
    "MISMATCH_EXIT_CODE",
    "NON_REPRODUCIBLE_EXIT_CODE",
    "NO_FRAMES_EXIT_CODE",
    "main",
]
