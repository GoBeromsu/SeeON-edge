#!/usr/bin/env python3
"""Score a packaged fall model against recorded live-camera traces.

Clean 300-frame training clips never exercise PTS resampling, track-id churn,
or reconnect padding -- the exact continuity bugs this bundle exists to catch.
This script instead replays recorded ``replay-trace-v2`` JSONL captures (real
NvDCF track lifecycles, real gaps) through ``worker.replay.engine.replay()``,
the same production compositor the worker boots, so the effective transition
threshold (receipt vs. policy default) is resolved exactly as it is live.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contracts.replay_trace import ReplayRow, decode_jsonl
from shared.detection_policies import FallPolicyV2, make_effective_policy
from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.replay.engine import ReplayRun, replay

DEFAULT_HIT_WINDOW_SEC = 10.0
DEFAULT_EXCLUSION_WINDOW_SEC = 20.0


def _read_frame_rows(path: Path) -> tuple[str, tuple[ReplayRow, ...]]:
    _, rows = decode_jsonl(path.read_text(encoding="utf-8"))
    frame_rows = tuple(row for row in rows if row.source_event == "frame")
    if not frame_rows:
        raise ValueError(f"{path} has no frame rows")
    camera_id = frame_rows[0].camera_id
    return camera_id, frame_rows


def _duration_hours(rows: tuple[ReplayRow, ...]) -> float:
    pts = [row.pts_ns for row in rows]
    return (max(pts) - min(pts)) / 1_000_000_000 / 3600


def _fall_event_offsets_sec(run: ReplayRun, t0_ns: int) -> list[float]:
    return [
        frame.pts_ns / 1_000_000_000 - t0_ns / 1_000_000_000
        for frame in run.frames
        for event in frame.events
        if event.event_type == "fall" and frame.pts_ns is not None
    ]


def _peak_score(run: ReplayRun, t0_ns: int, center_sec: float, window_sec: float) -> float | None:
    best: float | None = None
    for frame in run.frames:
        if frame.pts_ns is None:
            continue
        offset = frame.pts_ns / 1_000_000_000 - t0_ns / 1_000_000_000
        if abs(offset - center_sec) > window_sec:
            continue
        for snapshot in frame.snapshots:
            value = snapshot.values.get("fall_transition_probability")
            if isinstance(value, (int, float)):
                best = value if best is None else max(best, float(value))
    return best


def _window_fraction(run: ReplayRun) -> tuple[int, int]:
    """Count live-track frames the classifier scored vs. still warming up on.

    Only these two reasons are "the classifier was asked to score this live
    track" -- every other missing reason (stride-not-due, resample-gap,
    track-no-longer-live, ...) is not a warm-up/full-window distinction.
    """
    scored = 0
    warmup = 0
    for frame in run.frames:
        for snapshot in frame.snapshots:
            if snapshot.track_id is None:
                continue
            if "fall_transition_probability" in snapshot.values:
                scored += 1
            elif snapshot.missing_values.get("fall_transition_probability") == "classifier-warmup":
                warmup += 1
    return scored, warmup


def _build_policy():
    return make_effective_policy(
        module_id="fall",
        module_version=2,
        values=FallPolicyV2(),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


def score_traces(
    *,
    bundle: Path,
    traces_dir: Path,
    positive_trace: str,
    owner_fall_offsets_sec: list[float],
    hit_window_sec: float = DEFAULT_HIT_WINDOW_SEC,
    exclusion_window_sec: float = DEFAULT_EXCLUSION_WINDOW_SEC,
    runner_factory: Callable[[Path], Any] = OrtPoseBbox56Runner.from_artifact_dir,
    replay_factory: Callable[..., ReplayRun] = replay,
) -> dict[str, Any]:
    trace_files = sorted(traces_dir.glob("*.jsonl"))
    if not trace_files:
        raise ValueError(f"no .jsonl traces found in {traces_dir}")
    if positive_trace not in {path.name for path in trace_files}:
        raise ValueError(f"positive trace {positive_trace!r} not found in {traces_dir}")

    runner = runner_factory(bundle)
    policy = _build_policy()

    owner_fall_hits: list[dict[str, Any]] = []
    false_positive_events = 0
    exposed_hours = 0.0
    scored_total = 0
    warmup_total = 0
    gap_rows_total = 0
    lineage_adopted_total = 0
    per_trace: list[dict[str, Any]] = []

    for path in trace_files:
        camera_id, rows = _read_frame_rows(path)
        t0_ns = min(row.pts_ns for row in rows)
        run = replay_factory(
            camera_id=camera_id, rows=rows, module_id="fall", policy=policy, fall_model=runner
        )
        fall_offsets = _fall_event_offsets_sec(run, t0_ns)
        duration_hours = _duration_hours(rows)
        scored, warmup = _window_fraction(run)
        scored_total += scored
        warmup_total += warmup
        gap_rows_total += run.resample_gap_rows_total
        lineage_adopted_total += run.buffer_lineage_adopted_total

        is_positive = path.name == positive_trace
        if is_positive:
            trace_exposed_hours = duration_hours - len(owner_fall_offsets_sec) * (
                2 * exclusion_window_sec / 3600
            )
            for offset in owner_fall_offsets_sec:
                owner_fall_hits.append(
                    {
                        "offset_sec": offset,
                        "hit": any(abs(t - offset) <= hit_window_sec for t in fall_offsets),
                        "peak_fall_transition_score": _peak_score(
                            run, t0_ns, offset, hit_window_sec
                        ),
                    }
                )
            trace_false_positives = sum(
                1
                for t in fall_offsets
                if all(abs(t - offset) > exclusion_window_sec for offset in owner_fall_offsets_sec)
            )
        else:
            trace_exposed_hours = duration_hours
            trace_false_positives = len(fall_offsets)

        false_positive_events += trace_false_positives
        exposed_hours += max(trace_exposed_hours, 0.0)
        per_trace.append(
            {
                "file": path.name,
                "camera_id": camera_id,
                "frame_count": len(rows),
                "duration_hours": duration_hours,
                "is_positive_trace": is_positive,
                "fall_event_offsets_sec": fall_offsets,
                "false_positive_episode_count": trace_false_positives,
                "resample_gap_rows_total": run.resample_gap_rows_total,
                "buffer_lineage_adopted_total": run.buffer_lineage_adopted_total,
                "track_id_switch_total": run.track_id_switch_total,
            }
        )

    scored_denominator = scored_total + warmup_total
    return {
        "receipt_version": 1,
        "status": "measured",
        "method": (
            "worker.replay.engine.replay() over recorded replay-trace-v2 JSONL captures "
            "(production resampling, track lifecycle, and reconnect padding), not clean "
            "300-frame training clips"
        ),
        "bundle": str(bundle),
        "model_receipt_threshold": getattr(runner, "receipt_threshold", None),
        "model_promotion_eligible": getattr(runner, "promotion_eligible", None),
        "traces_dir": str(traces_dir),
        "positive_trace": positive_trace,
        "hit_window_sec": hit_window_sec,
        "exclusion_window_sec": exclusion_window_sec,
        "owner_fall_hits": owner_fall_hits,
        "false_positive_episode_count": false_positive_events,
        "exposed_camera_hours": exposed_hours,
        "false_positive_episodes_per_camera_hour": (
            false_positive_events / exposed_hours if exposed_hours else None
        ),
        "live_track_frames_scored": scored_total,
        "live_track_frames_classifier_warmup": warmup_total,
        "fraction_live_track_frames_with_full_window": (
            scored_total / scored_denominator if scored_denominator else None
        ),
        "resample_gap_rows_total": gap_rows_total,
        "buffer_lineage_adopted_total": lineage_adopted_total,
        "per_trace": per_trace,
        "scope_note": (
            "False-positive exposure on the positive trace excludes the "
            f"+/-{exclusion_window_sec:g}s window around each declared owner fall; "
            "episodes inside that window are scored only as owner-fall hits, never "
            "double-counted as false positives."
        ),
    }


def write_receipt(out: Path, receipt: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("models/fall/pose-bbox56-gru"))
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument(
        "--positive-trace",
        type=str,
        required=True,
        help="filename within --traces-dir that contains the known owner falls",
    )
    parser.add_argument(
        "--owner-fall-offsets-sec",
        type=float,
        nargs="+",
        required=True,
        help="seconds from trace t0 (min pts_ns) for each known owner fall",
    )
    parser.add_argument("--hit-window-sec", type=float, default=DEFAULT_HIT_WINDOW_SEC)
    parser.add_argument("--exclusion-window-sec", type=float, default=DEFAULT_EXCLUSION_WINDOW_SEC)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = score_traces(
        bundle=args.bundle,
        traces_dir=args.traces_dir,
        positive_trace=args.positive_trace,
        owner_fall_offsets_sec=args.owner_fall_offsets_sec,
        hit_window_sec=args.hit_window_sec,
        exclusion_window_sec=args.exclusion_window_sec,
    )
    write_receipt(args.out, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "per_trace"}, indent=2))


if __name__ == "__main__":
    main()
