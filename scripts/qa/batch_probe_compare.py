"""Compare two `scripts/qa/batch_probe.py` row sidecars frame by frame.

Diagnostic only. Subject = the engine/batch under test, control = the batch-1
engine on the same file sources (identical decoded frames, keyed by
(pad, frame_number)). Reports the FP16 parity numbers the #503 acceptance
uses: detection-count mismatches (with the scores of the odd box, which should
sit at the pre-cluster gate), matched-box IoU, |dscore|, and keypoint L2 in
network pixels. `--pad-permutation` remaps a run made with PAD_PERMUTATION so
it can be compared against an unpermuted control.

Usage:
    python scripts/qa/batch_probe_compare.py --subject out.json.rows.jsonl \
        --control ctrl.json.rows.jsonl [--pad-permutation 12,3,7,0,9,1,11,5,2,10,4,8,6] \
        [--iou 0.9 --dscore 0.05 --kpt-px 2.0]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_KEYPOINTS = 17
_ROW_SCORE = 4
_ROW_KEYPOINTS = 5
_TRACK_GATE = 0.2


def _load(path: Path, remap: list[int] | None) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            pad = row["pad"] if remap is None else remap[row["pad"]]
            rows[(pad, row["frame_number"])] = row
    return rows


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax2, ay2 = a["left"] + a["width"], a["top"] + a["height"]
    bx2, by2 = b["left"] + b["width"], b["top"] + b["height"]
    iw = max(0.0, min(ax2, bx2) - max(a["left"], b["left"]))
    ih = max(0.0, min(ay2, by2) - max(a["top"], b["top"]))
    inter = iw * ih
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return sorted(values)[min(len(values) - 1, int(len(values) * q))]


def compare(
    subject: dict[tuple[int, int], dict[str, Any]],
    control: dict[tuple[int, int], dict[str, Any]],
    *,
    iou_floor: float,
    dscore_ceiling: float,
    kpt_px: float,
) -> dict[str, Any]:
    keys = sorted(set(subject) & set(control))
    mismatches: list[list[float]] = []
    ious: list[float] = []
    dscores: list[float] = []
    kpts: list[float] = []
    for key in keys:
        c_objects = sorted(control[key]["objects"], key=lambda o: -o["confidence"])
        s_objects = sorted(subject[key]["objects"], key=lambda o: -o["confidence"])
        if len(c_objects) != len(s_objects):
            c_scores = {round(o["confidence"], 4) for o in c_objects}
            s_scores = {round(o["confidence"], 4) for o in s_objects}
            mismatches.append(sorted(c_scores ^ s_scores)[:2])
        for c_obj, s_obj in zip(c_objects, s_objects, strict=False):
            ious.append(_iou(c_obj["bbox"], s_obj["bbox"]))
            dscores.append(abs(c_obj["confidence"] - s_obj["confidence"]))
        c_rows = sorted(control[key]["raw_pose_rows"] or [], key=lambda r: -r[_ROW_SCORE])
        s_rows = sorted(subject[key]["raw_pose_rows"] or [], key=lambda r: -r[_ROW_SCORE])
        for c_row, s_row in zip(c_rows, s_rows, strict=False):
            if c_row[_ROW_SCORE] < _TRACK_GATE or s_row[_ROW_SCORE] < _TRACK_GATE:
                continue
            kpts.append(
                max(
                    math.hypot(
                        c_row[_ROW_KEYPOINTS + 3 * i] - s_row[_ROW_KEYPOINTS + 3 * i],
                        c_row[_ROW_KEYPOINTS + 3 * i + 1] - s_row[_ROW_KEYPOINTS + 3 * i + 1],
                    )
                    for i in range(_KEYPOINTS)
                )
            )
    return {
        "frames_compared": len(keys),
        "frame_keys_only_in_one_run": len(set(subject) ^ set(control)),
        "count_mismatch_frames": len(mismatches),
        "count_mismatch_odd_box_scores": mismatches[:20],
        "matched_boxes": len(ious),
        "iou_p1": _quantile(ious, 0.01),
        "iou_median": _quantile(ious, 0.5),
        "boxes_below_iou_floor": sum(v < iou_floor for v in ious),
        "dscore_max": max(dscores) if dscores else None,
        "boxes_above_dscore_ceiling": sum(v > dscore_ceiling for v in dscores),
        "kpt_px_p99": _quantile(kpts, 0.99),
        "kpts_above_px": sum(v > kpt_px for v in kpts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument(
        "--pad-permutation", default=None, help="PAD_PERMUTATION used for the subject run"
    )
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--dscore", type=float, default=0.05)
    parser.add_argument("--kpt-px", type=float, default=2.0)
    args = parser.parse_args()
    remap = None
    if args.pad_permutation:
        remap = [int(value) for value in args.pad_permutation.split(",")]
    result = compare(
        _load(args.subject, remap),
        _load(args.control, None),
        iou_floor=args.iou,
        dscore_ceiling=args.dscore,
        kpt_px=args.kpt_px,
    )
    print(json.dumps(result, indent=1))
    return 0 if result["frame_keys_only_in_one_run"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
