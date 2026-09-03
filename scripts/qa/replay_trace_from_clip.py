"""Generate replay-trace-v2 JSONL from a recorded clip using the pose adapter."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contracts.observation import BoundingBox  # noqa: E402
from contracts.replay_trace import ReplayTraceHeader, ReplayTraceRow, encode_jsonl  # noqa: E402
from worker.adapters.model.yolo_pose import YoloPoseRunner  # noqa: E402
from worker.domains.fall.pose_bbox56 import pose_bbox56_row  # noqa: E402
from worker.pipeline.perception.tracker import GreedyIouTracker  # noqa: E402


def generate(
    clip: Path, output: Path, camera_id: str, device: str, max_seconds: float | None
) -> tuple[int, float]:
    """Run pose inference and write an epoch-zero legacy association trace."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv is required") from exc
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {clip}")
    runner, tracker = YoloPoseRunner(device=device), GreedyIouTracker()
    rows: list[ReplayTraceRow] = []
    seen: set[int] = set()
    started = time.monotonic()
    seq = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        pts_ns = int(capture.get(cv2.CAP_PROP_POS_MSEC) * 1_000_000)
        if max_seconds is not None and pts_ns > max_seconds * 1_000_000_000:
            break
        result = runner.run(frame[:, :, ::-1])
        boxes = tuple(BoundingBox(*box[:4], confidence=box[4]) for box in result.boxes)
        ids = tracker.observe(boxes)
        for pose, box, track_id in zip(result.poses, result.boxes, ids, strict=True):
            rows.append(
                ReplayTraceRow(
                    source="legacy",
                    camera_id=camera_id,
                    stream_epoch=0,
                    seq=seq,
                    pts_ns=pts_ns,
                    track_id=track_id,
                    track_lifecycle="new" if track_id not in seen else "tracked",
                    pose_bbox56=pose_bbox56_row(pose, box[:4], frame.shape[1], frame.shape[0]),
                )
            )
            seen.add(track_id)
        seq += 1
    capture.release()
    elapsed = time.monotonic() - started
    output.write_text(encode_jsonl(ReplayTraceHeader(), rows))
    return seq, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    try:
        frames, elapsed = generate(
            args.clip, args.out, args.camera_id, args.device, args.max_seconds
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"frames={frames} runtime_s={elapsed:.3f} fps={frames / elapsed if elapsed else 0:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
