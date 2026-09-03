"""Diagnostic-only offline replay trace generation; not a production capture path."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contracts.observation import BoundingBox  # noqa: E402
from contracts.replay_trace import (  # noqa: E402
    ReplayRow,
    ReplayTraceHeader,
    ReplayTrack,
    encode_jsonl,
)
from worker.adapters.model.yolo_pose import YoloPoseRunner  # noqa: E402
from worker.pipeline.perception.tracker import GreedyIouTracker  # noqa: E402


def generate(
    clip: Path, output: Path, camera_id: str, device: str, max_seconds: float | None
) -> tuple[int, float]:
    """Diagnostic-only pose inference, never a substitute for production capture."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv is required") from exc
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {clip}")
    runner, tracker = YoloPoseRunner(device=device), GreedyIouTracker()
    rows: list[ReplayRow] = []
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
        height, width = frame.shape[:2]
        ids = tracker.observe(boxes)
        tracks = tuple(
            ReplayTrack(
                track_id=track_id,
                lifecycle="new" if track_id not in seen else "tracked",
                bbox=(
                    box[0] / width,
                    box[1] / height,
                    box[2] / width,
                    box[3] / height,
                    box[4],
                ),
                keypoints=tuple((point[0] / width, point[1] / height, point[2]) for point in pose),
            )
            for pose, box, track_id in zip(result.poses, result.boxes, ids, strict=True)
        )
        seen.update(ids)
        rows.append(
            ReplayRow(
                camera_id=camera_id,
                seq=len(rows),
                pts_ns=pts_ns,
                epoch=0,
                source_event="frame",
                source="legacy-association",
                tracks=tracks,
                bed_polygon_id=None,
                bed_polygon=None,
                night_window_active=False,
                frame_width=width,
                frame_height=height,
            )
        )
        seq += 1
    capture.release()
    elapsed = time.monotonic() - started
    output.write_text(encode_jsonl(ReplayTraceHeader(), rows))
    return seq, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic-only offline trace generator.")
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
