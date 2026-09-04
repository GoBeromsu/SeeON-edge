"""Capture real nvinfer pose rows beside the tracker boxes they belong to.

The letterbox inverse in ``worker/adapters/deepstream/metadata.py`` was wrong for
an entire bring-up while its unit test passed, because the test restated the
implementation's own assumption about where nvinfer puts the padding. This tool
records ground truth instead: for a handful of live frames it writes the raw
57-wide pose rows exactly as nvinfer produced them, together with the frame-space
boxes nvtracker attached to the same frame. A test can then assert that the
inverse maps one onto the other without anyone having to assert the padding
convention from memory.

Runs against a live RTSP source inside the shipped image. Never imported by the
worker; ``LIVE_URIS`` supplies the source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, "/app")

from worker.adapters.deepstream.tensor_rows import rows_from_tensor  # noqa: E402

_SCORE_MIN = 0.25


class _Capture:
    def __init__(self, frames_wanted: int) -> None:
        self.frames_wanted = frames_wanted
        self.samples: list[dict[str, Any]] = []
        self.done = threading.Event()

    def handle_metadata(self, batch_meta: Any) -> None:
        if self.done.is_set():
            return
        for frame in batch_meta.frame_items:
            rows = None
            for tensor_meta in frame.tensor_items:
                rows = rows_from_tensor(tensor_meta.as_tensor_output().get_layers()["output0"])
                break
            if rows is None:
                continue
            boxes = []
            for obj in frame.object_items:
                rect = obj.rect_params
                boxes.append(
                    {
                        "track_id": int(obj.object_id),
                        "left": float(rect.left),
                        "top": float(rect.top),
                        "width": float(rect.width),
                        "height": float(rect.height),
                        "confidence": float(obj.confidence),
                    }
                )
            scored = [
                {"index": index, "row": [float(value) for value in row]}
                for index, row in enumerate(rows)
                if float(row[4]) > _SCORE_MIN
            ]
            if not boxes or not scored:
                continue
            self.samples.append({"boxes": boxes, "rows": scored})
            if len(self.samples) >= self.frames_wanted:
                self.done.set()
                return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer-config", required=True)
    parser.add_argument("--tracker-config", required=True)
    parser.add_argument("--tracker-library", required=True)
    parser.add_argument("--frame-width", type=int, required=True)
    parser.add_argument("--frame-height", type=int, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from pyservicemaker import BatchMetadataOperator, Flow, Pipeline, Probe, RenderMode

    uris = [line.strip() for line in os.environ["LIVE_URIS"].splitlines() if line.strip()]
    capture = _Capture(args.frames)

    class _Operator(BatchMetadataOperator):
        def handle_metadata(self, batch_meta: Any) -> None:
            capture.handle_metadata(batch_meta)

    pipeline = Pipeline("letterbox-fixture")
    flow = (
        Flow(pipeline)
        .batch_capture(uris, width=args.frame_width, height=args.frame_height)
        .infer(args.infer_config)
        .track(ll_config_file=args.tracker_config, ll_lib_file=args.tracker_library)
        .attach(what=Probe("capture", _Operator()))
        .render(mode=RenderMode.DISCARD, enable_osd=False, sync=False)
    )
    thread = threading.Thread(target=flow, daemon=True)
    thread.start()
    capture.done.wait(timeout=args.seconds)
    pipeline.stop()

    payload = {
        "source": "live nvinfer + nvtracker capture inside the shipped image",
        "frame_width": args.frame_width,
        "frame_height": args.frame_height,
        "net_size": 640,
        "note": (
            "Rows are raw 57-wide nvinfer outputs in network space; boxes are what nvtracker "
            "attached in frame space for the same frame. A correct letterbox inverse maps a row "
            "onto its box."
        ),
        "samples": capture.samples[: args.frames],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"captured {len(payload['samples'])} frames -> {args.out}", flush=True)
    return 0 if payload["samples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
