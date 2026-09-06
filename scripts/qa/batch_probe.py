"""Diagnostic-only DeepStream batch probe; never imported by the worker.

Runs an isolated multi-source Flow topology and records frame/pad observations.
It is intended for disposable containers, not production serving.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict
from itertools import pairwise
from typing import Any

from pyservicemaker import BatchMetadataOperator, Pipeline, Probe

_SCORE_MIN = 0.05
N = int(os.environ["N_SOURCES"])
URIS = [uri.strip() for uri in os.environ["DIAG_URIS"].split(",")]
INFER_CFG = os.environ["INFER_CFG"]
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
SECONDS = float(os.environ.get("SECONDS", "12"))
OUT = os.environ["OUT"]


def _pad_permutation() -> list[int]:
    value = os.environ.get("PAD_PERMUTATION")
    if value is None:
        return list(range(N))
    permutation = [int(index.strip()) for index in value.split(",")]
    if sorted(permutation) != list(range(N)):
        raise SystemExit("PAD_PERMUTATION must contain each source index exactly once")
    return permutation


def _tensor_rows(layer: Any) -> list[list[float]] | None:
    """Return pose rows using the shipped adapter when this container has it.

    The diagnostic has no production import dependency.  Containers without the
    adapter retain a serializable raw tensor marker instead of failing the probe.
    """
    try:
        from worker.adapters.deepstream.tensor_rows import rows_from_tensor
    except ImportError:
        try:
            return [[float(value) for value in row] for row in layer if float(row[4]) > _SCORE_MIN]
        except TypeError:
            return None
    rows = rows_from_tensor(layer)
    return [[float(value) for value in row] for row in rows if float(row[4]) > _SCORE_MIN]


def _frame_pose_rows(frame_meta: Any) -> list[list[float]] | None:
    for tensor_meta in frame_meta.tensor_items:
        layers = tensor_meta.as_tensor_output().get_layers()
        layer = layers.get("output0")
        if layer is None:
            continue
        return _tensor_rows(layer)
    return None


class _Recorder(BatchMetadataOperator):
    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []
        self.batches = 0

    def handle_metadata(self, batch_meta: Any) -> None:
        # The SDK iterator hands out one borrowed wrapper that it reuses per
        # step; materializing the iterator aliases every entry to the last frame
        # and dereferences freed metadata. Read each frame inside the loop.
        observations: list[dict[str, Any]] = []
        for frame_meta in batch_meta.frame_items:
            pose_rows = _frame_pose_rows(frame_meta)
            objects = []
            for item in frame_meta.object_items:
                rect = item.rect_params
                objects.append(
                    {
                        "track_id": int(item.object_id),
                        "bbox": {
                            "left": float(rect.left),
                            "top": float(rect.top),
                            "width": float(rect.width),
                            "height": float(rect.height),
                        },
                        "confidence": float(item.confidence),
                    }
                )
            observations.append(
                {
                    "pad": int(frame_meta.pad_index),
                    "frame_number": int(frame_meta.frame_number),
                    "objects": objects,
                    "keypoints": None if pose_rows is None else [row[5:] for row in pose_rows],
                    "raw_pose_rows": pose_rows,
                }
            )
        for observation in observations:
            observation["batch_length"] = len(observations)
        with self.lock:
            self.batches += 1
            self.rows.extend(observations)


def main() -> int:
    if N != 13:
        raise SystemExit("N_SOURCES must be 13 for the batch-serving diagnostic")
    if len(URIS) != N or not all(URIS):
        raise SystemExit("DIAG_URIS must contain 13 comma-separated URIs")
    if len(set(URIS)) != N:
        raise SystemExit("DIAG_URIS entries must be distinct")
    permutation = _pad_permutation()
    no_tracker = os.environ.get("NO_TRACKER") == "1"
    tracker_cfg = None if no_tracker else os.environ["TRACKER_CFG"]

    pipeline = Pipeline("batch-continuity")
    for index, uri in enumerate(URIS):
        pipeline.add("nvurisrcbin", f"src{index}", {"uri": uri, "gpu-id": 0, "file-loop": False})
    pipeline.add(
        "nvstreammux",
        "mux",
        {
            "batch-size": N,
            "width": 640,
            "height": 360,
            "batched-push-timeout": 33000,
            "buffer-pool-size": 4,
            "drop-pipeline-eos": False,
            "live-source": False,
            "gpu-id": 0,
            "compute-hw": 1,
        },
    )
    infer_batch = int(os.environ.get("INFER_BATCH", str(N)))
    pipeline.add("nvinfer", "infer", {"config-file-path": INFER_CFG, "batch-size": infer_batch})
    tracker_element = os.environ.get("TRACKER_ELEMENT", "nvtrackerbin")
    if not no_tracker:
        pipeline.add(
            tracker_element, "tracker", {"ll-config-file": tracker_cfg, "ll-lib-file": TRACKER_LIB}
        )
    pipeline.add("fakesink", "sink", {"sync": False})
    for source_index in permutation:
        pipeline.link((f"src{source_index}", "mux"), ("vsrc_%u", ""))
    pipeline.link("mux", "infer", "sink") if no_tracker else pipeline.link(
        "mux", "infer", "tracker", "sink"
    )

    recorder = _Recorder()
    pipeline.attach("infer" if no_tracker else "tracker", Probe("probe", recorder))
    pipeline.start()
    time.sleep(SECONDS)
    pipeline.stop().wait()

    with recorder.lock:
        rows = list(recorder.rows)
        batches = recorder.batches
    with open(f"{OUT}.rows.jsonl", "w", encoding="utf-8") as sidecar:
        sidecar.writelines(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)

    per_pad: dict[int, list[dict[str, Any]]] = defaultdict(list)
    hist: dict[int, int] = defaultdict(int)
    for row in rows:
        per_pad[row["pad"]].append(row)
        hist[row["batch_length"]] += 1
    summary: dict[str, Any] = {
        "n_sources": N,
        "pad_permutation": permutation,
        "batches": batches,
        "frames_total": len(rows),
        "batch_len_hist": {str(key): value for key, value in sorted(hist.items())},
        "per_pad": {},
    }
    for pad, sequence in sorted(per_pad.items()):
        sequence.sort(key=lambda row: row["frame_number"])
        body = sequence[60:]
        frame_numbers = [row["frame_number"] for row in body]
        missing = sum(end - start - 1 for start, end in pairwise(frame_numbers) if end - start > 1)
        object_frames = [row for row in body if row["objects"]]
        identifiers: dict[int, list[int]] = defaultdict(list)
        for row in body:
            for item in row["objects"]:
                identifiers[item["track_id"]].append(row["frame_number"])
        presence = {
            str(track_id): round(len(numbers) / (numbers[-1] - numbers[0] + 1), 3)
            for track_id, numbers in identifiers.items()
            if numbers[-1] - numbers[0] + 1 >= 30
        }
        gaps: dict[int, int] = defaultdict(int)
        for numbers in identifiers.values():
            for start, end in pairwise(numbers):
                gaps[end - start] += 1
        summary["per_pad"][str(pad)] = {
            "frames": len(body),
            "frame_numbers": [frame_numbers[0], frame_numbers[-1]] if frame_numbers else None,
            "frame_number_holes": missing,
            "frames_with_object": len(object_frames),
            "ids": len(identifiers),
            "presence_per_id": presence,
            "gap_hist": {str(key): value for key, value in sorted(gaps.items())[:8]},
            "even_parity_share": round(
                sum(1 for row in object_frames if row["frame_number"] % 2 == 0)
                / len(object_frames),
                3,
            )
            if object_frames
            else None,
        }
    with open(OUT, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=1)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
