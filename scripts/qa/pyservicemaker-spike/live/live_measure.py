"""P1b gate items 2/3/6 on live facility cameras.

Production-shaped probe: per frame, copy bbox + 17x3 keypoints (from
output-tensor-meta, row-matched to each tracked object) + NvDCF id + lifecycle
+ source id into a bounded per-camera queue, and measure the callback cost.

Credentials arrive only through LIVE_URIS (newline-separated) in the environment.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.dlpack
from pyservicemaker import BatchMetadataOperator, Flow, Pipeline, Probe, RenderMode

REASSOC_SEC = 5.0


class ProductionShapedProbe(BatchMetadataOperator):
    def __init__(self, sources: int, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds
        self.deadline: float | None = None  # armed on the first frame, not at construction
        self.frames = 0
        self.objects = 0
        self.batches = 0
        self.latencies_ms: list[float] = []
        self.rows_matched = 0
        self.rows_unmatched = 0
        self.queues = [collections.deque(maxlen=64) for _ in range(sources)]
        self.drops = 0
        self.ids_seen: dict[int, set[int]] = collections.defaultdict(set)
        self.last_seen: dict[int, dict[int, float]] = collections.defaultdict(dict)
        self.id_births: list[dict] = []
        self.switch_heuristic = (
            0  # a birth within REASSOC_SEC of a disappearance on the same source
        )
        self.first_pts: dict[int, int] = {}
        self.last_pts: dict[int, int] = {}
        self.frames_by_source: dict[int, int] = collections.defaultdict(int)
        self.done = threading.Event()

    def handle_metadata(self, batch_meta) -> None:  # noqa: ANN001 - vendor type
        started = time.perf_counter_ns()
        now = time.monotonic()
        if self.deadline is None:
            self.deadline = now + self.seconds
        for frame in batch_meta.frame_items:
            src = int(frame.pad_index)
            self.frames += 1
            self.frames_by_source[src] += 1
            self.first_pts.setdefault(src, int(frame.buffer_pts))
            self.last_pts[src] = int(frame.buffer_pts)
            rows = None
            for tm in frame.tensor_items:
                layers = tm.as_tensor_output().get_layers()
                t = layers.get("output0") if hasattr(layers, "get") else None
                if t is not None:
                    # Zero-copy via DLPack when host-resident; the tensor is
                    # released with the buffer, so copy the rows we keep.
                    rows = _tensor_rows(t)
                break
            live_now: dict[int, float] = {}
            for obj in frame.object_items:
                self.objects += 1
                oid = int(obj.object_id)
                rp = obj.rect_params
                box = (float(rp.left), float(rp.top), float(rp.width), float(rp.height))
                kps = None
                if rows is not None:
                    # row-to-object association: nearest by box IoU-ish centre distance
                    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
                    best, bestd = None, 1e9
                    for r in rows:
                        if r[4] <= 0.05:
                            continue
                        rcx, rcy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
                        d2 = (rcx - cx) ** 2 + (rcy - cy) ** 2
                        if d2 < bestd:
                            best, bestd = r, d2
                    if best is not None:
                        kps = best[6:57].reshape(17, 3).tolist()
                        self.rows_matched += 1
                    else:
                        self.rows_unmatched += 1
                record = {
                    "source": src,
                    "id": oid,
                    "box": box,
                    "keypoints": kps,
                    "confidence": float(obj.confidence),
                    "pts": int(frame.buffer_pts),
                }
                q = self.queues[src]
                if len(q) == q.maxlen:
                    self.drops += 1
                q.append(record)
                live_now[oid] = now
                if oid not in self.ids_seen[src]:
                    self.ids_seen[src].add(oid)
                    recently_lost = [
                        i
                        for i, t in self.last_seen[src].items()
                        if i != oid and 0 < now - t <= REASSOC_SEC
                    ]
                    self.id_births.append(
                        {"t": round(now, 2), "source": src, "id": oid, "recent_lost": recently_lost}
                    )
                    if recently_lost:
                        self.switch_heuristic += 1
            for oid, t in live_now.items():
                self.last_seen[src][oid] = t
        self.batches += 1
        self.latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
        if self.deadline is not None and now >= self.deadline:
            self.done.set()


def _tensor_rows(tensor) -> np.ndarray:  # noqa: ANN001 - vendor type
    """Consume the binding's DLPack capsule the way NVIDIA's own sample does.

    The output tensor is GPU-resident; torch handles the device capsule and the
    D2H copy. The tensor is released with the buffer, so the rows are copied.
    """
    t = torch.utils.dlpack.from_dlpack(tensor)
    return t.detach().to("cpu", dtype=torch.float32).numpy().reshape(-1, 57).copy()


def write_report(
    probe: ProductionShapedProbe, sources: int, elapsed: float, error: str | None, out: str
) -> None:
    lat = probe.latencies_ms
    per_source_fps = {
        str(s): round(n / max(1e-9, (probe.last_pts[s] - probe.first_pts[s]) / 1e9), 2)
        for s, n in probe.frames_by_source.items()
        if s in probe.last_pts and probe.last_pts[s] > probe.first_pts[s]
    }
    report = {
        "sources": sources,
        "error": error,
        "elapsed_sec": round(elapsed, 1),
        "frames": probe.frames,
        "objects": probe.objects,
        "batch_callbacks": probe.batches,
        "batch_callbacks_per_sec": round(probe.batches / max(elapsed, 1e-9), 2),
        "frame_callbacks_per_sec": round(probe.frames / max(elapsed, 1e-9), 2),
        "per_source_fps_from_pts": per_source_fps,
        "probe_latency_ms": {
            "p50": pct(lat, 0.5),
            "p95": pct(lat, 0.95),
            "p99": pct(lat, 0.99),
            "max": max(lat) if lat else None,
            "mean": round(statistics.fmean(lat), 4) if lat else None,
        },
        "budget_ms_per_batch_at_15fps": round(1000 / 15, 2),
        "queue_drops": probe.drops,
        "keypoint_rows_matched": probe.rows_matched,
        "keypoint_rows_unmatched": probe.rows_unmatched,
        "distinct_ids_by_source": {str(s): len(v) for s, v in probe.ids_seen.items()},
        "id_births": len(probe.id_births),
        "id_switch_heuristic_total": probe.switch_heuristic,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({**report, "births": probe.id_births[:200]}, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)


def pct(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(fraction * (len(s) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-config", required=True)
    ap.add_argument("--tracker-config", required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--ort-fall-model", default=None, help="ONNX fall model to run co-resident on CPU ORT"
    )
    ap.add_argument("--cuda-apps-out", default=None)
    args = ap.parse_args()
    ort_stats: dict = {}
    if args.ort_fall_model:
        import subprocess

        import onnxruntime as ort

        session = ort.InferenceSession(args.ort_fall_model, providers=["CPUExecutionProvider"])
        ort_stats["providers"] = session.get_providers()

        def score_forever() -> None:
            window = np.zeros((1, 30, 56), dtype=np.float32)
            n = 0
            t0 = time.perf_counter()
            while True:
                session.run(None, {"window": window})
                n += 1
                ort_stats["inferences"] = n
                ort_stats["per_sec"] = round(n / max(1e-9, time.perf_counter() - t0), 1)
                time.sleep(
                    1 / 15 / 13
                )  # 13 cameras x one window per 15 fps stride is the P1a budget

        threading.Thread(target=score_forever, daemon=True).start()

        def snapshot_cuda() -> None:
            time.sleep(45)
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
                capture_output=True,
                text=True,
            ).stdout
            rows = [line.strip() for line in out.splitlines() if line.strip()]
            ort_stats["cuda_compute_apps"] = rows
            ort_stats["cuda_process_count"] = max(0, len(rows) - 1)
            ort_stats["ort_inferences_at_snapshot"] = ort_stats.get("inferences")
            if args.cuda_apps_out:
                Path(args.cuda_apps_out).write_text(json.dumps(ort_stats, indent=2))

        threading.Thread(target=snapshot_cuda, daemon=True).start()
    uris = [u.strip() for u in os.environ["LIVE_URIS"].splitlines() if u.strip()]
    probe = ProductionShapedProbe(len(uris), args.seconds)
    pipeline = Pipeline("p1b-live")
    flow = (
        Flow(pipeline)
        .batch_capture(uris, width=640, height=360)
        .infer(args.infer_config)
        .track(
            ll_config_file=args.tracker_config,
            ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        )
        .attach(what=Probe("prod-probe", probe))
        .render(mode=RenderMode.DISCARD, enable_osd=False, sync=False)
    )
    for i in range(len(uris)):
        pipeline[f"batch_capture-source-0_{i}"].set(
            {
                "select-rtp-protocol": 4,
                "latency": 200,
                "init-rtsp-reconnect-interval": 5,
                "rtsp-reconnect-interval": 5,
            }
        )

    def stopper() -> None:
        probe.done.wait()
        write_report(probe, len(uris), time.monotonic() - started, None, args.out)
        pipeline.stop()

    started = time.monotonic()
    threading.Thread(target=stopper, daemon=True).start()
    error = None
    try:
        flow()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    write_report(probe, len(uris), time.monotonic() - started, error, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
