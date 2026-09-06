"""Decisive P1b item-2/3/6 measurement: parsed nvinfer -> NvDCF over a Flow.

Counts frames, objects and distinct tracker ids seen through a real Flow probe,
plus per-callback latency, so the gate question ("can pyservicemaker carry the
perception path with tracker identity?") is answered with numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from pyservicemaker import BatchMetadataOperator, Flow, Pipeline, Probe


class Counter(BatchMetadataOperator):
    def __init__(self, cuda_apps_out: Path | None = None) -> None:
        super().__init__()
        self.frames = 0
        self.objects = 0
        self.track_ids: set[int] = set()
        self.latencies_ms: list[float] = []
        self.first_ns: int | None = None
        self.last_ns: int | None = None
        self.cuda_apps_out = cuda_apps_out

    def handle_metadata(self, batch_meta) -> None:  # noqa: ANN001 - vendor type
        started = time.perf_counter_ns()
        if self.first_ns is None:
            self.first_ns = started
            if self.cuda_apps_out is not None:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                header, *data = rows or [
                    "",
                ]
                # JSON, not CSV: the repository privacy gate refuses tracked
                # data-asset extensions, and a process count needs its caveat
                # travelling with it.
                self.cuda_apps_out.write_text(
                    json.dumps(
                        {
                            "command": (
                                "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
                            ),
                            "header": header,
                            "rows": data,
                            "process_count": len(data),
                            "note": (
                                "A process count is not a CUDA context count, and the CPU-ORT "
                                "fall model was not co-resident, so this does not satisfy the "
                                "single-CUDA-owner criterion."
                            ),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        for frame_meta in batch_meta.frame_items:
            self.frames += 1
            for object_meta in frame_meta.object_items:
                self.objects += 1
                self.track_ids.add(int(object_meta.object_id))
        finished = time.perf_counter_ns()
        self.last_ns = finished
        self.latencies_ms.append((finished - started) / 1e6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--sources", type=int, default=1)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--infer-config", required=True)
    parser.add_argument("--tracker-config", required=True)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cuda-apps-out", type=Path)
    args = parser.parse_args()

    counter = Counter(args.cuda_apps_out)
    flow = Flow(Pipeline("p1b-spike"))
    started = time.perf_counter()
    error: str | None = None
    try:
        (
            flow.batch_capture([args.uri] * args.sources)
            .infer(args.infer_config)
            .track(
                ll_config_file=args.tracker_config,
                ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
            )
            .attach(what=Probe("counter", counter))
            .render(enable_osd=False, sync=args.sync)()
        )
    except Exception as exc:  # noqa: BLE001 - the measurement records the failure
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started

    latencies = sorted(counter.latencies_ms)

    def pct(fraction: float) -> float | None:
        if not latencies:
            return None
        index = min(len(latencies) - 1, max(0, round(fraction * (len(latencies) - 1))))
        return latencies[index]

    report = {
        "uri": args.uri,
        "sources": args.sources,
        "sync": args.sync,
        "error": error,
        "elapsed_sec": round(elapsed, 3),
        "frames": counter.frames,
        "objects": counter.objects,
        "distinct_track_ids": len(counter.track_ids),
        "sample_track_ids": sorted(counter.track_ids)[:12],
        "probe_callbacks": len(latencies),
        "probe_latency_ms": {
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": latencies[-1] if latencies else None,
            "mean": round(statistics.fmean(latencies), 4) if latencies else None,
        },
        "observed_fps": (
            round(counter.frames / elapsed, 2) if elapsed > 0 and counter.frames else None
        ),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
