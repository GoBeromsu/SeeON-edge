#!/usr/bin/env python3
"""Measurement-only G8a pyservicemaker spike; never import this from worker code."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import threading
import time
from pathlib import Path

import yaml


def command(argv: list[str]) -> dict[str, object]:
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    return {
        "command": " ".join(argv),
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def percentile(values: list[int], fraction: float) -> int:
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def measure_smart_record() -> dict[str, object]:
    from pyservicemaker import Flow, Pipeline, RecordConfig, RenderMode

    recordings = Path("/output/records")
    recordings.mkdir(exist_ok=True)
    source_config = Path("/output/local-source.yml")
    source_config.write_text(
        "source-list:\n"
        "  - sensor-id: spike-camera-0\n"
        "    sensor-name: spike-camera-0\n"
        "    uri: file:///opt/nvidia/deepstream/deepstream-9.1/samples/streams/sample_720p.mp4\n"
    )
    pipeline = Pipeline("p1b-smart-record")
    Flow(pipeline).batch_capture(
        str(source_config),
        record_config=RecordConfig(
            recording_type="local", rec_cache=1, rec_dir_path=str(recordings)
        ),
        file_loop=True,
    ).render(RenderMode.DISCARD)
    events: list[dict[str, object]] = []
    thread = threading.Thread(target=lambda: pipeline.start().wait(), daemon=True)
    thread.start()
    time.sleep(3)

    def done(*callback_args: object) -> None:
        events.append({"event": "sr-done", "at_s": time.monotonic(), "args": repr(callback_args)})

    source_name = "spike-camera-0"
    session = pipeline.start_recording(source_name, 0, 4, done)
    events.append({"event": "start", "at_s": time.monotonic(), "session": session})
    time.sleep(2)
    events.append(
        {
            "event": "stop",
            "at_s": time.monotonic(),
            "result": pipeline.stop_recording(source_name),
        }
    )
    time.sleep(5)
    pipeline.stop()
    thread.join(timeout=5)
    return {
        "events": events,
        "files": sorted(str(path) for path in recordings.glob("*")),
        "source_name": source_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    started = time.monotonic_ns()
    from pyservicemaker import Flow, RecordConfig  # container-only import

    flow_source = inspect.getsource(Flow)
    exports = sorted(name for name in dir(__import__("pyservicemaker")) if not name.startswith("_"))
    flow_methods = sorted(name for name, member in inspect.getmembers(Flow, inspect.isfunction))
    record_source_lines = [
        line.strip()
        for line in flow_source.splitlines()
        if "sr" in line.lower() or "record" in line.lower()
    ]
    smart_record_measurement = measure_smart_record()

    # This is intentionally a simulated bounded-copy workload, not a Flow probe.
    # It supplies a reproducible baseline while an inference engine/parser is absent.
    callbacks = int(config["probe"]["callbacks"])
    sample = {"source_id": 0, "frame_num": 1, "objects": ((1, 0.9, (1, 2, 3, 4)),)}
    latencies_ns: list[int] = []
    for _sequence in range(callbacks):
        then = time.perf_counter_ns()
        copied = (sample["source_id"], sample["frame_num"], tuple(sample["objects"]))
        if copied[1] != 1:
            raise AssertionError("bounded-copy invariant failed")
        latencies_ns.append(time.perf_counter_ns() - then)
    elapsed_s = (time.monotonic_ns() - started) / 1_000_000_000

    model_pt = Path(config["model_pt"])
    report = {
        "harness": "pyservicemaker-p1b-spike",
        "config": config,
        "environment": {
            "python": command(["python3", "--version"]),
            "gpu": command(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
            ),
            "cuda_processes_after_pyservicemaker_import": command(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader",
                ]
            ),
            "pyservicemaker_file": __import__("pyservicemaker").__file__,
            "exports": exports,
        },
        "items": {
            "smart_record": {
                "verdict": "not proven",
                "measurement": {
                    "runtime": smart_record_measurement,
                    "record_config_signature": str(inspect.signature(RecordConfig)),
                    "flow_record_related_source": record_source_lines,
                    "flow_methods": flow_methods,
                    "has_extend_api": any("extend" in name.lower() for name in flow_methods),
                },
                "blocker": (
                    "Flow exposes start-sr/stop-sr attachment only in the cloud "
                    "RecordConfig branch; no public local start/stop/sr-done "
                    "invocation or extend call was found."
                ),
            },
            "metadata_path": {
                "verdict": "blocked",
                "measurement": {
                    "batch_capture_signature": str(inspect.signature(Flow.batch_capture)),
                    "infer_signature": str(inspect.signature(Flow.infer)),
                    "track_signature": str(inspect.signature(Flow.track)),
                    "source": config["source"],
                    "model_pt_exists": model_pt.exists(),
                },
                "blocker": (
                    "The mounted source is a .pt file; no ONNX/FP16 engine or "
                    "YOLO-pose custom parser configuration was available to "
                    "construct the required nvinfer chain. No object metadata "
                    "or 10-minute ID-switch number was fabricated."
                ),
            },
            "probe_latency": {
                "verdict": "not proven",
                "measurement": {
                    "mode": "simulated bounded metadata copy; not a Flow callback",
                    "callbacks": callbacks,
                    "target_callbacks_per_second": config["probe"]["cameras"]
                    * config["probe"]["fps"],
                    "elapsed_s_including_container_import": elapsed_s,
                    "latency_ns": {
                        "p50": percentile(latencies_ns, 0.50),
                        "p95": percentile(latencies_ns, 0.95),
                        "p99": percentile(latencies_ns, 0.99),
                        "max": max(latencies_ns),
                    },
                    "drops": "not measurable without a running Flow callback",
                },
                "blocker": (
                    "A populated Flow metadata callback requires the blocked "
                    "inference chain; this number is only a Python bounded-copy baseline."
                ),
            },
            "single_cuda_owner": {
                "verdict": "not proven",
                "measurement": {
                    "compute_processes": (
                        "see environment.cuda_processes_after_pyservicemaker_import"
                    )
                },
                "blocker": (
                    "Importing pyservicemaker without starting a DeepStream pipeline "
                    "does not establish context ownership, and no ORT CPU fall model "
                    "was supplied."
                ),
            },
            "cold_start_engine_build": {
                "verdict": "blocked",
                "measurement": {
                    "model_pt_exists": model_pt.exists(),
                    "model_pt_bytes": model_pt.stat().st_size if model_pt.exists() else None,
                    "ultralytics": command(
                        ["python3", "-c", "import ultralytics; print(ultralytics.__version__)"]
                    ),
                },
                "blocker": (
                    "The image lacks ultralytics, and the repository supplies no "
                    "exported ONNX, TensorRT engine, or parser config. Engine build "
                    "was therefore not attempted and no duration was invented."
                ),
            },
            "media_plane_counters": {
                "verdict": "blocked",
                "measurement": {
                    "flow_methods": flow_methods,
                    "counter_like_methods": [
                        name
                        for name in flow_methods
                        if any(
                            word in name.lower()
                            for word in ("counter", "frame", "drop", "state", "stat")
                        )
                    ],
                },
                "blocker": (
                    "No Flow public method named for frames-in/out, drops, or "
                    "per-source state was exposed by this installed API; a running "
                    "pipeline was also blocked by the missing engine/parser chain."
                ),
            },
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
