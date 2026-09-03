#!/usr/bin/env python3
"""Measurement-only G8a pyservicemaker spike; never import this from worker code."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path


def measure_smart_record() -> dict[str, object]:
    from pyservicemaker import Flow, Pipeline, RecordConfig, RenderMode

    recordings = Path("/work/records")
    recordings.mkdir(exist_ok=True)
    source_config = Path("/work/local-source.yml")
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "harness": "pyservicemaker-p1b-spike",
        "item_1_smart_record": measure_smart_record(),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
