"""Smart Record against a live RTSP source, per the DS docs.

The docs say: only RTSP sources are enabled for smart record; recording cannot
start until an I-frame is in the cache; start_time is seconds *before* now and
the cache must exceed it; overlapping records are unsupported.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from pyservicemaker import Flow, Pipeline, RecordConfig, RenderMode

URI = os.environ["SR_RTSP_URI"]
OUT = Path(os.environ.get("SR_OUT_DIR", "/output/records"))
OUT.mkdir(parents=True, exist_ok=True)

events: list[dict] = []
done = threading.Event()


def on_done(*args) -> None:  # noqa: ANN002 - vendor callback shape
    events.append({"t": time.monotonic(), "event": "sr-done", "args": [str(a) for a in args]})
    done.set()


pipeline = Pipeline("sr-live")
record = RecordConfig(
    recording_type="local",
    rec_dir_path=str(OUT),
    rec_cache=20,
    rec_container=0,
    rec_mode=1,
)
flow = (
    Flow(pipeline)
    .batch_capture([URI], record_config=record, width=640, height=360)
    .render(mode=RenderMode.DISCARD, enable_osd=False, sync=False)
)
# The DS docs: Smart Record requires an RTSP source. These cameras negotiate
# HEVC over UDP badly, so pin RTP over TCP (select-rtp-protocol=4) before the
# pipeline leaves NULL; also give the jitterbuffer a sane latency.
pipeline["batch_capture-source-0_0"].set(
    {"select-rtp-protocol": 4, "latency": 200, "low-latency-mode": True}
)

state = {"source": None}


def drive() -> None:
    time.sleep(12)
    source = "batch_capture-source-0_0"
    # Clean stop semantics: one session, explicit stop after 4 s of a 20 s window.
    s3 = pipeline.start_recording(source, 0, 20, on_done)
    events.append({"t": time.monotonic(), "event": "start(0,20)", "args": [s3]})
    time.sleep(4)
    stopped = pipeline.stop_recording(source)
    events.append({"t": time.monotonic(), "event": "stop-after-4s", "args": [stopped]})
    finished = done.wait(timeout=25)
    events.append({"t": time.monotonic(), "event": "done-wait", "args": [finished]})
    time.sleep(1)
    events.append(
        {"t": time.monotonic(), "event": "files", "args": sorted(p.name for p in OUT.glob("*"))}
    )
    _write_report()
    pipeline.stop()


def _write_report() -> None:
    report = {"events": events, "files": sorted(str(p) for p in OUT.glob("*"))}
    for p in OUT.glob("*"):
        report.setdefault("sizes", {})[p.name] = p.stat().st_size
    Path(os.environ.get("SR_REPORT", "/output/sr-live.json")).write_text(
        json.dumps(report, indent=2)
    )


threading.Thread(target=drive, daemon=True).start()
try:
    flow()
except Exception as exc:  # noqa: BLE001 - measurement records the failure
    events.append({"t": time.monotonic(), "event": "flow-error", "args": [repr(exc)]})

_write_report()
sys.exit(0)
