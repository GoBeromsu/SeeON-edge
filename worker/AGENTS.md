# worker

RTSP inference client: ingest, bounded bus, extract, decide, evidence, one-way relay egress.
Image `ml-worker`. Sole command: `python -m worker`.

## Layers

Run `uv run --group lint lint-imports`; a forbidden import means the design is wrong: add a Protocol in `interfaces/` and inject from `runtime/`.

| Package | Role | May import |
| --- | --- | --- |
| `types/` | internal envelopes | stdlib, `contracts` |
| `interfaces/` | one Protocol per seam | stdlib, `contracts`, `types` |
| `adapters/` | decode, model, encode | stdlib, `contracts`, `types`, `interfaces` |
| `pipeline/` | ingest, bus, perception, decision, output | everything except `runtime` |
| `domains/` | fall and bed-exit | `contracts`, `types`, `interfaces`, `pipeline.perception` |
| `runtime/` | composition root | everything |

Order: `runtime -> pipeline -> domains -> adapters -> interfaces -> types -> contracts`.
`contracts` contains cross-instance L0 data only. Worker-internal ports and envelopes live under `worker/`; never duplicate or shadow a vendored type, including `contracts/AGENTS.md`.
Allowed: `contracts`, `shared.events`, local `worker.*`. Forbidden: `backend`.

## FramePacket / DecisionInput

`FramePacket` is the only envelope that may carry an image. Four sinks: model extract, derivative evidence, overlay/MJPEG, alert snapshot. Publish immutable. Copy the image before draw or mutate.
`DecisionInput` is numeric: `observation`, `frame_width`, `frame_height`, `live_track_ids`, `time_sec`, `frame_index`, `bed_region`. No array, no buffer, no frame handle.
Domains take `DecisionInput` and return `BusinessEvent` tuples. A detector that needs pixels is a design error. Extract the number in `pipeline/perception` first.

## Shared model vs per-camera state

Shared once per process: models, extractors, profile/device, GPU lease, config/LKG, evidence outbox, clip-store lock.
Per camera, never shared: tracker, `SceneState`, window buffer, fall latch, bed assignment plus grace/hold, `IncidentManager`, bus slot, encoder ring, ingest backoff.
Hoisting a per-camera row leaks one resident into another. `tests/test_worker_per_camera_fall_state.py` asserts both halves.

## Sole CLI and composition root

`worker/__main__.py` owns argparse and exit codes. It constructs `WorkerRuntime` from `worker.runtime.worker` directly.
Don't add `worker.runtime.edge_worker`, an `edge` alias, a console script, or a second `python -m` target.
`runtime/` is the only package that may import everything. Nothing imports `runtime`. No business math and no model parsing live there.
Seam defaults are `None`. Missing wiring refuses to start. Always-fail stubs belong in tests only.
Exit codes: 0 clean, 1 generic, 2 config, 3 refuse-to-start, 4 fatal accelerator.
`--check-config` has no model, camera, or relay side effect.

## Hardware failure policy

Global stages are fatal and start zero cameras: profile/device, decode capability, model backend, real warmup, GPU lease. Exit 3.
`auto`, explicit blank, and unknown profiles fail loud; unset defaults to `cpu`. No silent CPU or OpenCV fallback. NVENC failure never starts `libx264`. `cpu` decode requires OpenCV `CAP_FFMPEG` in the videoio registry. Import success is not capability.
Mid-run device loss writes one first-fault record under `ML_WORKER_STATE_DIR`, stops every camera, hard-exits 4. The CUDA context is never recreated in-process.
Per-camera faults stay local. RTSP fail retries with backoff. Decode stall reopens the source. A full bus drops oldest. Extractor raise drops that frame. Clip encode fail still relays the alert. Relay POST lands in the durable outbox.
Source open failure is camera `DEGRADED` plus `camera.offline`. A later processing failure is not offline. Keep the categories distinct.
Missing evidence wiring or a locked local store refuses to start; remote delivery failures remain retryable. Two workers must not share one outbox.
Primary clips remux source packets from camera-local rings. Decoded frames are analysis and snapshot taps. Never describe a re-encode as the preserved source clip.

## Package navigation

Read the nearest `AGENTS.md` before changing that package.

| Path | Go here for |
| --- | --- |
| `types/` | `FramePacket`, `DecisionInput`, `ModuleResult`, `BusinessEvent` |
| `interfaces/` | decode, bus, extract, decide, encode, output, serving |
| `adapters/decode/` | `cpu_av`, `nvdec_cuvid` |
| `adapters/model/` | registry, YOLO/LSTM, `in_process` serving |
| `adapters/encode/` | per-camera FFmpeg segment muxer |
| `pipeline/ingest/` | RTSP/file/webcam, reconnect |
| `pipeline/bus/` | latest-only inference/live, FIFO evidence |
| `pipeline/perception/` | tracker, `SceneState`, features, `DecisionInput` build |
| `pipeline/inference_coordinator.py` | latest-only drain, every pose forward |
| `pipeline/camera_pipeline.py` | per-camera wiring, no business math |
| `pipeline/decision/` | `IncidentManager`, admission |
| `pipeline/output/` | EventSink, evidence, overlay, MJPEG |
| `domains/fall/` | window classifier, rising-edge latch |
| `domains/bed_exit/` | assignment, grace/hold |
| `runtime/worker.py` | composition root |
| `runtime/bootstrap.py` | named stages |
| `runtime/profile/` | `ML_WORKER_PROFILE` -> `(device, decode, encode)` |

New seam: Protocol plus two implementations, or one plus a test double.
Keep new pure-code modules at or below 250 logical LOC. Split by port or stage.
