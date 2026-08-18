# worker

RTSP inference client: ingest, bounded bus, extract, decide, evidence, one-way relay egress.
Image `ml-worker`. Sole command: `python -m worker`.

## Layers

Run `uv run --group lint lint-imports`; a forbidden import means the design is wrong: add a Protocol in `interfaces/` and inject from `runtime/`.

| Package | Role | Worker-layer ceiling |
| --- | --- | --- |
| `types/` | internal envelopes | no other `worker` layer |
| `interfaces/` | one Protocol per seam | `types` |
| `adapters/` | decode, model, encode | `types`, `interfaces` |
| `pipeline/` | ingest, bus, perception, decision, output | everything except `runtime` |
| `domains/` | fall and bed-exit | `types`, `interfaces`, `pipeline.perception` |
| `runtime/` | composition root | everything |

Order: `runtime -> pipeline -> domains -> adapters -> interfaces -> types -> contracts`.
`contracts` contains cross-instance L0 data only. Worker-internal ports and envelopes live under `worker/`; never duplicate or shadow a vendored type, including `contracts/AGENTS.md`.
Shared leaves are scope-owned: `detection_policies`, runtime `edge_db` APIs, `events`, and `rtsp_url_policy`. Worker never imports `backend` or edge-database DDL modules.

## Data and lifetime boundaries

`types/AGENTS.md` owns the pixel/numeric envelope contract. `runtime/AGENTS.md` owns process-shared versus per-camera allocation. Keep both boundaries intact; details stay in those scoped guides.

## Hardware failure policy

`runtime/AGENTS.md` owns boot exit codes, profile defaults, fault persistence, camera-local degradation, and lifecycle. `adapters/AGENTS.md` owns explicit backend fallback. `pipeline/AGENTS.md` owns queue overflow, and `pipeline/output/evidence/AGENTS.md` owns delivery durability. Read those scoped guides before changing failure behavior.

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
