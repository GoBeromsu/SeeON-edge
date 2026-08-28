# worker

RTSP inference client: ingest, bounded bus, extract, decide, evidence, one-way relay egress.
Image `ml-worker`. Sole production command: `python -m worker`.
Replay is backend-owned; the production worker has no replay CLI.

## Layers

Run `uv run --group lint lint-imports` for the configured contracts. A forbidden import means the design is wrong: add a Protocol in `interfaces/` and inject from `runtime/`. Native/tools ceilings are not configured there; focused dependency tests and manual path review own those checks.

| Package | Role | Worker-layer ceiling |
| --- | --- | --- |
| `types/` | internal envelopes | no other `worker` layer |
| `interfaces/` | one Protocol per seam | `types` |
| `adapters/` | decode, model, encode | `types`, `interfaces` |
| `pipeline/` | ingest, bus, perception, decision, output | everything except `runtime` |
| `domains/` | fall and bed-exit | `types`, `interfaces`, `pipeline.perception` |
| `native/deepstream/` | native child + Python wire/preflight seam | `types` |
| `runtime/` | composition root | everything |
| `tools/deepstream_canary/` | host-side operator qualification tool | out of the production import graph |
| `tools/fetch_models/` | pinned model provisioning (`edge-model-fetch`) | stdlib only; out of the production import graph |

Order: `runtime -> pipeline -> domains -> adapters -> interfaces -> types -> contracts`.
`native/deepstream/` is a parallel worker-internal leaf with a types-only ceiling,
not another rung in that order. `tools/` is out-of-band; import-linter forbids every worker layer from importing it.
`contracts` contains cross-instance L0 data only. Worker-internal ports and envelopes live under `worker/`; never duplicate or shadow a vendored type, including `contracts/AGENTS.md`.
Shared leaves are scope-owned: `detection_policies`, `events`, and
`rtsp_url_policy`. Worker never imports `backend` or database modules. The
`nvidia` production path consumes `native/deepstream/` from `runtime/`;
`tools/deepstream_canary/` remains out-of-band and is excluded from the image.

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
| `pipeline/inference_coordinator.py` | non-`nvidia` latest-only drain and host pose forwards; never constructed by `nvidia` |
| `pipeline/camera_pipeline.py` | per-camera wiring, no business math |
| `pipeline/decision/` | `IncidentManager`, admission |
| `pipeline/output/` | EventSink, evidence, overlay, MJPEG |
| `domains/fall/` | window classifier, rising-edge latch |
| `domains/bed_exit/` | assignment, grace/hold |
| `native/deepstream/` | native media/inference, `PerceptionFrameV1` wire, preflight, engine cache, association |
| `runtime/worker.py` | composition root |
| `runtime/deepstream/` | one-child Python supervision, metadata admission, source lifecycle, per-camera `NativePolicyPump` |
| `runtime/bootstrap.py` | named stages |
| `tools/deepstream_canary/` | isolated canary harness, off the production worker path |
| `tools/fetch_models/` | manifest-pinned model download + SHA-256 verify into `/app/models` |
| `runtime/profile/` | canonical backend/memory descriptor; `nvidia` routes to the native media plane |

New seam: Protocol plus two implementations, or one plus a test double.
Keep new pure-code modules at or below 250 logical LOC. Split by port or stage.
