# pyservicemaker P1b (G8a) spike

**Gate verdict: items 1–4 PASS on live facility cameras (items 2 and 3
with the two documented deviations below); items 5 and 6 are NOT yet signed
off.** Item 5 has its build-time measurement but not the bootstrap behaviour
(refuse until warmup, activation ordering, timeout, nonzero exit on corrupt
identity, no post-activation build); item 6's four named counters are
settled in meaning by the Smart Record measurements but do not yet exist. Both
are closed by G007 artefacts with their own receipts — the one-shot
`edge-engine-build` + boot refusal tests, and the recorder actor's exercised
counters — and this report is re-signed when they land. The owner has
acknowledged deviations A and B and directed G007 to proceed on that basis
(ultragoal ledger); the plan's "stop and record an owner decision" rule is
therefore satisfied by decision, not by silence.

Every number below is reproducible from a committed harness and has a
committed receipt. Camera URIs (with credentials) are supplied only through the
environment (`SR_RTSP_URI`, `LIVE_URIS`) and are never written to the tree; the
repository privacy gate enforces this.

## Reproduction

| what | command | receipts |
| --- | --- | --- |
| engine + parser + bundled-sample smoke | `scripts/qa/pyservicemaker-spike/run.sh` | `receipts/engine-build.log`, `receipts/item-2-metadata.json`, `receipts/item-3-fanout.json` |
| Smart Record on a live camera | `SR_RTSP_URI=… scripts/qa/pyservicemaker-spike/smart-record/run.sh` | `receipts/smart-record/plugin-*.log`, `receipts/smart-record/binding-stop.json` |
| full chain on live cameras | `LIVE_URIS=… scripts/qa/pyservicemaker-spike/live/run.sh` | `receipts/live/live-10min.json`, `receipts/live/live-13.json`, `receipts/live/cuda-ort-coresident.json` |

Pinned image: `nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994`.
Source for the live runs: the cameras' 640×360 HEVC sub-streams at their native
30 fps over RTP/TCP.

## What the DeepStream docs settle (read after the first, failed pass)

- Smart Record: **only RTSP sources are enabled**; recording **cannot start
  until an I-frame is in the cache**; `startTime` is seconds *before* now and the
  cache must exceed it; **overlapping smart record is not supported**.
- nvinfer: `output-tensor-meta=1` attaches the raw output tensor per frame as
  `NvDsInferTensorMeta`; `cluster-mode=4` disables clustering, which the YOLO26
  end-to-end head requires (it already resolved duplicates in-graph).

The first pass failed on a file source over UDP with an EGL render sink — three
harness errors, not DeepStream limitations.

## Per-item results

| item | criterion (plan wording) | measured | verdict |
| --- | --- | --- | --- |
| 1 Smart Record | per-source start/stop/sr-done with observed callback ordering and the delayed-stop extension behaviour | live camera, `nvurisrcbin` action signals: `start-sr(5,6)` → `sr-done` at +6.06 s, 10.94 s file; `start-sr(0,20)` + `stop-sr` at +4 s → `sr-done` **70–80 ms after stop**, 4.2–5.0 s file, **5/5** (`plugin-stop-attempt{1..5}.log`); a second `start-sr` while in flight is **absorbed into the same session, not extended** (10.2–10.6 s file) | **PROVEN**. "Delayed-stop extension" is not a DS primitive; it is an actor-owned delayed `stop-sr` (see G007 constraints) |
| 2 Full chain | one source at 15 fps for 10 minutes; populated `NvDsBatchMeta` with object metadata from the FP16 yolo26n-pose engine through the custom parser; NvDCF ids + lifecycle; measured id-switch count | 207호, **600.7 s, 18,033 frames at 29.99 fps**, **11,565 tracked objects**, every one matched one-to-one to a tensor row under an IoU ≥ 0.5 gate (**11,565 matched / 0 unmatched / 0 row reuse**, match IoU p05 = 0.99999), **40 NvDCF ids** with per-id lifecycle (first/last frame, frames present, gaps) recorded, **12 id switches** (a birth within 5 s of an id absent ≥ 2 frames whose last box overlaps the new one) under `NvDCF_perf` defaults — `live-10min.json` | **PROVEN, deviation A**: the source ran at its native 30 fps; the 15 fps model cadence is owned downstream by the P1a `PtsResampler` (`worker/pipeline/perception/pts_resample.py`), which the plan already designates as the single cadence owner. 12 switches / 10 min is the number G007 tunes down (`NvDCF_accuracy`, probation/shadow) |
| 3 Probe budget | 13 × 15 fps (195 callbacks/s) with a bounded metadata copy and per-camera actors | **all 13 facility cameras** in one batched pipeline, 154 s: **281 frame items/s in 24.5 batched callbacks/s**; production-shaped copy per object (bbox + 17×3 keypoints + NvDCF id + source + PTS) into bounded per-camera queues; probe **p50 3.85 ms / p95 7.19 ms / p99 9.73 ms / max 21.2 ms** per batch against a 33.3 ms batch interval — `live-13.json` | **PROVEN, deviation B**: DeepStream delivers frames in *batched* callbacks (13 frames per callback at 30 fps ≈ 24–30 callbacks/s), so the plan's "195 callbacks/s" is realised as ≥ 195 frame items/s through batched callbacks. The 10-minute run's 11,501 `queue_drops_no_consumer` are the bounded queue overwriting with no consumer attached — backpressure that is observable, not silent. P1b-AC6 must show zero unaccounted drops with real consumers |
| 4 Single CUDA owner | fall model on ORT CPU, no second context | the pose+bbox56 GRU (ONNX export of the packaged bundle) ran **co-resident on `CPUExecutionProvider` at 188.8 inferences/s** (P1a needs 13 cameras × 3/s = 39/s) while all 13 cameras streamed; `nvidia-smi --query-compute-apps` → **one process**; **`torch` is not imported** in the media-plane process (tensor rows are copied device→host through cudart, which uses the process's primary context — the same one the DeepStream plugins use) — `cuda-ort-coresident.json`, `live-13.json` | **PROVEN** at the process/import-surface level. A CUDA context *count* is not exposed by nvidia-smi; the import-surface assert (no Torch, no CuPy, cudart on the primary context) is the P1b-AC7 mechanism |
| 5 Cold start | engine build time and an explicit edge-engine-build step (ADR-0002) | FP16 build from the host ONNX export: **29.995 s / 30.397 s / 34 s** across three builds (`engine-build.log`) | **PROVEN** for the budget. Bootstrap semantics — refuse until warmup, activation ordering, timeout, nonzero exit on corrupt identity, no post-activation build — are P1b-AC4 implementation tests |
| 6 Counters | `smart_record_extended_total`, `smart_record_extension_raced_total`, `smart_record_start_refused_total`, `nvenc_sessions_active` | settled by item 1: DS **absorbs** an overlapping start (so `extended_total` must count actor-owned delayed-stop extensions, never plugin overlap); a `start-sr` before the first I-frame records nothing (so `start_refused_total` = readiness-gated refusals); Smart Record **caches encoded frames and never re-encodes**, so `nvenc_sessions_active == 0` for recording | **design settled**; implementation and exercise are P1b |

## Parser and transport boundary

`scripts/qa/pyservicemaker-spike/yolo26_pose_parser.cpp` accepts the pose
head's `[300,57]` layout with or without the batch dim (nvinfer strips it),
uses strict `score > 0.05`, preserves source order, copies xyxy boxes, and
performs no second NMS — the shipped decode in
`worker/native/deepstream/parity/parse.py`. It adds a positive-area rejection
the native pose parser does not have; G007 must drop that or prove it is
equivalent.

`NvDsInferObjectDetectionInfo` carries no keypoints. The live harness
(`live/live_measure.py`) shows the P1b transport that works: enable
`output-tensor-meta=1`, read `frame.tensor_items[…].as_tensor_output().get_layers()["output0"]`,
copy the rows device→host through cudart via the DLPack capsule (the binding's
`__dlpack__` takes a stream argument, so `numpy.from_dlpack` cannot consume it
directly and Torch must not be the consumer), map rows from network space
(letterbox 640×640, scale = min(NET/W, NET/H), asymmetric padding) into frame
space, and match rows to tracked objects **one-to-one under an IoU gate with
each row consumed at most once**; an object without a gated row is explicitly
unmatched. On these cameras the match IoU is ≥ 0.9999 because nvinfer's object
rect *is* the parser's box, which is why the gate is safe.

## Operational findings for G007 (from the measurements)

1. `pyservicemaker.Pipeline.stop_recording` is defective in this image: it
   returns `True`, never fires the completion callback, and leaves a moov-less
   stub (`binding-stop.json`). The `stop-sr` action signal finalises correctly.
   Drive stop through the signal (or the C API), never the convenience method.
2. `start-sr` must gate on **source liveness and a cached I-frame**, never on
   wall-clock. `nvurisrcbin` occasionally stalls after the SDP on these cameras;
   `init-rtsp-reconnect-interval=5` and `rtsp-reconnect-interval=5` recover it
   within one interval (5/5 thereafter).
3. These cameras do not negotiate HEVC over UDP: `select-rtp-protocol=4` (TCP)
   is required. Rapid reconnects exhaust the camera's RTSP sessions (`453 Not
   Enough Bandwidth`); one long-lived session per camera, with reconnect
   intervals, is the only sane shape.
4. `render()` defaults to an EGL sink that cannot configure caps headless; use
   `RenderMode.DISCARD` in the worker.

## Owner acknowledgements

- **Deviation A** (item 2) — acknowledged: native camera cadence into the media
  plane, 15 fps owned by `PtsResampler` downstream; G007 adds no second
  throttle.
- **Deviation B** (item 3) — acknowledged: "195 callbacks/s" is realised as
  ≥ 195 frame items/s through DeepStream's batched callbacks; P1b-AC6 must
  still show zero unaccounted drops with real consumers attached.
- **Deletion freeze**: G008 stays unauthorized and the shipping native profile
  and image digest are retained until G007 passes P1b-AC1–AC7 plus one hour of
  production stability.
