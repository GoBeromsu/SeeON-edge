# pyservicemaker P1b (G8a) spike

**Overall verdict: FAIL.** This is an all-or-stop gate. Smart Record produced
neither an `sr-done` callback nor a file, so P1b does not pass. The measurements
below are feasibility primitives, not acceptance evidence where their source,
load, payload, or observability differs from the criterion.

## Reproduction and receipts

`scripts/qa/pyservicemaker-spike/run.sh [output-directory]` is the single
end-to-end command. It locates `models/pose/yolo26n-pose.onnx` (or exports it
from the `.pt` model when absent), builds the parser and FP16 engine with
`trtexec`, runs Smart Record, then runs the one-source metadata and 13-source
fan-out probes. It writes all raw JSON/log receipts to its output directory.
The image is pinned to
`nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994`.

The committed captured receipts are:

* `scripts/qa/pyservicemaker-spike/receipts/engine-build.log` — successful
  FP16 build; elapsed wall time was 34 seconds in this capture.
* `scripts/qa/pyservicemaker-spike/receipts/item-2-metadata.json`.
* `scripts/qa/pyservicemaker-spike/receipts/item-3-fanout.json`.
* `scripts/qa/pyservicemaker-spike/receipts/nvidia-smi-compute-apps.json`.

They contain no media payloads. Re-running the harness overwrites equivalent
receipt names in the selected output directory; timings and counts are
observations, not fixed expected values.

| Item | Acceptance criterion | Actual measurement | Gate label |
|---|---|---|---|
| 1 Smart Record | A successful Smart Record result, including `sr-done` and a file. | `start_recording` returned session 0 and `stop_recording` returned true, but the captured run produced no callback and no file. | **criterion failed** |
| 2 Metadata and IDs | A 10-minute, synchronized 15-fps camera run with ID-switch counting, lifecycle data, and keypoints. | `receipts/item-2-metadata.json` is a short, unsynchronized bundled-file run: 1,443 frames, 2,397 objects, and 10 distinct NvDCF IDs. Object metadata and IDs appeared, but distinct IDs are not ID switches; no lifecycle or keypoints were read. | **primitive demonstrated; criterion failed** |
| 3 Fan-out callback budget | 13 sources paced at 15 fps (195 batch callbacks/s), with production-like payload copying and queue depth, drops, and backpressure measured. | `receipts/item-3-fanout.json` has 1,444 callbacks over 40.109 s, about 36 batch callbacks/s, not 195. Its callback-body p99 was 0.294477 ms, but it only iterated metadata and copied nothing like the production payload; it measured no queue depth, drops, or backpressure. | **primitive demonstrated; criterion failed** |
| 4 CUDA media-plane ownership | A verified context count with the CPU-ORT fall model co-resident. | `receipts/nvidia-smi-compute-apps.json` reports one GPU process (`85, 804 MiB`). A process count is not a CUDA-context count, and CPU-ORT was not co-resident. | **partial observation only** |
| 5 Cold engine build | ADR-0002 bootstrap behavior: refuse until warmup, source-activation ordering, timeout bound, nonzero exit for corrupt identity, and no post-activation builds. | `receipts/engine-build.log` records a successful FP16 engine build in about 34 seconds. None of the ADR-0002 behavior was exercised. | **primitive demonstrated; criterion failed** |
| 6 Named media-plane counters | Exercise `smart_record_extended_total`, `smart_record_extension_raced_total`, `smart_record_start_refused_total`, and `nvenc_sessions_active`. | No named counter exists in or is exercised by this harness. Its frames, objects, and callback counts are probe diagnostics, not those counters. | **criterion failed** |

## Parser and transport boundary

`yolo26_pose_parser.cpp` accepts the observed `[300,57]` stripped-batch layout
and `[1,300,57]`, uses strict `score > 0.05`, preserves source order, copies
xyxy boxes, and performs no second NMS. It also adds a positive-area rejection
that the native pose parser does **not** have.

`NvDsInferObjectDetectionInfo` carries no keypoints. A real port therefore
requires output tensor metadata and a row-to-track association design that
survives NvDCF reordering, shadow objects, and missed detections. This is a
substantial G007 transport/parity boundary, not a parser follow-up.

## Owner decisions required

1. **Smart Record path:** live-RTSP rerun vs redesign, with no native interim fallback.
2. **Gate definition:** rerun to criterion vs formally amend the acceptance text and accept the transferred risk.
3. **Pose-to-track transport boundary:** approve and budget a tensor-meta association/parity design.
4. **Deletion freeze:** G008 stays unauthorized and the shipping native profile/image digest is retained until G007 passes P1b acceptance plus one hour of production stability.
