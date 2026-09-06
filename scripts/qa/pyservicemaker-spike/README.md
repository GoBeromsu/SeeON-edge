# pyservicemaker P1b spike

Measurement-only G8a harness; production code does not import it. It is a
failed acceptance gate, but preserves reproducible feasibility measurements.

```bash
scripts/qa/pyservicemaker-spike/run.sh /tmp/pyservicemaker-p1b-spike-receipts
```

The command locates `models/pose/yolo26n-pose.onnx`, or exports it from the
local `.pt` model if necessary. In the pinned DeepStream image it compiles the
custom parser, builds the FP16 TensorRT engine, runs Smart Record, and runs the
one-source metadata and 13-source fan-out probes. It writes `engine-build.log`,
`parser-build.log`, `item-1-smart-record.json`, `item-2-metadata.json`,
`item-3-fanout.json`, and `nvidia-smi-compute-apps.json` to the output directory.

The pinned image is
`nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994`.
ONNX and TensorRT engine files are outputs and are never committed.

`yolo26_pose_parser.cpp` accepts `[300,57]` and `[1,300,57]`, uses strict
`score > 0.05`, preserves source order, copies xyxy boxes, does no second NMS,
and rejects non-positive-area boxes. The native pose parser does not have that
last rejection. The parser creates detection metadata only:
`NvDsInferObjectDetectionInfo` carries no keypoints. Production parity needs
output tensor metadata plus a row-to-track association design resilient to
NvDCF reordering, shadow objects, and missed detections.
