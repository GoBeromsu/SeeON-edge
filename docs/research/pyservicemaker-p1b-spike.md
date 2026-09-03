# pyservicemaker P1b (G8a) spike

**Verdict: STOP — owner decision required.** This measurement-only spike adds no
production code. The source was DeepStream's bundled `sample_720p.mp4`, not
RTSP or a facility camera.

## Reproduction

```bash
scripts/qa/pyservicemaker-spike/run.sh
```

The wrapper pins `nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994`, mounts the repository read-only, and writes raw JSON to `/tmp/pyservicemaker-p1b-spike.raw.json`.

## 1. Smart Record — not proven

**Command:** the reproduction command builds a local `RecordConfig` Flow and calls `Pipeline.start_recording` then `Pipeline.stop_recording`.

**Raw output:**

```text
Pipeline.start_recording(source_name, start_time, duration, callback) -> session id
Pipeline.stop_recording(source_name) -> bool
source_name: batch_capture-source-0_0
start_recording(source_name, 0, 4, callback) session: 0
stop_recording(source_name) result: true
event order: start at 1359777.699877917; stop at 1359779.699940489
sr-done callbacks: []
produced files: []
```

The application-controlled local source accepted a start and stop. It produced no callback and no file, so callback ordering, overlap extension/truncation, and recorded duration are not proven. The required owner finding is why a successful session/stop has no completion artifact in this image/source setup.

The harness now uses a SourceConfig with canonical `sensor-id:
spike-camera-0`, creates `/output/records` before pipeline start, waits three
seconds for PLAYING/history, and retains the callback closure through shutdown.
The corrected measurement still produced no callback or file.

## 2. Full-chain metadata — blocked

**Commands and raw output:**

```text
$ uv run python ... YOLO('models/pose/yolo26n-pose.pt').export(format='onnx', imgsz=640, opset=17, simplify=False, dynamic=False)
PyTorch input shape: (1, 3, 640, 640)
output shape(s): (1, 300, 57)
ONNX export success ✅ 0.4s

$ /usr/src/tensorrt/bin/trtexec --onnx=/work/yolo26n-pose.onnx --saveEngine=/work/yolo26n-pose-fp16.engine --fp16 --memPoolSize=workspace:2048
Output binding for output0 with dimensions 1x300x57 and type fp32 is created.
```

The engine is buildable, but no matching DeepStream YOLO-pose custom parser/configuration was available. The tensor is already NMS-decoded: a parser must convert its 300 candidates with 57 values (bbox, score/class, 17×3 keypoints) into `NvDsObjectMeta`. Therefore no populated `NvDsBatchMeta`, NvDCF lifecycle, 10-minute 15-fps run, or ID-switch count is claimed.

**Correction:** the spike now contains and compiles
`yolo26_pose_parser.cpp` as `libnvdsinfer_custom_yolo26_pose.so` in the pinned
container. It performs strict `score > 0.05`, source-order corner-box copying,
and no NMS. Raw compile output:

```text
$ ./build-parser.sh
-rwxr-xr-x 1 root root 16032 ... libnvdsinfer_custom_yolo26_pose.so
```

The parser is configured by `nvinfer-yolo26-pose.txt`. It only supplies bbox
metadata: keypoints are not representable through `NvDsInferObjectDetectionInfo`
and require a second tensor-meta/probe decoder. A parsed Flow/NvDCF run was not
completed in this bounded pass, so its metadata/lifecycle/id-switch measurements
remain unproven.

## 3. Probe latency budget — not proven

**Command:** the reproduction command. Without item 2's Flow metadata callback, it only runs a simulated bounded Python copy.

**Raw output:**

```text
mode: simulated bounded metadata copy; not a Flow callback
target_callbacks_per_second: 195
callbacks: 19500
latency_ns: p50=58 p95=72 p99=157 max=137991
drops: not measurable without a running Flow callback
```

This is not a Flow-probe or 13-source result.

## 4. Single CUDA owner — not proven

**Raw output while the local Smart Record Flow was running:**

```text
$ nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
1, python3, 274 MiB
```

One process was observed, but this does not prove exactly one CUDA context and no ORT CPU fall model was running. A valid proof needs the running parsed Flow plus the ORT CPU model and a context-aware measurement.

## 5. Cold-start engine build — proven

**Commands and raw output:**

```text
$ uv run python ... YOLO('models/pose/yolo26n-pose.pt').export(format='onnx', imgsz=640, opset=17, simplify=False, dynamic=False)
ONNX export success ✅ 0.4s

$ /usr/src/tensorrt/bin/trtexec --onnx=/work/yolo26n-pose.onnx --saveEngine=/work/yolo26n-pose-fp16.engine --fp16 --memPoolSize=workspace:2048
Engine generation completed in 29.8498 seconds.
Engine built in 30.397 sec.
Created engine with size: 7.93155 MiB
```

Cold FP16 engine build time is **30.397 s**. The explicit edge-engine-build sequence is host PT→ONNX export, then pinned-image `trtexec` build/verification before source activation. The generated artifacts are outside the repository and are not built at boot.

## 6. Media-plane counters — blocked

**Command:** the reproduction command enumerated public `Flow` methods.

**Raw output:**

```text
counter_like_methods: []
Flow methods: analyze, apply, attach, batch, batch_capture, capture, decode,
encode, fork, infer, inject, preprocess, publish, render, retrieve, select,
track
```

No public Flow counters for frames in/out, drops, or per-source state were exposed. Runtime discovery remains blocked by the missing parser chain.

## Required owner decision

Do not add a native fallback or interim association. Diagnose the local Smart Record session that accepts start/stop but creates no completion callback/file, and provide a parser/configuration that maps `output0: 1×300×57` into DeepStream pose metadata. Then re-run metadata, real probe, CUDA-owner, and counter measurements.
