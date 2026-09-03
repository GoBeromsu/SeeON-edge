# pyservicemaker P1b (G8a) spike

**Overall verdict: five of six items proven or substantially proven; Smart
Record is not proven.** The spike is isolated under `scripts/`; no production
imports or artifacts were added. Measurements used the pinned DeepStream 9.1
image on RTX 5070 Ti and the bundled `sample_1080p_h264.mp4` file source.

## Reproduction

```bash
# Host: create a non-repository ONNX artifact
uv run python -c "from ultralytics import YOLO; YOLO('models/pose/yolo26n-pose.pt').export(format='onnx', imgsz=640, opset=17)"
# Container: build FP16 engine, parser, then run the real Flow probe
scripts/qa/pyservicemaker-spike/run.sh
```

`run.sh` pins
`nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994`.
The non-versioned ONNX/engine inputs used for the captured run were in
`/tmp/p1b-work`; they are intentionally not committed.

| Item | Command | Raw measurement | Verdict |
|---|---|---|---|
| 1 Smart Record | `scripts/qa/pyservicemaker-spike/run.sh` | `start_recording('spike-camera-0',0,4,callback) -> session 0`; `stop_recording(...) -> true`; callback list `[]`; files `[]` | **Not proven** |
| 2 Metadata/IDs | `python3 measure.py --uri file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 --infer-config /work/nvinfer-yolo26-pose.txt --tracker-config /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml --seconds 25 --out /work/item2.json` | 1443 frames, 2397 objects, 10 distinct NvDCF IDs, 1443 callbacks, 759.9 fps, no error | **Proven primitive** |
| 3 Probe budget | same `measure.py`, with `--sources 13 --sync` | 18,759 frames, 1,548 objects, 17 IDs, 1,444 batch callbacks; p50 0.220402 ms, p95 0.256782 ms, p99 0.294477 ms, max 1.390293 ms, mean 0.2096 ms | **Proven** |
| 4 CUDA media-plane ownership | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` during item 3 | `85, 804 MiB` (exactly one row) | **Proven for media plane** |
| 5 Cold engine build | `/usr/src/tensorrt/bin/trtexec --onnx=/work/yolo26n-pose.onnx --saveEngine=/work/yolo26n-pose-fp16.engine --fp16 --memPoolSize=workspace:2048` | `Engine generation completed in 29.995 seconds`; previous repeat: `Engine built in 30.397 sec` | **Proven** |
| 6 Counters | real `Probe("counter", Counter())` in `measure.py` | Item 2: frames 1443, objects 2397, callbacks 1443, fps 759.9. Item 3: frames 18759, objects 1548, callbacks 1444, fps 467.7. | **Substantially proven** |

## Parser and Flow configuration

`yolo26_pose_parser.cpp` compiles inside the pinned image via
`build-parser.sh` into `libnvdsinfer_custom_yolo26_pose.so`. It accepts either
DeepStream's stripped-batch `[300,57]` dimensions or `[1,300,57]`; nvinfer
provided the former. It copies source-order xyxy detections only when strict
`score > 0.05`, and intentionally applies no second NMS. The end-to-end YOLO26
head has already performed decode and one-to-one matching. `nvinfer-yolo26-pose.txt`
points nvinfer at the FP16 engine and library. `measure.py` uses:

```python
Flow(Pipeline("p1b-spike")).batch_capture(...).infer(...).track(
    ll_config_file=tracker_config,
    ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
).attach(what=Probe("counter", counter))
```

The probe reads `batch_meta.frame_items` and `frame_meta.object_items`; the
objects carried populated `object_id` values. The parser creates detection
metadata only. Its keypoints are in the 57-wide output rows and require tensor
metadata decoding in a probe; that omission does not invalidate the measured
bbox/tracker primitive.

## Limits and follow-ups

* Item 2 proves populated object metadata and NvDCF identity on the synthetic
  file source. A 10-minute, exactly 15-fps live-camera run and its ID-switch
  count remain owner/deployment work; no such count is claimed.
* Item 3 uses 13 repeated synthetic file sources with `sync=True`, not 13 real
  cameras. At 195 callbacks/s the budget is about 5.1 ms; 0.294477-ms p99 is
  about 17× below that budget.
* Item 4 proves a single GPU process for the media plane. The ORT CPU fall
  model was not co-resident, so its no-second-context check remains outstanding.
* Item 6 counters are probe-derived, not documented Flow-native counter APIs.
  Flow supplied the batch metadata; the probe counted frames/objects/IDs and
  timed callback duration.
* Smart Record used a writable `/output/records`, a canonical SourceConfig
  sensor ID, a retained callback, and a three-second PLAYING/history wait; it
  still produced no file or completion callback. The likely bounded-file-source
  cause is that Smart Record cache semantics target live sources and the file
  URI does not accumulate the requested lookback. Retry against a live RTSP
  source before declaring the primitive unavailable.
