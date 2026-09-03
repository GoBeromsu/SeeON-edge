# pyservicemaker P1b spike

Measurement-only G8a harness. It remains under `scripts/`; production code does
not import it.

```bash
# Generate the ignored ONNX artifact on the host.
uv run python -c "from ultralytics import YOLO; YOLO('models/pose/yolo26n-pose.pt').export(format='onnx', imgsz=640, opset=17)"
# Compile the parser and run the local Smart Record measurement in pinned DeepStream.
scripts/qa/pyservicemaker-spike/run.sh
```

The pinned image is
`nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994`.
To run the full parsed Flow measurement, bind-mount the generated ONNX and
engine at `/work`, compile with `build-parser.sh`, then run `measure.py` with
`nvinfer-yolo26-pose.txt` and DeepStream's `config_tracker_NvDCF_perf.yml`.

`yolo26_pose_parser.cpp` accepts nvinfer's `[300,57]` stripped-batch output
(and `[1,300,57]`), uses strict `score > 0.05`, preserves source order, copies
xyxy boxes, and does no NMS. ONNX and TensorRT engines are never committed.
