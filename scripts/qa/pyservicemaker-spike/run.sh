#!/usr/bin/env bash
set -euo pipefail

readonly IMAGE='nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994'
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
readonly OUTPUT_DIR="${1:-/tmp/pyservicemaker-p1b-spike-receipts}"
readonly ONNX_SOURCE="${ONNX_PATH:-$ROOT/models/pose/yolo26n-pose.onnx}"

mkdir -p "$OUTPUT_DIR"
if [[ ! -f "$ONNX_SOURCE" ]]; then
  uv run python -c "from ultralytics import YOLO; YOLO('$ROOT/models/pose/yolo26n-pose.pt').export(format='onnx', imgsz=640, opset=17)"
fi
if [[ ! -f "$ONNX_SOURCE" ]]; then
  echo "ONNX export did not produce $ONNX_SOURCE" >&2
  exit 1
fi
cp "$ONNX_SOURCE" "$OUTPUT_DIR/yolo26n-pose.onnx"
docker run --rm --gpus all \
  --entrypoint bash \
  -v "$ROOT:/workspace:ro" \
  -v "$OUTPUT_DIR:/work" \
  "$IMAGE" \
  -lc 'set -euo pipefail
       pip install --quiet pyyaml
       cp /workspace/scripts/qa/pyservicemaker-spike/yolo26_pose_parser.cpp /work/
       cp /workspace/scripts/qa/pyservicemaker-spike/build-parser.sh /work/
       cp /workspace/scripts/qa/pyservicemaker-spike/nvinfer-yolo26-pose.txt /work/
       printf "person\n" > /work/labels.txt
       cd /work && ./build-parser.sh > parser-build.log 2>&1
       SECONDS=0
       /usr/src/tensorrt/bin/trtexec --onnx=/work/yolo26n-pose.onnx --saveEngine=/work/yolo26n-pose-fp16.engine --fp16 --memPoolSize=workspace:2048 > engine-build.log 2>&1
       printf "engine_build_elapsed_seconds=%s\n" "$SECONDS" >> engine-build.log
       python3 /workspace/scripts/qa/pyservicemaker-spike/spike.py --output /work/item-1-smart-record.json
       python3 /workspace/scripts/qa/pyservicemaker-spike/measure.py --uri file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 --infer-config /work/nvinfer-yolo26-pose.txt --tracker-config /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml --seconds 25 --out /work/item-2-metadata.json
       python3 /workspace/scripts/qa/pyservicemaker-spike/measure.py --uri file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 --sources 13 --sync --infer-config /work/nvinfer-yolo26-pose.txt --tracker-config /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml --seconds 25 --out /work/item-3-fanout.json --cuda-apps-out /work/nvidia-smi-compute-apps.json
       python3 - <<'"'"'PY'"'"'
from pathlib import Path
path = Path("/work/engine-build.log")
path.write_text("\n".join(path.read_text().splitlines()[-60:]) + "\n")
PY'
