#!/usr/bin/env bash
# Items 2/3/4/6 of the P1b gate on LIVE facility cameras.
# Camera URIs (with credentials) arrive only via LIVE_URIS, newline-separated; never commit them.
set -euo pipefail
: "${LIVE_URIS:?set LIVE_URIS to newline-separated camera RTSP URIs (kept in your environment)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIKE="$(cd "$HERE/.." && pwd)"
WORK="${1:-/tmp/pyservicemaker-spike-live}"
SECONDS_RUN="${SECONDS_RUN:-600}"
mkdir -p "$WORK"
cp "$HERE/live_measure.py" "$SPIKE/nvinfer-yolo26-pose.txt" "$SPIKE/yolo26_pose_parser.cpp" "$SPIKE/build-parser.sh" "$WORK/"
docker build -q -f "$SPIKE/smart-record/Dockerfile.sr" -t p1b-sr:local "$SPIKE/smart-record" >/dev/null
# engine + parser (see ../run.sh for the ONNX export and trtexec build; both must exist in $WORK)
[ -f "$WORK/yolo26n-pose-fp16.engine" ] || { echo "build the FP16 engine into $WORK first (see ../run.sh)"; exit 2; }
[ -f "$WORK/libnvdsinfer_custom_yolo26_pose.so" ] || docker run --rm --entrypoint bash -v "$WORK:/work" p1b-sr:local -lc 'cd /work && ./build-parser.sh'
docker run --rm --gpus all --network host --entrypoint bash -v "$WORK:/work" -e LIVE_URIS p1b-sr:local -lc \
  "cd /work && python3 live_measure.py --infer-config /work/nvinfer-yolo26-pose.txt \
     --tracker-config /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml \
     --seconds $SECONDS_RUN --out /work/live.json ${ORT_FALL_MODEL:+--ort-fall-model /work/fall-gru.onnx --cuda-apps-out /work/cuda.json}"
