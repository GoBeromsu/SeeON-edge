#!/usr/bin/env bash
set -euo pipefail

readonly IMAGE='nvcr.io/nvidia/deepstream@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994'
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
readonly OUTPUT="${1:-/tmp/pyservicemaker-p1b-spike.raw.json}"

mkdir -p "$(dirname "$OUTPUT")"
docker run --rm --gpus all \
  --entrypoint bash \
  -v "$ROOT:/workspace:ro" \
  -v "$(dirname "$OUTPUT"):/output" \
  "$IMAGE" \
  -lc 'pip install --quiet pyyaml &&
       mkdir -p /output/engine &&
       cp /workspace/scripts/qa/pyservicemaker-spike/yolo26_pose_parser.cpp /output/engine/ &&
       cp /workspace/scripts/qa/pyservicemaker-spike/build-parser.sh /output/engine/ &&
       cd /output/engine && ./build-parser.sh &&
       cp /workspace/scripts/qa/pyservicemaker-spike/nvinfer-yolo26-pose.txt . &&
       python3 /workspace/scripts/qa/pyservicemaker-spike/spike.py --config /workspace/scripts/qa/pyservicemaker-spike/spike.yaml --output /output/'"$(basename "$OUTPUT")"
