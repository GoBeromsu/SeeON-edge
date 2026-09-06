#!/usr/bin/env bash
set -euo pipefail

readonly DS=/opt/nvidia/deepstream/deepstream
g++ -std=c++17 -shared -fPIC -O2 \
  -I"$DS/sources/includes" \
  -I/usr/local/cuda-13.2/targets/x86_64-linux/include \
  yolo26_pose_parser.cpp -o libnvdsinfer_custom_yolo26_pose.so
ls -l libnvdsinfer_custom_yolo26_pose.so
