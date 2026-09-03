#!/usr/bin/env bash
set -euo pipefail

g++ -std=c++17 -shared -fPIC -O2 \
  -I/usr/local/cuda-13.2/targets/x86_64-linux/include \
  -I/opt/nvidia/deepstream/deepstream/sources/includes \
  yolo26_pose_parser.cpp -o libnvdsinfer_custom_yolo26_pose.so
