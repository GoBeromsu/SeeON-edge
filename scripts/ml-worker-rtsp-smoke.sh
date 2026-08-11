#!/bin/bash
# Shebang is pinned to /bin/bash deliberately (not `env bash`): Homebrew bash
# 5.3.15 writes heredoc bodies into a pipe before exec'ing the reader, so a
# body over 512 bytes deadlocks forever. This script has two such heredocs.
# See issue #9 and tests/test_shell_script_heredoc_contract.py.
set -euo pipefail

usage() {
  cat >&2 <<'MSG'
Usage: scripts/ml-worker-rtsp-smoke.sh

Probes every RTSP URL for the cameras currently registered in the local
camera registry (catalog.sqlite3 -- see backend/app/features/cameras/store.py,
the sole camera roster authority; register cameras via the Edge dashboard UI
before running this), then runs ml-worker for a bounded number of frames per
camera.

There is no config-file/env-var input any more: the registry is the only
source, matching what a real Edge worker pulls from ml-api at boot.

Common env:
  MAX_FRAMES_PER_CAMERA   Frames per camera before exit, default: 30
MSG
}

case "${1:-}" in
  -h|--help)
    usage
    exit 2
    ;;
esac

frames="${MAX_FRAMES_PER_CAMERA:-30}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/ml-worker-rtsp-smoke.XXXXXX")"
config_path="$tmpdir/ml-worker.yaml"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

# Registry is the sole camera roster authority (no EDGE_CAMERA_CONFIG env, no
# API_CAMERA_INVENTORY) -- render a throwaway static-config YAML from
# whatever is currently registered, the same escape hatch `--config` is
# documented for in worker/runtime/config/loader.py:resolve_config_path.
# facility_id is stamped "local": the registry does not track facility
# identity (see worker_config_snapshot's same placeholder in
# backend/app/features/cameras/router.py), so a bare probe run has no other
# value to put there.
urls_output="$(
  uv run python - "$config_path" <<'PY'
import sys

import yaml

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.shared.state_dir import resolve_state_dir
from worker.runtime.config.loader import load_worker_config

config_path = sys.argv[1]
store = CameraRegistryStore(resolve_state_dir("ml-api") / "catalog.sqlite3")
records = store.snapshot()["cameras"]

if not records:
    print(
        "카메라가 등록되지 않았습니다. Edge UI에서 카메라를 먼저 등록한 뒤 "
        "다시 실행하세요. (no cameras registered -- register at least one "
        "camera in the Edge dashboard UI first)",
        file=sys.stderr,
    )
    raise SystemExit(2)

cameras = []
urls = []
for record in records:
    rtsp_url = record.get("rtsp_url")
    if not isinstance(rtsp_url, str) or not rtsp_url.strip():
        continue
    camera_id = str(record.get("backend_camera_id") or record["id"])
    cameras.append(
        {
            "camera_id": camera_id,
            "facility_id": "local",
            "rtsp_url": rtsp_url,
            "label": record.get("label") or camera_id,
        }
    )
    urls.append(rtsp_url)

if not cameras:
    print("등록된 카메라 중 RTSP URL이 설정된 카메라가 없습니다.", file=sys.stderr)
    raise SystemExit(2)

with open(config_path, "w", encoding="utf-8") as handle:
    yaml.safe_dump({"version": 1, "cameras": cameras}, handle, sort_keys=False)

# Parse the rendered file back through the worker's own loader before the
# worker is launched, so a registry row that cannot become a valid
# WorkerConfig fails here with the loader's message instead of surfacing as
# an opaque worker boot crash after the ffprobe pass.
load_worker_config(config_path)

for url in urls:
    print(url)
PY
)"
mapfile -t urls <<<"$urls_output"
chmod 600 "$config_path"

if [[ "${#urls[@]}" -lt 1 ]]; then
  printf 'expected at least one registered camera with an RTSP URL\n' >&2
  exit 2
fi

for url in "${urls[@]}"; do
  ffprobe \
    -v error \
    -rtsp_transport tcp \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height \
    -of default=noprint_wrappers=1 \
    "$url" >/dev/null
done

uv run python -m worker \
  --config "$config_path" \
  --max-frames-per-camera "$frames" \
  --heartbeat-on-start
