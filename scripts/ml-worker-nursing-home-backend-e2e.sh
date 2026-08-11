#!/bin/bash
# Interpreter is pinned, not `#!/usr/bin/env bash`, on purpose.
#
# Homebrew bash 5.3.15 writes a heredoc body into a pipe before exec'ing the
# reader, so any body over PIPE_BUF (512 bytes on macOS) blocks forever against
# a pipe nothing is draining -- the command is never exec'd and the script hangs
# with no output. This file has two heredocs over that limit. bash 3.2.57 stages
# heredocs in a temp file and is unaffected at any size. See issue #9.
set -euo pipefail

compose_project="${COMPOSE_PROJECT_NAME:-ml-worker-nursing-home-backend-e2e}"
rtsp_url="${NURSING_HOME_RTSP_URL:?NURSING_HOME_RTSP_URL is required; start SeniorAILab/rtsp-generator separately and pass a worker-reachable URL}"
models_dir="${ML_MODELS_DIR:?ML_MODELS_DIR with pose/bed/fall weights is required}"
backend_base_url="${BACKEND_BASE_URL:-http://host.docker.internal:8080}"
relay_url="${RELAY_URL:-http://ml-api:8000}"
relay_token="${RELAY_TOKEN:-local-edge-relay-token}"
ml_api_image="${ML_API_IMAGE:-ghcr.io/seniorailab/eldercare-fall-ml/ml-api:local}"
ml_worker_image="${ML_WORKER_IMAGE:-ghcr.io/seniorailab/eldercare-fall-ml/ml-worker:local}"
profile="${ML_WORKER_PROFILE:-cpu}"
frames="${MAX_FRAMES_PER_CAMERA:-45}"
facility_id="${E2E_FACILITY_ID:?E2E_FACILITY_ID is required}"
resident_id="${E2E_RESIDENT_ID:?E2E_RESIDENT_ID is required}"
camera_id="${E2E_CAMERA_ID:?E2E_CAMERA_ID is required}"
db_container="${E2E_DB_CONTAINER:-eldercare-fall-db}"
postgres_user="${POSTGRES_USER:-fall}"
postgres_db="${POSTGRES_DB:-fall_dev}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${ML_EDGE_E2E_TMP_ROOT:-$repo_root/.omo/tmp}"
mkdir -p "$tmp_root"
tmpdir="$(mktemp -d "$tmp_root/ml-worker-nursing-home.XXXXXX")"
config="$tmpdir/ml-worker.yaml"
edge_models_dir="$tmpdir/models"
runtime_model_dir="$edge_models_dir/fall/lstm-runtime"
runtime_metadata="$tmpdir/lstm-runtime.env"

cleanup_containers() {
  ML_WORKER_PROFILE="$profile" ML_MODELS_DIR="$edge_models_dir" docker compose \
    -p "$compose_project" \
    -f compose.edge.yaml \
    down --remove-orphans >/dev/null 2>&1 || true
}

cleanup() {
  cleanup_containers
  rm -rf "$tmpdir"
}

write_lstm_runtime_artifact() {
  python3 - "$models_dir" "$runtime_model_dir" "$runtime_metadata" <<'PY'
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

models_dir = Path(sys.argv[1])
runtime_dir = Path(sys.argv[2])
runtime_metadata = Path(sys.argv[3])
source_dir = Path(os.environ.get("LSTM_ARTIFACT_DIR", models_dir / "fall" / "lstm"))
model_pt = Path(os.environ.get("LSTM_MODEL_PT", source_dir / "model.pt"))
metadata_json = source_dir / "metadata.json"
if not model_pt.exists():
    raise SystemExit(f"missing LSTM model.pt: {model_pt}")
metadata = {}
if metadata_json.exists():
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
window = int(os.environ.get("LSTM_WINDOW", metadata.get("window", 30)))
stride = int(os.environ.get("LSTM_STRIDE", metadata.get("stride", 5)))
threshold = float(os.environ.get("LSTM_OPERATING_THRESHOLD", metadata.get("operating_threshold", 0.5)))
runtime_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(model_pt, runtime_dir / "model.pt")
for family, filename in (("pose", "yolo26n-pose.pt"), ("bed", "yolo26m-seg.pt")):
    source = models_dir / family / filename
    if not source.exists():
        raise SystemExit(f"missing model artifact: {source}")
    target_dir = runtime_dir.parent.parent / family
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_dir / filename)
arch_json = os.environ.get("LSTM_ARCH_JSON")
if arch_json:
    shutil.copy2(arch_json, runtime_dir / "arch.json")
else:
    (runtime_dir / "arch.json").write_text(
        json.dumps({"hidden": 128, "layers": 2, "dropout": 0.3}, indent=2),
        encoding="utf-8",
    )
(runtime_dir / "metadata.yaml").write_text(
    "\n".join(
        [
            "type: lstm",
            "framework: pytorch",
            "mode: sequence",
            "artifact_dir: /app/models/fall/lstm-runtime",
            "weights: model.pt",
            "architecture: arch.json",
            "metadata: metadata.yaml",
            f"window: {window}",
            f"stride: {stride}",
            "input_shape: [30, 51]" if window == 30 else f"input_shape: [{window}, 51]",
            f"operating_threshold: {threshold}",
            "",
        ]
    ),
    encoding="utf-8",
)
print(f"runtime LSTM artifact: {runtime_dir}")
with runtime_metadata.open("w", encoding="utf-8") as handle:
    handle.write(f"LSTM_WINDOW={window}\n")
    handle.write(f"LSTM_STRIDE={stride}\n")
    handle.write(f"LSTM_OPERATING_THRESHOLD={threshold}\n")
PY
}

write_config() {
  source "$runtime_metadata"
  cat >"$config" <<YAML
version: 1
relay:
  url: ${relay_url}
  token: ${relay_token}
runtime:
  max_failures: 30
  open_timeout_ms: 5000
  read_timeout_ms: 5000
models:
  fall:
    type: lstm
    framework: pytorch
    mode: sequence
    artifact_dir: /app/models/fall/lstm-runtime
    weights: model.pt
    architecture: arch.json
    metadata: metadata.yaml
    window: ${LSTM_WINDOW}
    stride: ${LSTM_STRIDE}
    input_shape: [${LSTM_WINDOW}, 51]
    operating_threshold: ${LSTM_OPERATING_THRESHOLD}
domains:
  enabled:
    - fall
cameras:
  - camera_id: ${camera_id}
    facility_id: ${facility_id}
    resident_id: ${resident_id}
    rtsp_url: ${rtsp_url}
    heartbeat_interval_sec: 30
    frame_stride: 1
    label: nursing-home-fall
YAML
  chmod 600 "$config"
}

start_compose_network() {
  # The worker gets its camera roster via --config (see run_worker below),
  # not an env var -- a camera roster must never be provisionable through
  # the environment (compose is tracked in Git). Inventory env is gone.
  ML_API_IMAGE="$ml_api_image" \
    ML_WORKER_IMAGE="$ml_worker_image" \
    API_BACKEND_EVENTS_URL="${backend_base_url}/api/v1/events" \
    API_EDGE_RELAY_TOKEN="$relay_token" \
    API_DASHBOARD_USERNAME="${API_DASHBOARD_USERNAME:-admin}" \
    API_DASHBOARD_PASSWORD="${API_DASHBOARD_PASSWORD:-admin}" \
    CLIP_STORE_HOST_DIR="${CLIP_STORE_HOST_DIR:-/tmp/eldercare-e2e-clip-store}" \
    ML_WORKER_PROFILE="$profile" ML_MODELS_DIR="$edge_models_dir" docker compose \
    -p "$compose_project" \
    -f compose.edge.yaml \
    create ml-api ml-worker >/dev/null
}

run_worker() {
  ML_API_IMAGE="$ml_api_image" \
    ML_WORKER_IMAGE="$ml_worker_image" \
    API_BACKEND_EVENTS_URL="${backend_base_url}/api/v1/events" \
    API_EDGE_RELAY_TOKEN="$relay_token" \
    API_DASHBOARD_USERNAME="${API_DASHBOARD_USERNAME:-admin}" \
    API_DASHBOARD_PASSWORD="${API_DASHBOARD_PASSWORD:-admin}" \
    CLIP_STORE_HOST_DIR="${CLIP_STORE_HOST_DIR:-/tmp/eldercare-e2e-clip-store}" \
    ML_WORKER_PROFILE="$profile" ML_MODELS_DIR="$edge_models_dir" docker compose \
    -p "$compose_project" \
    -f compose.edge.yaml \
    run -T --rm \
    ml-worker \
    python -m worker \
    --config /run/secrets/ml-worker.yaml \
    --max-frames-per-camera "$frames" \
    --heartbeat-on-start
}

assert_backend_alert() {
  local started_at="$1"
  local count
  count="$(docker exec "$db_container" psql -U "$postgres_user" -d "$postgres_db" -tAc \
    "select count(*) from alerts where facility_id = '${facility_id}' and type = 'fall' and detected_at >= '${started_at}'::timestamptz;")"
  if [[ "${count//[[:space:]]/}" -lt 1 ]]; then
    printf 'backend did not persist a recent fall alert for %s\n' "$facility_id" >&2
    return 1
  fi
  printf 'backend recent fall alerts: %s\n' "${count//[[:space:]]/}"
}

trap cleanup EXIT
cleanup_containers
write_lstm_runtime_artifact
write_config
start_compose_network
run_started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_worker
assert_backend_alert "$run_started_utc"
printf 'production backend E2E ok: rtsp=%s backend=%s camera=%s\n' "$rtsp_url" "$backend_base_url" "$camera_id"
