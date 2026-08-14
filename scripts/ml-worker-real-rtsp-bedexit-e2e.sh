#!/bin/bash
# Interpreter is pinned, not `#!/usr/bin/env bash`, on purpose.
#
# Homebrew bash 5.3.15 writes a heredoc body into a pipe before exec'ing the
# reader, so any body over PIPE_BUF (512 bytes on macOS) blocks forever against
# a pipe nothing is draining. bash 3.2.57 stages heredocs in a temp file and is
# unaffected at any size. See issue #9.
set -euo pipefail
umask 077

# allow: SIZE_OK - this real-RTSP proof keeps preflight, authority updates, run,
# DB readback, and evidence writing in one audited operator command.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/ml-worker-real-rtsp-bedexit-e2e.sh [--dry-run]

Runs the real RTSP bed-exit worker proof using ml-api's runtime camera registry
and detection-settings authority. The worker pulls its config from ml-api; this
script never renders a worker YAML roster or model/domain/clip policy.

Options:
  -h, --help                 Show this help and exit.
  --dry-run                  Print the redacted execution plan without network or file writes.

Environment:
  BED_EXIT_RTSP_URL          RTSP stream URL to set on the registered camera.
  ML_MODELS_DIR              Model artifact root, default: <repo>/models.
  BACKEND_BASE_URL           Backend URL, default: http://127.0.0.1:8080.
  RELAY_URL                  ml-api relay URL, default: http://127.0.0.1:8000.
  RELAY_TOKEN                Relay bearer token.
  E2E_DASHBOARD_USERNAME     ml-api dashboard username.
  E2E_DASHBOARD_PASSWORD     ml-api dashboard password.
  E2E_FACILITY_ID            Facility id, required.
  E2E_CAMERA_ID              Existing registry/backend camera id, required.
  E2E_RESIDENT_ID            Resident id used by evidence labels, required.
  MAX_FRAMES_PER_CAMERA      Worker frame limit, default: 3200.
  EVIDENCE_DIR               Evidence output directory.
  BED_EXIT_NIGHT_WINDOW_TZ   Expected ml-api detection timezone, default: Asia/Seoul.
EOF
}

mode="run"
case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  --dry-run)
    [[ $# -eq 1 ]] || {
      echo "ERROR: --dry-run accepts no arguments" >&2
      exit 1
    }
    mode="dry-run"
    ;;
  "")
    ;;
  *)
    echo "ERROR: unknown option: $1" >&2
    usage >&2
    exit 1
    ;;
esac

backend_base_url="${BACKEND_BASE_URL:-http://127.0.0.1:8080}"
relay_base_url="${RELAY_URL:-http://127.0.0.1:8000}"
relay_token="${RELAY_TOKEN:-local-edge-relay-token}"
rtsp_url="${BED_EXIT_RTSP_URL:-rtsp://127.0.0.1:8554/s1/trackID-1/streamID-2}"
models_dir="${ML_MODELS_DIR:-$repo_root/models}"
facility_id="${E2E_FACILITY_ID:-}"
resident_id="${E2E_RESIDENT_ID:-}"
camera_id="${E2E_CAMERA_ID:-}"
dashboard_username="${E2E_DASHBOARD_USERNAME:-${API_DASHBOARD_USERNAME:-}}"
dashboard_password="${E2E_DASHBOARD_PASSWORD:-${API_DASHBOARD_PASSWORD:-}}"
frames="${MAX_FRAMES_PER_CAMERA:-3200}"
night_window_tz="${BED_EXIT_NIGHT_WINDOW_TZ:-Asia/Seoul}"
policy_cooldown_sec="${ALERT_COOLDOWN_SEC:-60}"
include_window_start=""
include_window_end=""
exclude_window_start=""
exclude_window_end=""
window_now=""
db_container="${E2E_DB_CONTAINER:-eldercare-fall-db}"
postgres_user="${POSTGRES_USER:-fall}"
postgres_db="${POSTGRES_DB:-fall_dev}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_value() {
  [[ -n "$2" ]] || fail "$1 is required"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

redact_url() {
  python3 - "$1" <<'PY'
from __future__ import annotations

import sys
from urllib.parse import urlsplit

scheme = urlsplit(sys.argv[1]).scheme or "rtsp"
print(f"{scheme}://<redacted>")
PY
}

require_value E2E_FACILITY_ID "$facility_id"
require_value E2E_CAMERA_ID "$camera_id"
require_value E2E_RESIDENT_ID "$resident_id"
require_value E2E_DASHBOARD_USERNAME "$dashboard_username"
require_value E2E_DASHBOARD_PASSWORD "$dashboard_password"
[[ "$frames" =~ ^[1-9][0-9]*$ ]] || fail "MAX_FRAMES_PER_CAMERA must be a positive integer"
[[ "$facility_id" != *"'"* && "$camera_id" != *"'"* ]] || fail "facility/camera ids must not contain quotes"

if [[ "$mode" == "dry-run" ]]; then
  redacted_rtsp_url="$(redact_url "$rtsp_url")"
  DRY_RELAY_URL="$relay_base_url" \
    DRY_FACILITY_ID="$facility_id" \
    DRY_CAMERA_ID="$camera_id" \
    DRY_RESIDENT_ID="$resident_id" \
    DRY_RTSP_URL="$redacted_rtsp_url" \
    DRY_FRAMES="$frames" \
    DRY_TZ="$night_window_tz" \
    python3 - <<'PY'
import json
import os

relay = os.environ["DRY_RELAY_URL"].rstrip("/")
print(
    json.dumps(
        {
            "mode": "dry-run",
            "authority": {
                "camera_registry": f"{relay}/api/v1/cameras",
                "detection_settings": f"{relay}/api/v1/detection-settings",
                "worker_config": f"{relay}/api/v1/relay/config",
            },
            "worker_yaml": False,
            "facility_id": os.environ["DRY_FACILITY_ID"],
            "camera_id": os.environ["DRY_CAMERA_ID"],
            "resident_id": os.environ["DRY_RESIDENT_ID"],
            "rtsp_url": os.environ["DRY_RTSP_URL"],
            "frames_per_pass": int(os.environ["DRY_FRAMES"]),
            "expected_detection_timezone": os.environ["DRY_TZ"],
        },
        sort_keys=True,
    )
)
PY
  exit 0
fi

psql_scalar() {
  docker exec "$db_container" psql -U "$postgres_user" -d "$postgres_db" -tAc "$1" | tr -d '[:space:]'
}

psql_json() {
  docker exec "$db_container" psql -U "$postgres_user" -d "$postgres_db" -tAc "$1"
}

require_services() {
  curl -fsS "${backend_base_url}/" >/dev/null 2>&1 ||
    fail "backend is not reachable at ${backend_base_url}"
  curl -fsS "${relay_base_url}/health/live" >/dev/null 2>&1 ||
    fail "ml-api relay is not healthy at ${relay_base_url}"
  docker exec "$db_container" psql -U "$postgres_user" -d "$postgres_db" -tAc "select 1;" >/dev/null ||
    fail "database container is not reachable: ${db_container}"
  psql_scalar "select count(*) from cameras where id = '${camera_id}' and facility_id = '${facility_id}';" |
    grep -qx '1' || fail "camera/facility seed not found: ${facility_id}/${camera_id}"
}

require_artifacts() {
  local source_lstm_dir="$models_dir/fall/lstm"
  local required=(
    "$repo_root/worker/models/person/yolo26n.pt"
    "$repo_root/models/pose/yolo26n-pose.pt"
    "$repo_root/models/bed/yolo26m-seg.pt"
    "$source_lstm_dir/model.pt"
    "$source_lstm_dir/arch.json"
    "$source_lstm_dir/metadata.yaml"
  )
  for path in "${required[@]}"; do
    [[ -f "$path" ]] || fail "missing real model artifact: $path"
  done
}

write_lstm_runtime_artifact() {
  local source_lstm_dir="$models_dir/fall/lstm"
  mkdir -p "$runtime_lstm_dir"
  cp "$source_lstm_dir/model.pt" "$runtime_lstm_dir/model.pt"
  cp "$source_lstm_dir/arch.json" "$runtime_lstm_dir/arch.json"
  cp "$source_lstm_dir/metadata.yaml" "$runtime_lstm_dir/metadata.yaml"
  {
    printf 'source_lstm_dir=%s\n' "$source_lstm_dir"
    printf 'runtime_lstm_dir=%s\n' "$runtime_lstm_dir"
    shasum -a 256 "$runtime_lstm_dir/model.pt" "$runtime_lstm_dir/arch.json" "$runtime_lstm_dir/metadata.yaml"
  } >"$evidence_dir/lstm-runtime-artifact.txt"
}

compute_windows() {
  python3 - "$night_window_tz" <<'PY'
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

now = datetime.now(ZoneInfo(sys.argv[1]))

def hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")

print(f'include_window_start="{hhmm(now - timedelta(hours=2))}"')
print(f'include_window_end="{hhmm(now + timedelta(hours=2))}"')
print(f'exclude_window_start="{hhmm(now + timedelta(hours=4))}"')
print(f'exclude_window_end="{hhmm(now + timedelta(hours=5))}"')
print(f'window_now="{now.isoformat()}"')
PY
}

api_request() {
  local method="$1"
  local path="$2"
  local output="$3"
  local input="${4:-}"
  local args=(
    -fsS
    -X "$method"
    --cookie "$session_cookie"
    -H 'Accept: application/json'
    -o "$output"
  )
  if [[ -n "$input" ]]; then
    args+=(-H 'Content-Type: application/json' --data-binary "@$input")
  fi
  curl "${args[@]}" "${relay_base_url}${path}"
}

login_dashboard() {
  local login_request="$tmpdir/dashboard-login.json"
  touch "$session_cookie" "$login_request"
  chmod 600 "$session_cookie" "$login_request"
  DASHBOARD_USERNAME="$dashboard_username" DASHBOARD_PASSWORD="$dashboard_password" \
    python3 - "$login_request" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "username": os.environ["DASHBOARD_USERNAME"],
            "password": os.environ["DASHBOARD_PASSWORD"],
        }
    ),
    encoding="utf-8",
)
PY
  curl -fsS \
    -X POST \
    -H 'Content-Type: application/json' \
    --data-binary "@$login_request" \
    --cookie-jar "$session_cookie" \
    -o /dev/null \
    "${relay_base_url}/api/v1/auth/session" || fail "ml-api dashboard authentication failed"
  rm -f "$login_request"
}

capture_and_update_camera() {
  local public_snapshot="$tmpdir/cameras-public.json"
  local worker_snapshot="$tmpdir/worker-config-before.json"
  local desired_patch="$tmpdir/camera-desired.json"
  local patch_response="$tmpdir/camera-patch-response.json"

  api_request GET /api/v1/cameras "$public_snapshot"
  curl -fsS \
    -H "@$relay_header_file" \
    -o "$worker_snapshot" \
    "${relay_base_url}/api/v1/relay/config"
  CAMERA_ID="$camera_id" DESIRED_RTSP_URL="$rtsp_url" python3 - \
    "$public_snapshot" "$worker_snapshot" "$camera_restore_request" "$desired_patch" "$camera_local_id_file" <<'PY'
import json
import os
import sys
from pathlib import Path

public_path, worker_path, restore_path, desired_path, local_id_path = map(Path, sys.argv[1:])
target = os.environ["CAMERA_ID"]
desired_url = os.environ["DESIRED_RTSP_URL"]
public = json.loads(public_path.read_text(encoding="utf-8"))
worker = json.loads(worker_path.read_text(encoding="utf-8"))
records = [
    camera
    for camera in public.get("cameras", [])
    if target in {camera.get("id"), camera.get("backend_camera_id")}
]
if len(records) != 1:
    raise SystemExit(
        f"expected exactly one existing ml-api registry camera for {target!r}, found {len(records)}; "
        "register and map it in the dashboard before running this proof"
    )
runtime = [camera for camera in worker.get("cameras", []) if camera.get("camera_id") == target]
if len(runtime) != 1 or not isinstance(runtime[0].get("rtsp_url"), str):
    raise SystemExit(f"runtime camera authority has no unique usable camera {target!r}")
restore_path.write_text(json.dumps({"rtsp_url": runtime[0]["rtsp_url"]}), encoding="utf-8")
desired_path.write_text(json.dumps({"rtsp_url": desired_url}), encoding="utf-8")
local_id_path.write_text(str(records[0]["id"]), encoding="utf-8")
PY
  chmod 600 "$worker_snapshot" "$camera_restore_request" "$desired_patch" "$camera_local_id_file"
  camera_local_id="$(cat "$camera_local_id_file")"
  api_request PATCH "/api/v1/cameras/${camera_local_id}" "$patch_response" "$desired_patch"
  camera_changed=1
}

write_detection_settings() {
  local path="$1"
  local window_start="$2"
  local window_end="$3"
  python3 - "$path" "$window_start" "$window_end" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "domains": {
                "fall": {"on": False, "mode": "always"},
                "bed_exit": {
                    "on": True,
                    "mode": "window",
                    "start": sys.argv[2],
                    "end": sys.argv[3],
                },
            }
        }
    ),
    encoding="utf-8",
)
PY
  chmod 600 "$path"
}

apply_detection_settings() {
  local request="$1"
  local response="$tmpdir/detection-settings-response.json"
  api_request PUT /api/v1/detection-settings "$response" "$request"
  settings_changed=1
}

capture_authority_evidence() {
  local label="$1"
  local expected_start="$2"
  local expected_end="$3"
  local raw="$tmpdir/${label}-worker-config.json"
  local redacted="$evidence_dir/${label}-runtime-authority.redacted.json"
  curl -fsS \
    -H "@$relay_header_file" \
    -o "$raw" \
    "${relay_base_url}/api/v1/relay/config"
  chmod 600 "$raw"
  CAMERA_ID="$camera_id" RTSP_URL="$rtsp_url" EXPECTED_TZ="$night_window_tz" python3 - \
    "$raw" "$redacted" "$expected_start" "$expected_end" <<'PY'
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

source, target = map(Path, sys.argv[1:3])
expected_start, expected_end = sys.argv[3:5]
payload = json.loads(source.read_text(encoding="utf-8"))
cameras = [camera for camera in payload.get("cameras", []) if camera.get("camera_id") == os.environ["CAMERA_ID"]]
if len(cameras) != 1 or cameras[0].get("rtsp_url") != os.environ["RTSP_URL"]:
    raise SystemExit("ml-api runtime camera registry did not publish the requested RTSP camera")
domains = payload.get("domains", {})
if domains.get("fall", {}).get("enabled") is not False or domains.get("bed_exit", {}).get("enabled") is not True:
    raise SystemExit("ml-api runtime domain authority did not publish fall=off, bed_exit=on")
window = payload.get("detection_windows", {}).get("bed_exit", {})
expected_window = {
    "start": expected_start,
    "end": expected_end,
    "tz": os.environ["EXPECTED_TZ"],
}
if window != expected_window:
    raise SystemExit(f"ml-api runtime bed-exit window mismatch: expected {expected_window!r}, got {window!r}")
for camera in payload.get("cameras", []):
    value = camera.get("rtsp_url")
    if isinstance(value, str):
        scheme = urlsplit(value).scheme or "rtsp"
        camera["rtsp_url"] = f"{scheme}://<redacted>"
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  chmod 600 "$redacted"
}

restore_authority() {
  if [[ "${camera_changed:-0}" -eq 1 && -s "${camera_restore_request:-}" && -s "${camera_local_id_file:-}" ]]; then
    camera_local_id="$(cat "$camera_local_id_file")"
    if ! api_request PATCH "/api/v1/cameras/${camera_local_id}" /dev/null "$camera_restore_request"; then
      printf 'WARNING: failed to restore camera registry RTSP URL for %s\n' "$camera_id" >&2
    fi
  fi
  if [[ "${settings_changed:-0}" -eq 1 && -s "${saved_detection_settings:-}" ]]; then
    if ! api_request PUT /api/v1/detection-settings /dev/null "$saved_detection_settings"; then
      printf 'WARNING: failed to restore ml-api detection settings\n' >&2
    fi
  fi
}

wait_for_policy_cooldown() {
  local wait_sec
  wait_sec="$(psql_scalar "select greatest(0, ${policy_cooldown_sec} - coalesce(extract(epoch from now() - max(created_at)), ${policy_cooldown_sec}))::int from alerts where facility_id = '${facility_id}' and camera_id = '${camera_id}' and type = 'bed-exit';")"
  if [[ "$wait_sec" =~ ^[0-9]+$ ]] && (( wait_sec > 0 )); then
    printf 'waiting %ss for backend alert cooldown before include pass\n' "$((wait_sec + 2))"
    sleep "$((wait_sec + 2))"
  fi
}

run_worker() {
  local log="$1"
  : >"$log"
  OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp" \
    PYTHONUNBUFFERED=1 \
    RELAY_URL="$relay_base_url" \
    RELAY_TOKEN="$relay_token" \
    ML_WORKER_FALL_MODEL_ARTIFACT_DIR="$runtime_lstm_dir" \
    ML_WORKER_FALL_MODEL_WINDOW=30 \
    ML_WORKER_FALL_MODEL_STRIDE=5 \
    ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD=0.5 \
    uv run \
    python -m worker \
    --max-frames-per-camera "$frames" \
    --heartbeat-on-start >"$log" 2>&1
}

count_events_since() {
  local started_at="$1"
  psql_scalar "select count(*) from events where facility_id = '${facility_id}' and camera_id = '${camera_id}' and type = 'bed-exit' and created_at >= '${started_at}'::timestamptz;"
}

count_alerts_since() {
  local started_at="$1"
  psql_scalar "select count(*) from alerts where facility_id = '${facility_id}' and camera_id = '${camera_id}' and type = 'bed-exit' and created_at >= '${started_at}'::timestamptz;"
}

capture_rows_since() {
  local label="$1"
  local started_at="$2"
  psql_json "select coalesce(json_agg(row_to_json(t)), '[]'::json) from (select id, facility_id, camera_id, space_id, type, confidence, detected_at, created_at from events where facility_id = '${facility_id}' and camera_id = '${camera_id}' and type = 'bed-exit' and created_at >= '${started_at}'::timestamptz order by created_at desc) t;" >"$evidence_dir/${label}-events.json"
  psql_json "select coalesce(json_agg(row_to_json(t)), '[]'::json) from (select alert_seq::text as alert_seq, id, facility_id, resident_id, camera_id, space_id, type, probability, detected_at, status, origin_event_id, created_at from alerts where facility_id = '${facility_id}' and camera_id = '${camera_id}' and type = 'bed-exit' and created_at >= '${started_at}'::timestamptz order by created_at desc) t;" >"$evidence_dir/${label}-alerts.json"
}

write_command_file() {
  local redacted_rtsp_url
  redacted_rtsp_url="$(redact_url "$rtsp_url")"
  cat >"$command_file" <<EOF
BED_EXIT_RTSP_URL='${redacted_rtsp_url}' \\
ML_MODELS_DIR='${models_dir}' \\
BACKEND_BASE_URL='${backend_base_url}' \\
RELAY_URL='${relay_base_url}' \\
RELAY_TOKEN='<redacted>' \\
E2E_DASHBOARD_USERNAME='<redacted>' \\
E2E_DASHBOARD_PASSWORD='<redacted>' \\
E2E_FACILITY_ID='${facility_id}' \\
E2E_CAMERA_ID='${camera_id}' \\
E2E_RESIDENT_ID='${resident_id}' \\
MAX_FRAMES_PER_CAMERA='${frames}' \\
scripts/ml-worker-real-rtsp-bedexit-e2e.sh
EOF
  chmod 600 "$command_file"
}

write_summary() {
  local include_started_at="$1"
  local include_events="$2"
  local include_alerts="$3"
  local exclude_started_at="$4"
  local exclude_events="$5"
  local exclude_alerts="$6"
  local redacted_rtsp_url
  redacted_rtsp_url="$(redact_url "$rtsp_url")"
  cat >"$summary" <<EOF
# Real RTSP worker bed-exit E2E

- Captured at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Worker: real \`python -m worker\`, pull-only runtime config
- Runtime authority: ml-api camera registry and detection settings
- RTSP source: \`${redacted_rtsp_url}\`
- Facility/camera/resident: \`${facility_id}\` / \`${camera_id}\` / \`${resident_id}\`
- Backend: \`${backend_base_url}\`
- Relay: \`${relay_base_url}\`
- Frames per pass: \`${frames}\`
- Clock timezone: \`${night_window_tz}\`
- Real model artifacts: \`${models_dir}\`
- LSTM runtime artifact proof: \`$evidence_dir/lstm-runtime-artifact.txt\`
- No mocks/stubs/fakes: no runner injection, no monkeypatching, no fake backend, no fake RTSP server in this repo.

## Night-window include pass

- Started at: \`${include_started_at}\`
- Window at real clock \`${window_now}\`: \`${include_window_start}-${include_window_end} ${night_window_tz}\`
- New backend events: \`${include_events}\`
- New backend alerts: \`${include_alerts}\`
- Worker log: \`$include_log\`
- Events readback: \`$evidence_dir/include-events.json\`
- Alerts readback: \`$evidence_dir/include-alerts.json\`

## Night-window exclude pass

- Started at: \`${exclude_started_at}\`
- Window at real clock \`${window_now}\`: \`${exclude_window_start}-${exclude_window_end} ${night_window_tz}\`
- New backend events: \`${exclude_events}\`
- New backend alerts: \`${exclude_alerts}\`
- Worker log: \`$exclude_log\`
- Events readback: \`$evidence_dir/exclude-events.json\`
- Alerts readback: \`$evidence_dir/exclude-alerts.json\`

## Runtime authority evidence

- Include snapshot redacted: \`$evidence_dir/include-runtime-authority.redacted.json\`
- Exclude snapshot redacted: \`$evidence_dir/exclude-runtime-authority.redacted.json\`
- Exact command: \`$command_file\`
EOF
}

evidence_dir="${EVIDENCE_DIR:-$repo_root/.omo/evidence/ml-event-alert-e2e-nokyang/worker-real-rtsp}"
tmp_root="${ML_EDGE_E2E_TMP_ROOT:-$repo_root/.omo/tmp}"
mkdir -p "$evidence_dir" "$tmp_root"
tmpdir="$(mktemp -d "$tmp_root/ml-worker-real-rtsp-bedexit.XXXXXX")"
runtime_lstm_dir="$tmpdir/models/fall/lstm-runtime"
include_settings="$tmpdir/detection-settings-include.json"
exclude_settings="$tmpdir/detection-settings-exclude.json"
saved_detection_settings="$tmpdir/detection-settings-before.json"
session_cookie="$tmpdir/dashboard-cookie.txt"
relay_header_file="$tmpdir/relay-header.txt"
camera_restore_request="$tmpdir/camera-restore.json"
camera_local_id_file="$tmpdir/camera-local-id.txt"
include_log="$evidence_dir/worker-include.log"
exclude_log="$evidence_dir/worker-exclude.log"
summary="$evidence_dir/summary.md"
command_file="$evidence_dir/command.txt"
camera_changed=0
settings_changed=0
printf 'X-Edge-Relay-Token: %s\n' "$relay_token" >"$relay_header_file"

cleanup() {
  restore_authority
  rm -rf "$tmpdir"
}
trap cleanup EXIT

require_command curl
require_command docker
require_command python3
require_command uv
require_services
require_artifacts
login_dashboard
api_request GET /api/v1/detection-settings "$saved_detection_settings"
chmod 600 "$saved_detection_settings"
capture_and_update_camera
write_lstm_runtime_artifact
eval "$(compute_windows)"
write_detection_settings "$include_settings" "$include_window_start" "$include_window_end"
write_detection_settings "$exclude_settings" "$exclude_window_start" "$exclude_window_end"
write_command_file

RELAY_URL="$relay_base_url" \
RELAY_TOKEN="$relay_token" \
ML_WORKER_FALL_MODEL_ARTIFACT_DIR="$runtime_lstm_dir" \
ML_WORKER_FALL_MODEL_WINDOW=30 \
ML_WORKER_FALL_MODEL_STRIDE=5 \
ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD=0.5 \
uv run python -m worker --check-config >/dev/null

apply_detection_settings "$include_settings"
capture_authority_evidence include "$include_window_start" "$include_window_end"
wait_for_policy_cooldown

include_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_worker "$include_log"
include_events="$(count_events_since "$include_started_at")"
include_alerts="$(count_alerts_since "$include_started_at")"
capture_rows_since include "$include_started_at"

if [[ "$include_events" -lt 1 || "$include_alerts" -lt 1 ]]; then
  write_summary "$include_started_at" "$include_events" "$include_alerts" "" "" ""
  printf 'include pass failed: events=%s alerts=%s\n' "$include_events" "$include_alerts" >&2
  printf 'worker include log follows:\n' >&2
  sed -n '1,220p' "$include_log" >&2
  exit 1
fi

apply_detection_settings "$exclude_settings"
capture_authority_evidence exclude "$exclude_window_start" "$exclude_window_end"
exclude_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_worker "$exclude_log"
exclude_events="$(count_events_since "$exclude_started_at")"
exclude_alerts="$(count_alerts_since "$exclude_started_at")"
capture_rows_since exclude "$exclude_started_at"

write_summary "$include_started_at" "$include_events" "$include_alerts" "$exclude_started_at" "$exclude_events" "$exclude_alerts"

if [[ "$exclude_events" -ne 0 || "$exclude_alerts" -ne 0 ]]; then
  printf 'exclude pass failed: events=%s alerts=%s\n' "$exclude_events" "$exclude_alerts" >&2
  printf 'worker exclude log follows:\n' >&2
  sed -n '1,220p' "$exclude_log" >&2
  exit 1
fi

printf 'real RTSP bed-exit worker E2E ok: include events=%s alerts=%s; exclude events=%s alerts=%s; evidence=%s\n' \
  "$include_events" "$include_alerts" "$exclude_events" "$exclude_alerts" "$evidence_dir"
