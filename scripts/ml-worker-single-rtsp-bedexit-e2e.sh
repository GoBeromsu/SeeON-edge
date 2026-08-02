#!/bin/bash
# Interpreter is pinned, not `#!/usr/bin/env bash`, on purpose.
#
# Homebrew bash 5.3.15 writes a heredoc body into a pipe before exec'ing the
# reader, so any body over PIPE_BUF (512 bytes on macOS) blocks forever against
# a pipe nothing is draining -- the command is never exec'd and the script hangs
# with no output. This file has a heredoc over that limit. bash 3.2.57 stages
# heredocs in a temp file and is unaffected at any size. See issue #9.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend_base_url="${BACKEND_BASE_URL:-http://127.0.0.1:8080}"
relay_base_url="${RELAY_URL:-http://127.0.0.1:8000}"
relay_token="${RELAY_TOKEN:-local-edge-relay-token}"
facility_id="${E2E_FACILITY_ID:?E2E_FACILITY_ID is required}"
resident_id="${E2E_RESIDENT_ID:?E2E_RESIDENT_ID is required}"
camera_id="${E2E_CAMERA_ID:?E2E_CAMERA_ID is required}"
night_now="${BED_EXIT_NIGHT_NOW:-2026-06-25T22:00:00+09:00}"
day_now="${BED_EXIT_DAY_NOW:-2026-06-25T13:00:00+09:00}"
night_window_start="${BED_EXIT_NIGHT_WINDOW_START:-21:00}"
night_window_end="${BED_EXIT_NIGHT_WINDOW_END:-05:00}"
night_window_tz="${BED_EXIT_NIGHT_WINDOW_TZ:-Asia/Seoul}"
frames="${MAX_FRAMES_PER_CAMERA:-8}"
rtsp_url="${BED_EXIT_RTSP_URL:?BED_EXIT_RTSP_URL is required; start SeniorAILab/rtsp-generator separately and pass a worker-reachable URL}"
compose_project="${COMPOSE_PROJECT_NAME:-ml-worker-single-rtsp-bedexit-e2e}"
db_container="${E2E_DB_CONTAINER:-eldercare-fall-db}"
postgres_user="${POSTGRES_USER:-fall}"
postgres_db="${POSTGRES_DB:-fall_dev}"
tmp_root="${ML_EDGE_E2E_TMP_ROOT:-$repo_root/.gjc/tmp}"
mkdir -p "$tmp_root"
tmpdir="$(mktemp -d "$tmp_root/ml-worker-single-bedexit.XXXXXX")"
config="$tmpdir/ml-worker.yaml"
api_log="$tmpdir/ml-api.log"
worker_log="$tmpdir/ml-worker.log"
api_pid=""

# Sandbox ml-api/ml-worker runtime state (GPU lease file, durable evidence
# outbox, camera/credentials catalog) under this run's disposable tmpdir
# instead of the developer's real ~/.local/state/{ml-api,ml-worker} -- both
# resolvers (worker/runtime/state_dir.py, backend/app/shared/state_dir.py)
# derive their path from Path.home() with no environment override, so HOME
# is the only sandboxing seam left. It is set only on the api/worker process
# invocations below (start_api, run_worker_with_clock), not for this whole
# script, so `uv run` here keeps resolving its own cache under the real
# HOME instead of an empty one. Also turn on relay export --
# EvidenceExportRuntime.from_environment() (worker/pipeline/output/evidence/
# evidence_runtime.py) is disabled unless ML_WORKER_EVENT_CLIP_EXPORT_ENABLED=1,
# in which case admitted events stage to the durable outbox but are never
# sent to the relay, and this harness's whole point is proving relay delivery.
sandbox_home="$tmpdir/home"
export ML_WORKER_EVENT_CLIP_EXPORT_ENABLED=1
mkdir -p "$sandbox_home"

cleanup() {
  if [[ -n "$api_pid" ]]; then
    kill "$api_pid" >/dev/null 2>&1 || true
    wait "$api_pid" >/dev/null 2>&1 || true
  fi
  rm -rf "$tmpdir"
}

require_backend() {
  if ! curl -fsS "${backend_base_url}/" >/dev/null 2>&1; then
    printf 'backend is not running or /health is unavailable: %s\n' "$backend_base_url" >&2
    printf 'start backend first (for example: pnpm dev:backend) and retry.\n' >&2
    return 1
  fi
}

write_config() {
  cat >"$config" <<YAML
version: 1
relay:
  url: ${relay_base_url}
  token: ${relay_token}
runtime:
  max_failures: 30
  open_timeout_ms: 20000
  read_timeout_ms: 20000
domains:
  bed_exit:
    enabled: true
    night_window:
      start: "${night_window_start}"
      end: "${night_window_end}"
      tz: ${night_window_tz}
cameras:
  - camera_id: ${camera_id}
    facility_id: ${facility_id}
    resident_id: ${resident_id}
    rtsp_url: ${rtsp_url}
    heartbeat_interval_sec: 30
    frame_stride: 1
    label: single-bed-exit
YAML
  chmod 600 "$config"
}

start_api() {
  HOME="$sandbox_home" \
  API_BACKEND_EVENTS_URL="${backend_base_url}/api/v1/events" \
  API_EDGE_RELAY_TOKEN="$relay_token" \
  API_CAMERA_INVENTORY="[{\"camera_id\":\"${camera_id}\",\"facility_id\":\"${facility_id}\",\"resident_id\":\"${resident_id}\"}]" \
  uv run uvicorn backend.app.main:app --host 127.0.0.1 --port "${relay_base_url##*:}" >"$api_log" 2>&1 &
  api_pid="$!"
  for _ in $(seq 1 60); do
    if curl -fsS "${relay_base_url}/health/live" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  printf 'ml-api did not become healthy; log follows:\n' >&2
  sed -n '1,120p' "$api_log" >&2
  return 1
}

alert_count_since() {
  local started_at="$1"
  docker exec "$db_container" psql -U "$postgres_user" -d "$postgres_db" -tAc \
    "select count(*) from alerts where facility_id = '${facility_id}' and type = 'bed-exit' and detected_at >= '${started_at}'::timestamptz;" | tr -d '[:space:]'
}

# Event admission (worker/domains/bed_exit) and relay export (worker/pipeline/
# output/evidence/evidence_runtime.py's EvidenceExportRuntime sender) run on
# separate background threads inside the worker process, so the alert can
# land in the backend a short interval after run_worker_with_clock returns.
# Retry the exact same readback query instead of a single point-in-time
# check.
wait_for_alert_count_at_least() {
  local started_at="$1" minimum="$2" count=0
  for _ in $(seq 1 20); do
    count="$(alert_count_since "$started_at")"
    if [[ "$count" -ge "$minimum" ]]; then
      printf '%s\n' "$count"
      return 0
    fi
    sleep 1
  done
  printf '%s\n' "$count"
  return 1
}

run_worker_with_clock() {
  local now="$1"
  # Body lives in scripts/single_rtsp_bedexit_worker_run.py, not a heredoc.
  # At 7162 bytes it exceeds PIPE_BUF on macOS (512) *and* Linux (4096), and
  # bash 5.x writes a heredoc body into a pipe before exec'ing the reader, so
  # this hung with no output. The #!/bin/bash pin only helps on macOS, where
  # /bin/bash is 3.2.57; on a Linux edge host it is a modern bash, so the body
  # had to move to a file. See issue #9.
  #
  # PYTHONPATH is required by that move, not incidental: `python -` puts the
  # working directory on sys.path, while running a file puts the *script's*
  # directory there instead -- so `import worker` resolved as a heredoc and
  # would not as a file. Setting it restores what the heredoc got implicitly.
  if ! HOME="$sandbox_home" PYTHONPATH="$repo_root" OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp" BED_EXIT_NOW="$now" EDGE_CAMERA_CONFIG="$config" uv run python "$repo_root/scripts/single_rtsp_bedexit_worker_run.py" "$frames" >>"$worker_log" 2>&1
  then
    printf 'worker run failed; log follows:\n' >&2
    sed -n '1,200p' "$worker_log" >&2
    return 1
  fi
}

trap cleanup EXIT
require_backend
write_config
start_api
printf 'clock night=%s day=%s night_window=%s-%s %s relay=%s backend_events=%s rtsp=%s\n' \
  "$night_now" "$day_now" "$night_window_start" "$night_window_end" "$night_window_tz" \
  "$relay_base_url" "$backend_base_url" "$rtsp_url"
night_started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_worker_with_clock "$night_now"
if ! night_count="$(wait_for_alert_count_at_least "$night_started_utc" 1)"; then
  printf 'night bed-exit did not reach backend Event API; worker log follows:\n' >&2
  sed -n '1,160p' "$worker_log" >&2
  printf '%s\n' '--- ml-api log ---' >&2
  sed -n '1,200p' "$api_log" >&2
  exit 1
fi
printf 'night alert count: %s\n' "$night_count"

day_started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_worker_with_clock "$day_now"
day_count="$(alert_count_since "$day_started_utc")"
if [[ "$day_count" -ne 0 ]]; then
  printf 'daytime bed-exit was not suppressed; new backend alert count=%s\n' "$day_count" >&2
  exit 1
fi
printf 'day suppress count: %s\n' "$day_count"
printf 'single RTSP bed-exit relay harness ok: camera=%s facility=%s\n' "$camera_id" "$facility_id"
