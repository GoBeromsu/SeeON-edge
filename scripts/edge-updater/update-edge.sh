#!/bin/sh
set -eu

DATA_DIR=${EDGE_UPDATER_DATA_DIR:-/var/lib/edge-updater}
ENV_FILE=${EDGE_UPDATER_ENV_FILE:-.env.edge.prod}
COMPOSE_FILE=${EDGE_UPDATER_COMPOSE_FILE:-compose.edge.yaml}
LOG_FILE=${EDGE_UPDATER_LOG_FILE:-$DATA_DIR/update.log}
OUTBOX_DIR=${EDGE_UPDATER_OUTBOX_DIR:-$DATA_DIR/outbox}
LOCK_DIR=${EDGE_UPDATER_LOCK_DIR:-$DATA_DIR/lock}
REPORT_URL=${EDGE_UPDATER_REPORT_URL:-}
DRY_RUN=${EDGE_UPDATER_DRY_RUN:-0}
SERVICES=${EDGE_UPDATER_SERVICES:-ml-api ml-worker}
VERIFY_TIMEOUT_MIN=${EDGE_UPDATER_VERIFY_TIMEOUT_MIN:-5}
VERIFY_TIMEOUT_SEC=${EDGE_UPDATER_VERIFY_TIMEOUT_SEC:-$((VERIFY_TIMEOUT_MIN * 60))}
VERIFY_INTERVAL_SEC=${EDGE_UPDATER_VERIFY_INTERVAL_SEC:-5}
HEALTH_PATH=${EDGE_UPDATER_HEALTH_PATH:-/health/ready}
STATUS_PATH=${EDGE_UPDATER_STATUS_PATH:-/api/v1/status}
SYSTEM_PATH=${EDGE_UPDATER_SYSTEM_PATH:-/api/v1/system}

TARGET_DIGESTS=""
PREVIOUS_DIGESTS=""
TARGET_IMAGES=""
ROLLBACK_RESULT="not_attempted"
FAILURE_PHASE=""
FAILURE_REASON=""

usage() {
  cat >&2 <<'MSG'
Usage: scripts/edge-updater/update-edge.sh

Required files/env:
  .env.edge.prod             Edge compose env file, override with EDGE_UPDATER_ENV_FILE.
  compose.edge.yaml          Edge compose file, override with EDGE_UPDATER_COMPOSE_FILE.

Common env:
  EDGE_UPDATER_DATA_DIR              Default: /var/lib/edge-updater
  EDGE_UPDATER_REPORT_URL            Optional backend report URL for result POSTs.
  EDGE_UPDATER_ML_API_BASE_URL       Default: http://127.0.0.1:${ML_SERVING_PORT:-8000}
  EDGE_UPDATER_VERIFY_TIMEOUT_MIN    Default: 5
  EDGE_UPDATER_DRY_RUN=1             Check and report only; no pull/up/env mutation.
MSG
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 2
fi

mkdir -p "$DATA_DIR" "$OUTBOX_DIR"

cleanup_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf '%s\n' "edge updater already running; lock=$LOCK_DIR" >&2
  exit 0
fi
trap cleanup_lock EXIT INT TERM

now_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

json_safe() {
  printf '%s' "$1" | tr '\n\r\t' '   ' | sed 's/["\\]/ /g'
}

log_event() {
  log_phase=$1
  log_status=$2
  log_message=$3
  log_ts=$(now_utc)
  printf '{"ts":"%s","phase":"%s","status":"%s","target_digest":"%s","previous_digest":"%s","failure_phase":"%s","rollback_result":"%s","message":"%s"}\n' \
    "$log_ts" \
    "$(json_safe "$log_phase")" \
    "$(json_safe "$log_status")" \
    "$(json_safe "$TARGET_DIGESTS")" \
    "$(json_safe "$PREVIOUS_DIGESTS")" \
    "$(json_safe "$FAILURE_PHASE")" \
    "$(json_safe "$ROLLBACK_RESULT")" \
    "$(json_safe "$log_message")" >>"$LOG_FILE"
}

write_payload() {
  payload_path=$1
  payload_status=$2
  payload_phase=$3
  payload_message=$4
  payload_ts=$(now_utc)
  cat >"$payload_path" <<EOF_PAYLOAD
{"ts":"$payload_ts","status":"$(json_safe "$payload_status")","phase":"$(json_safe "$payload_phase")","target_images":"$(json_safe "$TARGET_IMAGES")","target_digest":"$(json_safe "$TARGET_DIGESTS")","previous_digest":"$(json_safe "$PREVIOUS_DIGESTS")","failure_phase":"$(json_safe "$FAILURE_PHASE")","rollback_result":"$(json_safe "$ROLLBACK_RESULT")","message":"$(json_safe "$payload_message")"}
EOF_PAYLOAD
}

post_payload() {
  post_file=$1
  [ -n "$REPORT_URL" ] || return 1
  curl -fsS -X POST -H 'Content-Type: application/json' --data-binary "@$post_file" "$REPORT_URL" >/dev/null
}

flush_outbox() {
  [ -n "$REPORT_URL" ] || return 0
  for outbox_file in "$OUTBOX_DIR"/*.json; do
    [ -f "$outbox_file" ] || continue
    if post_payload "$outbox_file"; then
      rm -f "$outbox_file"
    fi
  done
}

report_result() {
  report_status=$1
  report_phase=$2
  report_message=$3
  log_event "$report_phase" "$report_status" "$report_message"
  flush_outbox || true
  tmp_report="$DATA_DIR/report.$$.$(date +%s).json"
  write_payload "$tmp_report" "$report_status" "$report_phase" "$report_message"
  if [ -n "$REPORT_URL" ]; then
    if post_payload "$tmp_report"; then
      rm -f "$tmp_report"
    else
      mv "$tmp_report" "$OUTBOX_DIR/report-$(date +%s)-$$.json"
    fi
  else
    rm -f "$tmp_report"
  fi
  printf '%s: %s\n' "$report_status" "$report_message"
}

fail_before_apply() {
  FAILURE_PHASE=$1
  FAILURE_REASON=$2
  report_result "failure" "$FAILURE_PHASE" "$FAILURE_REASON"
  exit 1
}

env_value() {
  env_name=$1
  awk -v name="$env_name" '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    {
      line=$0
      sub(/^[[:space:]]*export[[:space:]]+/, "", line)
      if (index(line, name "=") == 1) {
        sub("^[^=]*=", "", line)
        gsub(/\r$/, "", line)
        print line
      }
    }
  ' "$ENV_FILE" | tail -n 1
}

require_file() {
  req_file=$1
  req_name=$2
  [ -f "$req_file" ] || fail_before_apply "CHECK" "$req_name not found: $req_file"
}

hash_file() {
  hash_target=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$hash_target" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$hash_target" | awk '{print $1}'
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 -r "$hash_target" | awk '{print $1}'
  else
    printf '%s\n' 'missing sha256sum, shasum, or openssl' >&2
    return 1
  fi
}

input_hash() {
  hash_tmp="$DATA_DIR/input-hash.$$"
  {
    printf 'env:%s\n' "$ENV_FILE"
    awk '{ sub(/\r$/, ""); sub(/[[:space:]]+$/, ""); print }' "$ENV_FILE"
    printf 'compose:%s\n' "$COMPOSE_FILE"
    awk '{ sub(/\r$/, ""); sub(/[[:space:]]+$/, ""); print }' "$COMPOSE_FILE"
  } >"$hash_tmp"
  hash_value=$(hash_file "$hash_tmp")
  rm -f "$hash_tmp"
  printf 'sha256:%s' "$hash_value"
}

image_repo() {
  image_ref=$1
  case "$image_ref" in
    *@*) printf '%s' "${image_ref%@*}" ;;
    *:*) printf '%s' "${image_ref%:*}" ;;
    *) printf '%s' "$image_ref" ;;
  esac
}

service_env_var() {
  case "$1" in
    ml-api) printf '%s' 'ML_API_IMAGE' ;;
    ml-worker) printf '%s' 'ML_WORKER_IMAGE' ;;
    *) printf '%s' "$(printf '%s' "$1" | tr '[:lower:]-' '[:upper:]_')_IMAGE" ;;
  esac
}

remote_digest() {
  remote_ref=$1
  remote_out=$(docker buildx imagetools inspect "$remote_ref" 2>/dev/null || true)
  remote_value=$(printf '%s\n' "$remote_out" | awk '/^[[:space:]]*Digest:/ { print $2; exit }')
  if [ -n "$remote_value" ]; then
    printf '%s' "$remote_value"
    return 0
  fi
  remote_out=$(docker manifest inspect --verbose "$remote_ref" 2>/dev/null || true)
  remote_value=$(printf '%s\n' "$remote_out" | awk '{ if (match($0, /sha256:[0-9A-Fa-f][0-9A-Fa-f]*/)) { print substr($0, RSTART, RLENGTH); exit } }')
  [ -n "$remote_value" ] || return 1
  printf '%s' "$remote_value"
}

current_digest() {
  current_service=$1
  current_ref=$2
  current_repo=$(image_repo "$current_ref")
  current_cid=$(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps -q "$current_service" 2>/dev/null || true)
  [ -n "$current_cid" ] || return 1
  current_image_id=$(docker inspect --format '{{.Image}}' "$current_cid" 2>/dev/null || true)
  [ -n "$current_image_id" ] || return 1
  current_repo_digests=$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$current_image_id" 2>/dev/null || true)
  current_value=$(printf '%s\n' "$current_repo_digests" | awk -v repo="$current_repo" 'index($0, repo "@") == 1 { sub(/^.*@/, ""); print; exit }')
  if [ -z "$current_value" ]; then
    current_value=$(printf '%s\n' "$current_repo_digests" | awk '{ if (match($0, /sha256:[0-9A-Fa-f][0-9A-Fa-f]*/)) { print substr($0, RSTART, RLENGTH); exit } }')
  fi
  [ -n "$current_value" ] || return 1
  printf '%s' "$current_value"
}

append_csv() {
  csv_old=$1
  csv_new=$2
  if [ -n "$csv_old" ]; then
    printf '%s,%s' "$csv_old" "$csv_new"
  else
    printf '%s' "$csv_new"
  fi
}

collect_images() {
  TARGET_DIGESTS=""
  PREVIOUS_DIGESTS=""
  TARGET_IMAGES=""
  SNAPSHOT_ROWS=""
  all_same=0
  for service in $SERVICES; do
    env_name=$(service_env_var "$service")
    image_ref=$(env_value "$env_name")
    [ -n "$image_ref" ] || fail_before_apply "CHECK" "$env_name missing in $ENV_FILE"
    digest_target=$(remote_digest "$image_ref") || fail_before_apply "CHECK" "cannot read remote digest for $image_ref"
    digest_current=$(current_digest "$service" "$image_ref" || true)
    [ -n "$digest_current" ] || fail_before_apply "CHECK" "cannot read current running digest for $service"
    [ "$digest_current" = "$digest_target" ] || all_same=1
    TARGET_DIGESTS=$(append_csv "$TARGET_DIGESTS" "$service=$digest_target")
    PREVIOUS_DIGESTS=$(append_csv "$PREVIOUS_DIGESTS" "$service=$digest_current")
    TARGET_IMAGES=$(append_csv "$TARGET_IMAGES" "$service=$image_ref")
    SNAPSHOT_ROWS="$SNAPSHOT_ROWS
$service|$env_name|$image_ref|$digest_current|$digest_target"
  done
  return "$all_same"
}

write_snapshot() {
  snapshot_path=$1
  snapshot_mode=$2
  snapshot_hash=$(input_hash)
  snapshot_ts=$(now_utc)
  prev_tag_field=""
  prev_digest_field=""
  target_digest_field=""
  tmp_snapshot="$snapshot_path.$$"
  for service in $SERVICES; do
    env_name=$(service_env_var "$service")
    image_ref=$(env_value "$env_name")
    prev_digest=$(printf '%s\n' "$SNAPSHOT_ROWS" | awk -F'|' -v svc="$service" '$1 == svc { print $4; exit }')
    target_digest=$(printf '%s\n' "$SNAPSHOT_ROWS" | awk -F'|' -v svc="$service" '$1 == svc { print $5; exit }')
    snapshot_prev_digest=$prev_digest
    if [ "$snapshot_mode" = "committed" ]; then
      snapshot_prev_digest=$target_digest
    fi
    prev_tag_field=$(append_csv "$prev_tag_field" "$service=$image_ref")
    prev_digest_field=$(append_csv "$prev_digest_field" "$service=$snapshot_prev_digest")
    target_digest_field=$(append_csv "$target_digest_field" "$service=$target_digest")
  done
  {
    printf '{\n'
    printf '  "prev_tag": "%s",\n' "$(json_safe "$prev_tag_field")"
    printf '  "prev_digest": "%s",\n' "$(json_safe "$prev_digest_field")"
    printf '  "target_digest": "%s",\n' "$(json_safe "$target_digest_field")"
    printf '  "env_hash": "%s",\n' "$(json_safe "$snapshot_hash")"
    printf '  "ts": "%s",\n' "$snapshot_ts"
    printf '  "mode": "%s",\n' "$(json_safe "$snapshot_mode")"
    printf '  "images": [\n'
    first=1
    for service in $SERVICES; do
      env_name=$(service_env_var "$service")
      image_ref=$(env_value "$env_name")
      prev_digest=$(printf '%s\n' "$SNAPSHOT_ROWS" | awk -F'|' -v svc="$service" '$1 == svc { print $4; exit }')
      target_digest=$(printf '%s\n' "$SNAPSHOT_ROWS" | awk -F'|' -v svc="$service" '$1 == svc { print $5; exit }')
      snapshot_prev_digest=$prev_digest
      if [ "$snapshot_mode" = "committed" ]; then
        snapshot_prev_digest=$target_digest
      fi
      if [ "$first" -eq 0 ]; then
        printf ',\n'
      fi
      first=0
      printf '    {"service":"%s","env_var":"%s","prev_tag":"%s","prev_digest":"%s","target_digest":"%s"}' \
        "$(json_safe "$service")" "$(json_safe "$env_name")" "$(json_safe "$image_ref")" "$(json_safe "$snapshot_prev_digest")" "$(json_safe "$target_digest")"
    done
    printf '\n  ]\n}\n'
  } >"$tmp_snapshot"
  mv "$tmp_snapshot" "$snapshot_path"
}

compose_pull() {
  if [ "$DRY_RUN" = "1" ]; then
    log_event "PULL" "dry_run" "would run docker compose pull"
    return 0
  fi
  # shellcheck disable=SC2086
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" pull $SERVICES
}

compose_up() {
  if [ "$DRY_RUN" = "1" ]; then
    log_event "APPLY" "dry_run" "would run docker compose up -d"
    return 0
  fi
  # shellcheck disable=SC2086
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d $SERVICES
}

set_env_value() {
  set_name=$1
  set_value=$2
  set_tmp="$ENV_FILE.$$"
  awk -v name="$set_name" -v value="$set_value" '
    BEGIN { found=0 }
    {
      line=$0
      probe=line
      sub(/^[[:space:]]*export[[:space:]]+/, "", probe)
      if (index(probe, name "=") == 1) {
        print name "=" value
        found=1
        next
      }
      print line
    }
    END { if (found == 0) print name "=" value }
  ' "$ENV_FILE" >"$set_tmp"
  if [ "$DRY_RUN" = "1" ]; then
    rm -f "$set_tmp"
  else
    mv "$set_tmp" "$ENV_FILE"
  fi
}

restore_snapshot_images() {
  for service in $SERVICES; do
    env_name=$(service_env_var "$service")
    image_ref=$(env_value "$env_name")
    repo=$(image_repo "$image_ref")
    prev_digest=$(printf '%s\n' "$SNAPSHOT_ROWS" | awk -F'|' -v svc="$service" '$1 == svc { print $4; exit }')
    [ -n "$prev_digest" ] || return 1
    set_env_value "$env_name" "$repo@$prev_digest"
  done
}

base_url() {
  if [ -n "${EDGE_UPDATER_ML_API_BASE_URL:-}" ]; then
    printf '%s' "${EDGE_UPDATER_ML_API_BASE_URL%/}"
    return 0
  fi
  port=$(env_value ML_SERVING_PORT)
  [ -n "$port" ] || port=8000
  printf 'http://127.0.0.1:%s' "$port"
}

system_body_matches() {
  body=$1
  expected=$2
  expected_version=${EDGE_UPDATER_EXPECTED_VERSION:-$(env_value ML_EDGE_VERSION || true)}
  printf '%s' "$body" | python3 -c '
import json, sys
expected, version = sys.argv[1:]
value = json.load(sys.stdin)
assert isinstance(value, dict)
assert isinstance(value.get("image_digests"), dict)
if version:
    assert value.get("version") == version
keys = {"ml-api": "ml_api", "ml-worker": "ml_worker"}
for pair in expected.split(","):
    service, digest = pair.split("=", 1)
    assert value["image_digests"][keys.get(service, service.replace("-", "_"))] == digest
' "$expected" "$expected_version" >/dev/null 2>&1
}

status_body_matches() {
  printf '%s' "$1" | python3 -c '
import json, sys
value = json.load(sys.stdin)
assert isinstance(value, dict)
assert isinstance(value.get("cameras"), dict)
assert isinstance(value.get("runtime"), dict)
assert isinstance(value.get("stale_after_sec"), (int, float))
' >/dev/null 2>&1
}

verify_once() {
  expected_digests=$1
  root_url=$(base_url)
  health_code=$(curl -fsS -o /dev/null -w '%{http_code}' "$root_url$HEALTH_PATH" 2>/dev/null || printf '000')
  [ "$health_code" = "200" ] || return 1
  status_body=$(curl -fsS "$root_url$STATUS_PATH" 2>/dev/null || true)
  status_body_matches "$status_body" || return 1
  system_body=$(curl -fsS "$root_url$SYSTEM_PATH" 2>/dev/null || true)
  system_body_matches "$system_body" "$expected_digests" || return 1
  return 0
}

verify_until_ready() {
  verify_expected=$1
  start_epoch=$(date +%s)
  end_epoch=$((start_epoch + VERIFY_TIMEOUT_SEC))
  while :; do
    if verify_once "$verify_expected"; then
      return 0
    fi
    now_epoch=$(date +%s)
    [ "$now_epoch" -lt "$end_epoch" ] || return 1
    sleep "$VERIFY_INTERVAL_SEC"
  done
}

rollback() {
  FAILURE_PHASE=${FAILURE_PHASE:-VERIFY}
  log_event "ROLLBACK" "start" "restoring previous image digests into env file"
  if restore_snapshot_images && compose_up && verify_until_ready "$PREVIOUS_DIGESTS"; then
    ROLLBACK_RESULT="success"
    report_result "rollback_success" "ROLLBACK" "$FAILURE_REASON"
    exit 1
  fi
  ROLLBACK_RESULT="failure"
  report_result "rollback_failure" "ROLLBACK" "$FAILURE_REASON"
  exit 1
}

require_file "$ENV_FILE" ".env.edge.prod"
require_file "$COMPOSE_FILE" "compose.edge.yaml"
log_event "CHECK" "start" "checking remote manifest digests against running containers"
if collect_images; then
  report_result "no_change" "CHECK" "running digests already match target manifests"
  exit 0
fi
log_event "CHECK" "update_needed" "manifest digests differ from running containers"

if [ "$DRY_RUN" = "1" ]; then
  report_result "dry_run" "CHECK" "update available; dry run stopped before snapshot/pull/apply"
  exit 0
fi

write_snapshot "$DATA_DIR/snapshot.json" "pre_apply"
log_event "SNAPSHOT" "success" "snapshot written"

if ! compose_pull; then
  fail_before_apply "PULL" "docker compose pull failed"
fi
log_event "PULL" "success" "images pulled"

if ! compose_up; then
  FAILURE_PHASE="APPLY"
  FAILURE_REASON="docker compose up -d failed"
  rollback
fi
log_event "APPLY" "success" "compose stack restarted"

if ! verify_until_ready "$TARGET_DIGESTS"; then
  FAILURE_PHASE="VERIFY"
  FAILURE_REASON="readiness, heartbeat, or version verification did not converge"
  rollback
fi
log_event "VERIFY" "success" "readiness predicates passed"
write_snapshot "$DATA_DIR/snapshot.json" "committed"
report_result "success" "COMMIT" "update committed"
