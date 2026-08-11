#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
UPDATER=$ROOT_DIR/scripts/edge-updater/update-edge.sh
TMP_ROOT=${TMPDIR:-/tmp}/edge-updater-test.$$
MOCK_BIN=$TMP_ROOT/bin
STATE_DIR=$TMP_ROOT/state

API_OLD=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
API_NEW=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
WORKER_OLD=sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
WORKER_NEW=sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

mkdir -p "$MOCK_BIN" "$STATE_DIR"

cat >"$MOCK_BIN/docker" <<'MOCK_DOCKER'
#!/bin/sh
set -eu

last_arg=
for arg in "$@"; do
  last_arg=$arg
done

if [ "${1:-}" = "buildx" ] && [ "${2:-}" = "imagetools" ] && [ "${3:-}" = "inspect" ]; then
  case "$4" in
    *ml-api*) digest=$API_NEW ;;
    *ml-worker*) digest=$WORKER_NEW ;;
    *) exit 1 ;;
  esac
  printf 'Name: %s\nDigest: %s\n' "$4" "$digest"
  exit 0
fi

if [ "${1:-}" = "image" ] && [ "${2:-}" = "inspect" ]; then
  case "$last_arg" in
    *ml-api*) printf 'ghcr.io/acme/ml-api@%s\n' "$API_OLD" ;;
    *ml-worker*) printf 'ghcr.io/acme/ml-worker@%s\n' "$WORKER_OLD" ;;
    *) exit 1 ;;
  esac
  exit 0
fi

if [ "${1:-}" = "inspect" ]; then
  case "$last_arg" in
    cid-ml-api) printf 'image-ml-api\n' ;;
    cid-ml-worker) printf 'image-ml-worker\n' ;;
    *) exit 1 ;;
  esac
  exit 0
fi

if [ "${1:-}" = "compose" ]; then
  cmd=
  for arg in "$@"; do
    case "$arg" in
      ps|pull|up) cmd=$arg; break ;;
    esac
  done
  case "$cmd" in
    ps)
      printf 'cid-%s\n' "$last_arg"
      exit 0
      ;;
    pull)
      printf 'pull %s\n' "$*" >>"$TEST_STATE/docker.log"
      exit 0
      ;;
    up)
      count=0
      if [ -f "$TEST_STATE/up_count" ]; then
        count=$(cat "$TEST_STATE/up_count")
      fi
      count=$((count + 1))
      printf '%s\n' "$count" >"$TEST_STATE/up_count"
      printf 'up %s\n' "$*" >>"$TEST_STATE/docker.log"
      exit 0
      ;;
  esac
fi

printf 'unexpected docker invocation: %s\n' "$*" >&2
exit 1
MOCK_DOCKER
chmod +x "$MOCK_BIN/docker"

cat >"$MOCK_BIN/curl" <<'MOCK_CURL'
#!/bin/sh
set -eu

is_post=0
last_arg=
for arg in "$@"; do
  if [ "$arg" = "POST" ]; then
    is_post=1
  fi
  last_arg=$arg
done

if [ "$is_post" -eq 1 ]; then
  printf 'report %s\n' "$*" >>"$TEST_STATE/report.log"
  exit 0
fi

case "$last_arg" in
  */health/ready)
    printf '200'
    exit 0
    ;;
  */api/v1/status)
    up_count=0
    if [ -f "$TEST_STATE/up_count" ]; then
      up_count=$(cat "$TEST_STATE/up_count")
    fi
    if [ "${TEST_SCENARIO:-success}" = "bad_status" ] && [ "$up_count" -lt 2 ]; then
      printf '{"cameras":{"cam-edge-01":{"status":"online"}}}'
    else
      printf '{"cameras":{"cam-edge-01":{"status":"online"}},"stale_after_sec":90.0,"runtime":{}}'
    fi
    exit 0
    ;;
  */api/v1/system)
    up_count=0
    if [ -f "$TEST_STATE/up_count" ]; then
      up_count=$(cat "$TEST_STATE/up_count")
    fi
    if [ "${TEST_SCENARIO:-success}" = "rollback" ] && [ "$up_count" -lt 2 ]; then
      printf '{"version":"test-version","image_digests":{"ml_api":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","ml_worker":"sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},"backend":{"configured":false,"reachable":null,"last_ok_at":null},"storage":{"clip_store":{"total_bytes":null,"used_bytes":null,"used_pct":null}},"updated_at":"2026-07-07T00:00:00Z"}'
    elif [ "${TEST_SCENARIO:-success}" = "rollback" ] || [ "${TEST_SCENARIO:-success}" = "bad_status" ]; then
      printf '{"version":"test-version","image_digests":{"ml_api":"%s","ml_worker":"%s"},"backend":{"configured":false,"reachable":null,"last_ok_at":null},"storage":{"clip_store":{"total_bytes":null,"used_bytes":null,"used_pct":null}},"updated_at":"2026-07-07T00:00:00Z"}' "$API_OLD" "$WORKER_OLD"
    else
      printf '{"version":"test-version","image_digests":{"ml_api":"%s","ml_worker":"%s"},"backend":{"configured":false,"reachable":null,"last_ok_at":null},"storage":{"clip_store":{"total_bytes":null,"used_bytes":null,"used_pct":null}},"updated_at":"2026-07-07T00:00:00Z"}' "$API_NEW" "$WORKER_NEW"
    fi
    exit 0
    ;;
esac

printf 'unexpected curl invocation: %s\n' "$*" >&2
exit 1
MOCK_CURL
chmod +x "$MOCK_BIN/curl"

assert_contains() {
  file=$1
  needle=$2
  label=$3
  if ! grep -F "$needle" "$file" >/dev/null 2>&1; then
    printf 'missing %s in %s\n' "$label" "$file" >&2
    printf 'wanted: %s\n' "$needle" >&2
    exit 1
  fi
}

write_case_files() {
  case_dir=$1
  mkdir -p "$case_dir"
  cat >"$case_dir/.env.edge.prod" <<EOF_ENV
ML_SERVING_PORT=18080
ML_API_IMAGE=ghcr.io/acme/ml-api:new
ML_WORKER_IMAGE=ghcr.io/acme/ml-worker:new
API_EDGE_RELAY_TOKEN=not-secret-for-test
ML_EDGE_VERSION=test-version
EOF_ENV
  cat >"$case_dir/compose.edge.yaml" <<'EOF_COMPOSE'
services:
  ml-api:
    image: ${ML_API_IMAGE}
  ml-worker:
    image: ${ML_WORKER_IMAGE}
EOF_COMPOSE
}

run_success_case() {
  case_dir=$TMP_ROOT/success
  data_dir=$case_dir/data
  write_case_files "$case_dir"
  mkdir -p "$data_dir"
  : >"$STATE_DIR/docker.log"
  rm -f "$STATE_DIR/up_count" "$STATE_DIR/report.log"
  TEST_STATE=$STATE_DIR \
  TEST_SCENARIO=success \
  API_OLD=$API_OLD API_NEW=$API_NEW WORKER_OLD=$WORKER_OLD WORKER_NEW=$WORKER_NEW \
  PATH="$MOCK_BIN:$PATH" \
  EDGE_UPDATER_DATA_DIR=$data_dir \
  EDGE_UPDATER_ENV_FILE=$case_dir/.env.edge.prod \
  EDGE_UPDATER_COMPOSE_FILE=$case_dir/compose.edge.yaml \
  EDGE_UPDATER_REPORT_URL=http://backend.example.test/api/v1/edge-updater/report \
  EDGE_UPDATER_VERIFY_TIMEOUT_SEC=0 \
  sh "$UPDATER" >/dev/null
  assert_contains "$data_dir/update.log" '"status":"success"' 'success report'
  assert_contains "$data_dir/snapshot.json" "$API_NEW" 'committed api digest'
  assert_contains "$data_dir/snapshot.json" "$WORKER_NEW" 'committed worker digest'
  assert_contains "$STATE_DIR/docker.log" 'pull compose' 'compose pull'
  assert_contains "$STATE_DIR/docker.log" 'up compose' 'compose up'
  assert_contains "$STATE_DIR/report.log" 'report' 'backend report'
}

run_rollback_case() {
  case_dir=$TMP_ROOT/rollback
  data_dir=$case_dir/data
  write_case_files "$case_dir"
  mkdir -p "$data_dir"
  : >"$STATE_DIR/docker.log"
  rm -f "$STATE_DIR/up_count" "$STATE_DIR/report.log"
  set +e
  TEST_STATE=$STATE_DIR \
  TEST_SCENARIO=rollback \
  API_OLD=$API_OLD API_NEW=$API_NEW WORKER_OLD=$WORKER_OLD WORKER_NEW=$WORKER_NEW \
  PATH="$MOCK_BIN:$PATH" \
  EDGE_UPDATER_DATA_DIR=$data_dir \
  EDGE_UPDATER_ENV_FILE=$case_dir/.env.edge.prod \
  EDGE_UPDATER_COMPOSE_FILE=$case_dir/compose.edge.yaml \
  EDGE_UPDATER_REPORT_URL=http://backend.example.test/api/v1/edge-updater/report \
  EDGE_UPDATER_VERIFY_TIMEOUT_SEC=0 \
  sh "$UPDATER" >/dev/null
  code=$?
  set -e
  if [ "$code" -eq 0 ]; then
    printf 'rollback case unexpectedly exited 0\n' >&2
    exit 1
  fi
  assert_contains "$data_dir/update.log" '"status":"rollback_success"' 'rollback success report'
  assert_contains "$case_dir/.env.edge.prod" "ML_API_IMAGE=ghcr.io/acme/ml-api@$API_OLD" 'api rollback digest'
  assert_contains "$case_dir/.env.edge.prod" "ML_WORKER_IMAGE=ghcr.io/acme/ml-worker@$WORKER_OLD" 'worker rollback digest'
}

run_status_schema_case() {
  case_dir=$TMP_ROOT/bad-status
  data_dir=$case_dir/data
  write_case_files "$case_dir"
  mkdir -p "$data_dir"
  : >"$STATE_DIR/docker.log"
  rm -f "$STATE_DIR/up_count" "$STATE_DIR/report.log"
  set +e
  TEST_STATE=$STATE_DIR \
  TEST_SCENARIO=bad_status \
  API_OLD=$API_OLD API_NEW=$API_NEW WORKER_OLD=$WORKER_OLD WORKER_NEW=$WORKER_NEW \
  PATH="$MOCK_BIN:$PATH" \
  EDGE_UPDATER_DATA_DIR=$data_dir \
  EDGE_UPDATER_ENV_FILE=$case_dir/.env.edge.prod \
  EDGE_UPDATER_COMPOSE_FILE=$case_dir/compose.edge.yaml \
  EDGE_UPDATER_VERIFY_TIMEOUT_SEC=0 \
  sh "$UPDATER" >/dev/null
  code=$?
  set -e
  if [ "$code" -eq 0 ]; then
    printf 'invalid status schema unexpectedly exited 0\n' >&2
    exit 1
  fi
  assert_contains "$data_dir/update.log" '"status":"rollback_success"' 'status-schema rollback report'
}

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck "$UPDATER" "$0"
fi

run_success_case
run_rollback_case
run_status_schema_case
printf 'edge-updater tests passed\n'
