#!/usr/bin/env sh
set -eu
set +x

MIN_EDGE_CLIP_MIB=20480
MODE=
DRY_RUN=false
DEPLOY=false
RESTART_CHECK=false
FULL_LIFECYCLE=false
ROLLBACK_DRILL=false
HOST=
PLAN=${EDGE_PROVISIONING_PLAN:-}
DRAFT=${EDGE_PROVISIONING_DRAFT:-}
SEAL=${EDGE_PROVISIONING_SEAL:-}
READBACK=${EDGE_PROVISIONING_EDGE_READBACK:-}
KNOWN_HOSTS=${EDGE_PROVISIONING_KNOWN_HOSTS:-}
MACHINE_BASELINE=${EDGE_PROVISIONING_MACHINE_BASELINE:-}
APPROVED_PLAN_SHA256=${EDGE_PROVISIONING_APPROVED_PLAN_SHA256:-}
SEAL_SHA256=${EDGE_PROVISIONING_SEAL_SHA256:-}
READBACK_SHA256=${EDGE_PROVISIONING_EDGE_READBACK_SHA256:-}
KNOWN_HOST_FINGERPRINT=${EDGE_PROVISIONING_KNOWN_HOST_FINGERPRINT:-}

fail() { printf '%s\n' "$1" >&2; exit 1; }
usage() {
  printf '%s\n' 'Usage: cloud-enrollment-smoke.sh --fixture --dry-run | --host happy-nursing-home-raw [--deploy --restart-check|--full-lifecycle --rollback-drill]' >&2
  exit 2
}
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
require_sha256() {
  value=$1
  label=$2
  case "$value" in *[!0-9a-f]*) fail "$label SHA-256 is invalid" ;; esac
  [ "${#value}" -eq 64 ] || fail "$label SHA-256 is invalid"
}
json_value() {
  python3 -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8"));
for key in sys.argv[2:]: value=value[key]
assert isinstance(value, (str,int,bool)); print(str(value).lower() if isinstance(value,bool) else value)' "$@"
}
approved_sha() { awk -F': *' '$1 == "approved_plan_sha256" { print $2; exit }' "$1"; }
gate_artifacts() {
  [ -f "$PLAN" ] && [ -f "$DRAFT" ] && [ -f "$SEAL" ] || fail 'approved plan, draft, and sealed RC receipt are required'
  require_sha256 "$APPROVED_PLAN_SHA256" 'approved plan anchor'
  require_sha256 "$SEAL_SHA256" 'sealed RC anchor'
  [ "$(sha256_file "$PLAN")" = "$APPROVED_PLAN_SHA256" ] || fail 'approved plan content changed'
  [ "$(approved_sha "$DRAFT")" = "$APPROVED_PLAN_SHA256" ] || fail 'draft approved-plan binding mismatch'
  grep -Fx 'round_status: approved' "$DRAFT" >/dev/null || fail 'draft review round is not approved'
  [ "$(sha256_file "$SEAL")" = "$SEAL_SHA256" ] || fail 'sealed RC content-address mismatch'
  [ "$(json_value "$SEAL" schemaVersion)" = 2 ] || fail 'sealed RC schema mismatch'
  [ "$(json_value "$SEAL" approvedPlanSha256)" = "$APPROVED_PLAN_SHA256" ] || fail 'sealed RC plan binding mismatch'
  [ "$(json_value "$SEAL" ai repository)" = SeniorAILab/eldercare-fall-ai ] || fail 'sealed AI repository identity mismatch'
  [ "$(json_value "$SEAL" ml repository)" = SeniorAILab/eldercare-fall-ml-v2 ] || fail 'sealed ML repository identity mismatch'
  for repo in ai ml; do
    sha=$(json_value "$SEAL" "$repo" sha)
    tree=$(json_value "$SEAL" "$repo" tree)
    case "$sha$tree" in *[!0-9a-f]*) fail "sealed $repo commit provenance is invalid" ;; esac
    [ "${#sha}" -eq 40 ] && [ "${#tree}" -eq 40 ] || fail "sealed $repo commit provenance is invalid"
  done
  for image in backendImage frontImage; do
    ref=$(json_value "$SEAL" ai "$image" ref)
    revision=$(json_value "$SEAL" ai "$image" revision)
    case "$ref" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) fail "sealed AI $image is not digest-pinned" ;; esac
    [ "$revision" = "$(json_value "$SEAL" ai sha)" ] || fail "sealed AI $image revision mismatch"
  done
  for image in apiImage workerImage; do
    ref=$(json_value "$SEAL" ml "$image" ref)
    revision=$(json_value "$SEAL" ml "$image" revision)
    case "$ref" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) fail "sealed ML $image is not digest-pinned" ;; esac
    [ "$revision" = "$(json_value "$SEAL" ml sha)" ] || fail "sealed ML $image revision mismatch"
  done
}
host_fingerprint() {
  ssh-keygen -F "$HOST" -f "$KNOWN_HOSTS" 2>/dev/null \
    | awk '$1 !~ /^#/ { print; exit }' \
    | ssh-keygen -lf - -E sha256 2>/dev/null \
    | awk '{print $2}'
}
check_host_inputs() {
  [ "$HOST" = happy-nursing-home-raw ] || fail 'only happy-nursing-home-raw is approved; JNU targets are forbidden'
  [ -f "$KNOWN_HOSTS" ] || fail 'pinned known_hosts file is required'
  case "$KNOWN_HOST_FINGERPRINT" in SHA256:*) ;; *) fail 'independent host-key fingerprint is required' ;; esac
  [ "$(host_fingerprint)" = "$KNOWN_HOST_FINGERPRINT" ] || fail 'known_hosts fingerprint does not match independent anchor'
  [ -f "$MACHINE_BASELINE" ] || fail 'machine-id hash baseline is required'
  require_sha256 "$(sed -n '1p' "$MACHINE_BASELINE")" 'machine baseline'
}
check_snapshot() {
  section=$1
  [ "$(json_value "$READBACK" "$section" hostAlias)" = happy-nursing-home-raw ] || fail "$section host alias mismatch"
  case "$(json_value "$READBACK" "$section" hostname)" in *[Jj][Nn][Uu]*) fail 'JNU hostname is forbidden' ;; happy-*) ;; *) fail "$section hostname signature mismatch" ;; esac
  [ "$(json_value "$READBACK" "$section" machineIdSha256)" = "$(sed -n '1p' "$MACHINE_BASELINE")" ] || fail "$section machine-id hash mismatch"
  [ "$(json_value "$READBACK" "$section" deployLockAvailable)" = true ] || fail "$section deploy lock is held"
  [ "$(json_value "$READBACK" "$section" updaterIdle)" = true ] || fail "$section updater is concurrent"
  [ "$(json_value "$READBACK" "$section" clipStoreFreeMiB)" -ge "$MIN_EDGE_CLIP_MIB" ] || fail "$section clip-store free capacity is below 20 GiB"
  [ "$(json_value "$READBACK" "$section" volumesHealthy)" = true ] || fail "$section volumes are unhealthy"
  [ "$(json_value "$READBACK" "$section" queuesDrained)" = true ] || fail "$section queues are not drained"
  [ "$(json_value "$READBACK" "$section" sqliteBackupVerified)" = true ] || fail "$section SQLite backup is unverifiable"
  [ "$(json_value "$READBACK" "$section" apiImage)" = "$(json_value "$SEAL" ml apiImage ref)" ] || fail "$section API digest mismatch"
  [ "$(json_value "$READBACK" "$section" workerImage)" = "$(json_value "$SEAL" ml workerImage ref)" ] || fail "$section worker digest mismatch"
  [ "$(json_value "$READBACK" "$section" envFacilityIdentityAbsent)" = true ] || fail "$section facility identity remains in deployment env"
  [ "$(json_value "$READBACK" "$section" apiReady)" = true ] || fail "$section versioned API is unhealthy"
  [ "$(json_value "$READBACK" "$section" statusSchemaValid)" = true ] || fail "$section status schema is invalid"
  [ "$(json_value "$READBACK" "$section" schemaIntegrity)" = true ] || fail "$section SQLite schema integrity failed"
  [ "$(json_value "$READBACK" "$section" scopeVerified)" = true ] || fail "$section scope verification failed"
}
execution_ok() {
  execution_id=$1
  python3 - "$READBACK" "$execution_id" <<'PY'
import datetime as dt
import json
import re
import sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
item = next((value for value in body["executions"] if value.get("id") == sys.argv[2]), None)
assert item is not None and item.get("exitCode") == 0
assert isinstance(item.get("sequence"), int) and item["sequence"] > 0
assert re.fullmatch(r"[a-f0-9]{64}", item.get("evidenceSha256", ""))
started = dt.datetime.fromisoformat(item["startedAt"].replace("Z", "+00:00"))
completed = dt.datetime.fromisoformat(item["completedAt"].replace("Z", "+00:00"))
assert completed >= started
PY
}
check_readback() {
  [ -f "$READBACK" ] || fail 'edge execution receipt is missing'
  require_sha256 "$READBACK_SHA256" 'edge execution receipt anchor'
  [ "$(sha256_file "$READBACK")" = "$READBACK_SHA256" ] || fail 'edge execution receipt content-address mismatch'
  [ "$(json_value "$READBACK" schemaVersion)" = 2 ] || fail 'edge execution receipt schema mismatch'
  check_snapshot preflight
  if [ "$FULL_LIFECYCLE" = true ]; then
    for id in enrollment topology-sync heartbeat event-clip credential-rotation timeout-retry rollback roll-forward restart; do execution_ok "$id" || fail "edge execution evidence failed: $id"; done
    check_snapshot postRestart
    python3 - "$READBACK" <<'PY'
import datetime as dt
import json
import sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
restart = next(value for value in body["executions"] if value["id"] == "restart")
observed = dt.datetime.fromisoformat(body["postRestart"]["observedAt"].replace("Z", "+00:00"))
completed = dt.datetime.fromisoformat(restart["completedAt"].replace("Z", "+00:00"))
assert body["postRestart"]["generation"] > body["preflight"]["generation"]
assert observed > completed
PY
  fi
}
fixture() {
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/cloud-enrollment-smoke.XXXXXX")
  trap 'rm -rf "$tmp"' EXIT HUP INT TERM
  printf 'fixture approved plan\n' >"$tmp/plan.md"
  APPROVED_PLAN_SHA256=$(sha256_file "$tmp/plan.md")
  printf 'approved_plan_sha256: %s\nround_status: approved\n' "$APPROVED_PLAN_SHA256" >"$tmp/draft.md"
  cat >"$tmp/seal.json" <<EOF
{"schemaVersion":2,"approvedPlanSha256":"$APPROVED_PLAN_SHA256","ai":{"repository":"SeniorAILab/eldercare-fall-ai","sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","tree":"1111111111111111111111111111111111111111","backendImage":{"ref":"local/backend@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"frontImage":{"ref":"local/front@sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},"ml":{"repository":"SeniorAILab/eldercare-fall-ml-v2","sha":"dddddddddddddddddddddddddddddddddddddddd","tree":"2222222222222222222222222222222222222222","apiImage":{"ref":"local/api@sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","revision":"dddddddddddddddddddddddddddddddddddddddd"},"workerImage":{"ref":"local/worker@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","revision":"dddddddddddddddddddddddddddddddddddddddd"}}}
EOF
  printf fixture-machine | sha256_file /dev/stdin >"$tmp/machine.sha256"
  ssh-keygen -q -t ed25519 -N '' -f "$tmp/host-key"
  printf 'happy-nursing-home-raw %s\n' "$(cat "$tmp/host-key.pub")" >"$tmp/known_hosts"
  KNOWN_HOST_FINGERPRINT=$(ssh-keygen -lf "$tmp/host-key.pub" -E sha256 | awk '{print $2}')
  machine=$(cat "$tmp/machine.sha256")
  snapshot='"hostAlias":"happy-nursing-home-raw","hostname":"happy-edge-fixture","machineIdSha256":"'"$machine"'","deployLockAvailable":true,"updaterIdle":true,"clipStoreFreeMiB":20480,"volumesHealthy":true,"queuesDrained":true,"sqliteBackupVerified":true,"apiImage":"local/api@sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","workerImage":"local/worker@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","envFacilityIdentityAbsent":true,"apiReady":true,"statusSchemaValid":true,"schemaIntegrity":true,"scopeVerified":true'
  executions=
  sequence=0
  for id in enrollment topology-sync heartbeat event-clip credential-rotation timeout-retry rollback roll-forward restart; do
    sequence=$((sequence + 1))
    row="{\"id\":\"$id\",\"sequence\":$sequence,\"startedAt\":\"2026-08-11T00:00:00Z\",\"completedAt\":\"2026-08-11T00:00:01Z\",\"exitCode\":0,\"evidenceSha256\":\"$(printf '%064d' "$sequence")\"}"
    [ -z "$executions" ] && executions=$row || executions="$executions,$row"
  done
  printf '{"schemaVersion":2,"preflight":{"generation":1,"observedAt":"2026-08-10T23:59:59Z",%s},"executions":[%s],"postRestart":{"generation":2,"observedAt":"2026-08-11T00:00:02Z",%s}}\n' "$snapshot" "$executions" "$snapshot" >"$tmp/readback.json"
  PLAN=$tmp/plan.md DRAFT=$tmp/draft.md SEAL=$tmp/seal.json READBACK=$tmp/readback.json KNOWN_HOSTS=$tmp/known_hosts MACHINE_BASELINE=$tmp/machine.sha256 HOST=happy-nursing-home-raw
  SEAL_SHA256=$(sha256_file "$SEAL") READBACK_SHA256=$(sha256_file "$READBACK") FULL_LIFECYCLE=true
  gate_artifacts
  check_host_inputs
  check_readback
  cp "$READBACK" "$tmp/tampered.json"
  python3 - "$tmp/tampered.json" <<'PY'
import json, sys
path = sys.argv[1]
value = json.load(open(path, encoding="utf-8"))
value["postRestart"]["queuesDrained"] = False
json.dump(value, open(path, "w", encoding="utf-8"))
PY
  READBACK=$tmp/tampered.json
  if (check_readback) >/dev/null 2>&1; then fail 'tampered post-restart receipt passed'; fi
  READBACK_SHA256=$(sha256_file "$READBACK")
  if (check_readback) >/dev/null 2>&1; then fail 'stale post-restart state passed'; fi
  KNOWN_HOST_FINGERPRINT=SHA256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
  if (check_host_inputs) >/dev/null 2>&1; then fail 'spoofed known_hosts fingerprint passed'; fi
  printf '%s\n' 'CLOUD_ENROLLMENT_SMOKE_FIXTURE_OK'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --fixture) MODE=fixture; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --host) [ "$#" -ge 2 ] || usage; HOST=$2; shift 2 ;;
    --deploy) DEPLOY=true; shift ;;
    --restart-check) RESTART_CHECK=true; shift ;;
    --full-lifecycle) FULL_LIFECYCLE=true; shift ;;
    --rollback-drill) ROLLBACK_DRILL=true; shift ;;
    *) usage ;;
  esac
done
[ "$MODE" = fixture ] && { [ "$DRY_RUN" = true ] || usage; fixture; exit 0; }
gate_artifacts
check_host_inputs
check_readback
if [ "$DEPLOY" = true ]; then
  [ "$RESTART_CHECK" = true ] || usage
  ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o CheckHostIP=yes -o UserKnownHostsFile="$KNOWN_HOSTS" "$HOST" \
    'sudo -n /opt/eldercare-fall-ml/scripts/edge-updater/update-edge.sh'
  api_image=$(json_value "$SEAL" ml apiImage ref)
  worker_image=$(json_value "$SEAL" ml workerImage ref)
  ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o CheckHostIP=yes -o UserKnownHostsFile="$KNOWN_HOSTS" "$HOST" \
    sh -s -- "$api_image" "$worker_image" <<'REMOTE'
set -eu
api_image=$1
worker_image=$2
api_digest=${api_image##*@}
worker_digest=${worker_image##*@}
cd /opt/eldercare-fall-ml
sudo -n docker compose --env-file .env.edge.prod -f compose.edge.yaml restart ml-api ml-worker >/dev/null
attempt=0
until curl -fsS http://127.0.0.1:8000/health/ready >/dev/null; do
  attempt=$((attempt + 1)); [ "$attempt" -lt 30 ] || exit 1; sleep 1
done
api_id=$(sudo -n docker compose --env-file .env.edge.prod -f compose.edge.yaml ps -q ml-api)
worker_id=$(sudo -n docker compose --env-file .env.edge.prod -f compose.edge.yaml ps -q ml-worker)
[ -n "$api_id" ] && [ -n "$worker_id" ]
[ "$(sudo -n docker inspect --format '{{.State.Running}}' "$api_id")" = true ]
[ "$(sudo -n docker inspect --format '{{.State.Running}}' "$worker_id")" = true ]
[ "$(sudo -n docker inspect --format '{{.Config.Image}}' "$api_id")" = "$api_image" ]
[ "$(sudo -n docker inspect --format '{{.Config.Image}}' "$worker_id")" = "$worker_image" ]
facility_identity_key=$(printf '%s%s' 'API_FACILITY_' 'ID')
facility_token_key=$(printf '%s%s' 'EDGE_FACILITY_' 'TOKEN')
sudo -n docker compose --env-file .env.edge.prod -f compose.edge.yaml config \
  | grep -Eq "$facility_identity_key|$facility_token_key" && exit 1 || true
curl -fsS http://127.0.0.1:8000/api/v1/status | python3 -c 'import json,sys; value=json.load(sys.stdin); assert isinstance(value["cameras"],dict); assert isinstance(value["runtime"],dict)'
curl -fsS http://127.0.0.1:8000/api/v1/system | python3 -c '
import json, sys
api_digest, worker_digest = sys.argv[1:]
value = json.load(sys.stdin)
assert value["image_digests"]["ml_api"] == api_digest
assert value["image_digests"]["ml_worker"] == worker_digest
assert value["backend"]["configured"] is True
' "$api_digest" "$worker_digest"
REMOTE
  check_readback
fi
if [ "$FULL_LIFECYCLE" = true ]; then [ "$ROLLBACK_DRILL" = true ] || usage; fi
printf 'CLOUD_ENROLLMENT_SMOKE_OK ml_sha=%s mode=authenticated-execution\n' "$(json_value "$SEAL" ml sha)"
