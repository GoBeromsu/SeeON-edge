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

fail() { printf '%s\n' "$1" >&2; exit 1; }
usage() {
  printf '%s\n' 'Usage: cloud-enrollment-smoke.sh --fixture --dry-run | --host happy-nursing-home-raw [--deploy --restart-check|--full-lifecycle --rollback-drill]' >&2
  exit 2
}
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
json_value() {
  python3 -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8"));
for key in sys.argv[2:]: value=value[key]
assert isinstance(value, (str,int,bool)); print(str(value).lower() if isinstance(value,bool) else value)' "$@"
}
approved_sha() { awk -F': *' '$1 == "approved_plan_sha256" { print $2; exit }' "$1"; }
gate_artifacts() {
  [ -f "$PLAN" ] && [ -f "$DRAFT" ] && [ -f "$SEAL" ] || fail 'approved plan, draft, and sealed RC receipt are required'
  expected=$(approved_sha "$DRAFT")
  case "$expected" in *[!0-9a-f]*) fail 'approved plan SHA is invalid' ;; esac
  [ "${#expected}" -eq 64 ] || fail 'approved plan SHA is invalid'
  grep -Fx 'round_status: approved' "$DRAFT" >/dev/null || fail 'draft review round is not approved'
  [ "$(sha256_file "$PLAN")" = "$expected" ] || fail 'approved plan hash mismatch'
  [ "$(json_value "$SEAL" schemaVersion)" = 1 ] || fail 'sealed RC schema mismatch'
  [ "$(json_value "$SEAL" approvedPlanSha256)" = "$expected" ] || fail 'sealed RC plan binding mismatch'
  ml_sha=$(json_value "$SEAL" ml sha)
  case "$ml_sha" in *[!0-9a-f]*) fail 'sealed ML SHA is invalid' ;; esac
  [ "${#ml_sha}" -eq 40 ] || fail 'sealed ML SHA is invalid'
  for key in apiImage workerImage; do
    image=$(json_value "$SEAL" ml "$key")
    case "$image" in *@sha256:????????????????????????????????????????????????????????????????) ;; *) fail "sealed ML $key is not digest-pinned" ;; esac
  done
}
check_host_inputs() {
  [ "$HOST" = happy-nursing-home-raw ] || fail 'only happy-nursing-home-raw is approved; JNU targets are forbidden'
  [ -f "$KNOWN_HOSTS" ] || fail 'pinned known_hosts file is required'
  ssh-keygen -F "$HOST" -f "$KNOWN_HOSTS" >/dev/null 2>&1 || fail 'approved host key is not pinned'
  [ -f "$MACHINE_BASELINE" ] || fail 'machine-id hash baseline is required'
}
check_readback() {
  [ -f "$READBACK" ] || fail 'edge API readback receipt is missing'
  [ "$(json_value "$READBACK" schemaVersion)" = 1 ] || fail 'edge readback schema mismatch'
  [ "$(json_value "$READBACK" hostAlias)" = happy-nursing-home-raw ] || fail 'edge host alias mismatch'
  case "$(json_value "$READBACK" hostname)" in *[Jj][Nn][Uu]*) fail 'JNU hostname is forbidden' ;; happy-*) ;; *) fail 'edge hostname signature mismatch' ;; esac
  [ "$(json_value "$READBACK" hostKeyPinned)" = true ] || fail 'edge host key was not strictly pinned'
  [ "$(json_value "$READBACK" machineIdSha256)" = "$(sed -n '1p' "$MACHINE_BASELINE")" ] || fail 'edge machine-id hash mismatch'
  [ "$(json_value "$READBACK" deployLockAvailable)" = true ] || fail 'edge deploy lock is held'
  [ "$(json_value "$READBACK" updaterIdle)" = true ] || fail 'edge updater is concurrent'
  [ "$(json_value "$READBACK" clipStoreFreeMiB)" -ge "$MIN_EDGE_CLIP_MIB" ] || fail 'edge clip-store free capacity is below 20 GiB'
  [ "$(json_value "$READBACK" volumesHealthy)" = true ] || fail 'edge volumes are unhealthy'
  [ "$(json_value "$READBACK" queuesDrained)" = true ] || fail 'edge queues are not drained'
  [ "$(json_value "$READBACK" sqliteBackupVerified)" = true ] || fail 'edge SQLite backup is missing or unverifiable'
  [ "$(json_value "$READBACK" apiImage)" = "$(json_value "$SEAL" ml apiImage)" ] || fail 'edge API digest mismatch'
  [ "$(json_value "$READBACK" workerImage)" = "$(json_value "$SEAL" ml workerImage)" ] || fail 'edge worker digest mismatch'
  [ "$(json_value "$READBACK" envFacilityIdentityAbsent)" = true ] || fail 'facility identity or token remains in deployment env'
  [ "$(json_value "$READBACK" apiReady)" = true ] || fail 'edge versioned API is unhealthy'
  [ "$(json_value "$READBACK" schemaIntegrity)" = true ] || fail 'edge SQLite schema/integrity gate failed'
}
fixture() {
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/cloud-enrollment-smoke.XXXXXX")
  trap 'rm -rf "$tmp"' EXIT HUP INT TERM
  printf 'fixture approved plan\n' > "$tmp/plan.md"
  digest=$(sha256_file "$tmp/plan.md")
  printf 'approved_plan_sha256: %s\nround_status: approved\n' "$digest" > "$tmp/draft.md"
  cat > "$tmp/seal.json" <<EOF
{"schemaVersion":1,"approvedPlanSha256":"$digest","ai":{"sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","backendImage":"local/backend@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","frontImage":"local/front@sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},"ml":{"sha":"dddddddddddddddddddddddddddddddddddddddd","apiImage":"local/api@sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","workerImage":"local/worker@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}}
EOF
  machine=$(printf fixture-machine | sha256_file /dev/stdin)
  printf '%s\n' "$machine" > "$tmp/machine.sha256"
  printf 'happy-nursing-home-raw ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFixturePinnedHostKeyOnly\n' > "$tmp/known_hosts"
  cat > "$tmp/readback.json" <<EOF
{"schemaVersion":1,"hostAlias":"happy-nursing-home-raw","hostname":"happy-edge-fixture","hostKeyPinned":true,"machineIdSha256":"$machine","deployLockAvailable":true,"updaterIdle":true,"clipStoreFreeMiB":20480,"volumesHealthy":true,"queuesDrained":true,"sqliteBackupVerified":true,"apiImage":"local/api@sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","workerImage":"local/worker@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","envFacilityIdentityAbsent":true,"apiReady":true,"schemaIntegrity":true}
EOF
  PLAN=$tmp/plan.md DRAFT=$tmp/draft.md SEAL=$tmp/seal.json READBACK=$tmp/readback.json KNOWN_HOSTS=$tmp/known_hosts MACHINE_BASELINE=$tmp/machine.sha256 HOST=happy-nursing-home-raw
  gate_artifacts
  check_host_inputs
  check_readback
  for field in hostAlias hostname hostKeyPinned machineIdSha256 deployLockAvailable updaterIdle clipStoreFreeMiB volumesHealthy queuesDrained sqliteBackupVerified apiImage envFacilityIdentityAbsent apiReady schemaIntegrity; do
    python3 - "$tmp/readback.json" "$tmp/bad.json" "$field" <<'PY'
import json, sys
source, target, field = sys.argv[1:]
value = json.load(open(source, encoding="utf-8"))
replacements = {"hostAlias":"jnu-oss","hostname":"jnu-edge","hostKeyPinned":False,"machineIdSha256":"0"*64,"deployLockAvailable":False,"updaterIdle":False,"clipStoreFreeMiB":20479,"volumesHealthy":False,"queuesDrained":False,"sqliteBackupVerified":False,"apiImage":"mutable:latest","envFacilityIdentityAbsent":False,"apiReady":False,"schemaIntegrity":False}
value[field] = replacements[field]
json.dump(value, open(target, "w", encoding="utf-8"))
PY
    READBACK=$tmp/bad.json
    if (check_readback) >/dev/null 2>&1; then fail "$field rejection fixture passed"; fi
    READBACK=$tmp/readback.json
  done
  HOST=jnu-oss
  if (check_host_inputs) >/dev/null 2>&1; then fail 'JNU host fixture passed'; fi
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
fi
if [ "$FULL_LIFECYCLE" = true ]; then [ "$ROLLBACK_DRILL" = true ] || usage; fi
printf 'CLOUD_ENROLLMENT_SMOKE_OK ml_sha=%s mode=api-first\n' "$(json_value "$SEAL" ml sha)"
