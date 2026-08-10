#!/usr/bin/env sh
set -eu

SCRIPT=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)/cloud-enrollment-smoke.sh
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/cloud-enrollment-smoke-test.XXXXXX")
trap 'rm -rf "$TMP_ROOT"' EXIT HUP INT TERM

printf 'approved fixture plan\n' >"$TMP_ROOT/plan.md"
PLAN_DIGEST=$(shasum -a 256 "$TMP_ROOT/plan.md" | awk '{print $1}')
printf 'approved_plan_sha256: %s\nround_status: approved\n' "$PLAN_DIGEST" >"$TMP_ROOT/draft.md"
cat >"$TMP_ROOT/seal.json" <<EOF
{"schemaVersion":1,"approvedPlanSha256":"$PLAN_DIGEST","ai":{"sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","backendImage":"local/backend@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","frontImage":"local/front@sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},"ml":{"sha":"dddddddddddddddddddddddddddddddddddddddd","apiImage":"local/api@sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","workerImage":"local/worker@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}}
EOF
printf fixture-machine | shasum -a 256 | awk '{print $1}' >"$TMP_ROOT/machine.sha256"
ssh-keygen -q -t ed25519 -N '' -f "$TMP_ROOT/attacker-key"
printf 'happy-nursing-home-raw %s\n' "$(cat "$TMP_ROOT/attacker-key.pub")" >"$TMP_ROOT/known_hosts"
MACHINE_DIGEST=$(cat "$TMP_ROOT/machine.sha256")
cat >"$TMP_ROOT/spoofed-readback.json" <<EOF
{"schemaVersion":1,"hostAlias":"happy-nursing-home-raw","hostname":"happy-edge-fixture","hostKeyPinned":true,"machineIdSha256":"$MACHINE_DIGEST","deployLockAvailable":true,"updaterIdle":true,"clipStoreFreeMiB":999999,"volumesHealthy":true,"queuesDrained":true,"sqliteBackupVerified":true,"apiImage":"local/api@sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","workerImage":"local/worker@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","envFacilityIdentityAbsent":true,"apiReady":true,"schemaIntegrity":true,"fullLifecycleVerified":true,"rollbackDrillVerified":true}
EOF

if EDGE_PROVISIONING_PLAN="$TMP_ROOT/plan.md" \
  EDGE_PROVISIONING_DRAFT="$TMP_ROOT/draft.md" \
  EDGE_PROVISIONING_SEAL="$TMP_ROOT/seal.json" \
  EDGE_PROVISIONING_EDGE_READBACK="$TMP_ROOT/spoofed-readback.json" \
  EDGE_PROVISIONING_KNOWN_HOSTS="$TMP_ROOT/known_hosts" \
  EDGE_PROVISIONING_MACHINE_BASELINE="$TMP_ROOT/machine.sha256" \
  sh "$SCRIPT" --host happy-nursing-home-raw >/dev/null 2>&1; then
  printf '%s\n' 'attacker known_hosts and spoofed edge receipt were accepted' >&2
  exit 1
fi

printf '%s\n' 'CLOUD_ENROLLMENT_SPOOF_REJECTION_OK'
