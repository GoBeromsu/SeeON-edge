# Publish sealed edge images

This procedure publishes the ML release candidate only after the backward-
compatible AI release is healthy. It does not enroll a facility, edit topology,
or infer identity from environment variables.

## Gate

Before a workflow or updater command:

1. Rehash the approved plan and compare it with the dual-review draft.
2. Verify the independently recorded SHA-256 of `final-rc-seal.json`, then require
   its exact repository identity, commit tree, ML Git SHA, image IDs, platforms,
   and digest-pinned ML API/worker images with matching OCI provenance labels.
3. Confirm the task worktree is clean and at the sealed SHA.
4. Run all ML gates, both Docker builds, all five operational fixtures, review,
   security review, and the integration harness against that exact SHA.
5. If source changes, discard the seal and repeat the complete process.

The workflow artifact name is `edge-ml-image-refs-<sealed-sha>`. Its two values
must use `@sha256:` and must match locally built image digests recorded in the
seal. Never substitute `latest`, `dev`, a branch, or a hand-written digest.

## Host preflight

The only approved edge alias is `happy-nursing-home-raw`. SSH uses
`StrictHostKeyChecking=yes`, `CheckHostIP=yes`, `IdentitiesOnly=yes`, and a
dedicated pinned `UserKnownHostsFile`. Reject JNU signatures.

The first authorized preflight records only SHA-256 of `/etc/machine-id`.
Subsequent preflights require an exact match. Also require:

- edge deployment lock available and `edge-updater` idle
- clip-store free capacity at least 20 GiB
- existing API/worker state volumes healthy
- durable relay, evidence, and topology queues drained
- WAL-safe mode-0600 SQLite backups with integrity and fsync receipts
- current and previous API/worker image digests retained
- no facility ID or token in Compose/container environment

Run the local fixture before using a host:

```sh
sh scripts/ops/cloud-enrollment-smoke.sh --fixture --dry-run
```

## Publish and deploy

Builds are:

```sh
docker build -f Dockerfile.backend --build-arg "SOURCE_REVISION=$SEALED_ML_SHA" \
  -t "local/ml-api:$SEALED_ML_SHA" .
docker build --platform linux/amd64 -f Dockerfile.edge \
  --build-arg "SOURCE_REVISION=$SEALED_ML_SHA" \
  -t "local/ml-worker:$SEALED_ML_SHA" .
```

Publish only through `.github/workflows/edge-images.yml` for the sealed SHA.
Download the exact-SHA artifact and compare both digests with the seal before
updating the deployment receipt.

The host updater owns image replacement. It must run under the deployment lock,
preserve volumes, and reject a concurrent updater. Do not run `docker compose
down -v`, edit Python on the host, or place enrollment identity in the env file.

```sh
export EDGE_PROVISIONING_KNOWN_HOSTS=/secure/pointers/happy-known-hosts
export EDGE_PROVISIONING_KNOWN_HOST_FINGERPRINT='<independently recorded SHA256 fingerprint>'
export EDGE_PROVISIONING_MACHINE_BASELINE="$EVIDENCE/happy-machine-id.sha256"
export EDGE_PROVISIONING_EDGE_READBACK="$EVIDENCE/edge-preflight-readback.json"
export EDGE_PROVISIONING_EDGE_READBACK_SHA256='<independently recorded receipt digest>'
sh scripts/ops/cloud-enrollment-smoke.sh \
  --host happy-nursing-home-raw --deploy --restart-check
```

The wrapper verifies independently anchored plan, seal, host-key, and execution
receipt bytes before invoking the updater. After both services restart it parses
the actual `/api/v1/status` and `/api/v1/system` schemas, verifies running image
references, scans rendered Compose for facility identity residue, and revalidates
the complete post-restart state. Enrollment then occurs through local versioned
connection APIs, never through env mutation.

## Runtime enrollment and topology

The technician enters only facility code and one-time token. The local API
sequence is:

1. `PUT /api/v1/connection`
2. `POST /api/v1/connection/test`
3. restart and `GET /api/v1/connection`
4. `POST /api/v1/connection/sync-cameras`
5. `GET /api/v1/connection/topology-preview`
6. explicit `POST /api/v1/connection/topology-preview/confirm` only when the
   approved omission digest is exact

The cloud verification endpoint is `/api/v1/edge/enrollments/verify`. Canonical
facility identity, installation ID, and enrollment generation come from its
response and persist in SQLite. Complete topology comes from the camera registry.

## Stop conditions

Stop before mutation on any plan, SHA, digest, host key, hostname, machine ID,
lock, concurrency, capacity, volume, queue, backup, schema, or scope mismatch.
Do not continue with an old image or alternate host when publication fails.
