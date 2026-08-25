# NVIDIA native DeepStream worker cutover and C6 rollback

This is a manual single-service procedure. It does not automate cutover,
rollback, or camera enablement. Record immutable image digests, not mutable tags.
Never run it from the canary project and never use `down -v`.

## 1. Baseline and preflight

Before facility cameras are enabled, record the live Compose project, API and
worker image digests, container IDs/restart counts, Python/native/FFmpeg process
tree, GPU memory/utilization/processes, current-boot Xids, camera states,
evidence-drop counters, relay health, MJPEG/event/derivative checks, mounts, and
pinned C6 digest. Run the read-only six-stage C7 preflight:

```sh
sh scripts/edge-preflight/diagnose-edge.sh --with-container-probe
```

Stop on the first non-zero stage and preserve its machine-readable first-fault
receipt. The expected C7 topology is Python PID 1 with exactly one native
`seeon-deepstream-child`, no Python-parent CUDA context, and no per-camera
FFmpeg process. Preflight also proves driver, current-boot kernel log, runtime/
CDI, CUDA context, and relay status.

Run the isolated candidate loopback and independent verifier exactly as in
[`deepstream-canary-capacity.md`](deepstream-canary-capacity.md). Cutover requires
an exact `PASS`; a shared-host smoke, MARGINAL result, or `claim_eligible=false`
does not authorize facility cameras.

## 2. Manual single-service cutover

Set the sealed candidate digest through the existing protected deployment
environment; do not print that environment. Recreate only `ml-worker` in the
known live project and overlay:

```sh
docker compose --project-name "$COMPOSE_PROJECT_NAME" \
  --env-file .env.edge.prod -f compose.edge.yaml -f compose.edge.nvidia.yaml \
  up -d --pull never --no-deps --force-recreate ml-worker
```

Do not alter `ml-api`, volumes, camera registry, broadcast service, or another
Compose project.

## 3. Bounded verification

Verify all items before accepting the cutover:

- the running worker image ID equals the candidate digest;
- Python is PID 1 and owns exactly one native child;
- no per-camera FFmpeg exists and Python owns no second CUDA context;
- all previously enabled cameras return online within the recorded bound;
- each camera selects NVDEC with zero software fallback;
- relay heartbeat/runtime status and evidence-drop counters remain healthy;
- authenticated MJPEG preview works;
- one real event and its single-render derivative are playable and their
  evidence hashes verify;
- no new Xid/kernel signature, worker restart, stale live camera, evidence-drop
  increase, relay contamination, or protected-mount change occurs.

Abort immediately on any failed item, GPU memory/utilization gate, missing
source-PTS mapping, AU/config/timestamp discontinuity outside an injected fault
window, or native child/PID-1 topology mismatch.

## 4. One-command rollback

Rollback is pinned to the locally qualified C6 digest in
`scripts/ops/rollback-ml-worker-c6.sh`. It recreates only the live `ml-worker`;
it does not touch API, volumes, or camera enablement.

```sh
COMPOSE_PROJECT_NAME='<live project>' \
  sh scripts/ops/rollback-ml-worker-c6.sh
```

After rollback, verify the running image equals the pinned C6 digest, container
restart count is stable, all baseline cameras return online in the recorded
bound, relay/MJPEG/event/derivative work, evidence hashes remain valid, and no
protected mount or broadcast state changed. Preserve the candidate first-fault,
cutover transcript, rollback output, and post-rollback comparison together.
