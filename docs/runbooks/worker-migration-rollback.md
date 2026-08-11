# Roll back edge enrollment safely

Rollback is digest-based and ML-first. It preserves API state, worker state,
camera registry, outboxes, models, and clip storage. It is not a source revert
and never uses a mutable image tag, direct SQL, or environment-derived facility
identity. Deployment inputs remain `compose.edge.yaml` and the contract described
by `.env.edge.prod.example`.

The preserved storage identities are `ml-worker-state`, `ml-api-state`, and the
`clip-store` bind mount. A container may be recreated; these storage identities
must not be recreated or removed.

## Before deployment

Record the exact running and target API/worker image digests. Under the catalog
process lock:

1. checkpoint WAL with `PRAGMA wal_checkpoint(FULL)`
2. use the SQLite online backup API into a root-only `0700` directory and `0600`
   timestamped file
3. run `PRAGMA integrity_check`
4. fsync the backup file and parent directory
5. record only filename, SHA-256, and size

Retain the previous digests and backups through the rollback window. Never put
SQLite bytes or secret values in `.omo` evidence.

## Rollback gate

Rehash the approved plan, verify `final-rc-seal.json`, recheck the pinned host
key/hostname/machine baseline, acquire the edge deployment lock, and reject a
running updater. Require healthy volumes, drained queues, at least 20 GiB free
clip capacity, and a verified backup before changing containers.

## Order

1. Stop new v1 work and drain topology, relay, and evidence queues.
2. Restore `ML_API_IMAGE` and `ML_WORKER_IMAGE` to the recorded previous
   `@sha256:` references.
3. Pull and recreate only `ml-api` and `ml-worker`. Do not use `down -v`.
   Worker-only recovery uses `up -d --no-deps ml-worker` after the API is proven
   compatible.
4. Verify both previous digests, `/health/ready`, `/api/v1/system`, and all
   enumerated legacy AI routes.
5. Verify normal heartbeats and no duplicate topology or events.
6. Only after ML is proven compatible may AI perform a binary-only rollback.

AI database restore is prohibited after post-v1 traffic. Restore the pre-v1 ML
SQLite backup only if the old image cannot read the additive state. That restore
discards post-snapshot enrollment/topology-send state but preserves unrelated
camera registry, outbox, and media state.

## Roll forward

Deploy the sealed ML digests again, enroll through `PUT /api/v1/connection`, read
the current server revision, and trigger `POST /api/v1/connection/sync-cameras`.
The exact registry snapshot converges stable identities. Do not stamp a facility
ID into worker config or reconstruct it from a backup filename.

Run the lifecycle smoke only after the content-addressed execution receipt is
green. Boolean declarations are not lifecycle evidence: the receipt must contain
sequenced successful executions for enrollment, topology, heartbeat, event/clip,
rotation, timeout retry, ML-first rollback, roll-forward, and restart. Its fresh
post-restart generation repeats the complete lock/updater/capacity/volume/queue/
backup/image/env/status/schema/scope validation.

```sh
sh scripts/ops/cloud-enrollment-smoke.sh \
  --host happy-nursing-home-raw --full-lifecycle --rollback-drill
```

Success evidence contains only plan/SHA/digest bindings, backup metadata,
versioned API statuses, counts, and redacted lifecycle results.
