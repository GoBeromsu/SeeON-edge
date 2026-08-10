# Runbook: worker migration rollback (image digest)

Use this when a newly deployed `ml-worker` image misbehaves on an edge host and
you need the previous known-good worker back.

Rollback is **image-digest based, not a source revert**. Do not `git revert`,
rebuild, or edit Python on the edge host: the host pulls prebuilt GHCR images
only. You change one value — `ML_WORKER_IMAGE` — back to the prior digest and
restart that one service. Model, state, and clip volumes are preserved unchanged.

Inputs, exactly as described by
[`.env.edge.prod.example`](../../.env.edge.prod.example) and
[`compose.edge.yaml`](../../compose.edge.yaml):

```sh
cd /opt/eldercare-fall-ml   # the checkout that owns .env.edge.prod
# GPU host
DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml'

# CPU-only host: use this instead of the GPU value above.
# DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml -f compose.edge.cpu.yaml'
```

`compose.edge.yaml` requires `ML_WORKER_PROFILE` (`cuda|mps|cpu`). If it is
missing from `.env.edge.prod`, even `$DC ps` fails with
`${ML_WORKER_PROFILE:?set cuda|mps|cpu}`. Fix that before any compose action, or
you will misread a config error as a rollback failure.

## 1. Record the image you are leaving and the image you are returning to

```sh
# The configured reference of the currently running worker.
CURRENT_WORKER_IMAGE=$(docker inspect --format '{{.Config.Image}}' \
  "$($DC ps -q ml-worker)")
printf 'current worker image: %s\n' "$CURRENT_WORKER_IMAGE"

# Set this to the prior digest reference recorded before the failed deployment.
PRIOR_WORKER_IMAGE='ghcr.io/seniorailab/eldercare-fall-ml/ml-worker@sha256:<prior-digest>'
case "$PRIOR_WORKER_IMAGE" in
  *@sha256:*) ;;
  *) printf '%s\n' 'PRIOR_WORKER_IMAGE must be an @sha256: reference' >&2; exit 1 ;;
esac
```

Use the prior reference recorded in the deployment record or the prior successful
`edge-ml-image-refs-<sha>` workflow artifact. Do not infer a prior deployment
from a mutable tag.

`update-edge.sh`, when used independently, writes one overwritten file at
`${EDGE_UPDATER_DATA_DIR:-/var/lib/edge-updater}/snapshot.json`; it does not keep
a `snapshots/` directory or copy `.env.edge.prod` files. Its committed snapshot
describes the then-current target, so it is not the rollback source for this
manual procedure. Write both references down before continuing.

## 2. Preserve the volumes (read this before you touch compose)

The worker's durable state is not in the image. `compose.edge.yaml` mounts:

| Mount | Kind | Contents |
| --- | --- | --- |
| `ml-worker-state:/root/.local/state/ml-worker` | named volume | `ML_WORKER_STATE_DIR`: LKG config, first-fault records, GPU lease, outbox DB |
| `ml-api-state:/root/.local/state/ml-api` | named volume | ml-api local state |
| `${CLIP_STORE_HOST_DIR}:/var/lib/clip-store` | host bind | evidence clips and manifests (worker `rw`, api `ro`) |
| `${ML_MODELS_DIR:-./models}:/app/models:ro` | host bind | model artifacts, read-only |

Rules:

- **Never** run `docker compose down -v` / `docker volume rm` during a rollback.
  `-v` destroys `ml-worker-state` and `ml-api-state`, taking the last-known-good
  config, the durable evidence outbox, and the first-fault record with it.
- Recreating the container is fine and expected; recreating the **volume** is not.
- The clip store and models directory are host paths. Leave
  `CLIP_STORE_HOST_DIR` and `ML_MODELS_DIR` untouched so both image versions read
  the same evidence and the same weights.
- Take a copy of the state volume before rolling back if the failing image may
  have migrated the on-disk outbox schema forward. A forward-migrated SQLite
  outbox is not guaranteed readable by the older image:

   ```sh
   WORKER_CONTAINER=$($DC ps -q ml-worker)
   WORKER_STATE_VOLUME=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/root/.local/state/ml-worker"}}{{.Name}}{{end}}{{end}}' "$WORKER_CONTAINER")
   test -n "$WORKER_STATE_VOLUME"
   STATE_MOUNT=$(docker volume inspect --format '{{.Mountpoint}}' "$WORKER_STATE_VOLUME")
   sudo tar -C "$STATE_MOUNT" -cf /var/tmp/ml-worker-state-pre-rollback.tar .
   ```

  Keep that archive until the rollback is verified. It is the only way back if
  the older image rejects a migrated outbox.

## 3. Set the prior digest and restart only the worker

Leave every other key exactly as it is. The deployment identity is frozen: the
service and image stay `ml-worker`, the file names stay `compose.edge.yaml` /
`.env.edge.prod`, and the `ML_WORKER_*`, `WORKER_*`, `ML_API_*`, `API_*` keys are
unchanged across both image versions. A rollback that also renames something is
not a rollback.

```sh
umask 077
awk -v image="$PRIOR_WORKER_IMAGE" '
  /^ML_WORKER_IMAGE=/ { print "ML_WORKER_IMAGE=" image; found=1; next }
  { print }
  END { if (!found) exit 1 }
' .env.edge.prod > .env.edge.prod.rollback.tmp
mv .env.edge.prod.rollback.tmp .env.edge.prod

$DC pull ml-worker
$DC up -d --no-deps ml-worker
```

`--no-deps` keeps `ml-api` running. The worker→ml-api relay surface
(`/api/v1/relay/{config,restart,alerts,heartbeat,runtime-status}`) is unchanged
by the worker migration, so the worker can be rolled back on its own. Roll back
`ML_API_IMAGE` only if ml-api itself is the faulty component.

## 4. Verify

```sh
# The container now uses the exact prior reference, not merely an image ID.
ACTUAL_WORKER_IMAGE=$(docker inspect --format '{{.Config.Image}}' "$($DC ps -q ml-worker)")
test "$ACTUAL_WORKER_IMAGE" = "$PRIOR_WORKER_IMAGE"

# ml-api stayed healthy and is receiving worker facts
$DC ps
PORT=$(awk -F= '$1 == "ML_SERVING_PORT" { print $2 }' .env.edge.prod)
: "${PORT:=8000}"
curl -fsS http://127.0.0.1:"$PORT"/health/ready

# Worker boot: profile resolution, model warmup, per-camera activation
$DC logs --tail 200 ml-worker
```

Expected: the digest matches the prior digest from step 1, ml-api is `healthy`,
the worker logs a resolved profile and completed warmup, and heartbeats resume.
Volumes must be the same ones as before — confirm with `docker volume ls` that
no `*_ml-worker-state` volume was recreated.

If the older image fails to start against the current state volume, restore the
step-2 archive into the volume and retry. If it fails again, the fault is not in
the worker image and this runbook is finished; escalate with the worker logs and
the first-fault record from `ML_WORKER_STATE_DIR`.

## 5. Close the loop

- Record both digests, the reason, and the outcome in the operations log.
- Keep the pre-rollback state archive until the replacement image ships.
- Report the failing digest against the migration issue so the next image is not
  built on the same defect. A rollback is not a fix.

## Related

- [`docs/architecture.md`](../architecture.md) — worker layers, entrypoint
  (`python -m worker`), and the failure matrix that tells you whether a fault is
  global (process exits non-zero) or per camera (only that camera degrades).
- [`docs/runbooks/driver-cuda-alignment.md`](driver-cuda-alignment.md) — host
  driver/CUDA faults. An image rollback does not repair a GPU in a bad state.
