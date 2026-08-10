# Runbook: event thumbnail staged rollout and backfill

Use this runbook to deploy issue #268. It deploys `ml-worker` first, creates
missing thumbnail sidecars while normal cameras are stopped, then deploys
`ml-api`, which also contains the frontend. It does not change camera settings,
camera credentials, the roster, or `EDGE_CAMERA_CONFIG`.

The commands run from the operator machine through the existing edge SSH helper.
Set `EDGE_HOST`, `EDGE_USER`, and `EDGE_KEY` from the credential note.
`CAM_CHANNELS` is not needed for this procedure.

```sh
export EDGE_HOST=... EDGE_USER=... EDGE_KEY=~/.ssh/...
.claude/skills/edge-bringup/scripts/edge.sh open
.claude/skills/edge-bringup/scripts/edge.sh run 'hostname && docker ps -a'
```

Replace `/opt/eldercare-fall-ml` only when the edge host uses another checkout
path. Do not put an edge address, account, token, camera IP, RTSP URL, or
password in shell history, this document, or an issue.

## 1. Get the exact image artifact after merge

`Edge ML Images` runs automatically for every `main` push. The workflow resolves
the checked-out SHA and uploads one deploy artifact named
`edge-ml-image-refs-<resolved-sha>`. Wait for the run matching the merged `main`
SHA, then download that explicitly named artifact only.

```sh
set -eu
REPO=SeniorAILab/eldercare-fall-ml-v2
MAIN_SHA=$(gh api "repos/$REPO/commits/main" --jq .sha)
RUN=$(gh run list --repo "$REPO" --workflow edge-images.yml \
  --commit "$MAIN_SHA" --limit 1 --json databaseId --jq '.[0].databaseId')
test -n "$RUN"
gh run watch "$RUN" --repo "$REPO"

ARTIFACT="edge-ml-image-refs-$MAIN_SHA"
rm -rf /tmp/edge-refs
gh run download "$RUN" --repo "$REPO" --name "$ARTIFACT" --dir /tmp/edge-refs
REFS=/tmp/edge-refs/edge-ml-image-refs.env
test -f "$REFS"
cat "$REFS"
```

If `RUN` is empty, GitHub has not listed the push run yet. Wait briefly and
repeat the lookup. Do not use an artifact from another SHA or a mutable tag.
The artifact contains the two expected digest references:

```dotenv
ML_API_IMAGE=ghcr.io/seniorailab/eldercare-fall-ml/ml-api@sha256:...
ML_WORKER_IMAGE=ghcr.io/seniorailab/eldercare-fall-ml/ml-worker@sha256:...
```

## 2. Select one Compose form and record the rollback references

Select this once on the operator machine, before any edge command. Use the GPU
form on a healthy NVIDIA host. Use the CPU form only on a CPU-only host; it
removes the base compose file's mandatory NVIDIA reservation. Every command in
this runbook uses this same `EDGE_DC` value.

```sh
# GPU host
export EDGE_DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml'

# CPU-only host: use this instead of the GPU value above
# export EDGE_DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml -f compose.edge.cpu.yaml'
```

Do not use `scripts/edge-updater/update-edge.sh` in this staged rollout. Its
single compose-file setting cannot retain the selected ordered CPU overlay, and
its system-digest check is not the source of truth here. The exact
`docker inspect .Config.Image` comparisons below are the deployment checks.

Record both running references before changing either service. These are the
only rollback references for this staged deployment. Store the output in the
operations record, not in the repository.

```sh
.claude/skills/edge-bringup/scripts/edge.sh run "
  set -eu
  cd /opt/eldercare-fall-ml
  DC='$EDGE_DC'
  for service in ml-worker ml-api; do
    container=\$(\$DC ps -q \"\$service\")
    test -n \"\$container\"
    docker inspect --format \"\$service: {{.Config.Image}}\" \"\$container\"
  done
"
```

Keep existing `ml-worker-state:/root/.local/state/ml-worker`,
`ml-api-state:/root/.local/state/ml-api`, `CLIP_STORE_HOST_DIR`, and model mounts
unchanged. Do not add the development overlay to this production procedure.

## 3. Deploy and verify only `ml-worker`

Read the worker reference from the named artifact. The remote block changes only
`ML_WORKER_IMAGE`, pulls only `ml-worker`, recreates only that service without
dependencies, then proves the container's configured image reference equals the
artifact value.

```sh
NEW_WORKER_IMAGE=$(awk -F= '$1 == "ML_WORKER_IMAGE" { print $2 }' "$REFS")
test -n "$NEW_WORKER_IMAGE"

.claude/skills/edge-bringup/scripts/edge.sh run "
  set -eu
  cd /opt/eldercare-fall-ml
  DC='$EDGE_DC'
  umask 077
  awk -v image='$NEW_WORKER_IMAGE' '
    /^ML_WORKER_IMAGE=/ { print \"ML_WORKER_IMAGE=\" image; found=1; next }
    { print }
    END { if (!found) exit 1 }
  ' .env.edge.prod > .env.edge.prod.268.tmp
  mv .env.edge.prod.268.tmp .env.edge.prod
  \$DC pull ml-worker
  \$DC up -d --no-deps ml-worker
  container=\$(\$DC ps -q ml-worker)
  test -n \"\$container\"
  actual=\$(docker inspect --format '{{.Config.Image}}' \"\$container\")
  test \"\$actual\" = '$NEW_WORKER_IMAGE'
  port=\$(awk -F= '\$1 == \"ML_SERVING_PORT\" { print \$2 }' .env.edge.prod)
  : \"\${port:=8000}\"
  curl -fsS http://127.0.0.1:\"\$port\"/health/ready
  \$DC logs --tail 100 ml-worker
"
```

If this command fails after the image replacement, use the worker-only rollback
in step 7 before running a backfill.

## 4. Run the backfill without normal cameras

The normal `ml-worker` must not run during the one-off command. This uses the
same selected `EDGE_DC` form, stops the normal worker, runs only
`python -m worker --backfill-thumbnails`, and restarts the normal worker even
when the backfill exits nonzero. `--no-deps` keeps `ml-api` running and prevents
Compose from starting a second normal worker or camera loop.

```sh
.claude/skills/edge-bringup/scripts/edge.sh run "
  set -u
  cd /opt/eldercare-fall-ml
  DC='$EDGE_DC'
  if ! \$DC stop ml-worker; then
    exit 1
  fi
  if \$DC run --rm --no-deps ml-worker python -m worker --backfill-thumbnails; then
    backfill_status=0
  else
    backfill_status=\$?
  fi
  if ! \$DC up -d --no-deps ml-worker; then
    exit 1
  fi
  exit \"\$backfill_status\"
"
```

Success requires both exit status `0` and a final summary ending in `missing=0`:

```text
thumbnail backfill: scanned=... playable=... generated=... skipped=... failed=0 missing=0
```

Any nonzero exit, or `missing` greater than zero, is a failed backfill. The
normal worker has already been restarted. Capture the summary and one-off
output, retain all generated sidecars, and follow step 6. Do not retry while a
normal worker is running.

## 5. Deploy `ml-api` and frontend, then check cards

Only after a successful backfill, read and apply the API value from the same
artifact. This changes `ml-api` and its bundled frontend without recreating the
verified worker.

```sh
NEW_API_IMAGE=$(awk -F= '$1 == "ML_API_IMAGE" { print $2 }' "$REFS")
test -n "$NEW_API_IMAGE"

.claude/skills/edge-bringup/scripts/edge.sh run "
  set -eu
  cd /opt/eldercare-fall-ml
  DC='$EDGE_DC'
  umask 077
  awk -v image='$NEW_API_IMAGE' '
    /^ML_API_IMAGE=/ { print \"ML_API_IMAGE=\" image; found=1; next }
    { print }
    END { if (!found) exit 1 }
  ' .env.edge.prod > .env.edge.prod.268.tmp
  mv .env.edge.prod.268.tmp .env.edge.prod
  \$DC pull ml-api
  \$DC up -d --no-deps ml-api
  container=\$(\$DC ps -q ml-api)
  test -n \"\$container\"
  actual=\$(docker inspect --format '{{.Config.Image}}' \"\$container\")
  test \"\$actual\" = '$NEW_API_IMAGE'
  port=\$(awk -F= '\$1 == \"ML_SERVING_PORT\" { print \$2 }' .env.edge.prod)
  : \"\${port:=8000}\"
  curl -fsS http://127.0.0.1:\"\$port\"/health/ready
  \$DC ps
"
```

Open the dashboard through its normal operator route, sign in normally, and open
a camera history with existing playable clips. Verify that cards show thumbnails,
a thumbnail opens the existing clip playback flow, and a missing or failed
thumbnail leaves its playable card selectable. Do not fetch camera configuration
or RTSP URLs to perform this check.

## 6. Failed or partial backfill

The backfill is idempotent. A partial run may leave valid `thumbnail.jpg`
sidecars beside some clips. Preserve those files, their clip video, and their
`manifest.json` files. A retry skips valid thumbnails and processes only the
remaining playable clips.

1. Keep `ml-api` running and keep the restarted normal worker on the new image
   when it is otherwise healthy.
2. Record the nonzero exit status and final `scanned`, `playable`, `generated`,
   `failed`, and `missing` counts with the one-off command output.
3. Resolve the media or image fault, then repeat step 4. Do not delete valid
   thumbnails merely to reset counts.
4. If the new worker image is unhealthy, roll back only the worker in step 7.
   The prior worker ignores thumbnail sidecars, so image rollback does not
   require deleting them.

Never run `docker compose down -v`, remove either state volume, or remove the
host clip-store directory. Those actions destroy state or evidence and are not a
backfill rollback.

## 7. Roll back only the service that is faulty

Use the matching prior reference recorded in step 2. To roll back the worker,
set only `ML_WORKER_IMAGE` to that prior `@sha256:` reference and run the direct
single-service sequence in [worker-migration-rollback.md](worker-migration-rollback.md).
To roll back `ml-api` or its frontend, use the same selected `EDGE_DC` form with
`ML_API_IMAGE`, `pull ml-api`, `up -d --no-deps ml-api`, and an exact
`docker inspect .Config.Image` comparison to the recorded API reference.

After either rollback, keep the clip-store, manifests, and generated thumbnail
sidecars. They are evidence-adjacent artifacts, not camera credentials or camera
configuration. No camera credential work is part of rollout, backfill, retry, or
rollback.
