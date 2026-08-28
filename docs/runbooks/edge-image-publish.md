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

Neither image carries model weights. Models are a pinned external artifact:
`worker/tools/fetch_models/manifest.json` names every file, its upstream
(Hugging Face `Berom0227/eldercare-fall-models` at a 40-hex revision for the
LSTM fall model; `ultralytics/assets` release `v8.4.0` for the YOLO pose,
person, and bed weights), its size, and its SHA-256. The one-shot
`edge-model-fetch` compose service runs `python -m worker.tools.fetch_models`
from the sealed worker image into the `worker-models` named volume on every
`up`; `ml-worker` starts only after it exits 0, so a hash mismatch or a broken
pin holds the worker back instead of loading an unverified weight. Changing a
weight means changing the manifest in the sealed SHA, never editing the volume
by hand. The optional `HF_TOKEN` in `.env.edge.prod` reaches this service
only; leave it empty for the public pins. There is no host `./models` bind
mount any more.

Publish only through `.github/workflows/edge-images.yml` for the sealed SHA.
Download the exact-SHA artifact and compare both digests with the seal before
updating the deployment receipt.

### What the image workflow builds and when

`edge-images.yml` is the single build path for both images. Exactly two images
exist: `Dockerfile.backend` -> `ml-api` (front + backend; the one-shot
migrator/consistency services run the same image) and `Dockerfile.edge` ->
`ml-worker`. Models are never baked into either image.

| Event                 | Builds both | Boot smoke | Pushes to GHCR | Tags pushed                          | Writes build cache |
|-----------------------|-------------|------------|----------------|--------------------------------------|--------------------|
| `pull_request`        | yes         | yes        | never          | none                                 | no (read-only)     |
| `push` to `main`      | yes         | yes        | yes            | `<full-sha>`, `main-<12-char sha>`   | yes (`mode=max`)   |
| `release` (published) | yes         | yes        | yes            | `<full-sha>`                         | yes (`mode=max`)   |
| `workflow_dispatch`   | yes         | yes        | yes            | `<full-sha>`                         | yes (`mode=max`)   |

`main-<short-sha>` is a staging tag for trying a pre-release build on a bench
device. It is not a deployment reference: the seal and `.env.edge.prod` pin
`@sha256:` digests only, never a tag. Every run prints both digests in the job
step summary and exposes them as job outputs (`ml-api-digest`,
`ml-worker-digest`, `deploy-sha`); the `edge-ml-image-refs-<sha>` artifact is
uploaded only by runs that pushed, because a digest that was never pushed
cannot be pulled.

Cache: `type=gha` scoped per image (`edge-ml-api`, `edge-ml-worker`). Pull
requests only read it. `uv sync` / `pnpm i --frozen-lockfile` layers sit before
the source `COPY` in both Dockerfiles, so a code-only change re-runs only the
copy and the image-identity layers.

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
receipt bytes before invoking the updater. On restart, `edge-model-fetch` must
show `done: 7 file(s) verified, nothing to do` (or `fetched N file(s)` on a
fresh volume) in `docker compose logs edge-model-fetch` before `ml-worker`
is considered up. After both services restart it parses
the actual `/api/v1/status` and `/api/v1/system` schemas, verifies running image
references, scans rendered Compose for facility identity residue, and revalidates
the complete post-restart state. Enrollment then occurs through local versioned
connection APIs, never through env mutation.

## Cutting a release

A release is cut by pushing an annotated tag, never by clicking "Draft a new
release" in the GitHub UI — the UI path skips the guard below.

**Tag scheme.** `seeon-edge-v<semver>`, e.g. `seeon-edge-v0.1.0`. Pushing a tag
matching `seeon-edge-v*` triggers `.github/workflows/release.yml`, which is the
only thing that creates a GitHub Release here.

**The version-carrier guard.** `scripts/release_guard.py` runs first and reads
the product version out of every file that states one:

```
pyproject.toml
backend/pyproject.toml
worker/pyproject.toml
shared/pyproject.toml
front/package.json
```

All five must be identical, and the tag must be exactly `seeon-edge-v` plus
that version. A mismatch fails the run and prints every carrier with its value,
so the file that is out of step is named rather than guessed. Bring the
carriers into lockstep and retag; do not weaken the guard.

`EDGE_DATABASE_FORMAT_IDENTITY = 'seeon-edge-v1'` in
`front/src/shared/releaseIdentity.ts` is **not** a version carrier. It is the
on-disk database *format* identity (paired with
`EDGE_DATABASE_SCHEMA_VERSION`), it only coincidentally spells like the tag,
and it moves when the SQLite format lineage changes — never when the product
ships. Bumping it to match a release tag would tell every edge device its
existing database belongs to a different lineage. The same goes for the
torch/CUDA/ultralytics versions reported by
`worker/runtime/provenance/environment.py` and
`worker/native/deepstream/export.py`: those are other people's versions.

Check the carriers locally before tagging:

```sh
python3 scripts/release_guard.py --tag seeon-edge-v0.1.0
```

**Rehearsal.** `release.yml` also accepts `workflow_dispatch` with a
`rehearsal` boolean (default `true`). A rehearsal runs the same guard and
composes the same notes, and creates no release:

```sh
gh workflow run release.yml -f rehearsal=true
```

Use it to find a carrier that is out of step *before* a tag exists to be
deleted.

**Release notes.** `scripts/release_notes.py` composes them: the hand-written
`docs/releases/<tag>.md` (when present), then the commit range since the
previous `seeon-edge-v*` tag, then a pointer to the image digests.

**The release event is what publishes the images.** The GitHub Release created
by `release.yml` is not a prerelease, deliberately. `edge-images.yml` runs
`on: release: types: [published]` and is gated on
`github.event.release.prerelease == false`; publishing the release is therefore
what builds and pushes `ml-api` and `ml-worker` to GHCR at the full commit SHA
and uploads the `edge-ml-image-refs-<sha>` artifact carrying both `@sha256:`
digests. A prerelease would create a release that publishes nothing. Take the
two digest-pinned references from that artifact (or the run's step summary) —
they are what `.env.edge.prod` pins.

The full sequence:

```sh
python3 scripts/release_guard.py --tag seeon-edge-v0.1.0   # carriers agree
git tag -a seeon-edge-v0.1.0 -m "SeeON Edge v0.1.0"        # on the merge commit
git push origin seeon-edge-v0.1.0
gh run watch "$(gh run list --workflow=release.yml --limit 1 --json databaseId -q '.[0].databaseId')"
gh run watch "$(gh run list --workflow=edge-images.yml --limit 1 --json databaseId -q '.[0].databaseId')"
gh run download <edge-images run id> -n "edge-ml-image-refs-<sha>"
```

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
