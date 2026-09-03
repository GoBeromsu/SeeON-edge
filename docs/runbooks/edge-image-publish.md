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
docker build --platform linux/amd64 -f Dockerfile.edge --target runtime \
  --build-arg "SOURCE_REVISION=$SEALED_ML_SHA" \
  -t "local/ml-worker:$SEALED_ML_SHA" .
```

Neither image carries model weights. Models are a pinned external artifact:
`worker/tools/fetch_models/manifest.json` names every file, its upstream
(Hugging Face `Berom0227/seeon-model-v0.1.0-pose-bbox56-proxy-research` at a
40-hex revision for the pose+bbox56 fall bundle; `ultralytics/assets` release
`v8.4.0` for the YOLO pose, person, and bed weights), its size, and its SHA-256. The one-shot
`edge-model-fetch` compose service runs `python -m worker.tools.fetch_models`
from the sealed worker image into the `worker-models` named volume on every
`up`; `ml-worker` starts only after it exits 0, so a hash mismatch or a broken
pin holds the worker back instead of loading an unverified weight. Changing a
weight means changing the manifest in the sealed SHA, never editing the volume
by hand. The optional `HF_TOKEN` in `.env.edge.prod` reaches this service
only; leave it empty for the public pins. There is no host `./models` bind
mount any more.

## Deploying an admitted model selection

`/app/model-selection.json` is a deployment-owned, read-only file; it is never
copied into `Dockerfile.edge`. To switch the running fall model, publish the
bundle into `worker-models`, place the matching selection document on the
deployment host, and enable the read-only selection mount in
`compose.edge.yaml`. Restart `ml-worker` only after `edge-model-fetch` has
verified the bundle. Removing that mount is the only normal path back to the
packaged fallback. Replacing a model requires only the bundle and selection
file, not an image rebuild.

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
| `release` (published) | per image   | yes        | yes            | `<full-sha>`, `<release tag>`        | yes (`mode=max`)   |
| `workflow_dispatch` on a release tag | per image | yes | yes       | `<full-sha>`, `<release tag>`        | yes (`mode=max`)   |
| `workflow_dispatch` on any other ref | yes       | yes | yes       | `<full-sha>`                         | yes (`mode=max`)   |

### Per-image isolation at release time

A release is ONE version, and its artefact is a manifest pinning BOTH images by
`@sha256:` — the seal. Isolation is not about versioning; it is about not
rebuilding an image whose inputs did not move. On a run that is publishing a
release, each image is decided independently:

- **inputs changed** → build and push as usual;
- **inputs unchanged** → do not build. The digest already published for that
  image is given this release's tags (`docker buildx imagetools create`, which
  copies the manifest instead of rebuilding) and re-emitted **unchanged** into
  the seal.

Reuse is not an optimisation you can get from reproducible builds. Both
Dockerfiles take `SOURCE_REVISION` and stamp `org.opencontainers.image.revision`,
so rebuilding an unchanged tree at a new commit still produces a **new** digest.
The only way to keep a digest is to not rebuild it.

"Publishing a release" is **not** the same as the `release` event. A release
created by `release.yml` with the default `GITHUB_TOKEN` raises no event that
starts another workflow run, so `release.yml` dispatches `edge-images.yml` on
the tag instead. `RELEASE_BUILD` therefore counts both shapes: the `release`
event (still real for a release a human or a PAT publishes) and a
`workflow_dispatch` whose `ref` is a `seeon-edge-v*` tag. Keying reuse on the
event alone would have made the isolation dead code that never once engaged,
and nothing would have failed to say so —
`test_release_isolation_keys_on_the_dispatch_not_only_the_release_event` is
what keeps the two halves wired together.

Everything else builds both: `pull_request` (that is the gate, and it is also
what catches drift in the floating apt packages `Dockerfile.edge` installs),
`push` to `main` (it is the BuildKit cache writer), and a dispatch on any ref
that is not a release tag — the deliberate "rebuild this ref" escape hatch.

**The inputs**, per image, are exactly what each Dockerfile copies out of the
build context, plus the Dockerfile, plus the files that shape the context:

| image | inputs |
|---|---|
| `ml-api` | `Dockerfile.backend`, `backend/`, `contracts/`, `front/`, `shared/`, `scripts/ops/`, `pyproject.toml`, `uv.lock` |
| `ml-worker` | `Dockerfile.edge`, `worker/` (incl. the pinned model manifest), `contracts/`, `shared/`, `pyproject.toml`, `uv.lock` |
| both | `.dockerignore`, `.github/workflows/edge-images.yml` |

`scripts/ops/` is an `ml-api` input because `Dockerfile.backend` copies it in —
easy to forget, so `tests/test_edge_image_isolation.py` re-derives every `COPY`
source from both Dockerfiles and fails if the sets drift.

Anything the classifier does not recognise **fails closed**: an unrecognised
path is treated as affecting both images, so a new top-level directory costs a
redundant rebuild rather than a silently skipped one.

**The comparison base is not the previous release's commit.** It is the commit
that actually built the digest being reused, read back from that image's own
`org.opencontainers.image.revision` label. The two differ as soon as an image is
reused twice: if `ml-api` was built at C1 for v0.1.0 and reused for v0.2.0, the
digest published at v0.3.0 is still the one built at C1, so the comparison runs
C1..C3. Comparing against v0.2.0's commit would skip everything that changed
between C1 and C2 and reuse a genuinely stale image.

**Reading the result.** The step summary and the `edge-ml-image-refs-<sha>`
artefact state, per image, `built` or `reused`, with this release's digest and
the previous release's digest side by side. A `reused` row shows the *same*
digest in both columns — that is the proof it is the same artefact. A mixed
release looks like this:

| image | action | this release | previous release |
|---|---|---|---|
| `ml-api` | reused | `sha256:aaa…` | `sha256:aaa…` |
| `ml-worker` | built | `sha256:fff…` | `sha256:bbb…` |

The boot smoke test always runs against the image that would be deployed. When
`ml-worker` was reused rather than rebuilt it is **pulled by digest** and booted,
so the seal never carries a worker digest that was not booted in that run.

A release build therefore builds `ml-worker` **without** `load:` and pulls it
back by digest for the smoke. A digest that a later release may reuse has to be an OCI
index (see below), and the docker exporter cannot export a manifest list, so the
two are mutually exclusive on that path. Every other event keeps the cheap local
load. A consequence worth knowing: `ml-worker` digests published *before* this
landed are plain manifests and cannot be reused — the guard below detects that
and rebuilds, and the next release publishes an index that later releases can
reuse.

Two failure modes are deliberately loud rather than silent:

- `imagetools create` preserves a digest when the source manifest is an OCI
  index (what `docker/build-push-action` pushes when it attaches provenance) but
  **re-wraps a plain manifest** into a new index under a new digest. The re-tag
  is therefore always re-inspected, and a changed digest fails the release
  instead of landing in the seal labelled "reused".
- Every way of *not* knowing — no previous tag, a reference the registry does
  not resolve, a missing or unusable revision label, a label naming a commit
  this repository does not have — resolves to "build".

To see the plan before tagging, run the release rehearsal
(`gh workflow run release.yml -f rehearsal=true`); it prints the per-image plan
and publishes nothing.

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

**Creating the release is what publishes the images — but not via the
`release:` trigger.** `edge-images.yml` does carry
`on: release: types: [published]`, and that trigger does **not** fire for a
release `release.yml` creates. GitHub's rule: *"With the exception of
`workflow_dispatch` and `repository_dispatch`, other `GITHUB_TOKEN`-triggered
events do not create workflow runs at all."* The first release
(`seeon-edge-v0.1.0`) proved it — the release was published and no image run
started.

So the release job dispatches `edge-images.yml` itself, on the tag
(`gh workflow run edge-images.yml --ref "$TAG" -f ref="$TAG"`), which is the
documented exception. Dispatching on the tag means the images are built from
the released commit, never from whatever `main` has drifted to. The `release:`
trigger stays in place for a release published by a human or a PAT.

The release is still not a prerelease, deliberately: `edge-images.yml` is gated
on `github.event.release.prerelease == false`, and a prerelease also reads to
every operator as "do not deploy this".

That run pushes `ml-api` and `ml-worker` to GHCR at the full commit SHA and
uploads the `edge-ml-image-refs-<sha>` artifact carrying both `@sha256:`
digests. Take the two digest-pinned references from that artifact (or the run's
step summary) — they are what `.env.edge.prod` pins.

If a release is ever published by hand, or the dispatch step fails, publish the
images manually for the tag:

```sh
gh workflow run edge-images.yml --ref seeon-edge-v0.1.0 -f ref=seeon-edge-v0.1.0
```

The full sequence:

```sh
python3 scripts/release_guard.py --tag seeon-edge-v0.1.0   # carriers agree
git tag -a seeon-edge-v0.1.0 -m "SeeON Edge v0.1.0"        # on the merge commit
git push origin seeon-edge-v0.1.0
gh run watch "$(gh run list --workflow=release.yml --event=push --limit 1 --json databaseId -q '.[0].databaseId')"
# the release job dispatches edge-images.yml; watch that run and take the digests
gh run watch "$(gh run list --workflow=edge-images.yml --event=workflow_dispatch --limit 1 --json databaseId -q '.[0].databaseId')"
gh run download <edge-images run id> -n "edge-ml-image-refs-<sha>"
```
### Exact readback and rollback

Record the deployed tuple as one atomic receipt:

```
(ml-worker image digest,
 fall/pose-bbox56-gru/arch.json sha256=ba6b7b3e83514aa9b6cea15012bb1fa96b66ef668a2ddb447635af5f97602726,
 fall/pose-bbox56-gru/bundle-manifest.json sha256=55108ac7e1bb427d0387c29b55caa5e2019e7616e4bcd05c615c4805e85f91d6,
 fall/pose-bbox56-gru/calibration.json sha256=0653c44577390488a84ff20388907e34a2c7a6cfe91c4c6f80b6da2ac3de1510,
 fall/pose-bbox56-gru/conformance/pose-bbox56-v1.json sha256=72d7e911acf39c7183bcdf10fad3c066dd93b7a45b7d28bef1950e1bf85b3a9c,
 fall/pose-bbox56-gru/evaluation-receipt.json sha256=3880181783b97708903c8c440d852cae32cc44d619128eabbf4972f255317fce,
 fall/pose-bbox56-gru/metadata.yaml sha256=f309ca0b08bc58a790519e3df5cda0a072cd6419558e28c09c3a7c5906a5e69e,
 fall/pose-bbox56-gru/model-card.md sha256=77c09bc3eacba53cadcaef1b8e91b5085efee38a04847908d436e431738166bd,
 fall/pose-bbox56-gru/model.pt sha256=7bb75a2932e1a1250dc900013b2c80b220de5e23f3ea568e05f1db21d0a757e3,
 fall.v2, fall.policy.v2, [30,56])
```

This is the packaged pose+bbox56 bundle, pinned member by member -- not a
synthetic "bundle SHA-256". Its own `bundle-manifest.json` is re-verified at
load time, so a tampered member is refused before any weights are read.
`model.pt` and `metadata.upstream.json` are the pinned artifacts in
`worker/tools/fetch_models/manifest.json`; `metadata.yaml` and `arch.json` are
the byte-identified sidecars that the fetcher installs alongside them. Read
the configured worker reference separately from the container's immutable image
ID, then attest that ID's repository digest and source revision. Compare all of
them to the sealed receipt before declaring the deployment healthy:

```sh
dc() {
  docker compose --env-file .env.edge.prod -f compose.edge.yaml -f compose.edge.nvidia.yaml "$@"
}
worker_id="$(dc ps -q ml-worker)"
test -n "$worker_id"
configured_ref="$(docker inspect --format '{{.Config.Image}}' "$worker_id")"
image_id="$(docker inspect --format '{{.Image}}' "$worker_id")"
printf 'ml-worker configured reference: %s\nml-worker immutable image ID: %s\n' \
  "$configured_ref" "$image_id"
docker image inspect --format \
  'RepoDigests={{json .RepoDigests}} source_revision={{index .Config.Labels "org.opencontainers.image.revision"}}' \
  "$image_id"
dc run --rm --no-deps --entrypoint python ml-worker -c '
import hashlib
from pathlib import Path
for path in (
    Path("/app/models/fall/pose-bbox56-gru/bundle-manifest.json"),
    Path("/app/models/fall/pose-bbox56-gru/model.pt"),
    Path("/app/models/fall/pose-bbox56-gru/arch.json"),
    Path("/app/models/fall/pose-bbox56-gru/calibration.json"),
    Path("/app/models/fall/pose-bbox56-gru/evaluation-receipt.json"),
    Path("/app/models/fall/pose-bbox56-gru/metadata.yaml"),
):
    print(f"{path.relative_to('/app/models')} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
'
```

The worker has dual mounts of the same model volume. `/app/models` preserves
the packaged pose+bbox56 bundle; `/models` holds admitted bundles at
`/models/bundles/<digest>`. The image ships neither a selection document nor a
bundle. Deployment supplies a read-only selection document when it chooses an
admitted model. Its manifest declares the payload member list; evaluation and
field receipts are externally hashed, non-recursive receipts. A selected bundle
must pass `admit → construct → warm → persist` before camera activation. Any
failure refuses boot; no fallback is permitted while a selection exists.

Runtime provenance is dedicated local state, never an alert payload. Each boot
writes an immutable mode-0600 record under
`/var/lib/seeon-state/runtime-provenance/`; an atomic mode-0600 latest readback
is published at `/var/lib/seeon-state/applied-runtime-manifest.json`. Runtime
provenance content stays local; relay payloads carry only its digest reference.
Verify the
directory, immutable per-boot record, latest readback, and their hashes:

```sh
dc exec -T ml-worker python -c '
import hashlib
import json
from pathlib import Path
state = Path("/var/lib/seeon-state")
records = state / "runtime-provenance"
latest = state / "applied-runtime-manifest.json"
assert records.is_dir()
assert records.stat().st_mode & 0o777 == 0o700
assert latest.stat().st_mode & 0o777 == 0o600
record_paths = tuple(records.glob("*.json"))
assert record_paths
for path in (*record_paths, latest):
    assert path.stat().st_mode & 0o777 == 0o600
    receipt = json.loads(path.read_text())
    canonical = receipt["canonical_json"]
    assert hashlib.sha256(canonical.encode()).hexdigest() == receipt["manifest_sha256"]
    assert json.dumps(json.loads(canonical), separators=(",", ":"), sort_keys=True) == canonical
print(latest.read_text())
'
```

The active contract is the packaged pose+bbox56 proxy bundle, `fall.v2`,
`fall.policy.v2`, and temporal input `[30,56]` scored on the CPU. An admitted
content-addressed candidate bundle remains dark and inert until it is selected:
it is not an active module or policy and is never part of the packaged receipt.

For rollback, use the previous **worker image** and the complete previous flat
bundle receipt as one atomic tuple. Fetch with that worker image,
read the tuple back, then recreate only `ml-worker`:

```sh
PREVIOUS_WORKER_IMAGE='ghcr.io/seniorailab/eldercare-fall-ml/ml-worker@sha256:<previous-digest>'
ML_WORKER_IMAGE="$PREVIOUS_WORKER_IMAGE" dc pull ml-worker
ML_WORKER_IMAGE="$PREVIOUS_WORKER_IMAGE" dc run --rm --no-deps edge-model-fetch
ML_WORKER_IMAGE="$PREVIOUS_WORKER_IMAGE" dc up -d --no-deps --force-recreate ml-worker
```

Run the readback block again with `ML_WORKER_IMAGE="$PREVIOUS_WORKER_IMAGE"`
in its environment. Do not restart `ml-api` or any other service. Never use
`docker compose down -v`, delete model files or content-addressed candidates,
or hand-edit the model volume: retention and `edge-model-fetch` are required
for an atomic rollback. This procedure does not retain or promote candidates
automatically. Candidate activation and Phase 7 are outside this runbook.

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
