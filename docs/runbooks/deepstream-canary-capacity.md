# DeepStream isolated canary and capacity qualification

This runbook qualifies one recorded workload. It does not authorize cameras by
itself. The canary uses fixed Compose project `seeon-ds-canary`; its only shared
physical resource is the GPU. `GpuLease` is scoped to the canary state directory
and is advisory, so it does **not** protect the production worker.

## Modes and prerequisites

- `commissioning`: run before facility cameras are enabled. A capacity claim is
  eligible only when the recorded corpus exactly matches the claimed codec,
  resolution, FPS, GOP, models, policies, and camera count.
- `shared-host-smoke`: use on a development/operations host where a live worker
  already carries facility cameras. It proves coexistence safety only, always
  has `claim_eligible=false`, and must never be converted into a capacity claim.
  Such a host must not run commissioning or stop its live cameras. Commissioning
  belongs on the target facility appliance before its cameras are enabled.

Never load `compose.edge.yaml`, read `.env.edge.prod`, reuse a live mount, or
supply facility RTSP URLs for the default run. Confirm Docker, NVIDIA runtime,
FFmpeg, Chromium, models, and the digest-pinned worker image are present. The
harness refuses an existing `seeon-ds-canary` project, mount overlap, and a
non-digest worker image. All published ports bind `127.0.0.1`.

## Default pre-facility gate

Choose a directory that does not exist. Previous runs are immutable and must
never be reused.

```sh
uv run python -m worker.tools.deepstream_canary run \
  --rungs zero,loopback \
  --evidence-dir .omo/evidence/deepstream-nvidia-worker-migration/task-8-canary-default
uv run python scripts/qa/verify_deepstream_delivery.py canary \
  --evidence-root .omo/evidence/deepstream-nvidia-worker-migration/task-8-canary-default \
  --output .omo/evidence/deepstream-nvidia-worker-migration/task-8-canary-green.json
```

Before either rung, the harness explicitly runs the profiled `engine-builder`.
It verifies and reuses the matching `c7-<plan-key>` cache or builds the three
content-addressed engines and stores `raw/engine-prepare.json`. During this
one-time phase, live worker restarts, camera stale transitions, evidence-drop
increases, and new Xids remain abort conditions. The steady-state utilization
and slack gates arm only after preparation succeeds; their thresholds are not
relaxed.

Zero-camera runs for 2 minutes after the C7 preflight. Loopback uses the fixed
15 FPS H.264 1280x720 GOP-30 corpus for 15 clean minutes. The generated Compose
has one publisher service per camera; use `docker compose -p seeon-ds-canary -f
<run>/compose.rendered.yaml stop|pause|kill publisher-NN` for exact camera-local
fault injection. Test kill/EOS, bad relay auth, silent publisher stall,
geometry change, and remove/add. Fault windows are reported separately and are
never averaged into clean windows.

Exercise preview and one event derivative while loaded:

```sh
CANARY_RELAY_TOKEN='<run token from the protected run environment>' \
  node scripts/qa/deepstream_canary_browser.mjs loop-01 <run>/raw <event-clip> <internal-viewer-url>
```

Do not put the token in a tracked file or command transcript. The derivative
must use the C6 single-render token; a second render is a gate failure.

## Facility authorization template

Facility 1 -> 4 -> 8 requires an unexpired explicit owner artifact bound to:
appliance identity, exact worker image digest, exact ordered camera IDs, owner,
issue URL, expiry, and allowed rungs. It contains no RTSP credentials. Keep any
connection material outside the artifact and evidence. `claim_eligible=false`
never authorizes cameras.

```sh
uv run python -m worker.tools.deepstream_canary run --rungs 1,4,8 \
  --mode commissioning --authorization /run/operator/c8-authorization.json \
  --appliance-id '<appliance identity>' --camera-ids '<id1,...,id8>' \
  --evidence-dir <new-dir>/task-8-canary-live
```

Rungs 1 and 4 require 10-minute warmup plus 30 clean minutes. Candidate `N_pass`
requires 10-minute warmup plus 2 clean hours; MARGINAL is not `N_pass`. Rung 13
is never automatic. It additionally requires an 8-pass report SHA-256 in the
owner artifact and projected slack >=3 GiB. Projection admits the attempt; it
is not the result. Store each rung under `rung-<N>/`.

## Binary gates and aborts

The versioned policy is
`scripts/qa/deepstream-canary/gate-policy.v1.json`; its SHA-256 appears in every
report. The verifier recomputes per-camera 10-second FPS p05/p50/p95 and
source-PTS-to-decision p50/p95/p99/max. Aggregate FPS never passes a rung and a
missing PTS mapping is FAIL. It also requires child-PID and global GPU memory
warmup/steady/recovery/slack, utilization, no new Xid, exactly N NVDEC branches,
no software fallback, clean AU/config/timestamp continuity, metadata overwrite
below policy, playable hashed event/evidence timeline, preview and derivative,
and quiet live-protection counters.

The host watchdog atomically records the first fault and runs only:

```sh
docker compose -p seeon-ds-canary -f <run>/compose.rendered.yaml down --remove-orphans
```

After engine preparation, abort on any new Xid/kernel fault, utilization or
VRAM policy breach, live container replacement/restart, online-to-stale live
camera, evidence-drop increase, relay sentinel leak, or mount intersection.
During preparation, all those live-health signals remain armed except the
steady-state utilization/slack limits. Never run `down` against
the live project.

## Cleanup proof

After every outcome, record `docker compose ls`, port listeners for 18090/18554/
18888, canary-labeled containers/networks, live container IDs/restarts, live
camera states, evidence-drop counters, GPU processes, and protected mount
hashes. No canary container/network/socket/temp state may remain; live and
broadcast state must equal baseline.
