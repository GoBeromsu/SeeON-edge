# worker/tools/deepstream_canary: isolated qualification harness

Host-side operator tool, excluded from the production image and never imported
by `python -m worker`. It qualifies recorded workloads in the fixed Compose
project `seeon-ds-canary`; it does not authorize facility cameras.

## Safety boundary

- Published ports bind `127.0.0.1`; the GPU is the only shared physical
  resource. The nested production worker container uses
  `worker.runtime.lease.GpuLease` against its isolated canary state directory;
  that lease does not protect the live worker.
- Refuse an existing canary project, overlapping live mounts, and a worker image
  not explicitly bound as `repository@sha256` to the expected OCI source revision.
- `commissioning` refuses healthy live runtime cameras. `shared-host-smoke` is
  coexistence-only and always produces `claim_eligible=false`.
- The default pre-facility invocation explicitly requests `zero,loopback` and
  requires no facility RTSP.
  Live rungs `1,4,8,13` require an unexpired owner artifact bound to appliance,
  image digest, ordered camera IDs, and allowed rungs. Rung 13 must be last and
  additionally requires an 8-pass report digest and projected slack >= 3 GiB.

Never load `compose.edge.yaml`, read `.env.edge.prod`, reuse a live mount, or
run `down -v`. Cleanup is limited to project `seeon-ds-canary` and its isolated
network/socket/temp state.

## Gate policy and evidence

Canonical policy is `scripts/qa/deepstream-canary/gate-policy.v1.json`.
Recompute its current identity with
`sha256sum scripts/qa/deepstream-canary/gate-policy.v1.json`; the request records
that digest and the verifier recomputes it. Engine preparation
writes `raw/engine-prepare.json`; telemetry and immutable receipts write
`raw/telemetry-<rung>.json` and `raw/rung-<rung>.json`.

Engine preparation disarms steady-state utilization/slack gates only. Live
container restart, camera-stale, evidence-drop, and new-Xid signals remain abort
conditions. `PASS` is capacity-claim eligible only in commissioning mode;
`MARGINAL` and shared-host smoke are not claims.

## Commands

Default gate (with `CANARY_WORKER_IMAGE=repository@sha256:<digest>` and
`CANARY_EXPECTED_REVISION=<40-hex revision>` set):
`uv run python -m worker.tools.deepstream_canary run --rungs zero,loopback --evidence-dir <new-dir>`.
Verify immutable receipts:
`uv run python scripts/qa/verify_deepstream_delivery.py canary --evidence-root <dir> --output <green.json>`;
its JSON verdict must be exactly `PASS`.

Browser evidence uses `node scripts/qa/deepstream_canary_browser.mjs ...`.
Operational details live in `docs/runbooks/deepstream-canary-capacity.md` and
`docs/runbooks/deepstream-nvidia-cutover.md`. Focused tests:
`uv run pytest -q tests/test_deepstream_canary.py tests/test_deepstream_canary_runtime.py tests/test_deepstream_canary_telemetry.py`.
