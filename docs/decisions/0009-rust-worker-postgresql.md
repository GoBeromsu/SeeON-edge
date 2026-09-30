# 0009 — Rust inference runtime and API-owned PostgreSQL

- Status: Accepted target; implementation and deployment **not yet qualified**.
- Issue: [#607](https://github.com/SeniorAILab/SeeON-edge/issues/607).
- Governing approval: stage-04-final SHA-256
  `d85da142704cc466184ec8295aae87bb7ea6022a938ff2f1ef38ecce5c27d7d3`.
- Preserves: [0001](0001-preserve-source-stream-for-evidence-clips.md),
  [0005](0005-edge-component-ownership.md),
  [0006](0006-vendored-contracts-typed-vocabulary.md), and
  [0008](0008-evidence-overlay-is-a-sidecar-not-burned-pixels.md).
- Supersedes in part: [0006](0006-vendored-contracts-typed-vocabulary.md)
  (byte mirror into the archived eldercare-dataset-ops).

## Context and observed baseline

The admitted source baseline is commit
`5fa64b040700a291906ab1b6bbcc68ce31bab003`. The running API has that revision;
Worker revision `08c88988eb96ff0f756af69a6fe63b11b09f605b` has identical
committed Worker build inputs. Source hashes, actual deployment-owned Compose,
16 provisioned model artifacts, engine/native input hashes, applied provenance,
and all three independently reproduced frontend assets have been reconciled
onsite. These checks establish provenance, not target behavior or performance.

The actual baseline has 13 camera slots, `fall.v2` and `bed_exit.v1`, DeepStream
GPU pose, CPU ONNX Runtime fall classification and bed segmentation. Stored-clip
analysis is a separate model caller; inventory/provisioning is not proof that
any particular caller has executed. There is no deployment-selected alternate
fall bundle. Model ONNX shapes are dynamic-batch pose to `[batch,300,57]`, fall
`[1,30,56]` to `[1,1]`, and bed `[1,3,1280,1280]` to detection and mask outputs.
The exact published bytes and the existing preprocessing remain authoritative.

The product SQLite schema is 19. A separate diagnostics SQLite schema is 1;
retained execution tables in the product file are not a second active telemetry
writer. Historical source/archive/backup databases exist and must be preserved.
The Worker has a publish-once file delivery queue, not a database or DB outbox.
Current liveness/boot readiness alone do not prove that product reads work.

## Decision

### Ownership and runtime

Keep the component and image identities `front`, `backend`, `worker`, `ml-api`,
and `ml-worker`, and the established environment prefixes. Replace the
inference-runtime implementation with Rust; no Python interpreter, Python
control-plane wrapper, subprocess model helper, or stub may remain in its
production execution path. Keep DeepStream's native GPU media ownership behind
a narrow native adapter; Rust owns bounded scheduling, policy and egress.

Every executed model role uses the onsite NVIDIA GPU: live pose and fall,
on-demand bed recognition, and stored-clip pose/bed analysis. CPU control flow,
clock handling, small metadata and policy arithmetic are not CPU model
inference. Provider selection, every model call and failure paths must prove
that there is no CPU inference fallback. Missing GPU/provider/engine wiring
refuses activation rather than starting a degraded or always-failing feature.

Preserve runtime camera registration, model-selection authority, bytes,
threshold overrides, preprocessing, calibration eligibility, rounding,
class order, PTS resampling, per-track generations, reconnect/reset behavior,
confirmation windows, cooldown and detection windows. Preserve API payloads,
IDs, observability vocabulary, live views, diagnostic coverage and causal
joins. Golden features alone do not establish decode-to-event equivalence.

### PostgreSQL acceptance authority

PostgreSQL is opened only by the API and its explicit migration owner. Worker
receives neither database credentials nor a product database mount. Runtime
roles cannot perform DDL; migration roles cannot be used as serving defaults.
Use separate product and diagnostics authorities/resource budgets so telemetry
pruning cannot monopolize event acceptance. Retire SQLite runtime fallback and
basename-selected legacy catalogs after the full transition is verified.

One transaction owns immutable event identity, the corresponding durable
outbox disposition, and required audit changes. The API emits a durable ACK
only after COMMIT has returned successfully. A lost connection around COMMIT
is an unknown outcome, not permission to acknowledge or generate a new event
ID. Same-ID retries return the already committed outcome; conflicting immutable
content is refused explicitly. Media publication is not inside this transaction.

Preserve the distinction between local acceptance and upstream acceptance.
A currently local-only event gets an explicit terminal local-only outbox
disposition; later connectivity or mapping changes must not silently replay it
upstream. Only eligible pending rows are claimable by the sender. Delivery
attempts, receipts, refusal and retry state remain explicit and durable.

Writer and sender fencing, generation-bound claims, bounded transactions and
unique constraints establish authority. A process-local flag, container name,
or best-effort stop command is not an exclusive writer fence.

### Clips and bounded state

Local clips remain always on, defaulting to `/var/lib/clip-store` with an
explicit path override. Preserve source media, integrity sidecars, publication,
retention and disk-pressure behavior. Remote clip-export enablement is not a
local recording enablement switch. Use the existing six approved Worker file
surfaces in ADR-0005; do not invent a second mutable state database/index.

Queues have fixed record and byte capacities, visible overflow/refusal, fair
scheduling and bounded retry/backoff. Release buffers on every consume, drop,
shutdown and error path. Diagnostics retain exact coverage gaps where known;
resource pressure must not fabricate complete causal histories or preferentially
discard a lexically first camera. Batches include the entire encoded UTF-8 wire
envelope in their byte limit; increasing the relay cap is not a fix.

### Migration, recovery and deployment authority

Inventory all mutators/senders, product rows, diagnostics, configuration,
credentials, reviews, audit/history, enrollment/topology, runtime provenance,
pending delivery/clip publication and media references. Export/import and
verification remain onsite; credentials and resident media are not test inputs
or report content. Preserve pre-existing archives and concurrent development.

Quiescence fences every writer and sender, not just inference. After a
consistent snapshot/import, perform a final delta reconciliation before one
explicit authority transfer. Preserve primary keys, uniqueness, revision/CAS
semantics, audit chains and file references. Rehearse failures before/after
snapshot, import, final reconciliation, authority transfer and ACK.

Recovery after PostgreSQL has accepted writes must preserve target-only events,
configuration/history changes and delivery state, including a run with zero
new events. Restoring an old SQLite snapshot and discarding target-only writes
is not rollback. Recovery finishes with exactly one writer/sender authority and
no second event ID or external delivery for a replayed event.

Both integrated and staged activation must be rehearsed. Select the route from
measured behavior and recovery safety, not convenience. Production replacement
occurs only after one final 5-minute predeployment check passes with zero
skipped, incomplete or failed items and the same image digests that will be
deployed. That check replaces the 30-minute window of the governing approval;
it is not an implementation time cap or indefinite soak.
After replacement, independently verify image IDs, effective configuration,
GPU model execution, DB authority, user-facing reads, clips and durable state.

## Qualification rules

Before observing candidate performance, freeze the same-input replay corpus,
all model/policy/build identities, offered load, 13-camera and peak-track
scenarios, event oracle, clock boundaries/resolution, warm-up, five matched
baseline/candidate pairs, run order, sample windows and raw artifact schema.
Define each metric's direction, quantization and baseline-variance-derived
resolution bound before candidate results. No changed sample exclusion,
threshold, budget or confidence rule is permitted to rescue a candidate.

Correctness, event count/identity, golden replay and GPU-only execution are exact
or explicitly numerically bounded gates, not statistical speed gates. A
statistical improvement cannot offset a correctness failure. Indefinite
"inconclusive" is not success: allow at most one evidenced full-budget retry;
terminal inconclusive is no-go. Freeze numeric RAM/VRAM/queue/disk/latency and
isolation budgets separately; resource use must remain inside them even if a
latency comparison passes.

Real GPU inference, original-run observability, always-on clips, database
fault/restart/unknown-COMMIT cases, saturation, writer/sender fencing and recovery
are verified on the actual onsite host. Mock or macOS tests are not substitutes.
No partial Rust scaffold, toolchain test, source hash, or artifact-existence check
constitutes product or deployment acceptance. Final native cleanup, review, QA
and terminal gates apply to the cumulative implementation and its real evidence.

## Implementation decisions

Diagnostics live in their own PostgreSQL schema, `<API_POSTGRES_SCHEMA>_diagnostics`,
with a separate migration ledger and a separate connection-pool budget, so
telemetry writes and pruning draw from a different pool than event acceptance.
The runtime role cannot perform DDL in either schema.

GPU raw-output parity is gated on the rows the decision code consumes: rows at
or above the unchanged 0.25 score threshold, in order, with equal counts, within
`max(1e-4, one float32 ULP of the oracle value)`. The ULP floor applies only
where 1e-4 is not representable (bed coordinates above 1024). Bed prototype
masks are gated by equality of the derived mask and zone inputs that the
decider reads; raw prototype differences and the full top-300 comparison are
reported as diagnostics, not gates. TensorRT and ONNX Runtime CUDA FP32 kernels
differ, so unconsumed low-score rows can reorder. Event and state mismatches in
same-input replay must still be zero.

There is no PostgreSQL-to-SQLite export. Restoring the pre-cutover SQLite
snapshot is permitted only when the rollback check proves that PostgreSQL holds
no target-only writes. Otherwise PostgreSQL stays authoritative and faults are
fixed forward after an explicit unfreeze of the fenced generation.

Before any candidate run and before `R_m` was frozen, the Stage 1 baseline
protocol (sha256 `864ce6d6…`) received one recorded instrument correction,
`baseline-protocol-addendum-1.md` (sha256 `4a081a80…`), as plan Stage 1
(L256-273) allows; it is not the one post-INCONCLUSIVE rerun. The medians
`evt_decision_to_local_p50`, `evt_decision_to_central_p50` and
`evt_probe_to_central_p50` are still reported but no longer gate. The baseline
Worker's evidence sender idles up to 1 s between delivery attempts, so these
medians follow a ~1 s sawtooth phase (validation CV about 0.47, `R_m` about
580 ms). Their p95 counterparts (CV about 0.02) and every `frm_*` metric still
gate, and no other metric, estimator, `R_m` method, input, corpus or budget
changed. The change is `analyze.py` `57f2edb7…` → `03e174c9…` (harness digest
`05b04881…` → `50a4f486…`). The GPU-contaminated runs base1..5 remain harness
validation only, and `R_m` is frozen from the first five uncontaminated runs of
the clean series under the new digest.

## DeepStream graceful stop and RTSP reconnect

`pyservicemaker` `Pipeline.stop()` only sends EOS. With `nvurisrcbin` reconnect
intervals above zero, the plugin answers that EOS as a lost stream ("Resetting
source", reconnect attempt) and the pipeline EOS never reaches `wait()`, so the
Flow thread does not exit and `DeepStreamFlowStopTimeout` follows. Production has
no graceful stop path: it hard-exits with the fatal accelerator exit code, so
the interaction is a latent defect reachable only through graceful stop, which
the real-stack observability lane uses.

Resolution: the reconnect interval is configurable
(`DeepStreamMediaPlaneConfig.rtsp_reconnect_interval_sec`, worker env
`ML_WORKER_FLOW_RTSP_RECONNECT_INTERVAL_SEC`). The default stays 5 seconds, so
production behavior is unchanged. The load harness sets 0, where reconnect is
idle on its stable looping fixture. A runtime toggle was rejected because the
reconnect properties are only settable in the NULL/READY states and the binding
exposes no state control.

## Consequences

This is a coupled implementation, behavior-preservation, migration and deployment
change. A smaller feature-only port would be easier but would not satisfy the
approved request. Existing production remains authoritative until qualification;
new source and isolated test images alone do not change authority. Reports must
separate observed baseline facts, completed checks, unresolved gates, actual
replacement and measured benefits without claiming unobserved gains.
