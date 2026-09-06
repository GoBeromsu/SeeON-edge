# worker/domains: event interpretation

Own fall and bed-exit judgment. Numeric observations become typed `BusinessEvent` values. `DOMAIN_REGISTRY` is the only public extension surface.

## Ownership rule

**`worker.domains` must not import `worker.pipeline.output` or `worker.runtime`.**
A domain never opens a stream, never sends a relay request, never writes a clip, and never learns how it was scheduled.
It receives `DecisionInput` and returns `BusinessEvent` values. The pipeline decides delivery.

Allowed: `contracts`, `shared.detection_policies`, `worker.types`, `worker.interfaces`, `worker.pipeline.perception`.
Forbidden I/O and composition: ingest, output, runtime. Import-linter contracts *"worker domains do not depend on ingest, output, or runtime"* and *"worker runtime is the sole composition root"* enforce this.

## Local ownership

- `base.py`: `Decider` alias, `DomainDetector`, audit snapshot types.
- `registry.py` / `__init__.py`: `DetectionModuleDefinition`, `DOMAIN_REGISTRY`, enabled-domain helpers.
- `fall/`: window classifier, per-track pose buffers, probability, rising-edge latch.
- `bed_exit/`: sticky own-bed assignment, containment, grace/hold counters, night window, debug snapshot.

## Numeric input

`DecisionInput` is numeric: observation, frame size, live track ids, time, frame index, bed region.
No array, no buffer, no frame handle. A detector that needs pixels is a design error.
Extract the number in `worker/pipeline/perception` and add it to the observation first.

## Per-camera temporal state

Classifier buffers, per-track windows, fall probability, fall latch, bed assignments, and grace/hold counters stay per camera.
The fall model object is shared once per process. Hoisting a per-camera row leaks one resident into another.
`coast()` holds last-known state across a missing person inference. It emits nothing.

## Explicit policies

Fall uses `FallPolicyV2.transition_threshold` (fall.policy.v2). Bed-exit uses `BedExitPolicyV1.min_containment`, `hold_frames`, `grace_frames`.
Policies are typed and versioned in `shared.detection_policies`. Unknown documents don't degrade into defaults.
Night-window and cross-midnight behavior take an injected, timezone-aware clock. Don't read wall time.

## Events

`Decider.update(DecisionInput) -> tuple[BusinessEvent, ...]`. Return typed events, never a dict and never a raw frame.
Fall emits `event_type="fall"`. Bed-exit emits `event_type="bed-exit"`.
Rising-edge only: repeated positive frames emit one onset. Downstream delivery is not a domain concern.
Keep event schema changes additive.

## Registry-driven extension

Register an enabled detector through `DOMAIN_REGISTRY` with its input view, event types, debug adapter, and audit metadata provider.
The registry derives production domain configuration and the relay event allowlist. Don't build a parallel allowlist.
Registry-only scope, stated precisely. A judgment module that consumes the existing `DecisionInput` is fully registry-only: register it, enable it through `domains.<name>.enabled`, and its events reach the relay with no runtime, config, or relay edits.
New model or runner work is not registry-only. Fall-model provisioning still assumes a single fall model. Don't describe the registry as covering that.

## Focused tests

- `tests/test_domains_fall.py`, `tests/test_domains_bed_exit.py`
- `tests/test_domain_registry_scaffolds_disabled.py`, `tests/test_worker_domain_registry.py`
- `tests/test_worker_per_camera_fall_state.py`
- `tests/test_worker_domains_bed_exit.py` plus assignment, geometry, and time siblings
- Boundary: `uv run --group lint lint-imports`

## Change boundary

Cover hold, grace, and containment thresholds together.
Prove per-camera isolation whenever detector state changes shape.
