# worker/runtime/deepstream: native-child supervision

Python PID-1 composition for the `nvidia` profile. Own one native child, IPC,
containment, restart-based source lifecycle, and CPU-only temporal policy over
native metadata. GPU work stays in the child; Python-owned temporal/domain
policy remains in this process.

## Composition

`NvidiaMediaPlane` owns exactly one `DeepStreamChildSupervisor` and projects its
packet and preview stores into existing worker surfaces. `spawn_child` creates
control/failure SEQPACKET sockets, AU/preview STREAM sockets, a wake socket, and
identity/ready pipes. Any spawn failure closes all sockets and descriptors.
Only the child environment receives `CUDA_VISIBLE_DEVICES=0`.

`DeepStreamChildSupervisor.start` validates private paths, acquires the GPU
lease when bootstrap does not own it, spawns the child, connects IPC, and starts
metadata, AU, preview, failure, and exit monitors. A non-zero child exit raises
`ChildFatalError` as exit code 4; `NvidiaMediaPlane` calls the process fatal-exit
hook with 4.

## Source lifecycle

`DarkSourceController` serializes `add`, `rebuild`, and `remove`. A source is
ready only after an accepted native frame has positive source geometry/time,
all three channels are not `SKIPPED`, and association is present. The timeout
is 10 seconds. Failed add rolls back and tombstones; failed rebuild degrades;
an AU gap requests rebuild and an unrecoverable rebuild is fatal.

States are `ABSENT`, `ADDING`, `STARTING`, `SOURCE_READY`, `REBUILDING`,
`DEGRADED`, `REMOVING`, and `TOMBSTONED`. A control acknowledgement alone never
establishes readiness.

## Policy and evidence

One `NativePolicyPump` is created per activated camera. It converts image-free
metadata into the existing observation, `SceneState`, decision, and event paths.
It requires complete geometry and an association assignment for every person
cue. It creates `NativeEvidenceTrigger` and calls
`AlertEvidenceAttacher.attach_native`. It executes Python-owned
`decision.update`, including the configured CPU fall LSTM and temporal policy.
It never invokes a Python tracker on the `nvidia` runtime path, although
preflighted `CameraDetectionPlan` objects still contain an unused tracker.

`NativeAuReceiver` feeds the shared `PacketRingRepository` and
requests a source rebuild on AU discontinuity. `NativePreviewReceiver` feeds
`LatestFrameStore`; viewer demand and explicit snapshot requests travel through
the child control channel, and returned JPEGs reuse that store. Do not add a
second delivery queue, per-camera FFmpeg, host-frame IPC, or GPU work to the parent.

## Commands and focused tests

- Production composition: `python -m worker` with `ML_WORKER_PROFILE=nvidia`.
- Isolated QA/dark runner: `python -m worker.runtime.deepstream --state-dir <dir>`; it supervises one child outside normal `WorkerRuntime` composition.
- Focused: `uv run pytest -q tests/test_deepstream_child_supervisor.py tests/test_deepstream_dark_runner.py tests/test_nvidia_native_policy_pump.py tests/test_deepstream_review_regressions.py`.
- Boundary: `uv run --group lint lint-imports`.
