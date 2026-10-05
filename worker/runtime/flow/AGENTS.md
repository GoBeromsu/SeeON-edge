# worker/runtime/flow: Flow media-plane runtime

Composition-side half of the DeepStream Flow profile: prebuilt-engine admission,
the process-shared plane wrapper, capacity-one metadata handoff, the per-camera
native policy pump, and Smart Record evidence binding. Earned its own file:
24 external importing files, 20 test files (score 9, distinct domain).

## Where to look

| Task | File | Notes |
| --- | --- | --- |
| Engine admission | `cold_start.py` | `FlowColdStart.run` = `verify_engine_identity` then warmup; `EngineIdentityError`, `FlowWarmupTimeout`, `verify_flow_boot_inputs` |
| ONNX shape | `onnx_shape.py` | `input_dims`, `batch_axis_is_dynamic`; shared with `tools/edge_engine_build.py` and `tools/export_pose_onnx.py` |
| Plane wrapper | `media_plane.py` | `FlowMediaPlane` over the adapter plane; sources, snapshots, `smart_recorder`, `bind_live_frames` |
| Metadata handoff | `metadata_slot.py` | `LatestMetadataSlot`, `AcceptanceToken` |
| Per-camera policy | `policy_pump.py` | 635 LOC hotspot; `NativePolicyPump`, `NativePolicyContext` |
| Input coverage | `observation_coverage.py` | `ObservationCoverage`, gap and recovery records |
| Execution records | `execution_record_emit.py` | `emit_policy_consume`, `emit_model_and_decision`; read-only |
| Evidence binding | `evidence.py` | `FlowEvidenceBinding.emit_for_frame`, `on_sealed`, `replay_sealed` |
| Status tick | `lifecycle_supervisor.py` | `FlowLifecycleSupervisor.tick`, `FlowLifecycleCounters` |

## Conventions

- `verify_engine_identity` checks every digest in the identity file and the
  deployed batch. An absent engine names `edge-engine-build` in the error.
- A pyservicemaker Flow fixes its sources when built, so the plane is not
  started at composition time.
- `LatestMetadataSlot` holds one frame. A newer frame overwrites and counts;
  every rejection has a named counter (`late`, `unknown_source`,
  `generation_mismatch`, `epoch_mismatch`, `boot_mismatch`, `child_mismatch`,
  `transform_mismatch`, `malformed`, `pull_failures`, `pts_missing`).
- The pump declares its seams as local Protocols (`NativeEventSink`,
  `NativeSnapshotControl`, `NativeDiagnostics`). A missing seam refuses at
  wiring time, never mid-stream.
- The pump is image-free: it consumes perception metadata and hands a
  `NativeEvidenceTrigger` to the evidence path.
- Package `__all__` exports eight names (cold start, evidence, plane). Import
  the pump, slot, coverage, and supervisor from their modules.

## Anti-patterns

- Importing `pyservicemaker` or `pyds` here. Go through
  `worker.adapters.deepstream`; import-linter bans the direct import.
- Queueing metadata deeper than one frame.
- Showing a never-scored (warmup) track as "normal" in `preview_states`.
- Dropping an alert because its snapshot failed: degrade to no snapshot.
- Stamping an old-frame execution record with the current frame identity.
- Stamping an alert with another module's decision identity.
- Letting one bad sealed sidecar block recovery of the others.

## Focused tests

`tests/test_flow_cold_start.py`, `tests/test_flow_lifecycle_supervisor.py`,
`tests/test_flow_observation_coverage.py`, `tests/test_flow_policy_pump_preview.py`,
`tests/test_flow_sealed_recovery.py`, `tests/test_flow_live_frames.py`,
`tests/test_flow_single_cuda_context.py`, `tests/test_worker_flow_evidence_binding.py`,
`tests/test_execution_record_wiring.py`.
