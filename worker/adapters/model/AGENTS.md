# worker/adapters/model: model runners and task registry

Concrete model execution behind `worker.interfaces.serving` and
`worker.interfaces.fall_model`. Earned its own file: 23 modules, 46 external
importing files, its own artifact-verification rules (score 12, distinct domain).

## Where to look

| Task | File | Notes |
| --- | --- | --- |
| Task -> runner registry | `registry.py` | `ModelRegistry`, `default_registry()` (pose/person/bed, ultralytics), `flow_registry()` (bed only, ORT) |
| Serving seam impl | `in_process.py` | `InProcessServingClient`, `InProcessBatchServingClient` |
| Error vocabulary | `errors.py` | `ModelLoadError`, `ModelInputError`, `FatalAcceleratorError(camera_id, task)` |
| Fall bundle (pose+bbox56) | `ort_pose_bbox56.py` | 534 LOC hotspot: `load_packaged_fall_bundle`, `OrtPoseBbox56Runner`, conformance |
| Bundle verification | `pose_bbox56_bundle_support.py` | `verify_bundle`, `member_digest`; also used by `tools/export_fall_onnx.py` |
| Fall model families | `fall_family_registry.py` | `FallModelFamilyRegistry`; `gru_manifest.py`, `sklearn_fall.py`, `sklearn_metadata.py` |
| Bed segmentation | `ort_bed_seg.py`, `seg_postprocess.py` | CPU ORT runner plus numpy decode; `yolo_bed_seg.py` is the ultralytics twin |
| Ultralytics wrappers | `yolo_api.py`, `yolo_batch.py` | typed facade used by `yolo_pose.py`, `yolo_person.py` |
| Stored-clip re-analysis | `clip_reanalysis.py`, `ort_clip_pose.py` | bounded CPU analysis of one sealed clip |
| Artifact digests | `artifact.py` | sidecar read, digest, `verify_artifact_digest` |
| Batched input checks | `batch_input.py` | `validated_batch_images`, fail-closed per row |
| Warmup | `warmup.py` | `warmup_to_ready`, `synthetic_rgb_frame` |

## Conventions

- Package `__init__` exports nothing. Import from the concrete module.
- Registry factories import their runner inside the factory function. Under the
  flow profile the process asserts torch and ultralytics are never imported
  (P1b-AC7), so a module-level runner import in `registry.py` breaks boot.
- `"fall"` is deliberately absent from both registries. The fall model has no
  registry fallback; `WorkerRuntime._create_fall_model` is the fail-closed owner.
- `flow_registry()` omits `"pose"` and `"person"`: the media plane detects.
  Its `"bed"` is the ORT CPU segmenter.
- Adapters cannot import the runtime `FallModelConfig`. Family factories take
  the local structural `FallModelConfigLike` Protocol; unknown family types
  raise `UnknownFallModelTypeError`.
- `FatalAcceleratorError` means the GPU context is dead: never reuse or recreate
  it in-process. Runtime writes one first-fault record and exits 4.

## Anti-patterns

- Loading an artifact before its digest or bundle membership is verified.
- Registering a `"fall"` task, or a host pose/person runner in `flow_registry()`.
- Catching `FatalAcceleratorError` to retry a forward pass.
- Letting one bad row reach a batched forward: validate with
  `validated_batch_images` first (ADR-0002 fail-fast).
- Importing `torch`, `ultralytics`, or `onnxruntime` at module scope in a module
  the flow profile imports.

## Focused tests

`tests/test_runners_registry.py`, `tests/test_worker_model_serving.py`,
`tests/test_serving_batch_client.py`, `tests/test_ort_pose_bbox56_runner.py`,
`tests/test_ort_bed_seg_runner.py`, `tests/test_worker_yolo_adapters.py`,
`tests/test_worker_fall_adapters.py`, `tests/test_worker_fall_model_selection.py`,
`tests/test_worker_no_torch_import_surface.py`.
