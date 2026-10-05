# worker/adapters/deepstream

This is the sole worker package allowed to import DeepStream vendor modules.
Keep those imports lazy so host tests import this package without an NVIDIA
runtime. Convert vendor metadata immediately into worker envelopes; callers use
only `worker.interfaces` and `worker.types`.

## Where to look

| File | Role |
| --- | --- |
| `service_maker.py` | 742 LOC hotspot. `DeepStreamMediaPlane(MediaPlane)`: start/stop, sources, snapshot, Smart Record; `DeepStreamMediaPlaneConfig`, `FlowFactory`, `DeepStreamFlowStopTimeout` |
| `metadata.py` | `convert_frame`, `association_pass`: Flow batch metadata to `PerceptionFrameV1` plus association |
| `sources.py` | `SourceTable`: stable camera identities and Flow source names |
| `tensor_rows.py` | host copies of the fixed-shape pose tensor: `rows_from_tensor`, `host_array_from_tensor`, `CudaRuntime` |
| `configs/nvinfer-yolo26-pose.txt` | nvinfer config: ONNX and engine paths, custom parser lib, `output-tensor-meta=1` |
| `configs/config_tracker_NvDCF_{accuracy,perf}.yml`, `configs/labels.txt` | NvDCF tracker profiles and labels |
| `native_parser/yolo26_pose_parser.cpp` | C++ parser `NvDsInferParseCustomYolo26Pose`, shipped as `libnvdsinfer_custom_yolo26_pose.so` |

## Conventions

- `pyservicemaker` is imported inside functions in `service_maker.py`. The
  `BatchMetadataOperator` subclass is built lazily for the same reason.
- A Flow fixes its sources when it is built. The port carries that limit as
  `SourceRosterFixed`.
- The SDK cannot stop a Smart Record session early
  (`docs/research/pyservicemaker-p1b-spike.md`); `stop_recording` says so
  instead of pretending.
- Tracker config and library paths arrive through `DeepStreamMediaPlaneConfig`.
  nvinfer paths (`/app/models/pose/...`, `/var/cache/seeon/tensorrt/...`,
  `/opt/seeon/deepstream-flow/...`) are image-owned.
- The tracker's identities are final. The worker consumes them and never
  re-tracks.

## Anti-patterns

- Module-level `import pyservicemaker` or `import pyds`.
- Returning `NvDs*` objects or tensors that alias SDK memory. Copy to host rows.
- Renaming the parser function or output blob without the matching change in
  `nvinfer-yolo26-pose.txt` and the engine build.
- Hardcoding a tracker or engine path in Python.

More tests: `tests/test_flow_single_cuda_context.py`,
`tests/test_flow_live_frames.py`, `tests/test_worker_no_torch_import_surface.py`.
