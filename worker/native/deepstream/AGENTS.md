# worker/native/deepstream: supervised NVIDIA child

Native C++ media/inference for the `nvidia` profile plus the Python-owned
cross-language contract. The pinned DeepStream image builds the binaries;
CPU, MPS, and iGPU profiles do not run them.

## Ownership

- `src/` owns per-source RTSP/depay/parser and the encoded-AU tee before decode;
  the decode branch owns `nvv4l2decoder`/NVMM, custom TensorRT inference, tensor
  parse, inverse geometry, association, and bounded preview output. Production
  uses custom `TrtPerception`, not `nvinfer` or `nvtracker`;
  `nvstreammux batch-size=1` appears only in the preflight warmup.
- Python modules here own control/IPC codecs, metadata admission, manifest
  preflight, engine-cache identity, and parity helpers. PID-1 supervision and
  source lifecycle belong to `worker/runtime/deepstream/`; C++
  `ChildServer`/`SourceRuntime` execute child commands and source graphs.
- `PerceptionFrameV1` is the image-free boundary. Its identity is worker boot,
  camera, stream epoch, sequence, and optional source PTS; independent person,
  pose, and bed channels each report `inferred`, `inferred_empty`, or `skipped`.
  Raw tensors and host frames do not cross this boundary.
- The child alone receives `CUDA_VISIBLE_DEVICES=0`; the Python parent must not
  create a second CUDA context.

## Association and media invariants

`association/registry.py` enables only `legacy-greedy-bbox-iou.v1`.
`pose-aware-bbox-iou.v1` is registered but disabled. Keep source-order pose
rows, the legacy score threshold/integer conversion, and the no-second-NMS
rule unless a separate behavior change is approved.

The encoded-AU tee is source-primary and precedes decode. Primary evidence
remuxes that stream; preview JPEGs are bounded derivatives.
Do not add stock `nvtracker`, a second encoder on the primary path, or a public
`PerceptionFrameV1` contract.

## Preflight and engines

`run_configured_deepstream_preflight` runs only when
`SEEON_DEEPSTREAM_MANIFEST` is set. It checks GPU compatibility, pinned runtime
and plugin identity, read-only models, native binary identity, both engine-cache
identities, and one bounded loopback inference receipt. Failure writes the
configured first-fault JSON and refuses boot.

Engine plans are content-addressed from weights, exporter, `fp32` precision,
and builder identity under `c7-<plan-key[:32]>`. Deploy builds explicitly;
boot calls `verify_plan_cache` and never builds an engine.

## Commands and focused tests

- Build/verify cache: `uv run python -m worker.native.deepstream.engine_cache build <manifest>` / `uv run python -m worker.native.deepstream.engine_cache verify <manifest>`.
- Standalone preflight: `uv run python -m worker.native.deepstream.preflight <manifest>`.
- Native compile and CTest run in `docker build -f Dockerfile.edge --target runtime .`.
  (`--target` matters: the file's last stage is the CI-only `bootsmoke`.)
- Focused: `uv run pytest -q tests/test_perception_frame_v1.py tests/test_deepstream_full_wire.py tests/test_deepstream_preflight.py tests/test_deepstream_model_parity.py tests/test_native_association_registry.py`.
- `uv run --group lint lint-imports` checks configured contracts; the native
  types-only ceiling is checked by focused dependency tests and manual import
  path review.
