# worker/interfaces

Protocol-only seams. One `typing.Protocol` per swappable capability.
Callers stay on ports. Concrete backends live in `worker/adapters`.

## Ownership rule

`worker.interfaces` imports only the standard library, `contracts`, and `worker.types`.
No concrete logic, I/O, model parse, or vendor import.
A Protocol body is `...`. Behavior belongs in adapters or pipeline.
Import-linter: *"worker.interfaces imports only contracts and worker.types from the internal graph"* and *"worker runtime is the sole composition root"*.
Forbidden here: `backend`, `shared`, `worker.adapters`, `worker.pipeline`, `worker.domains`, `worker.runtime`.

## Injection

`worker/runtime` constructs adapters and injects them through these ports.
`pipeline/` and `domains/` take the Protocol, never a concrete class.
Seam defaults are `None`. Missing wiring refuses to start (ADR-0002).
Always-fail stubs belong in tests only.

## Decode, bus, extract, decide

- `decode.py`: `DecodeAdapter.open(config) -> DecodeSession`. `read() -> FramePacket | None`, `close()`. Optional `StreamIdentityDecodeSession.set_stream_identity`.
- `bus.py`: named, bounded, per-camera `FrameBus.subscribe` / `publish`. `publish` consumes the caller's lease. `FrameSubscription.take` / `close`.
- `extract.py`: `Extractor.extract(FramePacket) -> ModuleResult`.
- `decision.py`: `Decider.update(DecisionInput) -> tuple[BusinessEvent, ...]`. No pixels.
- `perception.py`: `PerceptionFrameAdapter` adapts capability outputs or compact
  payloads into worker-internal `PerceptionFrameV1` envelopes or typed failures.
  This C1 boundary is not a `contracts` type or a backend/public wire.

## Serving

- `serving.py`: `ServingClient.create(task, ...) -> RunnerProtocol`.
- `BatchServingClient.infer_batch(task, frames) -> tuple[RunnerResult, ...]`. Batch-input contract so a future networked serving service can swap in. That swap is deferred (ADR-0002). Do not land Triton or HTTP serving here.
- `BatchServingProvider.batch_serving_client` exposes a model-sharing batch view. A wrapper without it silently downgrades.

## Encode, frame, output

- `encode.py`: `ClipEncoder.open(camera, profile, geometry) -> EncoderSession`. `write(FramePacket)`, `close()`. `ClipFinalizer.finalize(segments, event) -> artifact`. Keep this wide enough for a future packet-copy adapter.
- `DeviceInputEncoder.submit(FrameLease)` is device-resident only. No silent host readback. Host frames go through a named `FrameMaterializer` first.
- `frame.py`: `FrameMaterializer.materialize(FrameLease) -> FrameLease`. `HostFrameView.view` is host-only and must fail on device input.
- `output.py`: `EventSink.emit(BusinessEvent)`.
- `render.py`: `OverlaySceneRenderer.render_scene(packet, OverlayScene)`. Hardware seam. No domain decisions.
- `source_packet.py`: `SourcePacketSink.append(SourcePacket)`. `EpochRollingSourcePacketSink.roll_epoch`.
- `thumbnail.py`: `ThumbnailGenerator.generate(video, thumb, duration) -> Path`.

## Conventions

`@runtime_checkable` when a test asserts substitutability.
Name the capability, not the vendor: `DecodeAdapter`, not `OpenCVAdapter`.
Every seam needs two implementations, or one plus a test double.
Never leak FFmpeg, cv2, CUDA, or torch types into a signature.
Adding a Protocol is cheap. Changing an existing signature is not.
Prefer a new Protocol over a breaking change.
Keep `__init__.__all__` honest with the public ports.

## Forbidden

Vendor kits stay out: OpenCV, FFmpeg, CUDA, NVDEC, NVENC, Ultralytics, sklearn, torch.
Skip default implementations and production `NotImplementedError`.
Keep filesystem, HTTP, and subprocess out of this package.
Profile policy, camera identity, schedule, and fallback branches belong in runtime.
Implicit host-device transfer is a leak; name a materializer instead.

## Tests

- `tests/test_worker_interfaces.py`: checkable fakes, envelope types, one-caller swap.
- `tests/test_perception_frame_v1.py`: `PerceptionFrameAdapter` exports,
  signature, substitutability, and worker-internal envelope behavior.
- `tests/test_import_dependency_ladder.py` and `uv run --group lint lint-imports`.
- `tests/test_serving_batch_client.py`, `tests/test_worker_model_serving.py`: batch vs single-frame.
- `tests/test_nvidia_device_resident_prototype.py`: surviving experimental adapter prototypes.
