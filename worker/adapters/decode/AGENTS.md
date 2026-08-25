# worker/adapters/decode: RTSP and remux ingest backends

Own every vendor decode path behind `DecodeAdapter` / `DecodeSession`.
Runtime picks the backend. This package executes it. No `pipeline`,
`domains`, or `runtime` import. Need a sink, clock, or spawner?
Constructor argument. Vendor kits live here: OpenCV, PyAV, ffmpeg.

## Local map

- `cpu_av/`: OpenCV `CAP_FFMPEG` RTSP. `cpu-host` / `apple-mps-host`. Never CUDA.
- `vaapi/`: iGPU ffmpeg child, host rgb24. OpenCV demotion is a runtime profile call, not an adapter probe.
- `nvdec_cuvid/`: fail-loud NVDEC/CUVID. FFprobe, `*_cuvid` codec, bounded stdout queue, child reap.
- `pyav_*.py`: packet-preserving demux and NVDEC tee. Remux path only.
- `nvdec_device/`: experimental device-resident pool and diagnostics. No current production profile constructs it; `nvidia` uses the native DeepStream child instead.
- `native_au_receiver.py`: bounded child AU stream into the shared
  `PacketRingRepository`; retires camera generations and escalates stream death.
- `native_preview_receiver.py`: bounded child JPEG stream into `LatestFrameStore`.

## Port and backend selection

`open(config) -> DecodeSession`. Same `FramePacket` metrics on every decoder: non-negative `decode_time_ms`, monotonic `seq`, real width/height, stream time. Preserving sessions also implement `StreamIdentityDecodeSession`. Don't leak cv2, av, or ffmpeg through the port.

Runtime profile owns `(device, decode, encode)`. Adapters do not probe onto another backend. `decoder_for` lives in runtime. Packet sink selects `PyAvPreservingAdapter`. No sink selects `CpuAvAdapter`, `NvdecCuvidAdapter`, or `VaapiAdapter`. Unknown token raises. `decode_backend='nvdec'` on a non-nvdec profile raises. A new backend is a new package behind the existing port, registered by runtime. Not a branch inside an adapter.

## CPU / PyAV

`CpuAvAdapter` opens exactly `(url, CAP_FFMPEG, params)`. No `TypeError` retry that drops timeouts (ADR-0003). Timeouts travel as constructor params. No post-open `set()`. `probe_opencv_ffmpeg_capability` requires import plus `videoio_registry.hasBackend(CAP_FFMPEG)`. Import success is not capability. Without a packet sink, CPU stays on OpenCV. With a sink, PyAV decodes in-process.

## VAAPI

`VaapiAdapter` spawns `ffmpeg -hwaccel vaapi` and downloads rgb24 to host. Probe checks the render node exists, then `-init_hw_device vaapi=va:<device>`. Grepping `-hwaccels` is not enough. Adapter never demotes to OpenCV. Runtime may call `resolve_decode_or_fallback` at boot. PyAV VAAPI uses `HWAccel(..., allow_software_fallback=False)`.

## NVDEC / CUVID

`NvdecCuvidAdapter` probes metadata, maps codec via `cuvid_decoder_for`, then spawns `ffmpeg -hwaccel cuda -c:v <codec>_cuvid`. Unknown codec raises `UnsupportedCodecError`. Spawn or probe failure is `NvdecUnavailableError`. No silent OpenCV path. Capability probe answers the ffmpeg build (`cuda`/`cuvid`/`nvdec` or a `*_cuvid` decoder), not GPU presence. Stdout queue capacity is 2. Incomplete rgb24 closes the session and raises `NvdecReadError`.

## Packet preservation and epochs

Primary clips remux source packets. Decoded frames are analysis and snapshot taps. `PyAvPacketDemuxer` publishes byte-identical compressed packets to `SourcePacketSink`. Exactly one video stream. A full ring increments `packet_drop_count`. Never echo a credentialed URL. Under `nvidia`, `NativeAuReceiver` feeds the same packet repository from the child's pre-decode AU tee and `NativePreviewReceiver` feeds the live-view store; no per-camera FFmpeg runs on that path.

NVDEC preservation requires `EpochRollingSourcePacketSink` and opens `NvdecPacketTeeSession`: demux in-process, all video decode in one ffmpeg child on stdin. `DecoderInputQueue` is capacity 32. Overflow discards backlog and resumes at a keyframe.

`set_stream_identity(worker_boot_id, stream_epoch)` assigns `StreamEpoch` once, then starts demux. Preserving packets carry `worker_boot_id`, `stream_epoch`, `source_pts`, `source_dts`, `source_time_base`. NVDEC tee calls `roll_epoch` on identity and on configuration change, then reaps and respawns the decoder. Identity missing or assigned twice raises.

## Resource closure, no silent fallback

`close()` is idempotent and reaps every child. NVDEC and VAAPI terminate, then kill, then join the stdout reader (5s). Tee sessions also abort the input queue and join the writer and demux thread. PyAV preserving joins the demux thread and closes the container. Open failure closes the container before re-raise. Sanitize errors. Report camera id and exception class. Never format the RTSP URL.

Fail closed. OpenCV needs `CAP_FFMPEG` in the videoio registry. NVDEC preflight fails loud. VAAPI OpenCV demotion is a named runtime decision, logged once at boot. Adapters do not implement it. `available=True` is this process. Warmup and ingest own camera reachability.

## Focused tests

`tests/test_worker_decode_cpu.py`, `tests/test_worker_opencv_decode_probe.py`, `tests/test_worker_nvdec_adapter.py`, `tests/test_worker_nvdec_probe.py`, `tests/test_worker_nvdec_process.py`, `tests/test_worker_vaapi_adapter.py`, `tests/test_worker_vaapi_probe.py`, `tests/test_decode_seam_selection_matrix.py`, `tests/test_decode_seam_packet_tee.py`, `tests/test_decode_seam_epoch_roll.py`, `tests/test_decode_seam_nvdec_subprocess.py`, `tests/test_pyav_preserving_nvdec_tee.py`, `tests/test_native_au_receiver.py`, `tests/test_native_preview_receiver.py`, `tests/test_packet_ring.py`. Boundary: `uv run --group lint lint-imports`.

Default tests stay hardware-free. Assert probe invariants and selection, not "this machine has no GPU".
