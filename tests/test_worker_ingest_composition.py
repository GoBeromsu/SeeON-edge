from __future__ import annotations

from typing import final

import pytest

import worker.runtime.ingest_composition as ingest_composition_module
import worker.runtime.worker as worker_module
from worker.adapters.device.cuda.probe import CudaCapability
from worker.adapters.device.nvml.probe import NvmlGpuStatus
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import CapturePolicy, IngestEvent
from worker.pipeline.ingest.registry import SourceRegistryError
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig, WorkerRuntimeConfig
from worker.runtime.ingest_composition import (
    CpuAvConfig,
    NvdecCuvidConfig,
    VaapiConfig,
    build_camera_source_registry,
    compose_camera_ingest_loop,
    decoder_for,
)
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import PROFILE_REGISTRY
from worker.runtime.worker import WorkerRuntime
from worker.types.source_packet import SourcePacket


@final
class _FakeServingClient:
    def create(self, task: str, **_options: object) -> object:
        raise AssertionError(f"ingest composition tests must not create a serving model: {task}")


@final
class _PacketSink:
    def append(self, packet: SourcePacket) -> bool:
        del packet
        return True


@final
class _Reporter:
    def mark_starting(self, camera_id: str) -> None:
        del camera_id

    def mark_ready(self, camera_id: str) -> None:
        del camera_id

    def mark_degraded(self, camera_id: str, *, category: str) -> None:
        del camera_id, category

    def emit(self, event: IngestEvent) -> None:
        del event


def _config(*camera_ids: str) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": camera_id,
                    "facility_id": f"facility-{camera_id.removeprefix('camera-')}",
                    "rtsp_url": f"rtsp://8.8.8.8/{camera_id}",
                    "heartbeat_interval_sec": 30.0,
                }
                for camera_id in camera_ids
            ],
        }
    )


def _camera(camera_id: str = "camera-a") -> CameraRuntimeConfig:
    return _config(camera_id).cameras[0]


def _boot_context_for(profile_name: str) -> BootContext:
    spec = PROFILE_REGISTRY[profile_name]
    return BootContext(profile=spec, device=spec.device, decode=spec.decode, encode=spec.encode)


def _forbid(name: str) -> object:
    def raise_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"{name} must not be constructed for this decode token")

    return raise_if_called


# -- decoder_for: the fail-fast selection seam --------------------------------


def test_decoder_for_opencv_token_constructs_only_the_cpu_av_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[str] = []

    def fake_cpu_av_adapter() -> object:
        calls.append("cpu_av")
        return sentinel

    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", fake_cpu_av_adapter)
    monkeypatch.setattr(
        ingest_composition_module, "NvdecCuvidAdapter", _forbid("NvdecCuvidAdapter")
    )

    result = decoder_for("opencv")

    assert result is sentinel
    assert calls == ["cpu_av"]


def test_decoder_for_nvdec_token_constructs_only_the_nvdec_cuvid_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[str] = []

    def fake_nvdec_adapter() -> object:
        calls.append("nvdec")
        return sentinel

    monkeypatch.setattr(ingest_composition_module, "NvdecCuvidAdapter", fake_nvdec_adapter)
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", _forbid("CpuAvAdapter"))

    result = decoder_for("nvdec")

    assert result is sentinel
    assert calls == ["nvdec"]


def test_decoder_for_nvdec_with_a_packet_sink_selects_the_tee_preserving_adapter() -> None:
    # Todo 4 moved preservation + NVDEC to PyAvPreservingAdapter's demux-only
    # NvdecPacketTeeSession branch. A sink-free NVDEC source above still uses
    # NvdecCuvidAdapter; this deliberate distinction must not regress.
    adapter = decoder_for("nvdec", packet_sink=_PacketSink())

    assert type(adapter) is ingest_composition_module.PyAvPreservingAdapter
    assert adapter._decode_backend == "nvdec"  # noqa: SLF001 - selection seam


def test_decoder_for_vaapi_token_constructs_only_the_vaapi_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[str] = []

    def fake_vaapi_adapter() -> object:
        calls.append("vaapi")
        return sentinel

    monkeypatch.setattr(ingest_composition_module, "VaapiAdapter", fake_vaapi_adapter)
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", _forbid("CpuAvAdapter"))
    monkeypatch.setattr(
        ingest_composition_module, "NvdecCuvidAdapter", _forbid("NvdecCuvidAdapter")
    )

    result = decoder_for("vaapi")

    assert result is sentinel
    assert calls == ["vaapi"]


def test_decoder_for_unknown_token_raises_without_constructing_any_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", _forbid("CpuAvAdapter"))
    monkeypatch.setattr(
        ingest_composition_module, "NvdecCuvidAdapter", _forbid("NvdecCuvidAdapter")
    )
    monkeypatch.setattr(ingest_composition_module, "VaapiAdapter", _forbid("VaapiAdapter"))

    with pytest.raises(RuntimeError, match="unsupported decode policy"):
        decoder_for("mystery")  # type: ignore[arg-type]


# -- decoder_for: per-camera decode_backend override (issue #42) --------------


@pytest.mark.parametrize("override", [None, "auto"])
def test_decoder_for_none_or_auto_override_defers_to_the_profile_token(
    monkeypatch: pytest.MonkeyPatch, override: str | None
) -> None:
    sentinel = object()
    monkeypatch.setattr(ingest_composition_module, "NvdecCuvidAdapter", lambda: sentinel)
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", _forbid("CpuAvAdapter"))

    result = decoder_for("nvdec", override)

    assert result is sentinel


@pytest.mark.parametrize("override", ["opencv", "cpu"])
def test_decoder_for_cpu_override_wins_over_an_nvdec_profile(
    monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    # A CPU-decode override is always compatible -- it never needs the GPU
    # the profile probed for, so it may downgrade an "nvdec" profile freely.
    sentinel = object()
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", lambda: sentinel)
    monkeypatch.setattr(
        ingest_composition_module, "NvdecCuvidAdapter", _forbid("NvdecCuvidAdapter")
    )

    result = decoder_for("nvdec", override)

    assert result is sentinel


def test_decoder_for_nvdec_override_matching_an_nvdec_profile_is_not_a_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(ingest_composition_module, "NvdecCuvidAdapter", lambda: sentinel)
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", _forbid("CpuAvAdapter"))

    result = decoder_for("nvdec", "nvdec")

    assert result is sentinel


def test_decoder_for_nvdec_override_on_a_non_nvdec_profile_raises_without_constructing_any_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fail-fast per ADR-0002: the boot profile never verified an NVDEC
    # device, so silently falling back to CPU decode would hide a
    # misconfiguration instead of surfacing it at composition time.
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", _forbid("CpuAvAdapter"))
    monkeypatch.setattr(
        ingest_composition_module, "NvdecCuvidAdapter", _forbid("NvdecCuvidAdapter")
    )

    with pytest.raises(RuntimeError, match="nvdec"):
        decoder_for("opencv", "nvdec")


# -- compose_camera_ingest_loop: end-to-end composition wiring ----------------


def test_compose_camera_ingest_loop_wires_the_cpu_av_adapter_and_its_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", lambda: sentinel)
    camera = _camera("camera-a")
    registry = build_camera_source_registry((camera,))

    loop = compose_camera_ingest_loop(
        camera, BoundedFrameBus(), _Reporter(), decode="opencv", registry=registry
    )

    assert loop.camera_id == "camera-a"
    assert loop._ports.decoder is sentinel  # noqa: SLF001 - composition wiring under test
    assert loop._ports.registry is registry  # noqa: SLF001
    resolved = registry.resolve(source_id="camera-a")
    config = loop._spec.make_decode_config("camera-a", resolved)  # noqa: SLF001
    assert config == CpuAvConfig(camera_id="camera-a", url=camera.inference_rtsp_url)


def test_compose_camera_ingest_loop_wires_the_nvdec_cuvid_adapter_and_its_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(ingest_composition_module, "NvdecCuvidAdapter", lambda: sentinel)
    camera = _camera("camera-b")
    registry = build_camera_source_registry((camera,))

    loop = compose_camera_ingest_loop(
        camera, BoundedFrameBus(), _Reporter(), decode="nvdec", registry=registry
    )

    assert loop._ports.decoder is sentinel  # noqa: SLF001
    resolved = registry.resolve(source_id="camera-b")
    config = loop._spec.make_decode_config("camera-b", resolved)  # noqa: SLF001
    assert config == NvdecCuvidConfig(camera_id="camera-b", url=camera.inference_rtsp_url)


def test_compose_camera_ingest_loop_wires_the_vaapi_adapter_and_its_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(ingest_composition_module, "VaapiAdapter", lambda: sentinel)
    camera = _camera("camera-c")
    registry = build_camera_source_registry((camera,))

    loop = compose_camera_ingest_loop(
        camera, BoundedFrameBus(), _Reporter(), decode="vaapi", registry=registry
    )

    assert loop._ports.decoder is sentinel  # noqa: SLF001
    resolved = registry.resolve(source_id="camera-c")
    config = loop._spec.make_decode_config("camera-c", resolved)  # noqa: SLF001
    assert config == VaapiConfig(camera_id="camera-c", url=camera.inference_rtsp_url)


def test_compose_camera_ingest_loop_honors_a_per_camera_decode_backend_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Issue #42: camera.decode_backend must win over the profile-global
    # "nvdec" token (downgrading to CPU decode is always compatible), not be
    # silently dropped at composition time.
    sentinel = object()
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", lambda: sentinel)
    monkeypatch.setattr(
        ingest_composition_module, "NvdecCuvidAdapter", _forbid("NvdecCuvidAdapter")
    )
    camera = _camera("camera-a").model_copy(update={"decode_backend": "opencv"})
    registry = build_camera_source_registry((camera,))

    loop = compose_camera_ingest_loop(
        camera, BoundedFrameBus(), _Reporter(), decode="nvdec", registry=registry
    )

    assert loop._ports.decoder is sentinel  # noqa: SLF001
    resolved = registry.resolve(source_id="camera-a")
    config = loop._spec.make_decode_config("camera-a", resolved)  # noqa: SLF001
    assert isinstance(config, CpuAvConfig)


def test_compose_camera_ingest_loop_rejects_an_incompatible_decode_backend_override() -> None:
    # Issue #42: fail-fast, not a silent fallback to the profile's decoder.
    camera = _camera("camera-a").model_copy(update={"decode_backend": "nvdec"})
    registry = build_camera_source_registry((camera,))

    with pytest.raises(RuntimeError, match="nvdec"):
        compose_camera_ingest_loop(
            camera, BoundedFrameBus(), _Reporter(), decode="opencv", registry=registry
        )


@pytest.mark.parametrize("decode", ["opencv", "nvdec"])
def test_compose_camera_ingest_loop_threads_runtime_config_into_capture_policy_and_decode_config(
    decode: str,
) -> None:
    # Issue #69: a non-default WorkerRuntimeConfig must actually reach the
    # constructed CapturePolicy and decode config -- before the fix, ingest
    # composition ignored it entirely and always took the adapter/lifecycle
    # dataclass defaults (max_failures=30, open/read_timeout_ms=5000)
    # regardless of what the yaml configured.
    camera = _camera("camera-a")
    registry = build_camera_source_registry((camera,))
    runtime = WorkerRuntimeConfig(max_failures=7, open_timeout_ms=1234, read_timeout_ms=2345)

    loop = compose_camera_ingest_loop(
        camera,
        BoundedFrameBus(),
        _Reporter(),
        decode=decode,  # type: ignore[arg-type]
        registry=registry,
        runtime=runtime,
    )

    assert loop._spec.policy.max_failures == 7  # noqa: SLF001
    assert loop._spec.policy.max_failures != CapturePolicy().max_failures  # noqa: SLF001
    resolved = registry.resolve(source_id="camera-a")
    config = loop._spec.make_decode_config("camera-a", resolved)  # noqa: SLF001
    assert config.open_timeout_ms == 1234
    assert config.read_timeout_ms == 2345
    assert config.open_timeout_ms != CpuAvConfig(camera_id="x", url="rtsp://x").open_timeout_ms


def test_compose_camera_ingest_loop_without_runtime_falls_back_to_dataclass_defaults() -> None:
    camera = _camera("camera-a")
    registry = build_camera_source_registry((camera,))

    loop = compose_camera_ingest_loop(
        camera, BoundedFrameBus(), _Reporter(), decode="opencv", registry=registry
    )

    assert loop._spec.policy.max_failures == CapturePolicy().max_failures  # noqa: SLF001
    resolved = registry.resolve(source_id="camera-a")
    config = loop._spec.make_decode_config("camera-a", resolved)  # noqa: SLF001
    assert config == CpuAvConfig(camera_id="camera-a", url=camera.inference_rtsp_url)


# -- compose_camera_ingest_loop: fps/frame_stride decoupling (QA fps bug) ----


@pytest.mark.parametrize("frame_stride", [1, 3, 30])
def test_compose_camera_ingest_loop_capture_policy_fps_is_independent_of_frame_stride(
    frame_stride: int,
) -> None:
    # QA measured live-view fps at ~1/3 of the source's 15fps and flagged
    # frame_stride as a possible (unconfirmed) cause. Root cause turned out to
    # be unrelated: camera.fps itself silently defaults to 5.0 when nothing
    # sets it. This test pins down the other half of that finding -- whatever
    # frame_stride is set to must never change the CapturePolicy target_fps
    # that paces ingest (and therefore the live view, which is published for
    # every packet CameraPipelinePump takes off the bus, unconditionally).
    camera = _camera("camera-a").model_copy(update={"fps": 15.0, "frame_stride": frame_stride})
    registry = build_camera_source_registry((camera,))

    loop = compose_camera_ingest_loop(
        camera, BoundedFrameBus(), _Reporter(), decode="opencv", registry=registry
    )

    assert loop._spec.policy.target_fps == 15.0  # noqa: SLF001


def test_compose_camera_ingest_loop_logs_requested_and_actual_decode_backend(
    caplog: pytest.LogCaptureFixture,
) -> None:
    camera = _camera("camera-a")
    registry = build_camera_source_registry((camera,))

    with caplog.at_level("INFO", logger="worker.runtime.ingest_composition"):
        compose_camera_ingest_loop(
            camera,
            BoundedFrameBus(),
            _Reporter(),
            decode="nvdec",
            registry=registry,
            packet_sink=_PacketSink(),
        )

    record = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("camera ingest decode selected:")
    )
    message = record.getMessage()
    assert "camera_id=camera-a" in message
    assert "requested_profile_decode=nvdec" in message
    assert "resolved_backend=nvdec" in message
    assert "actual_adapter_class=PyAvPreservingAdapter" in message


def test_compose_camera_ingest_loop_logs_the_effective_pacing_fps(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The values must land in the *rendered* message text (record.message,
    # i.e. what `%(message)s` actually substitutes), not merely in `extra=`
    # -- worker/__main__.py's console formatter only renders `%(message)s`,
    # so an extra=-only log is invisible there and defeats the point of this
    # log line (seeing the effective fps without reading source). `extra=`
    # stays too, for structured consumers, but is not what this asserts.
    camera = _camera("camera-a").model_copy(update={"fps": 15.0, "frame_stride": 3})
    registry = build_camera_source_registry((camera,))

    with caplog.at_level("INFO", logger="worker.runtime.ingest_composition"):
        compose_camera_ingest_loop(
            camera, BoundedFrameBus(), _Reporter(), decode="opencv", registry=registry
        )

    records = [r for r in caplog.records if r.name == "worker.runtime.ingest_composition"]
    assert records, "expected an INFO log from compose_camera_ingest_loop"
    for record in records:
        record.message = record.getMessage()
    assert any("camera_id=camera-a" in r.message for r in records)
    assert any("target_fps=15.0" in r.message for r in records)
    assert any("frame_stride=3" in r.message for r in records)


def test_build_camera_source_registry_allowlists_only_the_configured_cameras() -> None:
    registry = build_camera_source_registry((_camera("camera-a"),))

    resolved = registry.resolve(source_id="camera-a")

    assert resolved.record.trusted_live is True
    with pytest.raises(SourceRegistryError):
        registry.resolve(source_id="camera-not-configured")


def test_camera_config_and_source_registry_preserve_opaque_identity_bytes() -> None:
    camera_id = " \u2007camera/\u1100\u1161/e\u0301 "
    camera = CameraRuntimeConfig(
        camera_id=camera_id,
        facility_id="facility-1",
        rtsp_url="rtsp://8.8.8.8/stream",
    )

    registry = build_camera_source_registry((camera,))
    resolved = registry.resolve(source_id=camera_id)

    assert camera.camera_id.encode("utf-8") == camera_id.encode("utf-8")
    assert resolved.record.source_id.encode("utf-8") == camera_id.encode("utf-8")
    with pytest.raises(SourceRegistryError, match="unknown source id"):
        registry.resolve(source_id=camera_id.strip())


@pytest.mark.parametrize("camera_id", ["", " \t\u2007 ", "camera\x00id", "camera\nid"])
def test_camera_config_rejects_empty_blank_or_unsafe_identity(camera_id: str) -> None:
    with pytest.raises(ValueError, match="camera_id|blank|unsafe|length"):
        CameraRuntimeConfig(
            camera_id=camera_id,
            facility_id="facility-1",
            rtsp_url="rtsp://8.8.8.8/stream",
        )


# -- WorkerRuntime: the composition-root seam ----------------------------------


def test_worker_runtime_constructs_without_a_loop_factory() -> None:
    # This is the exact construction line worker/__main__.py uses: no loop_factory.
    runtime = WorkerRuntime(_config("camera-a"), serving_client=_FakeServingClient())

    assert isinstance(runtime, WorkerRuntime)


def test_default_loop_factory_selects_the_cpu_av_adapter_for_the_opencv_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", lambda: sentinel)
    monkeypatch.setattr(
        ingest_composition_module, "NvdecCuvidAdapter", _forbid("NvdecCuvidAdapter")
    )
    config = _config("camera-a")
    runtime = WorkerRuntime(config, serving_client=_FakeServingClient())
    runtime._boot = _boot_context_for("cpu")  # noqa: SLF001 - simulate post-model-init state

    loop = runtime._default_loop_factory(config.cameras[0], BoundedFrameBus(), _Reporter())

    assert loop._ports.decoder is sentinel  # noqa: SLF001


def test_default_loop_factory_selects_the_nvdec_cuvid_adapter_for_the_nvdec_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(ingest_composition_module, "NvdecCuvidAdapter", lambda: sentinel)
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", _forbid("CpuAvAdapter"))
    config = _config("camera-a")
    runtime = WorkerRuntime(config, serving_client=_FakeServingClient())
    runtime._boot = _boot_context_for("cuda")  # noqa: SLF001

    loop = runtime._default_loop_factory(config.cameras[0], BoundedFrameBus(), _Reporter())

    assert loop._ports.decoder is sentinel  # noqa: SLF001


def test_default_loop_factory_selects_the_vaapi_adapter_for_the_igpu_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(ingest_composition_module, "VaapiAdapter", lambda: sentinel)
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", _forbid("CpuAvAdapter"))
    monkeypatch.setattr(
        ingest_composition_module, "NvdecCuvidAdapter", _forbid("NvdecCuvidAdapter")
    )
    config = _config("camera-a")
    runtime = WorkerRuntime(config, serving_client=_FakeServingClient())
    runtime._boot = _boot_context_for("igpu")  # noqa: SLF001

    loop = runtime._default_loop_factory(config.cameras[0], BoundedFrameBus(), _Reporter())

    assert loop._ports.decoder is sentinel  # noqa: SLF001


def test_default_loop_factory_records_the_resolved_decode_selection_in_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The resolved decode backend (what `decoder_for`/`_decode_config_factory`
    # actually built the camera's real adapter from) never reached
    # `WorkerDiagnostics`, so `runtime.cameras[*].decode.selected` stayed
    # permanently null in the runtime-status payload no matter what backend
    # was really running.
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", lambda: object())
    monkeypatch.setattr(
        ingest_composition_module, "NvdecCuvidAdapter", _forbid("NvdecCuvidAdapter")
    )
    config = _config("camera-a")
    runtime = WorkerRuntime(config, serving_client=_FakeServingClient())
    runtime._boot = _boot_context_for("cpu")  # noqa: SLF001 - simulate post-model-init state

    _ = runtime._default_loop_factory(  # noqa: SLF001
        config.cameras[0], BoundedFrameBus(), _Reporter()
    )

    selection = runtime.diagnostics.decode_selection("camera-a")
    assert selection is not None
    assert selection.selected == "opencv"


def test_default_loop_factory_preserves_the_prior_requested_and_fallback_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `_build_camera` calls `register_decode` (requested=..., fallback_count=0)
    # before the loop factory ever runs in production. The loop factory's
    # `update_decode` must layer `selected` on top of that record, not
    # overwrite `requested`/`fallback_count` with fresh defaults.
    monkeypatch.setattr(ingest_composition_module, "CpuAvAdapter", lambda: object())
    monkeypatch.setattr(
        ingest_composition_module, "NvdecCuvidAdapter", _forbid("NvdecCuvidAdapter")
    )
    config = _config("camera-a")
    runtime = WorkerRuntime(config, serving_client=_FakeServingClient())
    runtime._boot = _boot_context_for("cpu")  # noqa: SLF001
    runtime.diagnostics.register_decode("camera-a", "auto")
    runtime.diagnostics.record_decode_open_failure("camera-a", "spawn_failed")

    _ = runtime._default_loop_factory(  # noqa: SLF001
        config.cameras[0], BoundedFrameBus(), _Reporter()
    )

    selection = runtime.diagnostics.decode_selection("camera-a")
    assert selection is not None
    assert selection.requested == "auto"
    # `record_decode_open_failure` doesn't bump `fallback_count` (unlike its
    # encode counterpart) -- out of scope here; this only pins that the loop
    # factory's `update_decode` doesn't reset the field back to 0 itself.
    assert selection.fallback_count == 0
    assert selection.last_reason == "spawn_failed"
    assert selection.selected == "opencv"


def test_worker_runtime_init_records_gpu_status_at_boot_via_the_production_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #132: `WorkerDiagnostics.set_gpu_status` existed with zero production
    # callers -- `runtime.device` in `/status` stayed permanently empty no
    # matter what GPU was really attached. Same failure class as #124
    # (`update_decode` never called in production, fixed in 583d02e): the
    # regression to guard against is the producer silently going missing
    # again, not any particular probe value.
    monkeypatch.setattr(
        worker_module,
        "probe_nvml_gpu_status",
        lambda: NvmlGpuStatus(
            nvml_available=True,
            reason="NVML reports a usable GPU device",
            driver_version="550.90.07",
            device_name="Tesla T4",
        ),
    )
    monkeypatch.setattr(
        worker_module,
        "probe_cuda_capability",
        lambda: CudaCapability(
            available=True, reason="cuda available", device_count=1, arch_list=("sm_90",)
        ),
    )
    config = _config("camera-a")

    runtime = WorkerRuntime(config, serving_client=_FakeServingClient())

    gpu = runtime.diagnostics.to_payload("facility-1", None, 0)["gpu"]
    assert gpu["nvml_available"] is True
    assert gpu["cuda_context_ok"] is True
    assert gpu["driver_version"] == "550.90.07"
    assert gpu["device_name"] == "Tesla T4"
    assert gpu["nvml_error"] is None
    assert isinstance(gpu["captured_at_sec"], float)


def test_worker_runtime_init_records_gpu_status_false_when_nvml_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The macOS/no-NVML path: `probe_nvml_gpu_status` fails closed (never
    # raises) rather than leaving `runtime.device` unset, so an operator can
    # still see *why* the GPU section is empty via `nvml_error`.
    monkeypatch.setattr(
        worker_module,
        "probe_nvml_gpu_status",
        lambda: NvmlGpuStatus(
            nvml_available=False,
            reason="nvmlInit failed: NVMLError_LibraryNotFound: NVML Shared Library Not Found",
        ),
    )
    monkeypatch.setattr(
        worker_module,
        "probe_cuda_capability",
        lambda: CudaCapability(available=False, reason="torch import failed: ModuleNotFoundError"),
    )
    config = _config("camera-a")

    runtime = WorkerRuntime(config, serving_client=_FakeServingClient())

    gpu = runtime.diagnostics.to_payload("facility-1", None, 0)["gpu"]
    assert gpu["nvml_available"] is False
    assert gpu["cuda_context_ok"] is False
    assert gpu["driver_version"] is None
    assert gpu["device_name"] is None
    assert gpu["nvml_error"] == (
        "nvmlInit failed: NVMLError_LibraryNotFound: NVML Shared Library Not Found"
    )


def test_default_loop_factory_without_a_resolved_boot_profile_raises() -> None:
    config = _config("camera-a")
    runtime = WorkerRuntime(config, serving_client=_FakeServingClient())

    with pytest.raises(RuntimeError, match="resolved boot profile"):
        runtime._default_loop_factory(  # noqa: SLF001
            config.cameras[0], BoundedFrameBus(), _Reporter()
        )
