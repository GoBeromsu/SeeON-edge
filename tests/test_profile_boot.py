from __future__ import annotations

import pytest

from worker.runtime.profile.boot import (
    resolve_boot_context,
    resolve_capability_graph_or_raise,
    resolve_decode_or_fallback,
    resolve_encode_or_fallback,
    resolve_profile,
)
from worker.runtime.profile.device import CudaProbe
from worker.runtime.profile.registry import (
    DEFAULT_PROFILE_NAME,
    ML_WORKER_PROFILE_ENV,
    PROFILE_REGISTRY,
    BootDependencies,
    ProfileError,
    ProfileVerifyError,
    VerifyResult,
    default_decode_probe,
    default_encode_probe,
    default_verifiers,
)
from worker.types import (
    ConverterCapabilities,
    FrameCapability,
    MemoryKind,
    PipelineProfile,
    PixelFormat,
    StageCapabilities,
)


def test_profile_boot_rejects_mismatch_unless_named_converter_is_supplied() -> None:
    host_rgb = FrameCapability(MemoryKind.HOST, PixelFormat.RGB24)
    cuda_nv12 = FrameCapability(MemoryKind.CUDA_DEVICE, PixelFormat.NV12)
    pipeline = PipelineProfile(
        name="test-mismatch",
        stages=(
            StageCapabilities("decode", frozenset({cuda_nv12}), cuda_nv12),
            StageCapabilities("inference", frozenset({host_rgb}), host_rgb),
        ),
    )
    spec = PROFILE_REGISTRY["cpu"].with_pipeline(pipeline)

    with pytest.raises(ProfileVerifyError, match="capability graph"):
        resolve_capability_graph_or_raise(spec)

    graph = resolve_capability_graph_or_raise(
        spec,
        converters=(
            ConverterCapabilities(
                "explicit-device-host-materializer",
                cuda_nv12,
                host_rgb,
                True,
            ),
        ),
    )
    assert graph.converter_names == ("explicit-device-host-materializer",)
    assert graph.full_frame_copy_count == 1


def _result(profile: str, *, ok: bool = True, stage: str = "device") -> VerifyResult:
    return VerifyResult(ok=ok, profile=profile, stage=stage, reason="test result")


def _deps(profile: str, *, ok: bool = True) -> BootDependencies:
    return BootDependencies({profile: lambda: _result(profile, ok=ok)})


def _decode_ok(decode: str) -> VerifyResult:
    return VerifyResult(True, "", "decode", f"{decode} available")


@pytest.mark.parametrize("env", [{}, {ML_WORKER_PROFILE_ENV: "  "}])
def test_resolve_profile_missing_env_defaults_to_cpu(env: dict[str, str]) -> None:
    """Issue #133: the worker must boot with zero env vars, so an unset/blank
    ML_WORKER_PROFILE no longer refuses to boot -- it defaults to
    DEFAULT_PROFILE_NAME ("cpu")."""
    spec = resolve_profile(env)
    assert DEFAULT_PROFILE_NAME == "cpu"
    assert spec.name == "cpu-host"
    assert spec == PROFILE_REGISTRY[DEFAULT_PROFILE_NAME]


def test_resolve_profile_unknown_raises() -> None:
    with pytest.raises(ProfileError):
        resolve_profile({ML_WORKER_PROFILE_ENV: "tpu"})


def test_nvidia_profile_verify_true() -> None:
    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "nvidia"}, _deps("nvidia"), _decode_ok
    )

    assert context.device == "cuda"
    assert context.decode == "nvdec"
    assert context.canonical_profile == "nvidia"


def test_nvidia_profile_verify_false() -> None:
    with pytest.raises(ProfileVerifyError):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "nvidia"}, _deps("nvidia", ok=False), _decode_ok
        )


def test_mps_profile_verify_true() -> None:
    context = resolve_boot_context({ML_WORKER_PROFILE_ENV: "mps"}, _deps("mps"), _decode_ok)

    assert context.device == "mps"
    assert context.decode == "opencv"


def test_mps_profile_verify_false() -> None:
    with pytest.raises(ProfileVerifyError):
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "mps"}, _deps("mps", ok=False), _decode_ok)


def test_cpu_profile_ok() -> None:
    context = resolve_boot_context({ML_WORKER_PROFILE_ENV: "cpu"}, _deps("cpu"), _decode_ok)

    assert context.device == "cpu"
    assert context.decode == "opencv"


def test_verifier_exception_becomes_profile_verify_error() -> None:
    def verifier() -> VerifyResult:
        raise RuntimeError("unavailable")

    with pytest.raises(ProfileVerifyError):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "cpu"}, BootDependencies({"cpu": verifier}), _decode_ok
        )


def test_decode_preflight_incompat_raises() -> None:
    def decode_probe(decode: str) -> VerifyResult:
        assert decode == "nvdec"
        return VerifyResult(False, "cuda", "decode", "NVDEC unavailable")

    with pytest.raises(ProfileVerifyError):
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "nvidia"}, _deps("nvidia"), decode_probe)


def test_resolve_boot_context_aggregates_multiple_gate_failures() -> None:
    """Issue #79 (track 2): a bad device *and* an incompatible legacy decode
    override are independent gates over the same profile -- both failures
    must be named in the single raised error instead of only the device
    failure (the first gate checked) being visible."""

    with pytest.raises(ProfileVerifyError) as excinfo:
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "nvidia", "ML_RTSP_BACKEND": "opencv"},
            _deps("nvidia", ok=False),
            _decode_ok,
        )

    message = str(excinfo.value)
    assert "2 boot gate(s) failed for profile 'nvidia'" in message
    assert "device verification failed" in message
    assert "conflicts with profile 'nvidia' decode" in message


def test_legacy_decode_conflict_rejected() -> None:
    with pytest.raises(ProfileVerifyError):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "cpu", "ML_RTSP_BACKEND": "nvdec"},
            _deps("cpu"),
            _decode_ok,
        )


def test_legacy_matching_allowed() -> None:
    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "cpu", "ML_RTSP_BACKEND": "opencv"},
        _deps("cpu"),
        _decode_ok,
    )

    assert context.decode == "opencv"


def test_profile_registry_exact_keys() -> None:
    assert set(PROFILE_REGISTRY) == {
        "cpu-host",
        "intel-vaapi-host",
        "apple-mps-host",
        "nvidia",
        "mps",
        "cpu",
        "igpu",
    }
    assert PROFILE_REGISTRY["nvidia"].device == "cuda"
    assert PROFILE_REGISTRY["nvidia"].decode == "nvdec"
    assert PROFILE_REGISTRY["mps"].device == "mps"
    assert PROFILE_REGISTRY["mps"].decode == "opencv"
    assert PROFILE_REGISTRY["cpu"].device == "cpu"
    assert PROFILE_REGISTRY["cpu"].decode == "opencv"
    # igpu keeps device="cpu" -- only decode moves to the iGPU in this PR;
    # OpenVINO GPU/NPU inference is a separate follow-up.
    assert PROFILE_REGISTRY["igpu"].device == "cpu"
    assert PROFILE_REGISTRY["igpu"].decode == "vaapi"


def _cuda_available() -> CudaProbe:
    return CudaProbe(available=True, reason="cuda available")


def test_nvidia_device_verify_uses_injected_resident_probe() -> None:
    deps = BootDependencies(
        default_verifiers(
            device_resident_source=lambda: VerifyResult(
                True, "nvidia", "device", "device-resident stages available"
            )
        )
    )

    context = resolve_boot_context({ML_WORKER_PROFILE_ENV: "nvidia"}, deps, _decode_ok)

    assert context.device == "cuda"
    assert context.canonical_profile == "nvidia"


def test_default_decode_probe_fails_closed_without_injected_probes() -> None:
    result = default_decode_probe("nvdec")

    assert not result.ok
    assert result.stage == "decode"
    assert result.profile == "cuda"
    assert result.reason == "nvdec capability probe is not configured"


def test_default_decode_probe_fails_closed_for_opencv_without_injected_probes() -> None:
    result = default_decode_probe("opencv")

    assert not result.ok
    assert result.profile == ""
    assert result.reason == "opencv capability probe is not configured"


def test_default_decode_probe_uses_injected_probe_when_configured() -> None:
    probes = {"nvdec": lambda: VerifyResult(True, "cuda", "decode", "nvdec available")}

    result = default_decode_probe("nvdec", probes)

    assert result.ok
    assert result.reason == "nvdec available"


def test_nvidia_profile_fails_closed_when_no_decode_probe_configured() -> None:
    deps = BootDependencies(
        default_verifiers(
            device_resident_source=lambda: VerifyResult(
                True, "nvidia", "device", "device-resident stages available"
            )
        )
    )

    with pytest.raises(ProfileVerifyError, match="nvdec capability probe is not configured"):
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "nvidia"}, deps)


def _nvenc_ok() -> VerifyResult:
    return VerifyResult(True, "cuda", "encode", "h264_nvenc encoder is available")


def _nvenc_unavailable() -> VerifyResult:
    return VerifyResult(False, "cuda", "encode", "ffmpeg has no h264_nvenc encoder")


def test_resolve_encode_or_fallback_keeps_nvenc_when_probe_succeeds() -> None:
    selection = resolve_encode_or_fallback(PROFILE_REGISTRY["nvidia"], _nvenc_ok)

    assert selection.requested == "h264_nvenc"
    assert selection.selected == "h264_nvenc"
    assert selection.fallback_count == 0
    assert selection.last_reason is None


def test_resolve_encode_or_fallback_never_probes_nvidia_without_declared_fallback() -> None:
    def unexpected_probe() -> VerifyResult:
        raise AssertionError("nvidia must not silently fall back from NVENC")

    selection = resolve_encode_or_fallback(PROFILE_REGISTRY["nvidia"], unexpected_probe)

    assert selection.requested == "h264_nvenc"
    assert selection.selected == "h264_nvenc"
    assert selection.fallback_count == 0
    assert selection.last_reason is None


def test_resolve_encode_or_fallback_never_probes_non_nvenc_profiles() -> None:
    def unexpected_probe() -> VerifyResult:
        raise AssertionError("libx264 profiles must never be probed")

    for profile_name in ("mps", "cpu"):
        selection = resolve_encode_or_fallback(PROFILE_REGISTRY[profile_name], unexpected_probe)
        assert selection.requested == "libx264"
        assert selection.selected == "libx264"
        assert selection.fallback_count == 0


def test_default_encode_probe_fails_closed_without_injected_probe() -> None:
    result = default_encode_probe()

    assert not result.ok
    assert result.profile == "cuda"
    assert result.stage == "encode"
    assert result.reason == "h264_nvenc capability probe is not configured"


def test_resolve_boot_context_keeps_nvenc_when_probe_succeeds() -> None:
    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "nvidia"},
        _deps("nvidia"),
        _decode_ok,
        _nvenc_ok,
    )

    assert context.encode == "h264_nvenc"
    assert context.encode_selection is not None
    assert context.encode_selection.fallback_count == 0


def test_resolve_boot_context_never_probes_encode_for_mps_or_cpu_profiles() -> None:
    def unexpected_probe() -> VerifyResult:
        raise AssertionError("mps/cpu profiles must never probe encode")

    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "cpu"},
        _deps("cpu"),
        _decode_ok,
        unexpected_probe,
    )

    assert context.encode == "libx264"


def _vaapi_ok(decode: str) -> VerifyResult:
    assert decode == "vaapi"
    return VerifyResult(True, "igpu", "decode", "VAAPI device init succeeded")


def _vaapi_unavailable(decode: str) -> VerifyResult:
    assert decode == "vaapi"
    return VerifyResult(False, "igpu", "decode", "VAAPI render device not found")


def test_resolve_decode_or_fallback_keeps_vaapi_when_probe_succeeds() -> None:
    selection = resolve_decode_or_fallback(PROFILE_REGISTRY["igpu"], _vaapi_ok)

    assert selection.requested == "vaapi"
    assert selection.selected == "vaapi"
    assert selection.fallback_count == 0
    assert selection.last_reason is None


def test_resolve_decode_or_fallback_never_raises_and_falls_back_to_opencv(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mirrors #53's resolve_encode_or_fallback: unlike nvdec/opencv's
    fail-fast preflight_decode_or_raise, a failed VAAPI probe must never
    abort boot -- it degrades to opencv (CPU) decode with a loud WARNING
    instead (issues #191/#194: a silent no-frames failure is the actual
    footgun, not a logged software-decode fallback)."""
    with caplog.at_level("WARNING"):
        selection = resolve_decode_or_fallback(PROFILE_REGISTRY["igpu"], _vaapi_unavailable)

    assert selection.requested == "vaapi"
    assert selection.selected == "opencv"
    assert selection.fallback_count == 1
    assert selection.last_reason == "vaapi_probe_failed"
    assert any("opencv" in record.message for record in caplog.records)


def test_resolve_decode_or_fallback_swallows_probe_exceptions() -> None:
    def raising_probe(decode: str) -> VerifyResult:
        del decode
        raise RuntimeError("ffmpeg exploded")

    selection = resolve_decode_or_fallback(PROFILE_REGISTRY["igpu"], raising_probe)

    assert selection.selected == "opencv"
    assert selection.fallback_count == 1
    assert selection.last_reason == "vaapi_probe_failed"


def test_resolve_decode_or_fallback_never_probes_non_vaapi_profiles() -> None:
    def unexpected_probe(decode: str) -> VerifyResult:
        del decode
        raise AssertionError("non-vaapi profiles must never be probed here")

    for profile_name in ("nvidia", "mps", "cpu"):
        selection = resolve_decode_or_fallback(PROFILE_REGISTRY[profile_name], unexpected_probe)
        requested = PROFILE_REGISTRY[profile_name].decode
        assert selection.requested == requested
        assert selection.selected == requested
        assert selection.fallback_count == 0


def test_resolve_decode_or_fallback_uses_default_decode_probe_when_none_injected() -> None:
    """Fail-closed default, mirroring default_encode_probe's precedent: no
    injected probe means "capability unknown", never a false positive."""
    selection = resolve_decode_or_fallback(PROFILE_REGISTRY["igpu"], None)

    assert selection.selected == "opencv"
    assert selection.fallback_count == 1


def test_igpu_profile_resolves_to_vaapi_when_probe_succeeds() -> None:
    context = resolve_boot_context({ML_WORKER_PROFILE_ENV: "igpu"}, _deps("igpu"), _vaapi_ok)

    assert context.device == "cpu"
    assert context.decode == "vaapi"
    assert context.decode_selection is not None
    assert context.decode_selection.selected == "vaapi"
    assert context.decode_selection.fallback_count == 0


def test_igpu_profile_falls_back_to_opencv_decode_instead_of_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Deliverable #5: VAAPI unavailable at boot must not crash the worker or
    produce zero frames -- the boot context resolves to opencv decode with a
    logged reason instead of resolve_boot_context raising."""
    with caplog.at_level("WARNING"):
        context = resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "igpu"}, _deps("igpu"), _vaapi_unavailable
        )

    assert context.device == "cpu"
    assert context.decode == "opencv"
    assert context.decode_selection is not None
    assert context.decode_selection.requested == "vaapi"
    assert context.decode_selection.selected == "opencv"
    assert context.decode_selection.last_reason == "vaapi_probe_failed"
    assert any("falling back to opencv" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    ("requested", "canonical"),
    [
        ("cpu", "cpu-host"),
        ("igpu", "intel-vaapi-host"),
        ("mps", "apple-mps-host"),
        ("cpu-host", "cpu-host"),
        ("intel-vaapi-host", "intel-vaapi-host"),
        ("apple-mps-host", "apple-mps-host"),
        ("nvidia", "nvidia"),
    ],
)
def test_profile_aliases_resolve_to_canonical_specs(requested: str, canonical: str) -> None:
    spec = resolve_profile({ML_WORKER_PROFILE_ENV: requested})

    assert spec.name == canonical
    assert requested in spec.accepted_names


def test_nvidia_reports_device_resident_runtime_path() -> None:
    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "nvidia"},
        _deps("nvidia"),
        _decode_ok,
        _nvenc_ok,
    )

    report = context.runtime_profile
    assert context.requested_profile == "nvidia"
    assert context.canonical_profile == "nvidia"
    assert context.profile.name == "nvidia"
    assert report.requested_profile == "nvidia"
    assert report.canonical_profile == "nvidia"
    assert report.requested_decode_backend == "nvdec"
    assert report.effective_decode_backend == "nvdec"
    assert report.requested_preprocess_backend == "cuda-nv12-to-rgb24"
    assert report.effective_preprocess_backend == "cuda-nv12-to-rgb24"
    assert report.requested_inference_backend == "tensorrt"
    assert report.effective_inference_backend == "tensorrt"
    assert report.requested_overlay_backend == "cuda-device"
    assert report.effective_overlay_backend == "cuda-device"
    assert report.requested_encode_backend == "h264_nvenc"
    assert report.effective_encode_backend == "h264_nvenc"
    assert report.memory_path == (
        "cuda-device/nv12",
        "cuda-device/nv12",
        "cuda-device/rgb24",
        "cuda-device/rgb24",
        "cuda-device/rgb24",
    )
    assert report.converter_chain == ("cuda-nv12-to-rgb24",)
    assert report.device_resident_after_decode is True
    assert report.concrete_stages_available is False
    assert report.full_frame_h2d_count == 0
    assert report.full_frame_d2h_count == 0
    assert context.capability_graph.converter_names == report.converter_chain
    assert context.capability_graph.full_frame_copy_count == 0
    assert all(edge.validated for edge in context.capability_graph.edges)
    assert report.degraded_reasons == ()


def test_old_nvidia_names_are_rejected_rather_than_aliased() -> None:
    for rejected in ("cuda", "nvidia-host-bridge", "nvidia-device-experimental"):
        with pytest.raises(ProfileError, match="unknown ML_WORKER_PROFILE"):
            resolve_profile({ML_WORKER_PROFILE_ENV: rejected})


def test_vaapi_profile_reports_host_download_and_fallback_truth() -> None:
    accelerated = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "igpu"}, _deps("igpu"), _vaapi_ok
    ).runtime_profile
    degraded = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "igpu"}, _deps("igpu"), _vaapi_unavailable
    ).runtime_profile

    assert accelerated.canonical_profile == "intel-vaapi-host"
    assert accelerated.memory_path[0:2] == ("host/rgb24", "host/rgb24")
    assert accelerated.converter_chain == ()
    assert accelerated.device_resident_after_decode is False
    assert accelerated.full_frame_h2d_count == 0
    assert accelerated.full_frame_d2h_count == 0
    assert accelerated.degraded_reasons == ()

    assert degraded.requested_decode_backend == "vaapi"
    assert degraded.effective_decode_backend == "opencv"
    assert degraded.memory_path[0:2] == ("host/rgb24", "host/rgb24")
    assert degraded.converter_chain == ()
    assert degraded.full_frame_d2h_count == 0
    assert degraded.degraded_reasons == ("vaapi_probe_failed",)


def test_nvidia_profile_unconfigured_verifier_fails_closed() -> None:
    spec = resolve_profile({ML_WORKER_PROFILE_ENV: "nvidia"})
    assert spec.name == "nvidia"
    assert spec.accepted_names == ("nvidia",)
    assert all(
        resolve_profile({ML_WORKER_PROFILE_ENV: alias}).name != spec.name
        for alias in ("cpu", "igpu", "mps")
    )

    with pytest.raises(ProfileVerifyError, match="no verifier configured"):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "nvidia"},
            BootDependencies({}),
            _decode_ok,
            _nvenc_ok,
        )


def test_nvidia_profile_fails_closed_on_negative_device_resident_probe() -> None:
    deps = BootDependencies(
        default_verifiers(
            device_resident_source=lambda: VerifyResult(
                False,
                "nvidia",
                "device",
                "no CUDA stream/event support on this host",
            )
        )
    )
    with pytest.raises(ProfileVerifyError, match="no CUDA stream/event support"):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "nvidia"}, deps, _decode_ok, _nvenc_ok
        )


def test_nvidia_profile_boots_once_device_resident_probe_is_positive() -> None:
    deps = BootDependencies(
        default_verifiers(
            device_resident_source=lambda: VerifyResult(
                True, "nvidia", "device", "device-resident stages available"
            )
        )
    )
    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "nvidia"}, deps, _decode_ok, _nvenc_ok
    )
    assert context.canonical_profile == "nvidia"
    assert context.runtime_profile.device_resident_after_decode is True
    assert context.runtime_profile.concrete_stages_available is False


def test_nvidia_plain_cuda_probe_never_satisfies_device_resident_gate() -> None:
    deps = BootDependencies(default_verifiers(cuda_source=_cuda_available))
    with pytest.raises(ProfileVerifyError, match="device-resident capability probe"):
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "nvidia"}, deps, _decode_ok, _nvenc_ok)


def test_igpu_profile_device_verify_false_still_reports_decode_gate() -> None:
    """Issue #79 (track 2) parity: igpu's device check failing must not skip
    running the (non-raising) decode resolution -- both are independent
    gates, same as every other profile."""
    decode_calls: list[str] = []

    def decode_probe(decode: str) -> VerifyResult:
        decode_calls.append(decode)
        return _vaapi_ok(decode)

    with pytest.raises(ProfileVerifyError, match="intel-vaapi-host"):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "igpu"}, _deps("igpu", ok=False), decode_probe
        )

    assert decode_calls == ["vaapi"]
