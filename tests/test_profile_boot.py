from __future__ import annotations

import pytest

from worker.runtime.profile.boot import (
    resolve_boot_context,
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
    assert spec.name == DEFAULT_PROFILE_NAME
    assert spec == PROFILE_REGISTRY[DEFAULT_PROFILE_NAME]


def test_resolve_profile_unknown_raises() -> None:
    with pytest.raises(ProfileError):
        resolve_profile({ML_WORKER_PROFILE_ENV: "tpu"})


def test_cuda_profile_verify_true() -> None:
    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "cuda"}, _deps("cuda"), _decode_ok
    )

    assert context.device == "cuda"
    assert context.decode == "nvdec"


def test_cuda_profile_verify_false() -> None:
    with pytest.raises(ProfileVerifyError):
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "cuda"}, _deps("cuda", ok=False), _decode_ok)


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
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "cuda"}, _deps("cuda"), decode_probe)


def test_resolve_boot_context_aggregates_multiple_gate_failures() -> None:
    """Issue #79 (track 2): a bad device *and* an incompatible legacy decode
    override are independent gates over the same profile -- both failures
    must be named in the single raised error instead of only the device
    failure (the first gate checked) being visible."""

    with pytest.raises(ProfileVerifyError) as excinfo:
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "cuda", "ML_RTSP_BACKEND": "opencv"},
            _deps("cuda", ok=False),
            _decode_ok,
        )

    message = str(excinfo.value)
    assert "2 boot gate(s) failed for profile 'cuda'" in message
    assert "device verification failed" in message
    assert "conflicts with profile 'cuda' decode" in message


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
    assert set(PROFILE_REGISTRY) == {"cuda", "mps", "cpu", "igpu"}
    assert PROFILE_REGISTRY["cuda"].device == "cuda"
    assert PROFILE_REGISTRY["cuda"].decode == "nvdec"
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


def test_cuda_device_verify_uses_injected_probe_source() -> None:
    deps = BootDependencies(default_verifiers(cuda_source=_cuda_available))

    context = resolve_boot_context({ML_WORKER_PROFILE_ENV: "cuda"}, deps, _decode_ok)

    assert context.device == "cuda"


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


def test_cuda_profile_fails_closed_when_no_decode_probe_configured() -> None:
    deps = BootDependencies(default_verifiers(cuda_source=_cuda_available))

    with pytest.raises(ProfileVerifyError, match="nvdec capability probe is not configured"):
        resolve_boot_context({ML_WORKER_PROFILE_ENV: "cuda"}, deps)


def _nvenc_ok() -> VerifyResult:
    return VerifyResult(True, "cuda", "encode", "h264_nvenc encoder is available")


def _nvenc_unavailable() -> VerifyResult:
    return VerifyResult(False, "cuda", "encode", "ffmpeg has no h264_nvenc encoder")


def test_resolve_encode_or_fallback_keeps_nvenc_when_probe_succeeds() -> None:
    selection = resolve_encode_or_fallback(PROFILE_REGISTRY["cuda"], _nvenc_ok)

    assert selection.requested == "h264_nvenc"
    assert selection.selected == "h264_nvenc"
    assert selection.fallback_count == 0
    assert selection.last_reason is None


def test_resolve_encode_or_fallback_never_raises_and_falls_back_to_libx264(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """#53: unlike decode's fail-fast preflight, a failed nvenc probe must
    never abort boot -- it degrades to libx264 with a loud WARNING instead."""
    with caplog.at_level("WARNING"):
        selection = resolve_encode_or_fallback(PROFILE_REGISTRY["cuda"], _nvenc_unavailable)

    assert selection.requested == "h264_nvenc"
    assert selection.selected == "libx264"
    assert selection.fallback_count == 1
    assert selection.last_reason == "nvenc_probe_failed"
    assert any("libx264" in record.message for record in caplog.records)


def test_resolve_encode_or_fallback_swallows_probe_exceptions() -> None:
    def raising_probe() -> VerifyResult:
        raise RuntimeError("ffmpeg exploded")

    selection = resolve_encode_or_fallback(PROFILE_REGISTRY["cuda"], raising_probe)

    assert selection.selected == "libx264"
    assert selection.fallback_count == 1


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


def test_resolve_boot_context_carries_encode_selection_and_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        context = resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "cuda"},
            _deps("cuda"),
            _decode_ok,
            _nvenc_unavailable,
        )

    assert context.encode == "libx264"
    assert context.encode_selection is not None
    assert context.encode_selection.requested == "h264_nvenc"
    assert context.encode_selection.selected == "libx264"
    assert context.encode_selection.fallback_count == 1


def test_resolve_boot_context_keeps_nvenc_when_probe_succeeds() -> None:
    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "cuda"},
        _deps("cuda"),
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

    for profile_name in ("cuda", "mps", "cpu"):
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


def test_igpu_profile_device_verify_false_still_reports_decode_gate() -> None:
    """Issue #79 (track 2) parity: igpu's device check failing must not skip
    running the (non-raising) decode resolution -- both are independent
    gates, same as every other profile."""
    decode_calls: list[str] = []

    def decode_probe(decode: str) -> VerifyResult:
        decode_calls.append(decode)
        return _vaapi_ok(decode)

    with pytest.raises(ProfileVerifyError, match="igpu"):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "igpu"}, _deps("igpu", ok=False), decode_probe
        )

    assert decode_calls == ["vaapi"]
