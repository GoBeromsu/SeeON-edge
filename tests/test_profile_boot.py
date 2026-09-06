from __future__ import annotations

import pytest

from worker.runtime.profile.boot import resolve_boot_context, resolve_profile
from worker.runtime.profile.registry import (
    DEFAULT_PROFILE_NAME,
    ML_WORKER_PROFILE_ENV,
    BootDependencies,
    ProfileError,
    ProfileVerifyError,
    VerifyResult,
)


def _flow_dependencies(*, available: bool = True) -> BootDependencies:
    return BootDependencies(
        {
            "flow": lambda: VerifyResult(
                available,
                "flow",
                "device",
                "flow engines verified" if available else "flow engine identity is invalid",
            )
        }
    )


def _nvdec_available(decode: str) -> VerifyResult:
    assert decode == "nvdec"
    return VerifyResult(True, "flow", "decode", "NVDEC available")


def test_flow_is_the_default_and_only_profile() -> None:
    assert DEFAULT_PROFILE_NAME == "flow"
    assert resolve_profile({}).name == "flow"
    assert resolve_profile({ML_WORKER_PROFILE_ENV: "flow"}).name == "flow"


@pytest.mark.parametrize("retired", ["auto", "cpu", "mps", "nvidia"])
def test_retired_production_profiles_refuse_with_adr_0002(retired: str) -> None:
    with pytest.raises(
        ProfileError,
        match=rf"ADR-0002: unsupported ML_WORKER_PROFILE '{retired}'; set flow",
    ):
        resolve_profile({ML_WORKER_PROFILE_ENV: retired})


def test_flow_boot_context_preserves_the_device_resident_contract() -> None:
    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "flow"}, _flow_dependencies(), _nvdec_available
    )

    assert context.canonical_profile == "flow"
    assert context.device == "cuda"
    assert context.decode == "nvdec"
    assert context.encode == "h264_nvenc"
    assert context.runtime_profile.device_resident_after_decode is True
    assert context.capability_graph.full_frame_copy_count == 0


def test_flow_refuses_to_boot_when_its_verifier_rejects_the_engine_inputs() -> None:
    with pytest.raises(ProfileVerifyError, match="flow engine identity is invalid"):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "flow"}, _flow_dependencies(available=False), _nvdec_available
        )
