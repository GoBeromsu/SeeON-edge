from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from contracts.encode_diagnostics import EncodeSelection
from worker.runtime.profile.registry import (
    ML_WORKER_PROFILE_ENV,
    PROFILE_REGISTRY,
    BootDependencies,
    DecodePolicy,
    DecodeProbe,
    DevicePolicy,
    EncodePolicy,
    EncodeProbe,
    ProfileError,
    ProfileSpec,
    ProfileVerifyError,
    VerifyResult,
    default_decode_probe,
    default_encode_probe,
    default_verifiers,
)

LOGGER: Final = logging.getLogger(__name__)

_LEGACY_DECODE_ENVIRONMENTS = ("ML_RTSP_BACKEND", "ML_DEFAULT_DECODE_BACKEND")


@dataclass(frozen=True, slots=True)
class BootContext:
    profile: ProfileSpec
    device: DevicePolicy
    decode: DecodePolicy
    encode: EncodePolicy
    encode_selection: EncodeSelection | None = None


def resolve_profile(
    env: Mapping[str, str],
    registry: Mapping[str, ProfileSpec] = PROFILE_REGISTRY,
) -> ProfileSpec:
    profile_name = env.get(ML_WORKER_PROFILE_ENV)
    if profile_name is None or not profile_name.strip():
        message = "ML_WORKER_PROFILE is required (no default); set cuda|mps|cpu"
        raise ProfileError(message)

    try:
        return registry[profile_name]
    except KeyError as error:
        choices = "|".join(sorted(registry))
        message = f"unknown ML_WORKER_PROFILE {profile_name!r}; set {choices}"
        raise ProfileError(message) from error


def verify_device_or_raise(spec: ProfileSpec, deps: BootDependencies) -> VerifyResult:
    try:
        verifier = deps.verifiers[spec.name]
        result = verifier()
    except (KeyError, RuntimeError) as error:
        message = f"profile {spec.name!r} device verification failed: {error}"
        raise ProfileVerifyError(message) from error

    if not result.ok:
        message = f"profile {spec.name!r} device verification failed: {result.reason}"
        raise ProfileVerifyError(message)
    return result


def preflight_decode_or_raise(spec: ProfileSpec, decode_probe: DecodeProbe) -> VerifyResult:
    try:
        result = decode_probe(spec.decode)
    except (OSError, RuntimeError) as error:
        message = f"profile {spec.name!r} decode preflight failed: {error}"
        raise ProfileVerifyError(message) from error

    if not result.ok:
        message = f"profile {spec.name!r} decode preflight failed: {result.reason}"
        raise ProfileVerifyError(message)
    return result


def resolve_encode_or_fallback(
    spec: ProfileSpec,
    encode_probe: EncodeProbe | None,
    *,
    now: Callable[[], float] = time.time,
) -> EncodeSelection:
    """Resolve the profile's clip encoder, demoting nvenc to libx264 on a failed preflight.

    Unlike `preflight_decode_or_raise`, a failed probe here never aborts boot:
    per #53's accepted design, NVENC and libx264 both emit the same H.264
    content, so trading GPU for CPU encode cost never changes what a clip
    records (unlike a fall-detector model swap, which #43 forbids from ever
    falling back silently). The demotion still has to be loud -- a WARNING is
    logged here, and the returned `EncodeSelection` is meant for local
    diagnostics exposure (`worker/runtime/telemetry/runtime_diagnostics.py`),
    not the byte-for-byte-frozen backend relay payload
    (`worker/runtime/telemetry/wire.py`).

    Profiles that already request `libx264` (`mps`, `cpu`) never probe --
    there is nothing to fall back from.
    """
    requested = spec.encode
    if requested != "h264_nvenc":
        return EncodeSelection(
            requested=requested,
            selected=requested,
            fallback_count=0,
            last_reason=None,
            updated_at_sec=now(),
        )

    probe = default_encode_probe if encode_probe is None else encode_probe
    try:
        result = probe()
    except (OSError, RuntimeError) as error:
        result = VerifyResult(
            False, spec.name, "encode", f"nvenc probe raised {type(error).__name__}: {error}"
        )

    if result.ok:
        return EncodeSelection(
            requested=requested,
            selected=requested,
            fallback_count=0,
            last_reason=None,
            updated_at_sec=now(),
        )

    LOGGER.warning(
        "profile %r encode preflight failed (%s); falling back to libx264",
        spec.name,
        result.reason,
    )
    return EncodeSelection(
        requested=requested,
        selected="libx264",
        fallback_count=1,
        last_reason="nvenc_probe_failed",
        updated_at_sec=now(),
    )


def reject_legacy_conflicts(spec: ProfileSpec, env: Mapping[str, str]) -> None:
    for key in _LEGACY_DECODE_ENVIRONMENTS:
        configured = env.get(key)
        if configured and configured != "auto" and configured != spec.decode:
            message = (
                f"{key}={configured!r} conflicts with profile "
                f"{spec.name!r} decode {spec.decode!r}"
            )
            raise ProfileVerifyError(message)


def resolve_boot_context(
    env: Mapping[str, str],
    deps: BootDependencies | None = None,
    decode_probe: DecodeProbe | None = None,
    encode_probe: EncodeProbe | None = None,
) -> BootContext:
    spec = resolve_profile(env)
    _ = verify_device_or_raise(spec, deps or BootDependencies(default_verifiers()))
    _ = preflight_decode_or_raise(spec, decode_probe or default_decode_probe)
    reject_legacy_conflicts(spec, env)
    encode_selection = resolve_encode_or_fallback(spec, encode_probe)
    return BootContext(
        profile=spec,
        device=spec.device,
        decode=spec.decode,
        encode=encode_selection.selected or spec.encode,
        encode_selection=encode_selection,
    )


__all__ = [
    "BootContext",
    "preflight_decode_or_raise",
    "reject_legacy_conflicts",
    "resolve_boot_context",
    "resolve_encode_or_fallback",
    "resolve_profile",
    "verify_device_or_raise",
]
