from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from contracts.decode_diagnostics import DecodeSelection
from contracts.encode_diagnostics import EncodeSelection
from worker.runtime.profile.registry import (
    DEFAULT_PROFILE_NAME,
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
    decode_selection: DecodeSelection | None = None


def resolve_profile(
    env: Mapping[str, str],
    registry: Mapping[str, ProfileSpec] = PROFILE_REGISTRY,
) -> ProfileSpec:
    """Resolve ``ML_WORKER_PROFILE``, defaulting to :data:`DEFAULT_PROFILE_NAME`.

    Issue #133: the worker must boot with zero env vars, so an unset/blank
    ``ML_WORKER_PROFILE`` no longer refuses to boot -- it falls back to
    ``DEFAULT_PROFILE_NAME`` ("cpu"), the only profile whose device
    verification always succeeds with no injected capability probe. An
    *explicit* but unrecognized value is still fail-closed: a typo like
    ``ML_WORKER_PROFILE=gpu`` must not be silently reinterpreted as the
    default.
    """
    raw = env.get(ML_WORKER_PROFILE_ENV)
    profile_name = DEFAULT_PROFILE_NAME if raw is None or not raw.strip() else raw

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


def resolve_decode_or_fallback(
    spec: ProfileSpec,
    decode_probe: DecodeProbe | None,
    *,
    now: Callable[[], float] = time.time,
) -> DecodeSelection:
    """Resolve iGPU VAAPI decode, demoting to opencv (CPU/software) decode on a failed preflight.

    Unlike `preflight_decode_or_raise` -- still used unchanged by nvdec/opencv,
    whose ADR-0002 fail-fast semantics this does not touch -- a failed VAAPI
    probe here never aborts boot. VAAPI and the ffmpeg/OpenCV software path
    both decode the same RTSP stream into identical RGB FramePackets, so
    trading iGPU offload for CPU decode cost never changes what a camera
    records or what downstream inference sees. Issues #191/#194 established
    that a *silent* no-frames failure is the actual footgun here, not a
    loudly-logged software-decode fallback -- so unlike an adapter probing
    its way to a different backend (disallowed per worker/adapters/AGENTS.md),
    this decision is made once, at the boot/profile composition root, exactly
    like `resolve_encode_or_fallback` (#53).

    Profiles that don't request vaapi (cuda, mps, cpu) never call this --
    they keep the existing fail-fast `preflight_decode_or_raise` path.
    """
    requested = spec.decode
    if requested != "vaapi":
        return DecodeSelection(
            requested=requested,
            selected=requested,
            fallback_count=0,
            last_reason=None,
            updated_at_sec=now(),
        )

    probe = default_decode_probe if decode_probe is None else decode_probe
    try:
        result = probe(requested)
    except (OSError, RuntimeError) as error:
        result = VerifyResult(
            False, spec.name, "decode", f"vaapi probe raised {type(error).__name__}: {error}"
        )

    if result.ok:
        return DecodeSelection(
            requested=requested,
            selected=requested,
            fallback_count=0,
            last_reason=None,
            updated_at_sec=now(),
        )

    LOGGER.warning(
        "profile %r decode preflight failed (%s); falling back to opencv (CPU) decode",
        spec.name,
        result.reason,
    )
    return DecodeSelection(
        requested=requested,
        selected="opencv",
        fallback_count=1,
        last_reason="vaapi_probe_failed",
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
    """Resolve the full boot gate: profile, device, decode, legacy-conflict.

    Issue #79 (track 2): the device check, the decode preflight, and the
    legacy-env conflict check are three independent gates over the same
    resolved ``spec`` -- none depends on another's outcome. Previously each
    raised immediately on its own failure, so an operator with e.g. both a
    bad device *and* an incompatible legacy decode override only ever saw
    the device failure, fixed it, reran, and only then discovered the
    decode conflict. All three now always run and every failure is
    collected into one raised ``ProfileVerifyError`` naming every failed
    gate instead of just the first.
    """
    spec = resolve_profile(env)
    failures: list[str] = []

    try:
        _ = verify_device_or_raise(spec, deps or BootDependencies(default_verifiers()))
    except ProfileVerifyError as error:
        failures.append(str(error))

    # vaapi is the one decode policy with an explicit, loud fallback (see
    # `resolve_decode_or_fallback`) instead of the fail-fast preflight every
    # other policy still uses -- nvdec/opencv keep raising on a failed probe.
    decode_selection: DecodeSelection | None = None
    if spec.decode == "vaapi":
        decode_selection = resolve_decode_or_fallback(spec, decode_probe)
    else:
        try:
            _ = preflight_decode_or_raise(spec, decode_probe or default_decode_probe)
        except ProfileVerifyError as error:
            failures.append(str(error))

    try:
        reject_legacy_conflicts(spec, env)
    except ProfileVerifyError as error:
        failures.append(str(error))

    if failures:
        summary = "; ".join(failures)
        raise ProfileVerifyError(
            f"{len(failures)} boot gate(s) failed for profile {spec.name!r}: {summary}"
        )

    encode_selection = resolve_encode_or_fallback(spec, encode_probe)
    return BootContext(
        profile=spec,
        device=spec.device,
        decode=(decode_selection.selected if decode_selection else spec.decode) or spec.decode,
        encode=encode_selection.selected or spec.encode,
        encode_selection=encode_selection,
        decode_selection=decode_selection,
    )


__all__ = [
    "BootContext",
    "preflight_decode_or_raise",
    "reject_legacy_conflicts",
    "resolve_boot_context",
    "resolve_decode_or_fallback",
    "resolve_encode_or_fallback",
    "resolve_profile",
    "verify_device_or_raise",
]
