from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final

from contracts.decode_diagnostics import DecodeSelection
from contracts.encode_diagnostics import EncodeSelection
from worker.runtime.profile.capability_graph import (
    CapabilityMismatchError,
    ValidatedCapabilityGraph,
    validate_capability_graph,
    validate_runtime_profile_descriptor,
)
from worker.runtime.profile.descriptor import RuntimeProfileDescriptor
from worker.runtime.profile.registry import (
    PROFILE_REGISTRY,
    BootDependencies,
    DecodePolicy,
    DecodeProbe,
    DevicePolicy,
    EncodePolicy,
    EncodeProbe,
    ProfileSpec,
    ProfileVerifyError,
    VerifyResult,
    default_decode_probe,
    default_encode_probe,
    default_verifiers,
    runtime_descriptor_for,
    select_profile,
)
from worker.types.capabilities import ConverterCapabilities, PipelineProfile

LOGGER: Final = logging.getLogger(__name__)

_LEGACY_DECODE_ENVIRONMENTS = ("ML_RTSP_BACKEND", "ML_DEFAULT_DECODE_BACKEND")


@dataclass(frozen=True, slots=True)
class BootContext:
    profile: ProfileSpec
    device: DevicePolicy
    decode: DecodePolicy
    encode: EncodePolicy
    requested_profile: str = ""
    degraded_reasons: tuple[str, ...] = ()
    pipeline_profile: PipelineProfile | None = None
    capability_graph: ValidatedCapabilityGraph = field(init=False)
    encode_selection: EncodeSelection | None = None
    decode_selection: DecodeSelection | None = None
    runtime_profile: RuntimeProfileDescriptor = field(init=False)

    def __post_init__(self) -> None:
        requested = self.requested_profile or self.profile.name
        object.__setattr__(self, "requested_profile", requested)
        runtime_profile = runtime_descriptor_for(
            self.profile,
            requested_profile=requested,
            effective_decode=self.decode,
            effective_encode=self.encode,
            degraded_reasons=self.degraded_reasons,
        )
        object.__setattr__(self, "runtime_profile", runtime_profile)
        object.__setattr__(
            self,
            "capability_graph",
            validate_runtime_profile_descriptor(runtime_profile),
        )

    @property
    def canonical_profile(self) -> str:
        return self.profile.name


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
    return select_profile(env, registry).spec


def resolve_capability_graph_or_raise(
    spec: ProfileSpec,
    *,
    converters: tuple[ConverterCapabilities, ...] = (),
) -> ValidatedCapabilityGraph:
    if spec.pipeline is None:
        raise ProfileVerifyError(f"profile {spec.name!r} has no legacy linear capability graph")
    try:
        return validate_capability_graph(spec.pipeline, converters=converters)
    except (CapabilityMismatchError, ValueError) as error:
        raise ProfileVerifyError(
            f"profile {spec.name!r} capability graph failed: {error}"
        ) from error


def verify_device_or_raise(spec: ProfileSpec, deps: BootDependencies) -> VerifyResult:
    verifier = next(
        (deps.verifiers[name] for name in spec.accepted_names if name in deps.verifiers),
        None,
    )
    if verifier is None:
        message = f"profile {spec.name!r} device verification failed: no verifier configured"
        raise ProfileVerifyError(message)
    try:
        result = verifier()
    except RuntimeError as error:
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
    if spec.encode_fallback is None:
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
        selected=spec.encode_fallback,
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
    if spec.decode_fallback is None:
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
        selected=spec.decode_fallback,
        fallback_count=1,
        last_reason="vaapi_probe_failed",
        updated_at_sec=now(),
    )


def effective_decode_policy(spec: ProfileSpec, selection: DecodeSelection | None) -> DecodePolicy:
    selected = selection.selected if selection is not None else spec.decode
    if selected not in ("nvdec", "opencv", "vaapi"):
        raise ProfileVerifyError(
            f"profile {spec.name!r} resolved unsupported decode backend {selected!r}"
        )
    return selected


def effective_encode_policy(spec: ProfileSpec, selection: EncodeSelection) -> EncodePolicy:
    selected = selection.selected or spec.encode
    if selected not in ("h264_nvenc", "libx264"):
        raise ProfileVerifyError(
            f"profile {spec.name!r} resolved unsupported encode backend {selected!r}"
        )
    return selected


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
    capability_converters: tuple[ConverterCapabilities, ...] = (),
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
    selection = select_profile(env)
    spec = selection.spec
    if capability_converters:
        raise ProfileVerifyError(
            "boot capability converters must be declared by the runtime profile descriptor"
        )
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
    degraded_reasons = tuple(
        reason
        for reason in (
            decode_selection.last_reason if decode_selection else None,
            encode_selection.last_reason,
        )
        if reason is not None
    )
    return BootContext(
        profile=spec,
        device=spec.device,
        decode=effective_decode_policy(spec, decode_selection),
        encode=effective_encode_policy(spec, encode_selection),
        requested_profile=selection.requested_name,
        degraded_reasons=degraded_reasons,
        encode_selection=encode_selection,
        decode_selection=decode_selection,
    )


__all__ = [
    "BootContext",
    "effective_decode_policy",
    "effective_encode_policy",
    "preflight_decode_or_raise",
    "reject_legacy_conflicts",
    "resolve_boot_context",
    "resolve_capability_graph_or_raise",
    "resolve_decode_or_fallback",
    "resolve_encode_or_fallback",
    "resolve_profile",
    "verify_device_or_raise",
]
