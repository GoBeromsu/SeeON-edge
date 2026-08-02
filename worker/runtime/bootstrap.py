"""Named 6-stage 2-tier bootstrap gates (ADR-0002, todo 26).

Two tiers with deliberately different blast radii:

- **Global stages** run once at process start, in one canonical order: gpu lease
  -> profile/device -> decode capability -> model backend init -> real warmup ->
  camera activation.  Any global-stage failure activates *zero* cameras, releases
  the GPU lease, and exits non-zero.  There is no CPU or software-encoder
  fallback anywhere on this path.
- **Per-camera stages** run once per camera.  A failure degrades only that camera
  (:func:`run_camera_stage` returns a failed outcome) and the process keeps
  running for the others.

Exit codes follow the canonical worker set (`0/1/2/3/4`):
:data:`GENERIC_RUNTIME_EXIT_CODE` (1) for an ordinary stage failure,
:data:`REFUSE_TO_START_EXIT_CODE` (3) when a precondition means we must not start
at all (lease contention, profile/device or decode mismatch), and
:data:`FATAL_ACCELERATOR_EXIT_CODE` (4) whenever a
:class:`~worker.adapters.model.errors.FatalAcceleratorError` surfaces — that code
wins over the stage's own code, because "the accelerator context is retired, do
not restart in place" must stay distinguishable from a plain crash.

Each :class:`Stage` is a *zero-argument* named callable.  Stages communicate
through the mutable :class:`BootstrapContext` they close over rather than by
threading a dict of prior outputs, so a stage that runs out of order fails
loudly on an unset context field instead of silently reading a missing key.  The
context is also what makes lease release possible from the failure path: the
process holds exactly one lease and exactly one place knows about it.

Warmup deliberately holds no CUDA knowledge.  The composition root injects
per-task callables (typically
``lambda runner: warmup_to_ready(runner, device=..., synchronize=...)``), so the
"synchronize after every forward on cuda" rule lives in
:func:`worker.adapters.model.warmup.warmup_to_ready` and is not reimplemented
here.
"""

from __future__ import annotations

import inspect
import logging
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, cast, final

from worker.adapters.model.errors import FatalAcceleratorError
from worker.runtime.faults.handler import FATAL_ACCELERATOR_EXIT_CODE
from worker.runtime.lease import GpuLease
from worker.runtime.profile.boot import (
    BootContext,
    preflight_decode_or_raise,
    reject_legacy_conflicts,
    resolve_profile,
    verify_device_or_raise,
)
from worker.runtime.profile.registry import (
    BootDependencies,
    DecodeProbe,
    ProfileSpec,
    default_decode_probe,
    default_verifiers,
)

LOGGER: Final = logging.getLogger(__name__)

GPU_LEASE_STAGE: Final = "gpu_lease"
PROFILE_DEVICE_STAGE: Final = "profile_device"
DECODE_CAPABILITY_STAGE: Final = "decode_capability"
MODEL_BACKEND_INIT_STAGE: Final = "model_backend_init"
WARMUP_STAGE: Final = "real_warmup"
CAMERA_ACTIVATION_STAGE: Final = "camera_activation"

GENERIC_RUNTIME_EXIT_CODE: Final = 1
REFUSE_TO_START_EXIT_CODE: Final = 3

BackendInitializer = Callable[[BootContext], object]
BackendWarmup = Callable[[object], object]
LeaseAcquirer = Callable[[], GpuLease]


@dataclass(frozen=True, slots=True)
class Stage:
    """One named global bootstrap stage.

    ``run`` takes no arguments and either returns this stage's output or raises.
    ``exit_code`` is the process exit code for *this* stage's ordinary failure; a
    :class:`FatalAcceleratorError` overrides it with
    :data:`FATAL_ACCELERATOR_EXIT_CODE`.
    """

    name: str
    run: Callable[[], object]
    exit_code: int = GENERIC_RUNTIME_EXIT_CODE


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Outputs of every stage that completed, keyed by stage name."""

    outputs: dict[str, object]


@dataclass(frozen=True, slots=True)
class CameraStageOutcome:
    """Result of one per-camera stage; ``ok=False`` degrades only this camera."""

    camera_id: str
    ok: bool
    reason: str | None = None


@final
class BootstrapStageError(RuntimeError):
    """A global bootstrap stage failed; the process must not continue."""

    __slots__ = ("exit_code", "stage")

    def __init__(self, stage: str, exit_code: int, reason: str) -> None:
        super().__init__(f"bootstrap stage {stage!r} failed: {reason}")
        self.stage = stage
        self.exit_code = exit_code


@dataclass(slots=True)
class BootstrapContext:
    """Everything the global stages resolve, in one mutable place.

    Deliberately mutable and deliberately not frozen: it is the single owner of
    the process-wide GPU lease, and the failure path must be able to release that
    lease without knowing which stage failed.
    """

    lease: GpuLease | None = None
    profile: ProfileSpec | None = None
    boot: BootContext | None = None
    runners: dict[str, object] = field(default_factory=dict)
    warmed: tuple[str, ...] = ()
    warmup_complete: bool = False
    cameras: tuple[CameraStageOutcome, ...] = ()

    def release_lease(self) -> None:
        """Release the held GPU lease.  Idempotent; safe to call twice.

        Clearing the attribute before closing means a second call is a no-op even
        if ``close()`` itself raises, so an operator retry is never blocked by a
        half-released lease.
        """
        lease = self.lease
        if lease is None:
            return
        self.lease = None
        lease.close()


def run_stages(stages: Iterable[Stage]) -> BootstrapResult:
    """Run global stages in order, stopping at the first failure.

    The offending stage name and cause are preserved so the operator gets a
    precise, non-silent reason instead of a bare traceback.
    """
    outputs: dict[str, object] = {}
    for stage in stages:
        try:
            outputs[stage.name] = _run_stage(stage, outputs)
        except BootstrapStageError:
            raise
        except FatalAcceleratorError as exc:
            # The accelerator context is unusable: code 4 outranks the stage's own
            # code so the operator and the updater can tell "do not restart in
            # place" apart from an ordinary runtime failure.
            raise BootstrapStageError(
                stage.name,
                FATAL_ACCELERATOR_EXIT_CODE,
                str(exc) or type(exc).__name__,
            ) from exc
        except Exception as exc:  # noqa: BLE001 - any global-stage failure is fatal
            raise BootstrapStageError(
                stage.name,
                stage.exit_code,
                str(exc) or type(exc).__name__,
            ) from exc
    return BootstrapResult(outputs)


def _run_stage(stage: Stage, outputs: dict[str, object]) -> object:
    parameters = inspect.signature(stage.run).parameters.values()
    takes_outputs = any(
        parameter.kind
        in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD, parameter.VAR_POSITIONAL}
        for parameter in parameters
    )
    if takes_outputs:
        run_with_outputs = cast(Callable[[dict[str, object]], object], stage.run)
        return run_with_outputs(outputs)
    return stage.run()


def bootstrap_or_exit(
    stages: Iterable[Stage],
    *,
    context: BootstrapContext | None = None,
    exit_fn: Callable[[int], None] = sys.exit,
    log: logging.Logger = LOGGER,
) -> BootstrapResult:
    """Run global bootstrap; on failure log loudly, free the lease, and exit non-zero.

    The lease is released *before* ``exit_fn`` so that an operator retry is not
    blocked by our own dying process, and :class:`BootstrapStageError` is
    re-raised afterwards so a non-terminating ``exit_fn`` cannot let the caller
    fall through into camera activation.
    """
    try:
        return run_stages(stages)
    except BootstrapStageError as exc:
        log.critical(
            "bootstrap stage %r failed; activating zero cameras and exiting %d: %s",
            exc.stage,
            exc.exit_code,
            exc,
        )
        if context is not None:
            context.release_lease()
        exit_fn(exc.exit_code)
        raise


def gpu_lease_stage(
    context: BootstrapContext,
    *,
    acquire: LeaseAcquirer = GpuLease.acquire,
) -> Stage:
    """First stage: take the repo-wide GPU lease before touching any device.

    Contention is a refuse-to-start condition, not something to wait out, so this
    runs ahead of every device probe — a losing entrant must exit before it
    constructs CUDA, NVDEC, or a model.
    """

    def _run() -> object:
        lease = acquire()
        context.lease = lease
        return lease

    return Stage(GPU_LEASE_STAGE, _run, REFUSE_TO_START_EXIT_CODE)


def profile_device_stage(
    context: BootstrapContext,
    env: Mapping[str, str],
    deps: BootDependencies | None = None,
) -> Stage:
    """Resolve the explicitly requested profile and verify its device.

    ``ML_WORKER_PROFILE`` is required; there is no implicit device probe and no
    CPU default.  ``context.profile`` is published only after verification passes,
    so a failed device never leaves a usable-looking profile behind.
    """

    def _run() -> object:
        spec = resolve_profile(env)
        _ = verify_device_or_raise(spec, deps or BootDependencies(default_verifiers()))
        context.profile = spec
        return spec

    return Stage(PROFILE_DEVICE_STAGE, _run, REFUSE_TO_START_EXIT_CODE)


def decode_capability_stage(
    context: BootstrapContext,
    env: Mapping[str, str],
    decode_probe: DecodeProbe | None = None,
) -> Stage:
    """Preflight the profile's decode backend and resolve the exact runtime triple.

    Publishing :class:`BootContext` here — after both the device and the decode
    checks — is what keeps a failed accelerator from ever resolving the software
    encoder.
    """

    def _run() -> object:
        spec = context.profile
        if spec is None:
            raise BootstrapStageError(
                DECODE_CAPABILITY_STAGE,
                REFUSE_TO_START_EXIT_CODE,
                "requires a completed profile/device stage",
            )
        _ = preflight_decode_or_raise(spec, decode_probe or default_decode_probe)
        reject_legacy_conflicts(spec, env)
        boot = BootContext(
            profile=spec,
            device=spec.device,
            decode=spec.decode,
            encode=spec.encode,
        )
        context.boot = boot
        return boot

    return Stage(DECODE_CAPABILITY_STAGE, _run, REFUSE_TO_START_EXIT_CODE)


def model_backend_init_stage(
    context: BootstrapContext,
    initializers: Mapping[str, BackendInitializer],
) -> Stage:
    """Construct every configured model backend against the resolved boot context."""

    def _run() -> object:
        boot = context.boot
        if boot is None:
            raise BootstrapStageError(
                MODEL_BACKEND_INIT_STAGE,
                REFUSE_TO_START_EXIT_CODE,
                f"requires a completed {DECODE_CAPABILITY_STAGE!r} stage",
            )
        runners = {task: initializer(boot) for task, initializer in initializers.items()}
        context.runners = runners
        return runners

    return Stage(MODEL_BACKEND_INIT_STAGE, _run, REFUSE_TO_START_EXIT_CODE)


def warmup_stage(
    context: BootstrapContext,
    warmups: Mapping[str, BackendWarmup],
) -> Stage:
    """Drive one real forward pass per configured backend, in declaration order.

    The injected callable owns the device semantics (including the CUDA
    synchronize after every forward), so this stage only sequences the tasks and
    records which ones reached ready.  ``warmup_complete`` flips exactly once, at
    the end, which is the gate camera activation checks.
    """

    def _run() -> object:
        missing = tuple(task for task in context.runners if task not in warmups)
        unexpected = tuple(task for task in warmups if task not in context.runners)
        if missing or unexpected:
            raise BootstrapStageError(
                WARMUP_STAGE,
                GENERIC_RUNTIME_EXIT_CODE,
                f"initializer and warmup task sets differ (missing={missing}, "
                f"unexpected={unexpected})",
            )
        warmed: list[str] = []
        for task, warmup in warmups.items():
            runner = context.runners.get(task)
            if runner is None:
                raise BootstrapStageError(
                    WARMUP_STAGE,
                    GENERIC_RUNTIME_EXIT_CODE,
                    f"no initialized model backend for warmup task {task!r}",
                )
            _ = warmup(runner)
            warmed.append(task)
            context.warmed = tuple(warmed)
        context.warmup_complete = True
        return tuple(warmed)

    return Stage(WARMUP_STAGE, _run)


def camera_activation_stage(
    context: BootstrapContext,
    activate: Callable[[BootContext], Iterable[CameraStageOutcome]],
) -> Stage:
    """Last stage: activate cameras, and only after every backend is warm.

    This is the invariant the whole sequence exists to protect — no camera may
    start against a model that has not completed a real forward pass.
    """

    def _run() -> object:
        if not context.warmup_complete:
            raise BootstrapStageError(
                CAMERA_ACTIVATION_STAGE,
                GENERIC_RUNTIME_EXIT_CODE,
                f"requires a completed {WARMUP_STAGE!r} stage",
            )
        boot = context.boot
        if boot is None:
            raise BootstrapStageError(
                CAMERA_ACTIVATION_STAGE,
                REFUSE_TO_START_EXIT_CODE,
                f"requires a completed {DECODE_CAPABILITY_STAGE!r} stage",
            )
        outcomes = tuple(activate(boot))
        context.cameras = outcomes
        return outcomes

    return Stage(CAMERA_ACTIVATION_STAGE, _run)


def named_stages(
    context: BootstrapContext,
    env: Mapping[str, str],
    *,
    initializers: Mapping[str, BackendInitializer],
    warmups: Mapping[str, BackendWarmup],
    activate: Callable[[BootContext], Iterable[CameraStageOutcome]],
    decode_probe: DecodeProbe | None = None,
    deps: BootDependencies | None = None,
    acquire: LeaseAcquirer | None = None,
    acquire_lease: LeaseAcquirer | None = None,
) -> tuple[Stage, ...]:
    """Build the canonical ordered global stage sequence.

    ``acquire`` and ``acquire_lease`` name the same injected lease factory; both
    spellings are accepted because both are already in use by callers, and
    passing them together is rejected rather than silently resolved. Omitting
    both uses the real repo-wide lease.
    """
    if acquire is not None and acquire_lease is not None:
        message = "pass either acquire or acquire_lease, not both"
        raise ValueError(message)
    resolved = acquire or acquire_lease or GpuLease.acquire
    return (
        gpu_lease_stage(context, acquire=resolved),
        profile_device_stage(context, env, deps),
        decode_capability_stage(context, env, decode_probe),
        model_backend_init_stage(context, initializers),
        warmup_stage(context, warmups),
        camera_activation_stage(context, activate),
    )


def run_camera_stage(camera_id: str, run: Callable[[], object]) -> CameraStageOutcome:
    """Run a per-camera stage; an ordinary failure degrades only this camera.

    A :class:`FatalAcceleratorError` is re-raised instead of degraded: the GPU
    context is process-wide, so treating it as one camera's problem would leave
    every other camera running against a retired context.
    """
    try:
        _ = run()
    except FatalAcceleratorError:
        raise
    except Exception as exc:  # noqa: BLE001 - a per-camera failure degrades one camera
        LOGGER.warning(
            "camera %s stage failed; degrading this camera only: %s",
            camera_id,
            exc,
        )
        return CameraStageOutcome(camera_id=camera_id, ok=False, reason=str(exc))
    return CameraStageOutcome(camera_id=camera_id, ok=True)


__all__ = [
    "CAMERA_ACTIVATION_STAGE",
    "DECODE_CAPABILITY_STAGE",
    "FATAL_ACCELERATOR_EXIT_CODE",
    "GENERIC_RUNTIME_EXIT_CODE",
    "GPU_LEASE_STAGE",
    "MODEL_BACKEND_INIT_STAGE",
    "PROFILE_DEVICE_STAGE",
    "REFUSE_TO_START_EXIT_CODE",
    "WARMUP_STAGE",
    "BackendInitializer",
    "BackendWarmup",
    "BootDependencies",
    "BootstrapContext",
    "BootstrapResult",
    "BootstrapStageError",
    "CameraStageOutcome",
    "LeaseAcquirer",
    "Stage",
    "bootstrap_or_exit",
    "camera_activation_stage",
    "decode_capability_stage",
    "gpu_lease_stage",
    "model_backend_init_stage",
    "named_stages",
    "profile_device_stage",
    "run_camera_stage",
    "run_stages",
    "warmup_stage",
]
