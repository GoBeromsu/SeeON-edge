# 0003 — Explicit Fallback Only

- Status: Accepted
- Date: 2026-08-01
- Relates to: [0002](0002-gpu-pipeline-failfast-modularization.md) (fail-fast GPU
  pipeline). Consistent with ADR-0004 (vendored `contracts/`), which this
  decision does not touch.

## Context

Two production gaps in this repository shared one shape: a degraded path was
selected automatically, and the degradation was invisible until a real worker
failed to boot. The decode capability probe and the CUDA device verifier were
both fail-closed dependency-injection stubs that passed lint, import contracts,
and the full test suite while the real worker could not start.

A third instance is still in the tree. `CpuAvAdapter` opens a capture with
`(url, CAP_FFMPEG, params)` and, on `TypeError`, retries as `(url, backend)` and
then `(url)`. `TypeError` is treated as a signal about the OpenCV API surface,
so a required read/open timeout and the explicit FFmpeg backend are dropped
silently. A capture that succeeds through that cascade has neither the timeout
contract nor the backend the adapter claims to require, and the operator sees a
working camera rather than a configuration error.

The common failure mode is not the fallback itself. It is that the caller never
asked for one, and the system cannot tell the difference afterwards.

## Decision

**Fallback is permitted only when it is explicit. Implicit fallback is
prohibited.**

A fallback is *explicit* when the caller selects it — a configuration value, a
named profile, an argument, or a documented policy object. A fallback is
*implicit* when code infers it from an exception, a missing attribute, a failed
capability query, or any other runtime accident.

This applies to:

- decode backend selection and capture-constructor parameters,
- capability probes (decode, device, encoder),
- device and encoder selection,
- configuration load and last-known-good handling.

Concretely prohibited:

- Catching `TypeError`, `AttributeError`, or a signature error and retrying with
  fewer arguments, a different backend, or dropped required options.
- Composing an OpenCV "safe fallback" behind a GPU-preferring path.
- Treating a failed or unavailable capability query as available.

Required instead: fail closed with a typed, sanitized error that names the
capability that was unavailable and the reason. Credentials and stream URLs are
never included in that message.

## Consequences

- The supported boundary must be declared rather than discovered. For CPU decode
  that boundary is `opencv-python-headless>=4.10` with the lockfile pin; versions
  outside it are refused, not silently accommodated.
- Probes report `available=False` with an explicit reason instead of degrading.
  A host without an FFmpeg-capable OpenCV build fails preflight rather than
  failing every camera later.
- Genuinely supporting multiple runtime variants requires an explicit
  version/capability adapter with tests for each path — a separate decision, not
  an exception handler.
- Operators see the real failure reason at boot instead of a partially
  configured runtime.

## Alternatives considered

**Keep the compatibility cascade.** Rejected: it converts a configuration error
into a silent capability downgrade, which is the exact defect this repository
has already shipped twice.

**Allow implicit fallback with a warning log.** Rejected: warnings do not gate
boot, and the resulting evidence ("the worker runs") is indistinguishable from a
correctly configured runtime.
