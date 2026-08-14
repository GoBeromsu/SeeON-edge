"""Standalone diagnostic CLI for the experimental device-input NVENC encoder seam.

Runnable directly on real edge hardware -- never imported by production boot,
never selected by any profile, and never executed by the default pytest
suite (the ``real_stack``-marked tests in
``tests/test_nvenc_device_real_stack.py`` cover the same probe
programmatically; this module is the operator-facing counterpart, mirroring
``worker.adapters.decode.nvdec_device.diagnostic`` exactly)::

    python -m worker.adapters.encode.nvenc_device.diagnostic --help
    python -m worker.adapters.encode.nvenc_device.diagnostic --probe
    python -m worker.adapters.encode.nvenc_device.diagnostic --fake-smoke-test
    python -m worker.adapters.encode.nvenc_device.diagnostic --fake-bad-pressure

``--probe`` (the default action) runs the real combined capability probe
(device-resident + ffmpeg NVENC build) and prints a truthful available/
unavailable verdict with its reason. ``--fake-smoke-test`` exercises the
deterministic fake session lifecycle end to end. ``--fake-bad-pressure``
exercises the same fake session past its declared capacity and confirms it
fails closed with a backpressure error rather than silently queuing or
dropping frames.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Final

import numpy as np

from contracts.frame import Frame
from worker.adapters.encode.nvenc_device.capability import (
    DeviceInputNvencCapability,
    probe_device_input_nvenc_capability,
)
from worker.adapters.encode.nvenc_device.errors import (
    DeviceEncoderPoolExhaustedError,
    DeviceEncoderRejectedInputError,
)
from worker.adapters.encode.nvenc_device.fake import (
    fake_device_input_nvenc_encoder,
    fake_device_resident_lease,
)
from worker.types import FrameLease, MemoryKind

PROBE_UNAVAILABLE_EXIT_CODE: Final = 1
BAD_PRESSURE_DID_NOT_FAIL_EXIT_CODE: Final = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m worker.adapters.encode.nvenc_device.diagnostic",
        description=(
            "Experimental device-input NVENC encoder diagnostic (Todo 18). "
            "Never selects a production profile; never runs during normal "
            "worker boot."
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--probe",
        action="store_true",
        help=(
            "Run the real combined device-resident + ffmpeg NVENC capability "
            "probe and print its verdict (default action). Exits non-zero "
            "when unavailable -- this is a truthful report, not a soft "
            "warning."
        ),
    )
    action.add_argument(
        "--fake-smoke-test",
        action="store_true",
        help=(
            "Exercise the deterministic fake device-input session lifecycle "
            "(no real hardware required) and print a pass/fail summary."
        ),
    )
    action.add_argument(
        "--fake-bad-pressure",
        action="store_true",
        help=(
            "Exercise the deterministic fake session past its declared "
            "capacity and a rejected host-memory submission; confirms both "
            "fail closed rather than silently falling back."
        ),
    )
    return parser


def _print_capability(capability: DeviceInputNvencCapability) -> None:
    print(f"available: {capability.available}")
    print(f"reason: {capability.reason}")
    print(
        f"device_resident.available: {capability.device_resident.available} "
        f"({capability.device_resident.reason})"
    )
    print(f"nvenc.available: {capability.nvenc.available} ({capability.nvenc.reason})")


def run_probe() -> int:
    capability = probe_device_input_nvenc_capability()
    _print_capability(capability)
    return 0 if capability.available else PROBE_UNAVAILABLE_EXIT_CODE


def run_fake_smoke_test() -> int:
    encoder = fake_device_input_nvenc_encoder(camera_id="diagnostic", capacity=2, width=4, height=4)
    selection = encoder.open()
    lease_a = fake_device_resident_lease(width=4, height=4, fill=10)
    lease_b = fake_device_resident_lease(width=4, height=4, fill=200)
    encoder.submit(lease_a)
    encoder.submit(lease_b)
    print(f"outstanding_before_retire={encoder.outstanding}")
    encoder.retire_all()
    print(f"outstanding_after_retire={encoder.outstanding}")
    with tempfile.TemporaryDirectory() as tmp:
        result = encoder.finalize(Path(tmp) / "artifact.bin")
        print(f"selected: {selection.selected_codec}/{selection.selected_container}")
        print(f"device_resident: {selection.device_resident}")
        print(f"artifact sha256: {result.sha256}")
        print(f"artifact size_bytes: {result.size_bytes}")
    lease_a.release()
    lease_b.release()
    snapshot = encoder.telemetry.snapshot()
    print(f"submissions_accepted={snapshot.submissions_accepted}")
    print(f"d2h_transfers={snapshot.d2h_transfers} d2h_bytes={snapshot.d2h_bytes}")
    encoder.close()
    print("fake smoke test: PASS")
    return 0


def run_fake_bad_pressure() -> int:
    encoder = fake_device_input_nvenc_encoder(camera_id="diagnostic", capacity=1, width=2, height=2)
    _ = encoder.open()

    host_lease = _host_lease()
    rejected_host_input = False
    try:
        encoder.submit(host_lease)
        host_lease.release()
    except DeviceEncoderRejectedInputError as error:
        rejected_host_input = True
        print(f"host-input rejection (expected): {error}")

    filling = fake_device_resident_lease(width=2, height=2)
    encoder.submit(filling)
    print(f"outstanding_at_capacity={encoder.outstanding}/{encoder.capacity}")

    exhausted = False
    overflow = fake_device_resident_lease(width=2, height=2)
    try:
        encoder.submit(overflow)
    except DeviceEncoderPoolExhaustedError as error:
        exhausted = True
        print(f"backpressure rejection (expected): {error}")
    finally:
        overflow.release()
        _ = encoder.cancel_pending()
        encoder.close()

    if not rejected_host_input or not exhausted:
        print("fake bad-pressure test: FAIL (expected rejection did not occur)")
        return BAD_PRESSURE_DID_NOT_FAIL_EXIT_CODE
    print("fake bad-pressure test: PASS")
    return 0


def _host_lease() -> FrameLease:
    frame = Frame(0, 0.0, np.zeros((2, 2, 3), dtype=np.uint8))
    lease = FrameLease.from_host(frame)
    assert lease.descriptor.memory_kind is MemoryKind.HOST  # noqa: S101 - diagnostic sanity check
    return lease


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.fake_smoke_test:
        return run_fake_smoke_test()
    if args.fake_bad_pressure:
        return run_fake_bad_pressure()
    return run_probe()


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    sys.exit(main())


__all__ = ["main", "run_fake_bad_pressure", "run_fake_smoke_test", "run_probe"]
