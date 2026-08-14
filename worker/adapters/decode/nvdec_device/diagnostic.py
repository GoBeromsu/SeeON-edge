"""Standalone diagnostic CLI for the experimental NVIDIA device-resident path.

Runnable directly on real edge hardware -- never imported by production boot,
never selected by any profile, and never executed by the default pytest
suite (the ``real_stack``-marked tests in
``tests/test_nvidia_device_resident_real_stack.py`` cover the same probe
programmatically; this module is the operator-facing counterpart for someone
sitting at an actual NVIDIA box)::

    python -m worker.adapters.decode.nvdec_device.diagnostic --help
    python -m worker.adapters.decode.nvdec_device.diagnostic --probe
    python -m worker.adapters.decode.nvdec_device.diagnostic --fake-smoke-test

``--probe`` (the default action) runs the real capability probe and prints a
truthful available/unavailable verdict with its reason -- it never guesses
and never reports available on a host it cannot prove supports every
concrete stage. ``--fake-smoke-test`` exercises the deterministic fake pool/
batcher lifecycle end to end (identical to the unit tests) so an operator can
confirm the prototype's Python surface imports and runs correctly even
before real hardware is available.
"""

from __future__ import annotations

import argparse
import sys
from typing import Final

import numpy as np

from worker.adapters.decode.nvdec_device.capability import (
    DeviceResidentCapability,
    probe_device_resident_capability,
)
from worker.adapters.decode.nvdec_device.fake import (
    FakeDeviceResidentBatcher,
    fake_device_resident_pool,
)

PROBE_UNAVAILABLE_EXIT_CODE: Final = 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m worker.adapters.decode.nvdec_device.diagnostic",
        description=(
            "Experimental NVIDIA device-resident analysis prototype diagnostic "
            "(Todo 17). Never selects a production profile; never runs during "
            "normal worker boot."
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--probe",
        action="store_true",
        help=(
            "Run the real device-resident capability probe and print its "
            "verdict (default action). Exits non-zero when unavailable -- "
            "this is a truthful report, not a soft warning."
        ),
    )
    action.add_argument(
        "--fake-smoke-test",
        action="store_true",
        help=(
            "Exercise the deterministic fake pool/batcher lifecycle (no real "
            "hardware required) and print a pass/fail summary."
        ),
    )
    return parser


def _print_capability(capability: DeviceResidentCapability) -> None:
    print(f"available: {capability.available}")
    print(f"reason: {capability.reason}")
    print(f"cuda.available: {capability.cuda.available} ({capability.cuda.reason})")
    print(f"nvml.nvml_available: {capability.nvml.nvml_available} ({capability.nvml.reason})")
    print(f"nvml.driver_version: {capability.nvml.driver_version}")
    print(f"nvml.device_name: {capability.nvml.device_name}")
    print(f"stream_event_supported: {capability.stream_event_supported}")
    print(f"dlpack_supported: {capability.dlpack_supported}")


def run_probe() -> int:
    capability = probe_device_resident_capability()
    _print_capability(capability)
    return 0 if capability.available else PROBE_UNAVAILABLE_EXIT_CODE


def run_fake_smoke_test() -> int:
    pool, allocator = fake_device_resident_pool(
        camera_id="diagnostic", capacity=2, width=4, height=4
    )
    batcher = FakeDeviceResidentBatcher(max_batch_size=2, allocator=allocator)

    lease_a = pool.acquire()
    lease_b = pool.acquire()
    allocator.upload(lease_a, np.full((4, 4, 3), 10, dtype=np.uint8))
    allocator.upload(lease_b, np.full((4, 4, 3), 200, dtype=np.uint8))
    batch = batcher.form_batch([lease_a, lease_b])
    means = batcher.infer_mean_rgb(batch)
    lease_a.release()
    lease_b.release()

    snapshot = pool.telemetry.snapshot()
    print(f"batch means: {means}")
    print(f"h2d_transfers={snapshot.h2d_transfers} h2d_bytes={snapshot.h2d_bytes}")
    print(f"pool_high_watermark={snapshot.pool_high_watermark}")
    print(f"pool_outstanding_after_release={pool.outstanding}")
    print("fake smoke test: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.fake_smoke_test:
        return run_fake_smoke_test()
    return run_probe()


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    sys.exit(main())


__all__ = ["main", "run_fake_smoke_test", "run_probe"]
