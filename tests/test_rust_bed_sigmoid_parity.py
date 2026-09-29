"""Exact numeric-result differential, not inference or whole-decoder qualification.

The oracle is the imported NumPy 2.4.6 sigmoid with float32 exp X86_V3 dispatch.
No exponential or sigmoid arithmetic is copied here. Rust consumes every f32
bit pattern, including nonfinite intermediates; this does not admit nonfinite
model outputs into a decoder. Warning flags are not part of this comparison.
Absent probe opt-in may skip ordinary CI; explicit invalid opt-in must fail.
"""

from __future__ import annotations

import hashlib
import os
import selectors
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from worker.adapters.model import seg_postprocess as segmentation

_ENV = "SEEON_TEST_BED_SIGMOID_PROBE"
_SOURCE_SHA256 = "1564c2c1cb68bece11dadcc84b305b3ad0e1b6bb937ab979d023cef3afe93a8a"
_MAGIC = b"BDSIGM01"
_MAX_COUNT = 1_048_576
_MAX_BYTES = 12 + 4 * _MAX_COUNT
_TIMEOUT = 20.0
_REJECTION = b"bed-sigmoid-probe: rejected\n"


def _probe_path() -> str:
    configured = os.environ.get(_ENV)
    if configured is None:
        pytest.skip("bed-sigmoid opt-in is absent; skip is not qualification")
    if not configured.strip() or "\0" in configured:
        pytest.fail("configured bed-sigmoid probe must name an executable", pytrace=False)
    try:
        executable = Path(configured).expanduser().resolve()
        usable = executable.is_file() and os.access(executable, os.X_OK)
    except (OSError, RuntimeError, ValueError):
        pytest.fail("configured bed-sigmoid probe path is unusable", pytrace=False)
    if not usable:
        pytest.fail("configured bed-sigmoid probe is missing or unexecutable", pytrace=False)
    return str(executable)


def _check_authority() -> None:
    if np.__version__ != "2.4.6":
        pytest.fail("bed-sigmoid differential requires NumPy 2.4.6", pytrace=False)
    try:
        digest = hashlib.sha256(Path(segmentation.__file__).read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        pytest.fail("cannot read the imported bed sigmoid source", pytrace=False)
    if digest != _SOURCE_SHA256:
        pytest.fail("bed sigmoid source differs from frozen SHA", pytrace=False)
    dispatch = np.lib.introspect.opt_func_info(func_name="^exp$")
    if dispatch.get("exp", {}).get("ff", {}).get("current") != "X86_V3":
        pytest.fail("bed-sigmoid differential requires float32 exp X86_V3", pytrace=False)


@pytest.fixture
def bed_sigmoid_probe() -> str:
    executable = _probe_path()
    _check_authority()
    return executable


def _invoke(probe: str, payload: bytes) -> subprocess.CompletedProcess[bytes]:
    # The extra byte is permitted only to exercise the input-budget rejection.
    assert len(payload) <= _MAX_BYTES + 1
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + _TIMEOUT
    try:
        with subprocess.Popen(
            [probe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            try:
                assert process.stdin is not None
                assert process.stdout is not None
                assert process.stderr is not None
                with selectors.DefaultSelector() as selector:
                    for name in ("stdin", "stdout", "stderr"):
                        stream = getattr(process, name)
                        os.set_blocking(stream.fileno(), False)
                        events = selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ
                        selector.register(stream, events, name)
                    sent = 0
                    view = memoryview(payload)
                    while selector.get_map():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired([probe], _TIMEOUT)
                        events = selector.select(remaining)
                        if not events:
                            raise subprocess.TimeoutExpired([probe], _TIMEOUT)
                        for key, _ in events:
                            stream = key.fileobj
                            if key.data == "stdin":
                                try:
                                    written = os.write(stream.fileno(), view[sent : sent + 65536])
                                except BlockingIOError:
                                    continue
                                except BrokenPipeError:
                                    written = 0
                                sent += written
                                if sent == len(payload) or written == 0:
                                    selector.unregister(stream)
                                    stream.close()
                            else:
                                target = buffers[key.data]
                                cap = _MAX_BYTES if key.data == "stdout" else 1024
                                try:
                                    chunk = os.read(
                                        stream.fileno(), min(65536, cap - len(target) + 1)
                                    )
                                except BlockingIOError:
                                    continue
                                if not chunk:
                                    selector.unregister(stream)
                                    stream.close()
                                else:
                                    target.extend(chunk)
                                    if len(target) > cap:
                                        pytest.fail(
                                            "bed-sigmoid probe exceeded output bounds",
                                            pytrace=False,
                                        )
                    returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pytest.fail("bed-sigmoid probe failed within its execution bound", pytrace=False)
    return subprocess.CompletedProcess(
        [probe], returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])
    )


def _request(values: np.ndarray) -> bytes:
    assert values.dtype == np.float32 and values.ndim == 1 and values.size <= _MAX_COUNT
    bits = values.view(np.uint32).astype("<u4", copy=False)
    return _MAGIC + struct.pack("<I", values.size) + bits.tobytes(order="C")


def _response(data: bytes, count: int) -> np.ndarray:
    assert len(data) >= 12 and data[:8] == _MAGIC
    assert struct.unpack_from("<I", data, 8)[0] == count
    assert len(data) == 12 + 4 * count
    return np.frombuffer(data, dtype="<u4", offset=12).copy()


def _same_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual, expected)


def _compare(probe: str, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype=np.float32)
    expected = segmentation._sigmoid(values)
    assert expected.shape == values.shape and expected.dtype == np.float32
    result = _invoke(probe, _request(values))
    assert result.returncode == 0 and result.stderr == b""
    _same_bits(_response(result.stdout, values.size), expected.view(np.uint32))


def _neighbours(centers: list[np.float32], steps: int) -> np.ndarray:
    points = []
    for center in centers:
        points.append(center)
        for direction in (np.float32("-inf"), np.float32("inf")):
            value = center
            for _ in range(steps):
                value = np.nextafter(value, direction)
                points.append(value)
    return np.asarray(points, dtype=np.float32)


def test_exact_bits_comparator_rejects_one_bit_change() -> None:
    expected = np.array([0x3F000000, 0x00000000], dtype=np.uint32)
    _same_bits(expected.copy(), expected)
    changed = expected.copy()
    changed[0] ^= np.uint32(1)
    with pytest.raises(AssertionError):
        _same_bits(changed, expected)


def test_signed_zero_subnormals_extrema_and_nonfinite_intermediates(bed_sigmoid_probe: str) -> None:
    bits = np.array(
        [
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0x00800000,
            0x80800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0xFFC00000,
            0x7FC00001,
            0xFFC12345,
            0x7FFFFFFF,
            0xFFFFFFFF,
            0x7F800001,
            0xFF800001,
            0x7FBFFFFF,
            0xFFBFFFFF,
        ],
        dtype=np.uint32,
    )
    _compare(bed_sigmoid_probe, bits.view(np.float32))


def test_zero_rounding_and_saturation_neighbours(bed_sigmoid_probe: str) -> None:
    centers = [
        np.float32(0),
        np.float32(-0.0),
        np.float32(2.9802322e-8),
        np.float32(-2.9802322e-8),
        np.float32(88.72283935546875),
        np.float32(-88.72283935546875),
        np.float32(103.97208404541015625),
        np.float32(-103.97208404541015625),
    ]
    _compare(bed_sigmoid_probe, _neighbours(centers, 256))


@pytest.mark.parametrize("length", [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64, 127, 128])
def test_branch_populations_and_vector_tails(bed_sigmoid_probe: str, length: int) -> None:
    values = np.resize(
        np.array([-100, -5, -1, -0.000001, -0.0, 0, 0.000001, 1, 5, 100], dtype=np.float32), length
    )
    _compare(bed_sigmoid_probe, values)
    _compare(bed_sigmoid_probe, np.sort(values))


def test_dense_finite_range(bed_sigmoid_probe: str) -> None:
    _compare(bed_sigmoid_probe, np.linspace(-104, 104, 200001, dtype=np.float32))


def test_seeded_finite_bit_patterns(bed_sigmoid_probe: str) -> None:
    bits = np.random.default_rng(607591).integers(0, 2**32, _MAX_COUNT, dtype=np.uint32)
    values = bits.view(np.float32)
    _compare(bed_sigmoid_probe, values[np.isfinite(values)])


def test_full_transport_capacity_preserves_all_results(bed_sigmoid_probe: str) -> None:
    _compare(bed_sigmoid_probe, np.zeros(_MAX_COUNT, dtype=np.float32))


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        _MAGIC,
        b"INVALID!" + struct.pack("<I", 0),
        _MAGIC + struct.pack("<I", 1),
        _MAGIC + struct.pack("<I", 1) + b"\0" * 3,
        _MAGIC + struct.pack("<I", 0) + b"x",
        _MAGIC + struct.pack("<I", 1) + b"\0" * 5,
        _MAGIC + struct.pack("<I", _MAX_COUNT + 1),
    ],
)
def test_malformed_transport_never_publishes_prefix(bed_sigmoid_probe: str, payload: bytes) -> None:
    result = _invoke(bed_sigmoid_probe, payload)
    assert (result.returncode, result.stdout, result.stderr) == (2, b"", _REJECTION)


def test_one_byte_above_transport_budget_is_rejected(bed_sigmoid_probe: str) -> None:
    payload = _MAGIC + struct.pack("<I", _MAX_COUNT) + b"\0" * (4 * _MAX_COUNT + 1)
    result = _invoke(bed_sigmoid_probe, payload)
    assert (result.returncode, result.stdout, result.stderr) == (2, b"", _REJECTION)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"BADMAGIC" + struct.pack("<I", 0),
        _MAGIC + struct.pack("<I", 1),
        _MAGIC + struct.pack("<I", 0) + b"x",
    ],
)
def test_response_decoder_rejects_incomplete_or_trailing_output(data: bytes) -> None:
    with pytest.raises(AssertionError):
        _response(data, 0)


def test_absent_opt_in_is_explicitly_not_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="not qualification"):
        _probe_path()


@pytest.mark.parametrize("value", ["", " \t\n", "synthetic\0probe"])
def test_explicit_invalid_probe_never_skips(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setattr(os, "environ", {_ENV: value})
    with pytest.raises(pytest.fail.Exception, match="must name an executable"):
        _probe_path()


@pytest.mark.parametrize("kind", ["missing", "directory", "nonexecutable"])
def test_unusable_explicit_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    path = tmp_path / "probe"
    if kind == "directory":
        path.mkdir()
    elif kind == "nonexecutable":
        path.write_bytes(b"not executable")
        path.chmod(0o600)
    monkeypatch.setenv(_ENV, str(path))
    with pytest.raises(pytest.fail.Exception, match="missing or unexecutable"):
        _probe_path()


def test_wrong_numpy_version_is_not_an_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(np, "__version__", "unqualified")
    with pytest.raises(pytest.fail.Exception, match="requires NumPy 2.4.6"):
        _check_authority()


def test_wrong_imported_source_is_not_an_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "changed.py"
    path.write_text("different authority\n", encoding="utf-8")
    monkeypatch.setattr(np, "__version__", "2.4.6")
    monkeypatch.setattr(segmentation, "__file__", str(path))
    with pytest.raises(pytest.fail.Exception, match="differs from frozen SHA"):
        _check_authority()


def test_wrong_exp_dispatch_is_not_an_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(np, "__version__", "2.4.6")
    monkeypatch.setattr(
        np.lib.introspect,
        "opt_func_info",
        lambda **kwargs: {"exp": {"ff": {"current": "baseline(X86_V2)"}}},
    )
    with pytest.raises(pytest.fail.Exception, match="requires float32 exp X86_V3"):
        _check_authority()
