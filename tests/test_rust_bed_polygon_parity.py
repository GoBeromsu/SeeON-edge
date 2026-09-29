"""Ordered geometry differential; not decoder, inference or runtime admission.

Only the actual imported simplify_polygon produces expected results. Checked
slow-path coordinate rejection is a Rust admission test, not Python parity
outside that domain. Private Rust unit tests cover DP zero-length endpoints and
fallback midpoint ties; public simplification does not necessarily reach them.
No probe is built here. Missing opt-in is not qualification.
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

_ENV = "SEEON_TEST_BED_POLYGON_PROBE"
_SHA256 = "1564c2c1cb68bece11dadcc84b305b3ad0e1b6bb937ab979d023cef3afe93a8a"
_MAGIC = b"BDPOLY01"
_MAX_BYTES = 1024 * 1024
_MAX_POINTS = 4096
_TIMEOUT = 20.0
_REJECTION = b"bed-polygon-probe: rejected\n"
Polygon = tuple[tuple[int, int], ...]


def _probe_path() -> str:
    configured = os.environ.get(_ENV)
    if configured is None:
        pytest.skip("bed-polygon opt-in is absent; skip is not qualification")
    if not configured.strip() or "\0" in configured:
        pytest.fail("configured bed-polygon probe must name an executable", pytrace=False)
    try:
        path = Path(configured).expanduser().resolve()
        usable = path.is_file() and os.access(path, os.X_OK)
    except (OSError, RuntimeError, ValueError):
        pytest.fail("configured bed-polygon probe path is unusable", pytrace=False)
    if not usable:
        pytest.fail("configured bed-polygon probe is missing or unexecutable", pytrace=False)
    return str(path)


def _check_authority() -> None:
    if np.__version__ != "2.4.6":
        pytest.fail("bed-polygon differential requires NumPy 2.4.6", pytrace=False)
    try:
        digest = hashlib.sha256(Path(segmentation.__file__).read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        pytest.fail("cannot read the imported polygon authority", pytrace=False)
    if digest != _SHA256:
        pytest.fail("polygon authority differs from frozen SHA", pytrace=False)


@pytest.fixture
def bed_polygon_probe() -> str:
    executable = _probe_path()
    _check_authority()
    return executable


def _invoke(probe: str, payload: bytes) -> subprocess.CompletedProcess[bytes]:
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
                        event = selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ
                        selector.register(stream, event, name)
                    view, sent = memoryview(payload), 0
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
                                            "bed-polygon probe exceeded output bounds",
                                            pytrace=False,
                                        )
                    returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pytest.fail("bed-polygon probe failed within its execution bound", pytrace=False)
    return subprocess.CompletedProcess(
        [probe], returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])
    )


def _record(points: Polygon, capacity: int) -> bytes:
    assert len(points) <= _MAX_POINTS
    return struct.pack("<qI", capacity, len(points)) + b"".join(
        struct.pack("<qq", x, y) for x, y in points
    )


def _request(calls: list[tuple[Polygon, int]]) -> bytes:
    assert 1 <= len(calls) <= 4
    return (
        _MAGIC
        + struct.pack("<I", len(calls))
        + b"".join(_record(points, capacity) for points, capacity in calls)
    )


def _response(data: bytes, count: int) -> list[tuple[int, Polygon | None]]:
    assert len(data) >= 12 and data[:8] == _MAGIC
    assert struct.unpack_from("<I", data, 8)[0] == count
    result, offset = [], 12
    for _ in range(count):
        assert offset < len(data)
        status = data[offset]
        offset += 1
        assert status in (0, 1, 2, 3)
        if status:
            result.append((status, None))
            continue
        assert offset + 4 <= len(data)
        size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        assert size <= _MAX_POINTS and offset + size * 16 <= len(data)
        points = tuple(struct.unpack_from("<qq", data, offset + 16 * i) for i in range(size))
        offset += size * 16
        result.append((0, points))
    assert offset == len(data)
    return result


def _same_points(actual: Polygon, expected: Polygon) -> None:
    assert actual == expected


def _observed(probe: str, calls: list[tuple[Polygon, int]]) -> list[tuple[int, Polygon | None]]:
    result = _invoke(probe, _request(calls))
    assert result.returncode == 0 and result.stderr == b""
    return _response(result.stdout, len(calls))


def _compare(probe: str, calls: list[tuple[Polygon, int]]) -> list[Polygon]:
    expected = [segmentation.simplify_polygon(points, capacity) for points, capacity in calls]
    observed = _observed(probe, calls)
    for (status, actual), reference in zip(observed, expected, strict=True):
        assert status == 0 and actual is not None
        assert isinstance(reference, tuple)
        assert all(type(x) is int and type(y) is int for x, y in reference)
        _same_points(actual, reference)
    return expected


def test_comparator_rejects_rotation_reversal_and_coordinate_mutation() -> None:
    expected = ((1, 2), (5, 2), (5, 7), (1, 7))
    _same_points(expected, expected)
    for mutant in (expected[1:] + expected[:1], tuple(reversed(expected)), ((2, 2), *expected[1:])):
        with pytest.raises(AssertionError):
            _same_points(mutant, expected)


@pytest.mark.parametrize(
    "points",
    [
        (),
        ((4, 7),),
        ((4, 7), (9, 12)),
        ((3, 5),) * 9,
        ((0, 0), (10, 0), (10, 10), (0, 10)),
        ((0, 0), (10, 0), (10, 10), (0, 10), (0, 0)),
        ((0, 0), (0, 0), (5, 7), (10, 0), (10, 0), (5, -7)),
    ],
)
def test_short_duplicate_closed_and_tied_polygons(bed_polygon_probe: str, points: Polygon) -> None:
    _compare(bed_polygon_probe, [(points, cap) for cap in (1, 2, 3, 48)])
    _compare(bed_polygon_probe, [(points, max(1, len(points))), (points, len(points) + 1)])


def test_capacity_one_fallback_keeps_original_first_not_furthest_anchor(
    bed_polygon_probe: str,
) -> None:
    points = ((1, 1), (0, 0), (10, 0), (10, 10), (0, 10))
    expected = _compare(bed_polygon_probe, [(points, 1)])[0]
    assert expected == (points[0],)


def test_identical_slow_path_can_return_empty_without_a_minimum_polygon(
    bed_polygon_probe: str,
) -> None:
    expected = _compare(bed_polygon_probe, [(((7, 4),) * 8, 1)])[0]
    assert expected == ()


def _contour(kind: str) -> Polygon:
    mask = np.zeros((64, 72), dtype=bool)
    if kind == "rectangle":
        mask[2:61, 3:69] = True
    elif kind == "concave":
        mask[3:60, 4:68] = True
        mask[3:44, 20:44] = False
    elif kind == "staircase":
        for row in range(2, 60):
            mask[row, 3 : row + 4] = True
    elif kind == "shared-corners":
        for row in range(2, 55):
            mask[row, row] = True
    elif kind == "hole":
        mask[2:61, 3:69] = True
        mask[17:48, 18:55] = False
    else:
        raise AssertionError("unknown contour fixture")
    return segmentation.largest_external_contour(mask)


@pytest.mark.parametrize("kind", ["rectangle", "concave", "staircase", "shared-corners", "hole"])
def test_imported_contours_preserve_simplification_order(bed_polygon_probe: str, kind: str) -> None:
    points = _contour(kind)
    assert len(points) > 48
    _compare(bed_polygon_probe, [(points, cap) for cap in (2, 3, 12, 48)])


@pytest.mark.parametrize("seed", [607, 608, 1048576])
def test_deterministic_integer_polygons_and_reverse_order(
    bed_polygon_probe: str, seed: int
) -> None:
    rows = np.random.default_rng(seed).integers(-2000, 2001, size=(129, 2)).tolist()
    points = tuple((x, y) for x, y in rows)
    _compare(
        bed_polygon_probe, [(points, 2), (points, 48), (tuple(reversed(points)), 48), (points, 128)]
    )


def test_inclusive_slow_coordinate_domain_and_unrestricted_fast_copy(
    bed_polygon_probe: str,
) -> None:
    limit = 2**53
    admitted = ((-limit, 0), (0, limit), (limit, 0), (0, -limit), (1, 3))
    extremes = ((-(2**63), 2**63 - 1), (2**63 - 1, -(2**63)))
    _compare(
        bed_polygon_probe, [(admitted, 3), (admitted, 1), (extremes, 2), (extremes, 2**63 - 1)]
    )
    assert _observed(bed_polygon_probe, [(extremes, 1)]) == [(2, None)]
    for outside in (-limit - 1, limit + 1):
        points = ((outside, 0), (0, 1))
        assert _observed(bed_polygon_probe, [(points, 1)]) == [(2, None)]


def test_capacity_validation_precedes_even_an_empty_fast_path(bed_polygon_probe: str) -> None:
    calls = [((), 0), ((), -1), (((0, 0),), -(2**63)), (((0, 0),), 1)]
    for points, capacity in calls[:-1]:
        with pytest.raises(ValueError, match="max_points must be positive"):
            segmentation.simplify_polygon(points, capacity)
    assert _observed(bed_polygon_probe, calls) == [(1, None), (1, None), (1, None), (0, ((0, 0),))]


def test_repeated_calls_and_full_point_transport_capacity(bed_polygon_probe: str) -> None:
    points = tuple((i - 2048, (i * 17) % 331) for i in range(_MAX_POINTS))
    _compare(bed_polygon_probe, [(points, _MAX_POINTS)] * 4)
    short = ((4, 4), (1, 0), (9, 0), (9, 8), (1, 8))
    _compare(bed_polygon_probe, [(short, 1), (short, 3), (short, 1), (short, 3)])


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        _MAGIC,
        b"INVALID!" + struct.pack("<I", 1),
        _MAGIC + struct.pack("<I", 0),
        _MAGIC + struct.pack("<I", 5),
        _MAGIC + struct.pack("<I", 1),
        _MAGIC + struct.pack("<I", 1) + struct.pack("<q", 1),
        _MAGIC + struct.pack("<I", 1) + struct.pack("<qI", 1, _MAX_POINTS + 1),
        _MAGIC + struct.pack("<I", 1) + struct.pack("<qI", 1, 1) + b"\0" * 15,
        _MAGIC + struct.pack("<I", 1) + struct.pack("<qI", 1, 0) + b"x",
    ],
)
def test_malformed_requests_never_publish_a_prefix(bed_polygon_probe: str, payload: bytes) -> None:
    result = _invoke(bed_polygon_probe, payload)
    assert (result.returncode, result.stdout, result.stderr) == (2, b"", _REJECTION)


def test_later_truncation_rejects_the_whole_request(bed_polygon_probe: str) -> None:
    payload = _MAGIC + struct.pack("<I", 2) + _record(((1, 2),), 1) + struct.pack("<qI", 1, 1)
    result = _invoke(bed_polygon_probe, payload)
    assert (result.returncode, result.stdout, result.stderr) == (2, b"", _REJECTION)


def test_input_budget_cannot_be_bypassed_by_valid_prefix(bed_polygon_probe: str) -> None:
    prefix = _request([((), 1)])
    result = _invoke(bed_polygon_probe, prefix + b"\0" * (_MAX_BYTES + 1 - len(prefix)))
    assert (result.returncode, result.stdout, result.stderr) == (2, b"", _REJECTION)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"INVALID!" + struct.pack("<I", 1),
        _MAGIC + struct.pack("<I", 2),
        _MAGIC + struct.pack("<I", 1) + b"\4",
        _MAGIC + struct.pack("<I", 1) + b"\0" + struct.pack("<I", 1),
        _MAGIC + struct.pack("<I", 1) + b"\1x",
    ],
)
def test_response_decoder_rejects_bad_framing(data: bytes) -> None:
    with pytest.raises(AssertionError):
        _response(data, 1)


def test_absent_opt_in_is_not_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="not qualification"):
        _probe_path()


@pytest.mark.parametrize("value", ["", " \t\n", "synthetic\0probe"])
def test_explicit_invalid_probe_configuration_never_skips(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
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


def test_changed_imported_source_is_not_an_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "changed.py"
    path.write_text("different authority\n", encoding="utf-8")
    monkeypatch.setattr(np, "__version__", "2.4.6")
    monkeypatch.setattr(segmentation, "__file__", str(path))
    with pytest.raises(pytest.fail.Exception, match="differs from frozen SHA"):
        _check_authority()
