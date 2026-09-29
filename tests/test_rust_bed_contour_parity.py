"""Differential for largest_external_contour only.

Not a decoder, GPU, or Worker qualification. No simplification, sigmoid, model,
or area/IoU oracle. SEEON_TEST_BED_CONTOUR_PROBE must name a parent-built
executable. Absent opt-in may skip ordinary CI and is NOT qualification.
Invalid explicit configuration fails. pytest never builds anything. The
protocol is documented in bed_contour_probe.rs.

The imported seg_postprocess.py source is pinned at SHA256
1564c2c1cb68bece11dadcc84b305b3ad0e1b6bb937ab979d023cef3afe93a8a with NumPy
2.4.6. Vertex equality is the imported function's ordered sequence, which
fixes the start, winding, and order together.
"""

from __future__ import annotations

import hashlib
import os
import selectors
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from worker.adapters.model import seg_postprocess as segmentation

_ENV = "SEEON_TEST_BED_CONTOUR_PROBE"
_FROZEN_SEG_SHA256 = "1564c2c1cb68bece11dadcc84b305b3ad0e1b6bb937ab979d023cef3afe93a8a"
_FROZEN_NUMPY = "2.4.6"
_MAGIC = b"BEDCNT01"
_REJECTED = b"bed-contour-probe: rejected\n"
_MAX_INPUT = 48 * 1024 * 1024
_MAX_OUTPUT = 1024 * 1024
_MAX_CALLS = 4
_TIMEOUT = 20.0
_OK = 0
_DIMENSIONS = 1
_OVERFLOW = 2
_LENGTH = 3
_VALUE = 4
_DOMAIN = 5
# Status 6 is allocation. A parity process cannot force it; it is not a quota.
_SEMANTIC = (_DIMENSIONS, _OVERFLOW, _LENGTH, _VALUE, _DOMAIN, 6)
_I64_MIN = -1 << 63
_I64_MAX = (1 << 63) - 1
_POINT = struct.Struct("<qq")


def _probe_path() -> str:
    configured = os.environ.get(_ENV)
    if configured is None:
        pytest.skip("bed-contour opt-in is absent; skip is not qualification")
    if not configured.strip() or "\0" in configured:
        pytest.fail("configured bed-contour probe must name an executable", pytrace=False)
    try:
        executable = Path(configured).expanduser().resolve()
        usable = executable.is_file() and os.access(executable, os.X_OK)
    except (OSError, RuntimeError, ValueError):
        pytest.fail("configured bed-contour probe path is unusable", pytrace=False)
    if not usable:
        pytest.fail("configured bed-contour probe is missing or unexecutable", pytrace=False)
    return str(executable)


def _check_numpy() -> None:
    if np.__version__ != _FROZEN_NUMPY:
        pytest.fail(
            f"bed-contour differential requires NumPy {_FROZEN_NUMPY}, got {np.__version__}",
            pytrace=False,
        )


def _check_source(module: ModuleType, expected: str, label: str) -> None:
    try:
        digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        pytest.fail(f"cannot read the actual imported bed-contour {label}", pytrace=False)
    if digest != expected:
        pytest.fail(f"bed-contour {label} differs from frozen SHA: {digest}", pytrace=False)


@pytest.fixture
def bed_contour_probe() -> str:
    executable = _probe_path()
    _check_numpy()
    _check_source(segmentation, _FROZEN_SEG_SHA256, "contour source")
    return executable


def _invoke(probe: str, payload: bytes) -> subprocess.CompletedProcess[bytes]:
    # One extra input byte is allowed here ONLY to test the probe's budget guard.
    assert len(payload) <= _MAX_INPUT + 1
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
                                cap = _MAX_OUTPUT if key.data == "stdout" else 1024
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
                                            "bed-contour probe exceeded output bounds; "
                                            f"stderr={bytes(buffers['stderr'])!r}",
                                            pytrace=False,
                                        )
                    returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pytest.fail(
            "bed-contour probe failed to execute within its bound; "
            f"stderr={bytes(buffers['stderr'])!r}",
            pytrace=False,
        )
    return subprocess.CompletedProcess(
        [probe], returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])
    )


def _proven(width: int, height: int) -> bool:
    if width < 0 or height < 0:
        return False
    pixels = width * height
    return 4 * pixels * pixels <= 1 << 53


def _smallest_unproven_pixels() -> int:
    proven = 0
    step = 1 << 26
    while step:
        candidate = proven + step
        if 4 * candidate * candidate <= 1 << 53:
            proven = candidate
        step >>= 1
    unproven = proven + 1
    assert 4 * proven * proven <= 1 << 53
    assert 4 * unproven * unproven > 1 << 53
    return unproven


def _imported(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    polygon = segmentation.largest_external_contour(mask)
    if not isinstance(polygon, tuple):
        pytest.fail("imported largest_external_contour did not return a tuple", pytrace=False)
    for point in polygon:
        if not isinstance(point, tuple) or len(point) != 2:
            pytest.fail("imported contour point is not a coordinate pair", pytrace=False)
        x_coord, y_coord = point
        coords_are_ints = (
            isinstance(x_coord, int)
            and isinstance(y_coord, int)
            and not isinstance(x_coord, bool)
            and not isinstance(y_coord, bool)
        )
        if not coords_are_ints:
            pytest.fail("imported contour coordinate is not an int", pytrace=False)
    return polygon


def _same_vertices(
    actual: tuple[tuple[int, int], ...],
    expected: tuple[tuple[int, int], ...],
) -> None:
    assert actual == expected


def _raw_record(width: int, height: int, mask: bytes) -> bytes:
    return struct.pack("<qqI", width, height, len(mask)) + mask


def _mask_record(mask: np.ndarray) -> bytes:
    assert mask.dtype == np.bool_ and mask.ndim == 2
    values = np.ascontiguousarray(mask, dtype=np.uint8)
    height, width = values.shape
    payload = values.tobytes(order="C")
    assert len(payload) == height * width
    return _raw_record(int(width), int(height), payload)


def _payload(records: list[bytes]) -> bytes:
    return _MAGIC + struct.pack("<I", len(records)) + b"".join(records)


@dataclass(frozen=True)
class _Observed:
    status: int
    vertices: tuple[tuple[int, int], ...] | None


def _observations(probe: str, records: list[bytes]) -> list[_Observed]:
    assert 1 <= len(records) <= _MAX_CALLS
    result = _invoke(probe, _payload(records))
    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    data = result.stdout
    assert data[:12] == _MAGIC + struct.pack("<I", len(records))
    offset = 12
    observations: list[_Observed] = []
    for _ in records:
        assert offset < len(data)
        status = data[offset]
        offset += 1
        if status == _OK:
            assert offset + 4 <= len(data)
            count = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            points: list[tuple[int, int]] = []
            for _ in range(count):
                assert offset + _POINT.size <= len(data)
                x_coord, y_coord = _POINT.unpack_from(data, offset)
                offset += _POINT.size
                points.append((x_coord, y_coord))
            vertices: tuple[tuple[int, int], ...] | None = tuple(points)
        else:
            assert status in _SEMANTIC
            vertices = None
        observations.append(_Observed(status, vertices))
    assert offset == len(data)
    return observations


def _exercise(probe: str, masks: list[np.ndarray]) -> list[tuple[tuple[int, int], ...]]:
    assert 1 <= len(masks) <= _MAX_CALLS
    expected: list[tuple[tuple[int, int], ...]] = []
    records: list[bytes] = []
    for mask in masks:
        height, width = mask.shape
        assert _proven(int(width), int(height))
        expected.append(_imported(mask))
        records.append(_mask_record(mask))
    observed = _observations(probe, records)
    assert len(observed) == len(expected)
    for actual, polygon in zip(observed, expected, strict=True):
        assert actual.status == _OK
        assert actual.vertices is not None
        _same_vertices(actual.vertices, polygon)
    return expected


def _border(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _ring(height: int, width: int, margin: int = 1) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    mask[margin:-margin, margin:-margin] = False
    return mask


def _diagonal(size: int, anti: bool) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for index in range(size):
        column = size - 1 - index if anti else index
        mask[index, column] = True
    return mask


def _both_diagonals(size: int) -> np.ndarray:
    return _diagonal(size, False) | _diagonal(size, True)


def _staircase(steps: int) -> np.ndarray:
    mask = np.zeros((steps, steps + 1), dtype=bool)
    for index in range(steps):
        mask[index, index] = True
        mask[index, index + 1] = True
    return mask


def _separated(second_width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canvas = 6 + second_width
    full = np.zeros((6, canvas), dtype=bool)
    full[1:4, 1:4] = True
    full[1:4, 6 : 6 + second_width] = True
    first = np.zeros_like(full)
    first[1:4, 1:4] = True
    second = np.zeros_like(full)
    second[1:4, 6 : 6 + second_width] = True
    return full, first, second


def test_comparator_rejects_rotation_reversal_and_changed_point() -> None:
    _check_numpy()
    _check_source(segmentation, _FROZEN_SEG_SHA256, "contour source")
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 2:6] = True
    expected = _imported(mask)
    assert len(expected) >= 4
    _same_vertices(expected, expected)
    mutants = (
        expected[1:] + expected[:1],
        tuple(reversed(expected)),
        ((expected[0][0] + 1, expected[0][1]), *expected[1:]),
    )
    for mutant in mutants:
        with pytest.raises(AssertionError):
            _same_vertices(mutant, expected)


def test_empty_and_one_pixel_match_imported_vertices(bed_contour_probe: str) -> None:
    empty = np.zeros((4, 5), dtype=bool)
    flat = np.zeros((0, 6), dtype=bool)
    tall = np.zeros((6, 0), dtype=bool)
    zero = np.zeros((0, 0), dtype=bool)
    _exercise(bed_contour_probe, [empty, flat, tall, zero])
    pixel = np.zeros((4, 5), dtype=bool)
    pixel[2, 3] = True
    single = np.ones((1, 1), dtype=bool)
    origin = np.zeros((1, 2), dtype=bool)
    origin[0, 0] = True
    _exercise(bed_contour_probe, [pixel, single, origin, empty])


def test_strips_and_rectangles_match_imported_vertices(bed_contour_probe: str) -> None:
    horizontal = np.zeros((5, 12), dtype=bool)
    horizontal[2, :] = True
    vertical = np.zeros((12, 4), dtype=bool)
    vertical[:, 1] = True
    solid = np.ones((4, 7), dtype=bool)
    inset = np.zeros((8, 9), dtype=bool)
    inset[2:6, 1:8] = True
    _exercise(bed_contour_probe, [horizontal, vertical, solid, inset])


@pytest.mark.parametrize(("height", "width"), [(3, 3), (3, 9), (8, 3), (6, 7)])
def test_all_border_rings_match_imported_vertices(
    bed_contour_probe: str, height: int, width: int
) -> None:
    _exercise(bed_contour_probe, [_border(height, width)])


def test_both_diagonals_match_imported_vertices(bed_contour_probe: str) -> None:
    _exercise(
        bed_contour_probe,
        [_both_diagonals(1), _both_diagonals(2), _both_diagonals(7), _both_diagonals(8)],
    )
    _exercise(
        bed_contour_probe,
        [
            _diagonal(6, False),
            _diagonal(6, True),
            _diagonal(5, False),
            _diagonal(5, True),
        ],
    )


def test_holes_match_imported_vertices(bed_contour_probe: str) -> None:
    inset = np.zeros((10, 12), dtype=bool)
    inset[1:9, 1:11] = True
    inset[3:7, 4:9] = False
    _exercise(
        bed_contour_probe,
        [_ring(5, 5), _ring(4, 9), _ring(9, 4), _ring(7, 8, 2)],
    )
    _exercise(bed_contour_probe, [inset, _border(5, 8), _ring(6, 6), _ring(8, 5, 2)])


def test_equal_area_disconnected_components_keep_the_first_maximum(
    bed_contour_probe: str,
) -> None:
    # Which component wins is decided by the imported function, not a local area.
    full, first, second = _separated(3)
    selected = _imported(full)
    assert selected == _imported(first)
    assert selected != _imported(second)
    assert _exercise(bed_contour_probe, [full]) == [selected]


def test_larger_disconnected_component_replaces_the_earlier_loop(
    bed_contour_probe: str,
) -> None:
    full, first, second = _separated(4)
    selected = _imported(full)
    assert selected == _imported(second)
    assert selected != _imported(first)
    assert _exercise(bed_contour_probe, [full]) == [selected]


def test_corner_touch_branches_match_imported_vertices(bed_contour_probe: str) -> None:
    rectangles = np.zeros((7, 7), dtype=bool)
    rectangles[0:3, 0:3] = True
    rectangles[3:6, 3:6] = True
    branch = np.zeros((5, 5), dtype=bool)
    branch[0, 0] = True
    branch[1, 1] = True
    branch[1, 2] = True
    branch[2, 1] = True
    pair = np.zeros((2, 2), dtype=bool)
    pair[0, 0] = True
    pair[1, 1] = True
    _exercise(bed_contour_probe, [rectangles, branch, pair, _both_diagonals(4)])


@pytest.mark.parametrize(
    ("height", "width"),
    [(1, 8), (8, 1), (2, 17), (17, 2), (5, 9), (9, 5), (3, 14), (14, 3)],
)
def test_fixed_landscape_and_portrait_masks_match_imported_vertices(
    bed_contour_probe: str, height: int, width: int
) -> None:
    mask = np.zeros((height, width), dtype=bool)
    mask[::2, ::2] = True
    solid = np.ones((height, width), dtype=bool)
    _exercise(bed_contour_probe, [mask, solid])


@pytest.mark.parametrize(
    ("seed", "height", "width", "threshold"),
    [
        (731, 9, 14, 0.45),
        (8128, 15, 6, 0.35),
        (65537, 7, 7, 0.5),
        (19, 2, 21, 0.55),
        (23, 21, 2, 0.55),
        (29, 11, 4, 0.4),
        (31, 4, 11, 0.4),
        (37, 3, 16, 0.5),
        (41, 16, 3, 0.5),
    ],
)
def test_seeded_landscape_and_portrait_masks_match_imported_vertices(
    bed_contour_probe: str, seed: int, height: int, width: int, threshold: float
) -> None:
    mask = np.random.default_rng(seed).random((height, width)) < threshold
    _exercise(bed_contour_probe, [mask])


def test_contours_above_48_points_are_unchanged(bed_contour_probe: str) -> None:
    masks = [np.ones((2, 30), dtype=bool), np.ones((30, 2), dtype=bool), _staircase(24)]
    for mask in masks:
        assert len(_imported(mask)) > 48
    _exercise(bed_contour_probe, masks)


def test_repeat_calls_are_stateless(bed_contour_probe: str) -> None:
    one = np.zeros((3, 4), dtype=bool)
    one[1, 2] = True
    block = np.ones((3, 5), dtype=bool)
    empty = np.zeros((2, 2), dtype=bool)
    hole = _ring(5, 6)
    first = _exercise(bed_contour_probe, [one, block, empty, one])
    assert first[0] == first[3]
    second = _exercise(bed_contour_probe, [hole, one, block, empty])
    assert second[1] == first[0]
    assert second[2] == first[1]
    assert second[3] == first[2]


def test_strided_mask_matches_logical_row_major_values(bed_contour_probe: str) -> None:
    base = np.zeros((8, 9), dtype=bool)
    base[1:7, 1:8] = True
    base[3:5, 3:6] = False
    view = base[::2, ::2]
    assert not view.flags.c_contiguous
    _exercise(bed_contour_probe, [view])


def test_empty_mask_outside_shoelace_domain_matches_imported_empty(
    bed_contour_probe: str,
) -> None:
    # No shoelace is evaluated. This is not area equivalence outside the domain.
    pixels = _smallest_unproven_pixels()
    assert not _proven(pixels, 1)
    mask = np.zeros((1, pixels), dtype=bool)
    assert _imported(mask) == ()
    record = _mask_record(mask)
    assert len(_payload([record])) <= _MAX_INPUT
    observed = _observations(bed_contour_probe, [record])
    assert observed[0].status == _OK
    assert observed[0].vertices == ()


def test_rust_only_nonempty_outside_shoelace_domain_is_refused(
    bed_contour_probe: str,
) -> None:
    # Finite ndarray would still trace this. That result is not the oracle here.
    pixels = _smallest_unproven_pixels()
    assert not _proven(pixels, 1)
    raw = bytearray(pixels)
    raw[0] = 1
    record = _raw_record(pixels, 1, bytes(raw))
    assert len(_payload([record])) <= _MAX_INPUT
    observed = _observations(bed_contour_probe, [record])
    assert observed[0].status == _DOMAIN
    assert observed[0].vertices is None


@pytest.mark.parametrize(
    ("width", "height", "mask", "status"),
    [
        (-1, 1, b"", _DIMENSIONS),
        (1, -1, b"\x00", _DIMENSIONS),
        (_I64_MIN, 0, b"", _DIMENSIONS),
        (_I64_MAX, _I64_MAX, b"", _OVERFLOW),
        (_I64_MAX, 3, b"", _OVERFLOW),
        (2, 2, b"\x01\x00\x01", _LENGTH),
        (1, 1, b"", _LENGTH),
        (0, 0, b"\x00", _LENGTH),
        (1, 1, b"\x02", _VALUE),
        (2, 1, b"\x01\xff", _VALUE),
    ],
)
def test_rust_only_semantic_bounds_are_not_python_ndarray_cases(
    bed_contour_probe: str, width: int, height: int, mask: bytes, status: int
) -> None:
    observed = _observations(bed_contour_probe, [_raw_record(width, height, mask)])
    assert observed[0].status == status
    assert observed[0].vertices is None


def test_semantic_error_keeps_earlier_exact_vertices(bed_contour_probe: str) -> None:
    mask = np.ones((2, 3), dtype=bool)
    observed = _observations(
        bed_contour_probe,
        [_mask_record(mask), _raw_record(2, 2, b"\x01\x00\x01"), _mask_record(mask)],
    )
    expected = _imported(mask)
    assert observed[0].vertices is not None
    assert observed[2].vertices is not None
    _same_vertices(observed[0].vertices, expected)
    assert observed[1].status == _LENGTH
    assert observed[1].vertices is None
    _same_vertices(observed[2].vertices, expected)


def test_exact_input_budget_is_transport_valid_and_semantic_length_is_observable(
    bed_contour_probe: str,
) -> None:
    overhead = len(_payload([_raw_record(1, 1, b"")]))
    assert overhead == 32
    record = _raw_record(1, 1, b"\x00" * (_MAX_INPUT - overhead))
    assert len(_payload([record])) == _MAX_INPUT
    observed = _observations(bed_contour_probe, [record])
    assert observed[0].status == _LENGTH
    assert observed[0].vertices is None


def test_probe_output_budget_rejects_without_truncation(bed_contour_probe: str) -> None:
    # The core has no vertex quota. This mask is in-domain and only the probe
    # print buffer is too small, so the response must not be a shortened contour.
    width = (_MAX_OUTPUT // 16) + 8
    mask = np.ones((1, width), dtype=bool)
    assert _proven(width, 1)
    polygon = _imported(mask)
    encoded = 12 + 1 + 4 + 16 * len(polygon)
    assert encoded > _MAX_OUTPUT
    result = _invoke(bed_contour_probe, _payload([_mask_record(mask)]))
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == _REJECTED


@pytest.mark.parametrize(
    "defect",
    [
        "empty",
        "magic",
        "short-magic",
        "short-count",
        "zero-calls",
        "call-bound",
        "short-width",
        "short-height",
        "short-length",
        "short-mask",
        "mask-count",
        "trailing",
        "late-truncation",
        "input-bound",
    ],
)
def test_malformed_transport_rejects_atomically_with_private_static_diagnostic(
    bed_contour_probe: str, defect: str
) -> None:
    record = _raw_record(1, 1, b"\x01")
    payload = _payload([record])
    if defect == "empty":
        payload = b""
    elif defect == "magic":
        payload = b"PRIVATE!" + payload[8:]
    elif defect in ("short-magic", "short-count", "short-width", "short-height", "short-length"):
        length = {
            "short-magic": 7,
            "short-count": 11,
            "short-width": 19,
            "short-height": 27,
            "short-length": 31,
        }[defect]
        payload = payload[:length]
    elif defect == "zero-calls":
        payload = _payload([])
    elif defect == "call-bound":
        payload = _MAGIC + struct.pack("<I", _MAX_CALLS + 1)
    elif defect == "short-mask":
        payload = payload[:-1]
    elif defect == "mask-count":
        payload = _payload([struct.pack("<qqI", 1, 1, 0xFFFFFFFF)])
    elif defect == "trailing":
        payload += b"private-path-and-token"
    elif defect == "late-truncation":
        payload = _MAGIC + struct.pack("<I", 2) + record
    else:
        assert defect == "input-bound"
        payload = payload.ljust(_MAX_INPUT + 1, b"x")
    result = _invoke(bed_contour_probe, payload)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == _REJECTED


def test_absent_opt_in_skips_and_is_not_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="not qualification"):
        _probe_path()


@pytest.mark.parametrize("setting", ["empty", "blank", "missing", "directory", "unexecutable"])
def test_invalid_explicit_opt_in_fails_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, setting: str
) -> None:
    nonexecutable = tmp_path / "not-executable"
    nonexecutable.write_bytes(b"not a probe")
    nonexecutable.chmod(0o600)
    value = {
        "empty": "",
        "blank": " \t\n",
        "missing": str(tmp_path / "absent-probe"),
        "directory": str(tmp_path),
        "unexecutable": str(nonexecutable),
    }[setting]
    monkeypatch.setenv(_ENV, value)
    with pytest.raises(pytest.fail.Exception, match="configured bed-contour probe"):
        _probe_path()


def test_nul_opt_in_reaches_the_validator_without_os_setenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # setenv rejects NUL before the validator runs; replace the mapping instead.
    monkeypatch.setattr(os, "environ", {**os.environ, _ENV: "private\0probe"})
    with pytest.raises(pytest.fail.Exception, match="configured bed-contour probe"):
        _probe_path()


@pytest.mark.parametrize("version", ["", "2.4.5", "2.4.6.dev0", "2.4.6+local"])
def test_nonfrozen_numpy_fails_instead_of_weakening_vertex_comparisons(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(np, "__version__", version)
    with pytest.raises(pytest.fail.Exception, match="requires NumPy 2.4.6"):
        _check_numpy()


@pytest.mark.parametrize("defect", ["different", "unreadable", "missing-file-attribute"])
def test_imported_source_must_match_the_frozen_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    path = tmp_path / "source.py"
    if defect == "different":
        path.write_bytes(b"not the frozen contour source\n")
    monkeypatch.setattr(
        segmentation, "__file__", None if defect == "missing-file-attribute" else str(path)
    )
    message = (
        "differs from frozen SHA" if defect == "different" else "cannot read the actual imported"
    )
    with pytest.raises(pytest.fail.Exception, match=message):
        _check_source(segmentation, _FROZEN_SEG_SHA256, "contour source")
