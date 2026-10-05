"""Original G002 stored-pose differential, NOT inference/runtime qualification.

Opt in with SEEON_TEST_STORED_POSE_PROBE pointing to the parent-built executable.
Absent opt-in skips; empty/invalid opt-in, a different consumer, or different NumPy
fails. No builds are performed here. The frozen consumer SHA and NumPy 2.4.6 come
from the parent-inspected CPU image
fffd430545508b7473644975c25e077367d8b42cf032085e96d8f5dcca49171f.
This suite binds source/numeric semantics, not the identity of the running image.

The ACTUAL imported OrtClipPoseRunner consumes an artifact-fixture model and a
recording session returning supplied float32 output0. Its tensor and ordered box
bits are the oracle; no resize/filter/unletterbox implementation is copied here.
Nonfinite rejection and transport limits are separate Rust admission tests, not
claims that Python rejects those inputs. No GPU gate or tolerance is changed.
The complete binary grammar is in worker/policy/examples/stored_pose_probe.rs.
"""

from __future__ import annotations

import hashlib
import math
import os
import selectors
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from worker.adapters.model import ort_clip_pose as canonical
from worker.adapters.model.errors import ModelLoadError

_ENV = "SEEON_TEST_STORED_POSE_PROBE"
_FROZEN_CONSUMER_SHA256 = "310ceb4276980c9adcc4a9b3ea58e8b0a41c0eafb54ce8d3d05819073fbf88e1"
_FROZEN_NUMPY = "2.4.6"
_MAGIC = b"STPOSE01"
_MAX_INPUT = 16 * 1024 * 1024
_MAX_OUTPUT = 40 * 1024 * 1024
_TIMEOUT = 10.0
_TENSOR_VALUES = 3 * 640 * 640
_OUTPUT_SHAPE = (1, 300, 57)


def _probe_path() -> str:
    configured = os.environ.get(_ENV)
    if configured is None:
        pytest.skip("stored-pose opt-in is absent; skip is not qualification")
    if not configured:
        pytest.fail("configured stored-pose probe must name an executable", pytrace=False)
    try:
        executable = Path(configured).expanduser().resolve()
        usable = executable.is_file() and os.access(executable, os.X_OK)
    except (OSError, RuntimeError, ValueError):
        pytest.fail("configured stored-pose probe path is unusable", pytrace=False)
    if not usable:
        pytest.fail("configured stored-pose probe is missing or unexecutable", pytrace=False)
    return str(executable)


@pytest.fixture
def stored_pose_probe() -> str:
    executable = _probe_path()
    if np.__version__ != _FROZEN_NUMPY:
        pytest.fail(
            f"stored-pose differential requires NumPy {_FROZEN_NUMPY}, got {np.__version__}",
            pytrace=False,
        )
    try:
        digest = hashlib.sha256(Path(canonical.__file__).read_bytes()).hexdigest()
    except (OSError, TypeError):
        pytest.fail("cannot read the actual imported stored-pose consumer", pytrace=False)
    if digest != _FROZEN_CONSUMER_SHA256:
        pytest.fail(f"stored-pose consumer differs from frozen SHA: {digest}", pytrace=False)
    return executable


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    # Same verified-artifact fixture pattern as test_ort_clip_pose.py, not weights.
    path = tmp_path / "pose.onnx"
    path.write_bytes(b"fake")
    path.with_suffix(".onnx.sha256").write_text(hashlib.sha256(b"fake").hexdigest() + "\n")
    return path


def _invoke(probe: str, payload: bytes) -> subprocess.CompletedProcess[bytes]:
    # The extra byte is used ONLY to test the probe's input-budget rejection.
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
                                            "stored-pose probe exceeded output bounds",
                                            pytrace=False,
                                        )
                    returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pytest.fail("stored-pose probe failed to execute within its bound", pytrace=False)
    return subprocess.CompletedProcess(
        [probe], returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])
    )


class _Input:
    name = "images"


class _RecordingSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.tensor: np.ndarray | None = None
        self.calls = 0

    def get_inputs(self):
        return [_Input()]

    def run(self, names, feed):
        assert names == ["output0"]
        assert tuple(feed) == ("images",)
        self.tensor = feed["images"]
        self.calls += 1
        return [self.output]


@dataclass(frozen=True)
class _Case:
    image: np.ndarray
    output: np.ndarray
    threshold: float = 0.25


def _reference(model: Path, case: _Case):
    session = _RecordingSession(case.output)
    runner = canonical.OrtClipPoseRunner(
        model, case.threshold, session_factory=lambda path, providers: session
    )
    boxes = runner.detect_persons(case.image)
    assert session.calls == 1 and session.tensor is not None
    assert session.tensor.shape == (1, 3, 640, 640) and session.tensor.dtype == np.float32
    assert all(math.isfinite(component) for box in boxes for component in box)
    return session.tensor, boxes


def _image(width: int, height: int, seed: int = 0) -> np.ndarray:
    # Distinct RGB triples identify every source pixel in these bounded fixtures.
    pixels = np.arange(width * height, dtype=np.uint32).reshape(height, width)
    return np.stack(
        (
            (pixels + seed) % 256,
            ((pixels >> 8) + seed + 37) % 256,
            ((pixels >> 16) + seed + 251) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _rows(*heads: tuple[float, ...]) -> np.ndarray:
    output = np.zeros(_OUTPUT_SHAPE, dtype=np.float32)
    # Deliberately not useful keypoints. These finite columns must be uninterpreted.
    output[0, :, 6:] = np.linspace(-1e12, 1e12, 51, dtype=np.float32)
    for index, head in enumerate(heads):
        output[0, index, :6] = head
    return output


def _record(
    width: int,
    height: int,
    rgb: bytes,
    output: np.ndarray,
    threshold: float = 0.25,
    shape: tuple[int, ...] | None = None,
) -> bytes:
    assert output.dtype == np.float32
    shape = output.shape if shape is None else shape
    return b"".join(
        (
            struct.pack("<qqdI", width, height, threshold, len(shape)),
            struct.pack(f"<{len(shape)}Q", *shape),
            struct.pack("<II", len(rgb), output.size),
            rgb,
            output.astype("<f4", copy=False).tobytes(order="C"),
        )
    )


def _case_record(case: _Case) -> bytes:
    height, width = case.image.shape[:2]
    return _record(width, height, case.image.tobytes(order="C"), case.output, case.threshold)


def _payload(records: list[bytes]) -> bytes:
    return _MAGIC + struct.pack("<I", len(records)) + b"".join(records)


@dataclass(frozen=True)
class _Observed:
    status: int
    tensor_bits: np.ndarray
    box_bits: tuple[tuple[int, ...], ...]


def _observations(probe: str, records: list[bytes]) -> list[_Observed]:
    result = _invoke(probe, _payload(records))
    assert result.returncode == 0 and result.stderr == b""
    data = result.stdout
    assert data[:12] == _MAGIC + struct.pack("<I", len(records))
    offset = 12
    observations = []
    for _ in records:
        assert offset + 1 + _TENSOR_VALUES * 4 + 4 <= len(data)
        status = data[offset]
        assert status in range(7)
        offset += 1
        tensor = np.frombuffer(data, dtype="<u4", count=_TENSOR_VALUES, offset=offset)
        offset += _TENSOR_VALUES * 4
        (count,) = struct.unpack_from("<I", data, offset)
        offset += 4
        assert count <= 300 and (status == 0 or count == 0)
        assert offset + count * 40 <= len(data)
        boxes = tuple(
            struct.unpack_from("<5Q", data, offset + index * 40) for index in range(count)
        )
        offset += count * 40
        observations.append(_Observed(status, tensor, boxes))
    assert offset == len(data)
    return observations


def _box_bits(box: tuple[float, ...]) -> tuple[int, ...]:
    return struct.unpack("<5Q", struct.pack("<5d", *box))


def _same_tensor(actual: _Observed, tensor: np.ndarray) -> None:
    expected = np.frombuffer(tensor.astype("<f4", copy=False).tobytes(order="C"), dtype="<u4")
    np.testing.assert_array_equal(actual.tensor_bits, expected)


def _exercise(probe: str, model: Path, cases: list[_Case]):
    actual = _observations(probe, [_case_record(case) for case in cases])
    references = []
    for observation, case in zip(actual, cases, strict=True):
        tensor, boxes = _reference(model, case)
        assert observation.status == 0
        _same_tensor(observation, tensor)
        assert observation.box_bits == tuple(_box_bits(box) for box in boxes)
        references.append(boxes)
    return references


@pytest.mark.parametrize(
    "width,height",
    [
        (1, 1),
        (5, 3),
        (640, 640),
        (997, 541),
        (541, 997),
        (853, 479),
        (1280, 1),
        (1, 1280),
        (1280, 3),
        (3, 1280),
        (1280, 5),
        (5, 1280),
        (1280, 7),
        (7, 1280),
        (65537, 1),
        (1, 65537),
    ],
)
def test_finite_tensor_bits_cover_indices_channels_ties_and_zero_axes(
    stored_pose_probe: str, model_path: Path, width: int, height: int
) -> None:
    _exercise(
        stored_pose_probe,
        model_path,
        [_Case(_image(width, height), _rows((0, 0, 640, 640, 0.8, 0)))],
    )


def test_padding_clears_across_consecutive_shapes_and_zero_axis(
    stored_pose_probe: str, model_path: Path
) -> None:
    output = _rows((0, 0, 640, 640, 0.75, 0))
    cases = [_Case(np.full((640, 640, 3), 255, dtype=np.uint8), output)]
    cases.extend(
        _Case(_image(width, height, seed), output)
        for seed, (width, height) in enumerate(
            [(997, 173), (173, 997), (1280, 1), (3, 1280), (8, 8)], start=1
        )
    )
    _exercise(stored_pose_probe, model_path, cases)


def test_packed_rgb_represents_logical_pixels_of_a_strided_python_image(
    stored_pose_probe: str, model_path: Path
) -> None:
    image = _image(17, 13)[::-1, ::2, ::-1]
    _exercise(stored_pose_probe, model_path, [_Case(image, _rows((0, 0, 640, 640, 0.5, 0)))])


@pytest.mark.parametrize(
    "threshold",
    [
        0.25,
        math.nextafter(0.25, math.inf),
        math.nextafter(0.25, -math.inf),
        0.1,
        math.nextafter(0.1, math.inf),
        0.50000001,
        0.9999999999999999,
        0.0,
        -0.0,
        1.0,
        2.0**-150,
        math.nextafter(2.0**-150, math.inf),
    ],
)
def test_inclusive_threshold_uses_float32_weak_scalar_promotion(
    stored_pose_probe: str, model_path: Path, threshold: float
) -> None:
    center = np.float32(threshold)
    scores = (
        center,
        np.nextafter(center, np.float32(-math.inf)),
        np.nextafter(center, np.float32(math.inf)),
        -0.0,
        0.0,
        0.25,
        1.0,
        np.finfo(np.float32).max,
    )
    output = _rows(*[(1, 2, 30, 40, score, 0) for score in scores])
    (boxes,) = _exercise(
        stored_pose_probe, model_path, [_Case(_image(997, 541), output, threshold)]
    )
    if threshold == math.nextafter(0.25, math.inf):
        assert any(box[4] == 0.25 for box in boxes)


def test_order_duplicate_boxes_scale_precision_clipping_and_signed_zero(
    stored_pose_probe: str, model_path: Path
) -> None:
    first = (321.25, 7.25, 500.75, 123.75, 0.4, 0)
    output = _rows(
        first,
        (1, 2, 30, 40, 0.99, 1),
        (0, 0, 100, 100, 0.9, -0.0),
        first,
        (-100, -200, 2000, 2000, 0.6, 0),
        (-0.0, -0.0, 1, 2, -0.0, -0.0),
        (800, 0, 900, 10, 1, 0),
        (100, 0, 90, 1, 1, 0),
        (0, -4, 1, -1, 1, 0),
        (1, 1, 2, 2, 1, np.nextafter(np.float32(0), np.float32(1))),
        (0, 0, 1, 1, 1, -1),
        (0, 0, 1, 1, 1, 0.5),
        (1, 1, 1, 2, 1, 0),
    )
    (boxes,) = _exercise(stored_pose_probe, model_path, [_Case(_image(997, 541), output, 0.0)])
    assert len(boxes) == 5
    assert boxes[0] == boxes[2]
    assert boxes[0][0] == 500.4472961425781
    assert boxes[3][:4] == (0.0, 0.0, 997.0, 541.0)
    assert _box_bits(boxes[4])[:2] == (0, 0)
    assert _box_bits(boxes[4])[4] == 1 << 63


def test_all_300_rows_remain_in_source_order_without_nms(
    stored_pose_probe: str, model_path: Path
) -> None:
    output = _rows(*[(299 - i, 1, 300 - i, 2, 0.5, 0) for i in range(300)])
    (boxes,) = _exercise(stored_pose_probe, model_path, [_Case(_image(640, 640), output, 0.5)])
    assert len(boxes) == 300
    assert boxes[0][0] == 299.0 and boxes[-1][0] == 0.0


def test_finite_coordinates_whose_division_overflows_still_clip_canonically(
    stored_pose_probe: str, model_path: Path
) -> None:
    maximum = np.finfo(np.float32).max
    output = _rows((-maximum, -maximum, maximum, maximum, 0.75, 0))
    with pytest.warns(RuntimeWarning, match="overflow encountered in scalar divide"):
        (boxes,) = _exercise(stored_pose_probe, model_path, [_Case(_image(997, 541), output)])
    assert boxes == ((0.0, 0.0, 997.0, 541.0, 0.75),)


def test_float32_division_underflow_precedes_strict_positive_box_admission(
    stored_pose_probe: str, model_path: Path
) -> None:
    tiny = np.nextafter(np.float32(0), np.float32(1))
    output = _rows(
        (0, 0, tiny, 640, 0.5, 0),
        (tiny, -tiny, 1, 1, -0.0, 0),
    )
    (boxes,) = _exercise(stored_pose_probe, model_path, [_Case(_image(1, 1), output, 0.0)])
    assert len(boxes) == 1
    assert _box_bits(boxes[0])[:2] == (0, 0)
    assert _box_bits(boxes[0])[4] == 1 << 63


def test_bad_rgb_admission_does_not_partially_mutate_reused_tensor(
    stored_pose_probe: str, model_path: Path
) -> None:
    initial = _Case(_image(3, 2), _rows((0, 0, 640, 640, 0.5, 0)))
    final = _Case(_image(2, 7, 31), initial.output)
    rgb = initial.image.tobytes()
    bad = [
        (_record(0, 2, b"", initial.output), 1),
        (_record(-1, 2, b"", initial.output), 1),
        (_record(3, 0, b"", initial.output), 1),
        (_record((1 << 63) - 1, 2, b"", initial.output), 2),
        (_record(3, 2, rgb[:-1], initial.output), 3),
        (_record(3, 2, rgb + b"\0", initial.output), 3),
    ]
    observed = _observations(
        stored_pose_probe,
        [_case_record(initial), *(record for record, _ in bad), _case_record(final)],
    )
    tensor, boxes = _reference(model_path, initial)
    assert observed[0].status == 0
    _same_tensor(observed[0], tensor)
    assert observed[0].box_bits == tuple(_box_bits(box) for box in boxes)
    for observation, (_, code) in zip(observed[1:-1], bad, strict=True):
        assert observation.status == code and observation.box_bits == ()
        _same_tensor(observation, tensor)
    tensor, boxes = _reference(model_path, final)
    assert observed[-1].status == 0
    _same_tensor(observed[-1], tensor)
    assert observed[-1].box_bits == tuple(_box_bits(box) for box in boxes)
    for image in (np.zeros((0, 3, 3), dtype=np.uint8), np.zeros((2, 0, 3), dtype=np.uint8)):
        with pytest.raises(ModelLoadError, match="dimensions must be positive"):
            _reference(model_path, _Case(image, initial.output))


def test_malformed_output_shapes_and_lengths_fail_closed_without_tensor_changes(
    stored_pose_probe: str, model_path: Path
) -> None:
    initial = _Case(_image(11, 7), _rows((0, 0, 640, 640, 0.5, 0)))
    arrays = [
        np.zeros(shape, dtype=np.float32)
        for shape in ((), (300, 57), (1, 299, 57), (1, 300, 56), (2, 300, 57))
    ]
    records = [_case_record(initial)]
    for output in arrays:
        records.append(_case_record(_Case(initial.image, output)))
        with pytest.raises(ModelLoadError, match="output0 must have shape"):
            _reference(model_path, _Case(initial.image, output))
    for size in (300 * 57 - 1, 300 * 57 + 1):
        # A declared shape/length mismatch cannot be represented by a NumPy array.
        records.append(
            _record(
                11,
                7,
                initial.image.tobytes(),
                np.zeros(size, dtype=np.float32),
                shape=_OUTPUT_SHAPE,
            )
        )
    observed = _observations(stored_pose_probe, records)
    tensor, boxes = _reference(model_path, initial)
    assert observed[0].status == 0 and observed[0].box_bits == tuple(
        _box_bits(box) for box in boxes
    )
    for observation in observed:
        _same_tensor(observation, tensor)
    assert [observation.status for observation in observed[1:]] == [5] * 7


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_output_is_separate_fail_closed_admission_even_in_ignored_fields(
    stored_pose_probe: str, model_path: Path, bad: float
) -> None:
    initial = _Case(_image(19, 5), _rows((0, 0, 640, 640, 0.5, 0)))
    records = [_case_record(initial)]
    # Low-score rows, nonperson rows, and unused keypoints must all be scanned.
    for row, component in ((0, 0), (0, 4), (0, 5), (0, 6), (298, 2), (299, 0), (299, 56)):
        output = initial.output.copy()
        output[0, 299, 4:6] = (1.0, 1.0)
        output[0, row, component] = bad
        records.append(_case_record(_Case(_image(5, 19, 99), output)))
    observed = _observations(stored_pose_probe, records)
    tensor, boxes = _reference(model_path, initial)
    assert observed[0].status == 0 and observed[0].box_bits == tuple(
        _box_bits(box) for box in boxes
    )
    for observation in observed:
        _same_tensor(observation, tensor)
    assert [observation.status for observation in observed[1:]] == [6] * 7


def test_invalid_thresholds_reject_before_mutating_previous_tensor(
    stored_pose_probe: str, model_path: Path
) -> None:
    initial = _Case(_image(9, 2), _rows((0, 0, 640, 640, 0.5, 0)))
    thresholds = (
        -0.1,
        math.nextafter(0.0, -math.inf),
        math.nextafter(1.0, math.inf),
        math.nan,
        math.inf,
        -math.inf,
    )
    records = [_case_record(initial)]
    for threshold in thresholds:
        case = _Case(_image(2, 9, 42), initial.output, threshold)
        records.append(_case_record(case))
        with pytest.raises(ModelLoadError, match="threshold must be in"):
            _reference(model_path, case)
    observed = _observations(stored_pose_probe, records)
    tensor, boxes = _reference(model_path, initial)
    assert observed[0].status == 0 and observed[0].box_bits == tuple(
        _box_bits(box) for box in boxes
    )
    for observation in observed:
        _same_tensor(observation, tensor)
    assert [observation.status for observation in observed[1:]] == [4] * len(thresholds)


@pytest.mark.parametrize(
    "defect",
    [
        "magic",
        "zero-calls",
        "call-bound",
        "rank-bound",
        "truncated",
        "trailing",
        "rgb-count",
        "value-count",
        "late-truncation",
        "input-bound",
    ],
)
def test_transport_rejects_malformed_input_atomically_not_domain_parity(
    stored_pose_probe: str, defect: str
) -> None:
    record = _record(1, 1, b"\0\x7f\xff", _rows())
    payload = _payload([record])
    if defect == "magic":
        payload = b"BADMAGIC" + payload[8:]
    elif defect == "zero-calls":
        payload = _payload([])
    elif defect == "call-bound":
        payload = _MAGIC + struct.pack("<I", 9)
    elif defect == "rank-bound":
        payload = _MAGIC + struct.pack("<IqqdI", 1, 1, 1, 0.25, 5)
    elif defect == "truncated":
        payload = payload[:-1]
    elif defect == "trailing":
        payload += b"\0"
    elif defect in ("rgb-count", "value-count"):
        header = struct.pack("<qqdI3Q", 1, 1, 0.25, 3, *_OUTPUT_SHAPE)
        counts = (0xFFFFFFFF, 0) if defect == "rgb-count" else (0, 0xFFFFFFFF)
        payload = _payload([header + struct.pack("<II", *counts)])
    elif defect == "late-truncation":
        payload = _MAGIC + struct.pack("<I", 2) + record
    else:
        assert defect == "input-bound"
        payload = payload.ljust(_MAX_INPUT + 1, b"x")
    result = _invoke(stored_pose_probe, payload)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == b"stored-pose-probe: rejected\n"


def test_absent_local_opt_in_is_a_skip_not_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="not qualification"):
        _probe_path()


@pytest.mark.parametrize("setting", ["empty", "missing", "unexecutable"])
def test_invalid_explicit_opt_in_fails_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_path: Path, setting: str
) -> None:
    value = {
        "empty": "",
        "missing": str(tmp_path / "absent-probe"),
        "unexecutable": str(model_path),
    }[setting]
    monkeypatch.setenv(_ENV, value)
    with pytest.raises(pytest.fail.Exception, match="configured stored-pose probe"):
        _probe_path()
