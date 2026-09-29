"""Original G002 RGB-to-bed-input differential; recording-session evidence only.

No inference, segmentation numerics, GPU admission, whole-Worker, or corpus claim.
SEEON_TEST_BED_INPUT_PROBE must name a parent-built executable. Absent opt-in may
skip ordinary CI and is NOT qualification. Invalid explicit configuration fails;
pytest never builds anything. The protocol is documented in bed_input_probe.rs.

The parent inspected the installed sources in frozen worker image sha256:
fffd430545508b7473644975c25e077367d8b42cf032085e96d8f5dcca49171f, NumPy 2.4.6.
We bind BOTH actual imported source files and the NumPy version, not the running
image identity. The oracle executes actual letterbox_rgb for metadata AND actual
OrtBedSegRunner.detect_beds with a recording session for the model input. Empty
segmentation outputs are test doubles, not model predictions. No resize or
normalization arithmetic is reconstructed, and all float32/f64 bits are compared.
Rust semantic rejection/transport bounds are separate admission tests, not claims
that arbitrary malformed byte buffers have a corresponding Python ndarray.
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

from worker.adapters.model import ort_bed_seg as canonical
from worker.adapters.model import seg_postprocess as segmentation

_ENV = "SEEON_TEST_BED_INPUT_PROBE"
_FROZEN_SEG_SHA256 = "1564c2c1cb68bece11dadcc84b305b3ad0e1b6bb937ab979d023cef3afe93a8a"
_FROZEN_CONSUMER_SHA256 = "bded38203d399c7acdae5686ee4b6e0a16d2d9bfa26f361ff731130bfa2b76b9"
_FROZEN_NUMPY = "2.4.6"
_MAGIC = b"BEDINP01"
_MAX_INPUT = 16 * 1024 * 1024
_MAX_OUTPUT = 80 * 1024 * 1024
_MAX_CALLS = 4
_TIMEOUT = 20.0
_NET_SIZE = 1280
_TENSOR_SHAPE = (1, 3, _NET_SIZE, _NET_SIZE)
_TENSOR_VALUES = 3 * _NET_SIZE * _NET_SIZE
_LETTERBOX = struct.Struct("<qqdIIII")


def _probe_path() -> str:
    configured = os.environ.get(_ENV)
    if configured is None:
        pytest.skip("bed-input opt-in is absent; skip is not qualification")
    if not configured.strip() or "\0" in configured:
        pytest.fail("configured bed-input probe must name an executable", pytrace=False)
    try:
        executable = Path(configured).expanduser().resolve()
        usable = executable.is_file() and os.access(executable, os.X_OK)
    except (OSError, RuntimeError, ValueError):
        pytest.fail("configured bed-input probe path is unusable", pytrace=False)
    if not usable:
        pytest.fail("configured bed-input probe is missing or unexecutable", pytrace=False)
    return str(executable)


def _check_numpy() -> None:
    if np.__version__ != _FROZEN_NUMPY:
        pytest.fail(
            f"bed-input differential requires NumPy {_FROZEN_NUMPY}, got {np.__version__}",
            pytrace=False,
        )


def _check_source(module: ModuleType, expected: str, label: str) -> None:
    try:
        digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        pytest.fail(f"cannot read the actual imported bed-input {label}", pytrace=False)
    if digest != expected:
        pytest.fail(f"bed-input {label} differs from frozen SHA: {digest}", pytrace=False)


@pytest.fixture
def bed_input_probe() -> str:
    executable = _probe_path()
    _check_numpy()
    _check_source(segmentation, _FROZEN_SEG_SHA256, "letterbox source")
    _check_source(canonical, _FROZEN_CONSUMER_SHA256, "consumer source")
    assert canonical.letterbox_rgb is segmentation.letterbox_rgb
    return executable


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    # Verified-artifact fixture, following test_ort_bed_seg_runner.py; NOT weights.
    payload = b"recording-session conversion fixture, no inference"
    path = tmp_path / "bed.onnx"
    path.write_bytes(payload)
    path.with_suffix(".onnx.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )
    return path


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
                                            "bed-input probe exceeded output bounds; "
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
            "bed-input probe failed to execute within its bound; "
            f"stderr={bytes(buffers['stderr'])!r}",
            pytrace=False,
        )
    return subprocess.CompletedProcess(
        [probe], returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])
    )


class _Input:
    shape = _TENSOR_SHAPE


class _RecordingSession:
    def __init__(self) -> None:
        self.tensor: np.ndarray | None = None
        self.calls = 0

    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, names, feed) -> list[np.ndarray]:
        assert names is None
        assert tuple(feed) == ("images",)
        self.tensor = feed["images"]
        self.calls += 1
        # No rows => no boxes/masks; the actual consumer still checks output shape.
        return [
            np.empty((1, 0, 38), dtype=np.float32),
            np.zeros((1, 32, 320, 320), dtype=np.float32),
        ]


def _reference(model: Path, image: np.ndarray) -> tuple[np.ndarray, segmentation.Letterbox]:
    direct_tensor, letterbox = segmentation.letterbox_rgb(image, _NET_SIZE)
    session = _RecordingSession()

    def session_factory(path: str, providers: list[str]) -> _RecordingSession:
        assert Path(path) == model.resolve()
        assert providers == ["CPUExecutionProvider"]
        return session

    runner = canonical.OrtBedSegRunner(
        str(model),
        confidence=0.25,
        max_points=48,
        device="cpu",
        providers=("CPUExecutionProvider",),
        session_factory=session_factory,
    )
    result = runner.detect_beds(image)
    assert result.kind == "bed" and tuple(result.boxes) == ()
    assert session.calls == 1 and session.tensor is not None
    tensor = session.tensor
    assert tensor.shape == _TENSOR_SHAPE and tensor.dtype == np.float32
    assert tensor.flags.c_contiguous and np.isfinite(tensor).all()
    np.testing.assert_array_equal(tensor.view(np.uint32), direct_tensor.view(np.uint32))
    return tensor, letterbox


def _image(width: int, height: int, seed: int = 0) -> np.ndarray:
    pixels = np.arange(width * height, dtype=np.uint32).reshape(height, width)
    return np.stack(
        (
            (pixels * 17 + seed) % 256,
            ((pixels // width) * 29 + pixels * 3 + seed + 37) % 256,
            ((pixels ^ (pixels >> 8)) + seed + 251) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _record(width: int, height: int, rgb: bytes) -> bytes:
    return struct.pack("<qqI", width, height, len(rgb)) + rgb


def _image_record(image: np.ndarray) -> bytes:
    height, width = image.shape[:2]
    assert image.dtype == np.uint8 and image.shape[2] == 3
    return _record(width, height, image.tobytes(order="C"))


def _payload(records: list[bytes]) -> bytes:
    return _MAGIC + struct.pack("<I", len(records)) + b"".join(records)


@dataclass(frozen=True)
class _Observed:
    status: int
    letterbox_bits: bytes | None
    tensor_bits: np.ndarray


def _observations(probe: str, records: list[bytes]) -> list[_Observed]:
    assert 1 <= len(records) <= _MAX_CALLS
    result = _invoke(probe, _payload(records))
    returncode, stderr = result.returncode, result.stderr
    assert returncode == 0, stderr
    assert stderr == b""
    data = result.stdout
    assert data[:12] == _MAGIC + struct.pack("<I", len(records))
    offset = 12
    observations = []
    for _ in records:
        assert offset < len(data)
        status = data[offset]
        assert status in (0, 1, 2, 3)
        offset += 1
        metadata = None
        if status == 0:
            assert offset + _LETTERBOX.size <= len(data)
            metadata = data[offset : offset + _LETTERBOX.size]
            offset += _LETTERBOX.size
        assert offset + _TENSOR_VALUES * 4 <= len(data)
        tensor = np.frombuffer(data, dtype="<u4", count=_TENSOR_VALUES, offset=offset)
        offset += _TENSOR_VALUES * 4
        observations.append(_Observed(status, metadata, tensor))
    assert offset == len(data)
    return observations


def _metadata_bits(letterbox: segmentation.Letterbox) -> bytes:
    # Serialization of actual metadata, including f64 scale bits; no arithmetic oracle.
    return _LETTERBOX.pack(
        letterbox.source_height,
        letterbox.source_width,
        letterbox.scale,
        letterbox.resized_height,
        letterbox.resized_width,
        letterbox.pad_top,
        letterbox.pad_left,
    )


def _same_tensor(actual: _Observed, expected: np.ndarray) -> None:
    assert expected.dtype == np.float32 and expected.flags.c_contiguous
    np.testing.assert_array_equal(actual.tensor_bits, expected.view(np.uint32).reshape(-1))


def _exercise(probe: str, model: Path, images: list[np.ndarray]) -> list[segmentation.Letterbox]:
    actual = _observations(probe, [_image_record(image) for image in images])
    metadata = []
    for observation, image in zip(actual, images, strict=True):
        tensor, letterbox = _reference(model, image)
        assert observation.status == 0
        _same_tensor(observation, tensor)
        assert observation.letterbox_bits == _metadata_bits(letterbox)
        metadata.append(letterbox)
    return metadata


@pytest.mark.parametrize(
    "width,height",
    [
        (1, 1),
        (5, 3),
        (17, 11),
        (319, 181),
        (997, 541),
        (541, 997),
        (1280, 1280),
        (1280, 719),
        (719, 1280),
        (1537, 887),
        (887, 1537),
        (2048, 1025),
        (65537, 1),
        (1, 65537),
    ],
)
def test_all_tensor_and_metadata_bits_for_up_down_same_size_and_odd_aspects(
    bed_input_probe: str, model_path: Path, width: int, height: int
) -> None:
    _exercise(bed_input_probe, model_path, [_image(width, height)])


@pytest.mark.parametrize("short,resized", [(1, 0), (3, 2), (5, 2), (7, 4)])
@pytest.mark.parametrize("transpose", [False, True])
def test_python_ties_even_including_valid_zero_resized_axes(
    bed_input_probe: str, model_path: Path, short: int, resized: int, transpose: bool
) -> None:
    width, height = (short, 2560) if transpose else (2560, short)
    (metadata,) = _exercise(bed_input_probe, model_path, [_image(width, height, 19)])
    assert (metadata.resized_width if transpose else metadata.resized_height) == resized
    assert struct.pack("<d", metadata.scale) == struct.pack("<d", 0.5)


def test_half_coordinate_and_uint8_truncation_edges_keep_channel_order(
    bed_input_probe: str, model_path: Path
) -> None:
    corners = np.array([[[0, 1, 254], [255, 254, 1]], [[1, 255, 0], [254, 0, 255]]], dtype=np.uint8)
    # 2560 -> 1280 gives half coordinates. Adjacent extrema/odd sums exercise
    # interpolation and uint8 truncation; expectations still come only from Python.
    downsample = np.tile(corners, (1, 1280, 1))
    _exercise(bed_input_probe, model_path, [corners, downsample])


@pytest.mark.parametrize("seed", [731, 8128, 65537])
def test_seeded_pixels_and_geometry_are_selected_independently_of_results(
    bed_input_probe: str, model_path: Path, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    width = int(rng.integers(2, 1601))
    height = int(rng.integers(2, 1101))
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    _exercise(bed_input_probe, model_path, [image])


def test_packed_rgb_preserves_logical_pixels_of_a_strided_python_image(
    bed_input_probe: str, model_path: Path
) -> None:
    image = _image(19, 13)[::-1, ::2, ::-1]
    assert not image.flags.c_contiguous
    _exercise(bed_input_probe, model_path, [image])


def test_four_call_reuse_clears_previous_content_and_padding_within_output_budget(
    bed_input_probe: str, model_path: Path
) -> None:
    _exercise(
        bed_input_probe,
        model_path,
        [
            np.full((1280, 1280, 3), 255, dtype=np.uint8),
            _image(1031, 173, 7),
            _image(173, 1031, 13),
            _image(2560, 1, 97),
        ],
    )


@pytest.mark.parametrize(
    "invalid",
    [
        [(0, 1, b"", 1), (1, -1, b"", 1), (-(2**63), 1, b"", 1)],
        [(1, 0, b"", 1), (-1, 1, b"", 1), (0, 2**63 - 1, b"", 1)],
        [
            (2**63 - 1, 1, b"", 2),
            (2**32, 2**32, b"", 2),
            ((2**63 - 1) // 3 + 1, 1, b"", 2),
        ],
        [(1, 1, b"", 3), (1, 1, b"\x01\x02", 3), (1, 1, b"\x01\x02\x03\x04", 3)],
        [(1280, 1280, b"", 3), ((2**63 - 1) // 3, 1, b"", 3), (2, 1, b"abc", 3)],
    ],
)
def test_typed_dimension_length_and_overflow_failures_preserve_every_previous_bit(
    bed_input_probe: str, model_path: Path, invalid: list[tuple[int, int, bytes, int]]
) -> None:
    image = _image(11, 7)
    records = [_image_record(image)]
    records.extend(_record(width, height, rgb) for width, height, rgb, _ in invalid)
    observed = _observations(bed_input_probe, records)
    tensor, letterbox = _reference(model_path, image)
    assert observed[0].status == 0
    assert observed[0].letterbox_bits == _metadata_bits(letterbox)
    for observation in observed:
        _same_tensor(observation, tensor)
    for observation, (_, _, _, status) in zip(observed[1:], invalid, strict=True):
        assert observation.status == status
        assert observation.letterbox_bits is None


def test_success_after_error_replaces_content_without_stale_padding(
    bed_input_probe: str, model_path: Path
) -> None:
    before, after = _image(1280, 1280, 3), _image(9, 71, 99)
    observed = _observations(
        bed_input_probe,
        [_image_record(before), _record(0, 1, b"private-invalid-rgb"), _image_record(after)],
    )
    initial, initial_metadata = _reference(model_path, before)
    final, final_metadata = _reference(model_path, after)
    assert [item.status for item in observed] == [0, 1, 0]
    _same_tensor(observed[0], initial)
    _same_tensor(observed[1], initial)
    _same_tensor(observed[2], final)
    assert observed[0].letterbox_bits == _metadata_bits(initial_metadata)
    assert observed[1].letterbox_bits is None
    assert observed[2].letterbox_bits == _metadata_bits(final_metadata)


def test_exact_input_budget_is_transport_valid_and_semantic_error_is_observable(
    bed_input_probe: str,
) -> None:
    record = _record(1, 1, b"x" * (_MAX_INPUT - 12 - 20))
    assert len(_payload([record])) == _MAX_INPUT
    (observed,) = _observations(bed_input_probe, [record])
    assert observed.status == 3 and observed.letterbox_bits is None
    assert np.count_nonzero(observed.tensor_bits) == 0


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
        "short-rgb",
        "rgb-count",
        "trailing",
        "late-truncation",
        "input-bound",
    ],
)
def test_malformed_transport_rejects_atomically_with_private_static_diagnostic(
    bed_input_probe: str, defect: str
) -> None:
    record = _record(1, 1, b"\x00\x7f\xff")
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
    elif defect == "short-rgb":
        payload = payload[:-1]
    elif defect == "rgb-count":
        payload = _payload([struct.pack("<qqI", 1, 1, 0xFFFFFFFF)])
    elif defect == "trailing":
        payload += b"private-path-and-token"
    elif defect == "late-truncation":
        payload = _MAGIC + struct.pack("<I", 2) + record
    else:
        assert defect == "input-bound"
        payload = payload.ljust(_MAX_INPUT + 1, b"x")
    result = _invoke(bed_input_probe, payload)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == b"bed-input-probe: rejected\n"


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
    with pytest.raises(pytest.fail.Exception, match="configured bed-input probe"):
        _probe_path()


def test_nul_opt_in_reaches_the_validator_without_os_setenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # setenv rejects NUL before the validator runs; replace the mapping instead.
    monkeypatch.setattr(os, "environ", {**os.environ, _ENV: "private\0probe"})
    with pytest.raises(pytest.fail.Exception, match="configured bed-input probe"):
        _probe_path()


@pytest.mark.parametrize("version", ["", "2.4.5", "2.4.6.dev0", "2.4.6+local"])
def test_nonfrozen_numpy_fails_instead_of_weakening_bit_comparisons(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(np, "__version__", version)
    with pytest.raises(pytest.fail.Exception, match="requires NumPy 2.4.6"):
        _check_numpy()


@pytest.mark.parametrize(
    "module,expected,label",
    [
        (segmentation, _FROZEN_SEG_SHA256, "letterbox source"),
        (canonical, _FROZEN_CONSUMER_SHA256, "consumer source"),
    ],
    ids=["letterbox", "consumer"],
)
@pytest.mark.parametrize("defect", ["different", "unreadable", "missing-file-attribute"])
def test_each_actual_imported_source_must_match_the_frozen_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: ModuleType,
    expected: str,
    label: str,
    defect: str,
) -> None:
    path = tmp_path / "source.py"
    if defect == "different":
        path.write_bytes(b"not the frozen consumer\n")
    monkeypatch.setattr(
        module, "__file__", None if defect == "missing-file-attribute" else str(path)
    )
    message = (
        "differs from frozen SHA" if defect == "different" else "cannot read the actual imported"
    )
    with pytest.raises(pytest.fail.Exception, match=message):
        _check_source(module, expected, label)
