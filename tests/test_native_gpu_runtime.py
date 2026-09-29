"""Native GPU runtime parity against the recorded ORT-CUDA oracle.

The oracle is onnxruntime 1.29 ``CUDAExecutionProvider`` (``use_tf32=0``, CPU EP fallback
disabled), recorded once by ``tests_support/native_yolo_parity.py`` on the frozen synthetic
frames of ``worker/runtime/rust/tests/fixtures/gpu``. The manifest records the provider, the
device and the recorder digest, and this module pins the manifest digest. The CPU EP is never
an oracle. These are parity inputs, not inference qualification: the corpus stays unverified.

The gates mirror ``worker/runtime/rust/tests/gpu_parity.rs`` through the C ABI and the
production Python decode: gated pose rows at 1e-4, gated bed rows at
``max(1e-4, 2 * spacing(oracle))``, then zero mismatches in the bed decision inputs. CI's ``-m``
deselects these tests, and a selected test without an asset fails, naming its variable.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import struct
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from worker.adapters.model import seg_postprocess
from worker.adapters.model.ort_clip_pose import _resize_rgb

pytestmark = pytest.mark.gpu

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "worker/runtime/rust/tests/fixtures/gpu/manifest.json"
# Changes only when the oracle is re-recorded, never to absorb a runtime regression.
_MANIFEST_SHA256 = "1184583122b90f5ecbe83a5f5b9c59e740a4c4701e4600b1a038c5db728cb82b"
_SCHEMA = "seeon-gpu-oracle-fixtures/v2"
_MAX_ABS = 1e-4
_ROWS_FROM_SCORE = 0.25
_SCORE = 4
_DEVICE = 0
_IDENTITY_FILE = "engine-identity.json"
_ENGINES = {
    "bed": "SEEON_TEST_BED_ENGINE",
    "stored_pose": "SEEON_TEST_STORED_POSE_ENGINE",
    "fall": "SEEON_TEST_FALL_ENGINE",
}
_BED_SIZE = 1280
_BED_ROW = 38
# The stored-clip bed runner's threshold, at which the goldens were decoded.
_BED_CONFIDENCE = 0.25
_BED_POINTS = (48, 16)
_POSE_SIZE = 640
_POSE_ROW = 57
_FALL_OUTPUT = "84"


class Tensor(ctypes.Structure):
    _fields_ = (
        ("name", ctypes.c_char_p),
        ("values", ctypes.POINTER(ctypes.c_float)),
        ("capacity", ctypes.c_size_t),
        ("dimensions", ctypes.c_int32 * 8),
        ("rank", ctypes.c_int32),
    )


class Metrics(ctypes.Structure):
    _fields_ = (
        ("attempted", ctypes.c_uint64),
        ("succeeded", ctypes.c_uint64),
        ("failed", ctypes.c_uint64),
        ("host_to_device_bytes", ctypes.c_uint64),
        ("device_to_host_bytes", ctypes.c_uint64),
        ("elapsed_ns", ctypes.c_uint64),
        ("device", ctypes.c_int32),
    )


def _required(var: str) -> Path:
    value = os.environ.get(var)
    if value is None:
        pytest.fail(f"{var} is required", pytrace=False)
    if not value.strip() or "\0" in value:
        pytest.fail(f"{var} must be a nonblank path", pytrace=False)
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{var} names a missing path: {path}", pytrace=False)
    return path


def _read(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        pytest.fail(f"read {label}: {error}", pytrace=False)


class _Oracle:
    """The recorded fixtures; each file is read only after its manifest size and digest."""

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root
        self.manifest = manifest

    def bytes(self, relative: str) -> bytes:
        recorded = self.manifest["fixtures"].get(relative)
        assert recorded is not None, f"fixture {relative} is not in the manifest"
        raw = _read(self.root / relative, relative)
        assert len(raw) == recorded["bytes"], f"fixture {relative} size"
        assert hashlib.sha256(raw).hexdigest() == recorded["sha256"], f"fixture {relative} sha256"
        return raw

    def f32s(self, spec: dict[str, Any]) -> np.ndarray:
        raw = self.bytes(spec["path"])
        shape = tuple(spec["shape"])
        assert len(raw) == 4 * math.prod(shape), f"{spec['path']} does not match its shape"
        return np.frombuffer(raw, "<f4").reshape(shape)

    def rgb(self, relative: str) -> np.ndarray:
        raw = self.bytes(relative)
        assert len(raw) >= 16 and raw[:8] == b"SPRGB001", f"{relative} header"
        width, height = struct.unpack("<II", raw[8:16])
        assert len(raw) == 16 + width * height * 3, f"{relative} pixels"
        return np.frombuffer(raw, np.uint8, offset=16).reshape(height, width, 3)


@pytest.fixture(scope="module")
def native() -> ctypes.CDLL:
    library = _required("SEEON_TEST_GPU_LIBRARY")
    try:
        api = ctypes.CDLL(str(library))
    except OSError as error:
        pytest.fail(f"load SEEON_TEST_GPU_LIBRARY: {error}", pytrace=False)
    api.seeon_gpu_open.argtypes = (
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
    )
    api.seeon_gpu_open.restype = ctypes.c_int
    api.seeon_gpu_run.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(Tensor),
        ctypes.POINTER(Tensor),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    )
    api.seeon_gpu_run.restype = ctypes.c_int
    api.seeon_gpu_metrics.argtypes = (ctypes.c_void_p, ctypes.POINTER(Metrics))
    api.seeon_gpu_metrics.restype = ctypes.c_int
    api.seeon_gpu_close.argtypes = (ctypes.c_void_p,)
    api.seeon_gpu_close.restype = None
    return api


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    raw = _MANIFEST.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _MANIFEST_SHA256, "manifest pin"
    manifest = json.loads(raw)
    assert manifest["schema"] == _SCHEMA, "manifest schema"
    assert manifest["tolerance"] == {
        "max_abs": _MAX_ABS,
        "ulp": 2,
        "rows_from_score": _ROWS_FROM_SCORE,
    }, "manifest tolerance"
    recorder = manifest["recorder"]
    recorded = hashlib.sha256((_ROOT / recorder["path"]).read_bytes()).hexdigest()
    assert recorded == recorder["sha256"], "manifest recorder"
    oracle = manifest["oracle"]
    assert oracle["provider"] == "CUDAExecutionProvider", "oracle provider"
    assert oracle["provider_options"]["use_tf32"] == "0", "oracle TF32"
    assert oracle["strict_session_config"] == {"session.disable_cpu_ep_fallback": "1"}, (
        "oracle CPU fallback"
    )
    for key, model in manifest["models"].items():
        placement = model["placement"]
        # Stored pose keeps shape nodes on the CPU EP; none of them reads image-derived data.
        assert placement["strict"] or placement["cpu_nodes"]["image_derived_inputs"] == 0, (
            f"{key} oracle placement"
        )
    # Parity inputs, never inference qualification.
    assert manifest["corpus"]["status"] == "unverified", "corpus admission"
    return manifest


@pytest.fixture(scope="module")
def oracle(manifest: dict[str, Any]) -> _Oracle:
    return _Oracle(_required("SEEON_TEST_GPU_FIXTURES"), manifest)


def _admitted(manifest: dict[str, Any], key: str) -> Path:
    """The engine for key, only when its identity ties it to the oracle's ONNX and GPU."""
    var = _ENGINES[key]
    path = _required(var)
    digest = hashlib.sha256(_read(path, var)).hexdigest()
    try:
        identity = json.loads(_read(path.parent / _IDENTITY_FILE, _IDENTITY_FILE))
    except json.JSONDecodeError as error:
        pytest.fail(f"{_IDENTITY_FILE} is not JSON: {error}", pytrace=False)
    entry = identity.get("engines", {}).get(key, {})
    assert entry.get("engine") == path.name, f"{var} is not the engine recorded for {key}"
    assert entry.get("engine_sha256") == digest, f"{key} digest"
    assert entry.get("onnx_sha256") == manifest["models"][key]["onnx_sha256"], (
        f"{key} engine was built from another ONNX"
    )
    assert entry.get("tf32_enabled") is False, f"{key} TF32"
    assert entry.get("compute_capability") == manifest["gpu"]["compute_capability"], (
        f"{key} compute capability"
    )
    return path


@pytest.fixture
def engine(
    native: ctypes.CDLL, manifest: dict[str, Any]
) -> Iterator[Callable[[str], ctypes.c_void_p]]:
    handles: list[ctypes.c_void_p] = []

    def open_engine(key: str) -> ctypes.c_void_p:
        path = _admitted(manifest, key)
        handle, error = ctypes.c_void_p(), ctypes.create_string_buffer(256)
        status = native.seeon_gpu_open(
            str(path).encode(), _DEVICE, ctypes.byref(handle), error, len(error)
        )
        assert status == 0 and handle.value is not None, f"open {key} engine: {error.value!r}"
        handles.append(handle)
        return handle

    try:
        yield open_engine
    finally:
        for handle in handles:
            native.seeon_gpu_close(handle)


def _tensor(name: str, values: np.ndarray) -> Tensor:
    assert values.dtype == np.float32 and values.flags.c_contiguous
    return Tensor(
        name.encode(),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.size,
        (ctypes.c_int32 * 8)(*values.shape),
        values.ndim,
    )


def _run(
    native: ctypes.CDLL,
    handle: ctypes.c_void_p,
    name: str,
    values: np.ndarray,
    outputs: dict[str, tuple[int, ...]],
) -> list[np.ndarray]:
    """One native run; outputs start as NaN with no shape, so nothing unwritten can pass."""
    source = np.ascontiguousarray(values, dtype=np.float32)
    buffers = [np.full(shape, np.nan, dtype=np.float32) for shape in outputs.values()]
    incoming = _tensor(name, source)
    outgoing = (Tensor * len(buffers))(
        *(_tensor(output, buffer) for output, buffer in zip(outputs, buffers, strict=True))
    )
    for descriptor in outgoing:
        descriptor.rank = 0
        descriptor.dimensions[:] = [0] * 8
    error = ctypes.create_string_buffer(256)
    status = native.seeon_gpu_run(
        handle, ctypes.byref(incoming), outgoing, len(buffers), error, len(error)
    )
    assert status == 0, error.value
    for descriptor, buffer in zip(outgoing, buffers, strict=True):
        assert tuple(descriptor.dimensions[: descriptor.rank]) == buffer.shape
    return buffers


def _metrics(native: ctypes.CDLL, handle: ctypes.c_void_p) -> Metrics:
    metrics = Metrics()
    assert native.seeon_gpu_metrics(handle, ctypes.byref(metrics)) == 0
    return metrics


def _gated(values: np.ndarray, width: int) -> np.ndarray:
    """Rows the decider consumes (score >= 0.25), strongest first; ties keep row order."""
    rows = values.reshape(-1, width)
    rows = rows[rows[:, _SCORE] >= _ROWS_FROM_SCORE]
    return rows[np.argsort(-rows[:, _SCORE], kind="stable")]


def _compare_rows(
    actual: np.ndarray, expected: np.ndarray, width: int, *, ulp: bool
) -> tuple[int, float]:
    """Gate 1: equal gated row counts, then every gated value within 1e-4.

    With ulp, a value may differ by up to two f32 spacings of the oracle value when that is
    wider than 1e-4, which on this model happens only at magnitudes of 512 or more.
    """
    if actual.size != expected.size or actual.size % width:
        raise AssertionError(f"lengths {actual.size} and {expected.size} for rows of {width}")
    for side, values in (("actual", actual), ("oracle", expected)):
        broken = np.flatnonzero(~np.isfinite(values))
        if broken.size:
            raise AssertionError(f"{side} value {broken[0]} is not finite")
    rows, oracle = _gated(actual, width), _gated(expected, width)
    if len(rows) != len(oracle):
        raise AssertionError(f"{len(rows)} gated rows, oracle has {len(oracle)}")
    worst = 0.0
    for row, (actual_row, oracle_row) in enumerate(zip(rows, oracle, strict=True)):
        for column, (value, reference) in enumerate(zip(actual_row, oracle_row, strict=True)):
            diff = abs(float(value) - float(reference))
            limit = _MAX_ABS
            if ulp:
                limit = max(_MAX_ABS, 2 * float(np.spacing(np.float32(abs(reference)))))
            if diff > limit:
                raise AssertionError(
                    f"gated row {row} column {column}: {value} vs {reference}, "
                    f"diff {diff:e} > {limit:e}"
                )
            worst = max(worst, diff)
    return len(rows), worst


def _max_abs(label: str, actual: np.ndarray, expected: np.ndarray) -> float:
    assert actual.size == expected.size, f"{label} length"
    assert np.isfinite(actual).all(), f"{label} produced a non-finite value"
    worst = float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max())
    assert worst <= _MAX_ABS, f"{label} max_abs {worst:e} exceeds {_MAX_ABS:e}"
    return worst


def _spread(actual: np.ndarray, expected: np.ndarray) -> str:
    """Disclosure only: prototypes and the full top-300 never gate."""
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return f"max_abs={diff.max():e} over_1e-4={int((diff > _MAX_ABS).sum())}"


def _hex_f64(value: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(value, 16)))[0]


def _hex_f32(value: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(value, 16)))[0]


def _decode(
    monkeypatch: pytest.MonkeyPatch,
    output0: np.ndarray,
    output1: np.ndarray,
    letterbox: seg_postprocess.Letterbox,
    points: int,
) -> tuple[tuple[seg_postprocess.BedInstance, ...], list[np.ndarray]]:
    """The production bed decode, keeping the thresholded mask it traces for each instance."""
    masks: list[np.ndarray] = []
    trace = seg_postprocess.largest_external_contour

    def keep(mask: np.ndarray) -> seg_postprocess.BedPolygon:
        masks.append(np.asarray(mask, dtype=bool).copy())
        return trace(mask)

    with monkeypatch.context() as patch:
        patch.setattr(seg_postprocess, "largest_external_contour", keep)
        instances = seg_postprocess.decode_end_to_end_segmentation(
            output0,
            output1,
            letterbox,
            model_size=_BED_SIZE,
            confidence=_BED_CONFIDENCE,
            max_points=points,
        )
    assert len(masks) == len(instances)
    return instances, masks


def _golden_beds(oracle: _Oracle, entry: dict[str, Any]) -> tuple[list[dict], np.ndarray]:
    instances = entry["instances"]
    assert len(instances) == entry["detections"], "bed detections"
    letterbox = entry["letterbox"]
    shape = [len(instances), letterbox["source_height"], letterbox["source_width"]]
    assert entry["masks"]["shape"] == shape, "bed mask shape"
    raw = oracle.bytes(entry["masks"]["path"])
    assert len(raw) == math.prod(shape), "bed masks size"
    masks = np.frombuffer(raw, np.uint8).reshape(shape)
    assert np.isin(masks, (0, 1)).all(), "bed masks are 0 or 1"
    return instances, masks


def _compare_beds(
    instances: tuple[seg_postprocess.BedInstance, ...],
    masks: list[np.ndarray],
    golden: list[dict],
    golden_masks: np.ndarray,
    points: int,
) -> None:
    """Gate 2: zero mismatches in what production consumes (box, mask, polygon)."""
    if len(instances) != len(golden):
        raise AssertionError(f"{len(instances)} bed instances, oracle has {len(golden)}")
    order = sorted(range(len(instances)), key=lambda index: -instances[index][4])
    recorded = sorted(range(len(golden)), key=lambda index: -_hex_f32(golden[index]["score"]))
    for rank, (index, other) in enumerate(zip(order, recorded, strict=True)):
        box, expected_box = list(instances[index][:4]), golden[other]["box"]
        if box != expected_box:
            raise AssertionError(f"instance {rank} box {box} vs {expected_box}")
        mask, expected_mask = masks[index], golden_masks[other].astype(bool)
        if mask.shape != expected_mask.shape:
            raise AssertionError(
                f"instance {rank} mask sizes {mask.shape} and {expected_mask.shape}"
            )
        flips = int((mask != expected_mask).sum())
        if flips:
            raise AssertionError(f"instance {rank} mask: {flips} pixels differ")
        polygon = [list(point) for point in instances[index][5]]
        if polygon != golden[other][f"polygon{points}"]:
            raise AssertionError(f"instance {rank} polygon{points} differs")


def _pose_tensor(rgb: np.ndarray) -> np.ndarray:
    """The stored-clip pose preprocessing: nearest resize, top-left zero padding."""
    height, width = rgb.shape[:2]
    scale = min(_POSE_SIZE / width, _POSE_SIZE / height)
    resized = _resize_rgb(rgb, round(width * scale), round(height * scale))
    canvas = np.zeros((_POSE_SIZE, _POSE_SIZE, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return np.transpose(canvas, (2, 0, 1))[None].astype(np.float32) / 255.0


def test_native_bed_matches_the_recorded_ort_cuda_oracle(
    native: ctypes.CDLL,
    oracle: _Oracle,
    engine: Callable[[str], ctypes.c_void_p],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beds = oracle.manifest["bed"]
    assert beds, "no bed fixtures"
    handle = engine("bed")
    failures: list[str] = []
    gated = uploaded = downloaded = 0
    for entry in beds:
        name = entry["id"]
        rgb = oracle.rgb(entry["frame"])
        images = oracle.f32s(entry["images"])
        expected0 = oracle.f32s(entry["output0"])
        expected1 = oracle.f32s(entry["output1"])
        tensor, letterbox = seg_postprocess.letterbox_rgb(rgb, _BED_SIZE)
        recorded = {**entry["letterbox"], "scale": _hex_f64(entry["letterbox"]["scale"])}
        assert letterbox == seg_postprocess.Letterbox(**recorded), f"bed {name} letterbox"
        assert np.array_equal(tensor, images), f"bed {name} images"
        output0, output1 = _run(
            native,
            handle,
            "images",
            tensor,
            {"output0": expected0.shape, "output1": expected1.shape},
        )
        uploaded += tensor.nbytes
        downloaded += output0.nbytes + output1.nbytes
        gated += len(_gated(expected0, _BED_ROW))
        worst = "not reached"
        try:
            _, value = _compare_rows(output0, expected0, _BED_ROW, ulp=True)
            worst = f"{value:e}"
        except AssertionError as error:
            failures.append(f"bed {name} output0: {error}")
        try:
            golden, golden_masks = _golden_beds(oracle, entry)
            for points in _BED_POINTS:
                instances, masks = _decode(monkeypatch, output0, output1, letterbox, points)
                _compare_beds(instances, masks, golden, golden_masks, points)
        except AssertionError as error:
            failures.append(f"bed {name} decision inputs: {error}")
        print(
            f"bed {name}: gated output0 max_abs={worst}; "
            f"top-300 {_spread(output0, expected0)}; protos {_spread(output1, expected1)}"
        )
    metrics = _metrics(native, handle)
    assert metrics.attempted == metrics.succeeded == len(beds)
    assert metrics.failed == 0 and metrics.device == _DEVICE and metrics.elapsed_ns > 0
    assert metrics.host_to_device_bytes == uploaded
    assert metrics.device_to_host_bytes == downloaded
    assert not failures, "\n".join(failures)
    assert gated, "no bed row reaches the gate"


def test_native_stored_pose_matches_the_recorded_ort_cuda_oracle(
    native: ctypes.CDLL, oracle: _Oracle, engine: Callable[[str], ctypes.c_void_p]
) -> None:
    poses = oracle.manifest["stored_pose"]
    assert poses, "no stored pose fixtures"
    handle = engine("stored_pose")
    gated = uploaded = downloaded = 0
    for entry in poses:
        name = entry["id"]
        tensor = _pose_tensor(oracle.rgb(entry["frame"]))
        assert np.array_equal(tensor, oracle.f32s(entry["images"])), f"stored pose {name} images"
        expected = oracle.f32s(entry["output0"])
        (output,) = _run(native, handle, "images", tensor, {"output0": expected.shape})
        uploaded += tensor.nbytes
        downloaded += output.nbytes
        try:
            rows, worst = _compare_rows(output, expected, _POSE_ROW, ulp=False)
        except AssertionError as error:
            raise AssertionError(f"stored pose {name} output0: {error}") from None
        assert rows == entry["detections"], f"stored pose {name} gated rows"
        gated += rows
        print(
            f"stored pose {name}: gated rows={rows} max_abs={worst:e}; "
            f"top-300 {_spread(output, expected)}"
        )
    placement = oracle.manifest["models"]["stored_pose"]["placement"]
    if not placement["strict"]:
        print(f"stored pose oracle: {placement['cpu_nodes']['count']} CPU EP shape nodes")
    metrics = _metrics(native, handle)
    assert metrics.attempted == metrics.succeeded == len(poses)
    assert metrics.failed == 0 and metrics.device == _DEVICE and metrics.elapsed_ns > 0
    assert metrics.host_to_device_bytes == uploaded
    assert metrics.device_to_host_bytes == downloaded
    assert gated, "no pose row reaches the gate"


def test_native_fall_matches_the_recorded_ort_cuda_oracle(
    native: ctypes.CDLL, oracle: _Oracle, engine: Callable[[str], ctypes.c_void_p]
) -> None:
    fall = oracle.manifest["fall"]
    windows = oracle.f32s(fall["windows"])
    logits = oracle.f32s(fall["logits"])
    assert logits.size, "no fall windows"
    assert windows.shape == (logits.size, 30, 56), "fall windows"
    handle = engine("fall")
    worst = 0.0
    for index, (window, logit) in enumerate(zip(windows, logits, strict=True)):
        (output,) = _run(native, handle, "window", window[None], {_FALL_OUTPUT: (1, 1)})
        label = f"fall window {index} logit"
        worst = max(worst, _max_abs(label, output.reshape(1), logit.reshape(1)))
    print(f"fall: windows={logits.size} logit max_abs={worst:e}")
    metrics = _metrics(native, handle)
    assert metrics.attempted == metrics.succeeded == logits.size
    assert metrics.failed == 0 and metrics.device == _DEVICE and metrics.elapsed_ns > 0
    assert metrics.host_to_device_bytes == windows.nbytes
    assert metrics.device_to_host_bytes == logits.nbytes


@pytest.mark.parametrize("defect", ["shape", "size", "name", "output_count", "output_size"])
def test_invalid_tensor_cannot_produce_success_or_resume_poisoned_context(
    native: ctypes.CDLL, engine: Callable[[str], ctypes.c_void_p], defect: str
) -> None:
    model = engine("fall")
    window, result = (
        np.zeros((1, 30, 56), dtype=np.float32),
        np.full((1, 1), -987.0, dtype=np.float32),
    )
    incoming, outgoing = _tensor("window", window), _tensor(_FALL_OUTPUT, result)
    if defect == "shape":
        incoming.dimensions[1] = 0
    elif defect == "size":
        incoming.capacity -= 1
    elif defect == "name":
        incoming.name = b"undeclared"
    elif defect == "output_size":
        outgoing.capacity = 0
    error = ctypes.create_string_buffer(256)
    count = 0 if defect == "output_count" else 1
    assert (
        native.seeon_gpu_run(
            model, ctypes.byref(incoming), ctypes.byref(outgoing), count, error, len(error)
        )
        == -1
    )
    assert error.value and result[0, 0] == -987.0
    incoming = _tensor("window", window)
    assert (
        native.seeon_gpu_run(
            model, ctypes.byref(incoming), ctypes.byref(outgoing), 1, error, len(error)
        )
        == -1
    )
    assert b"unavailable" in error.value
    metrics = _metrics(native, model)
    assert metrics.attempted == metrics.failed == 2 and metrics.succeeded == 0
    assert metrics.host_to_device_bytes == metrics.device_to_host_bytes == 0


def test_gpu_admission_refuses_missing_gpu_without_leaking_path(native: ctypes.CDLL) -> None:
    handle = ctypes.c_void_p(123)
    error = ctypes.create_string_buffer(256)
    assert (
        native.seeon_gpu_open(
            b"/private/credential-like-path.plan",
            2147483647,
            ctypes.byref(handle),
            error,
            len(error),
        )
        == -1
    )
    assert handle.value is None
    assert b"private" not in error.value and error.value


def test_gpu_admission_refuses_symlink_engine(
    native: ctypes.CDLL, manifest: dict[str, Any], tmp_path: Path
) -> None:
    linked = tmp_path / "model.plan"
    linked.symlink_to(_admitted(manifest, "fall"))
    handle = ctypes.c_void_p()
    error = ctypes.create_string_buffer(256)
    assert (
        native.seeon_gpu_open(str(linked).encode(), 0, ctypes.byref(handle), error, len(error))
        == -1
    )
    assert handle.value is None and b"readable" in error.value
