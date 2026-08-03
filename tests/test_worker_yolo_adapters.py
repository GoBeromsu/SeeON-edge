from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, assert_never, final

import numpy as np
import pytest
from numpy.typing import NDArray

from worker.adapters.model.warmup import WarmupFrameSpec, warmup_to_ready
from worker.adapters.model.yolo_api import (
    YoloArtifactError,
    YoloForwardError,
    YoloLoadError,
    YoloModel,
    YoloOutputError,
    load_yolo_model,
)
from worker.adapters.model.yolo_bed_seg import COCO_BED_CLASS_ID, YoloBedSegRunner
from worker.adapters.model.yolo_person import COCO_PERSON_CLASS_ID, YoloPersonRunner
from worker.adapters.model.yolo_pose import COCO_KEYPOINT_COUNT, YoloPoseRunner


@final
class _Tensor:
    def __init__(self, values: Sequence[float] | Sequence[Sequence[float]]) -> None:
        self._values: NDArray[np.float64] = np.asarray(values, dtype=np.float64)

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> NDArray[np.float64]:
        return self._values


@final
class _Boxes:
    def __init__(
        self,
        xyxy: Sequence[Sequence[float]] = (),
        confidence: Sequence[float] = (),
        classes: Sequence[float] = (),
    ) -> None:
        self.xyxy: _Tensor = _Tensor(xyxy)
        self.conf: _Tensor = _Tensor(confidence)
        self.cls: _Tensor = _Tensor(classes)

    def __len__(self) -> int:
        return len(self.conf.numpy())


@final
class _Keypoints:
    def __init__(
        self,
        xy: Sequence[Sequence[Sequence[float]]],
        confidence: Sequence[Sequence[float]] | None,
    ) -> None:
        self.xy: _Tensor3D = _Tensor3D(xy)
        self.conf: _Tensor | None = None if confidence is None else _Tensor(confidence)


@final
class _Tensor3D:
    def __init__(self, values: Sequence[Sequence[Sequence[float]]]) -> None:
        self._values: NDArray[np.float64] = np.asarray(values, dtype=np.float64)

    def cpu(self) -> _Tensor3D:
        return self

    def numpy(self) -> NDArray[np.float64]:
        return self._values


@final
class _Masks:
    def __init__(self, polygons: Sequence[NDArray[np.float64]]) -> None:
        self.xy: Sequence[NDArray[np.float64]] = polygons


@final
class _Result:
    def __init__(
        self,
        *,
        boxes: _Boxes | None = None,
        keypoints: _Keypoints | None = None,
        masks: _Masks | None = None,
    ) -> None:
        self.boxes: _Boxes | None = boxes
        self.keypoints: _Keypoints | None = keypoints
        self.masks: _Masks | None = masks


@final
class _Model:
    def __init__(
        self,
        result: _Result,
        *,
        names: Mapping[int, str] | None = None,
        failure: RuntimeError | None = None,
    ) -> None:
        self.names: Mapping[int, str] = {} if names is None else names
        self.result: _Result = result
        self.failure: RuntimeError | None = failure
        self.calls: list[tuple[tuple[int, ...], np.dtype[np.uint8], float, bool, str]] = []

    def predict(
        self,
        *,
        source: NDArray[np.uint8],
        conf: float,
        verbose: bool,
        device: str,
    ) -> tuple[_Result, ...]:
        self.calls.append((source.shape, source.dtype, conf, verbose, device))
        if self.failure is not None:
            raise self.failure
        return (self.result,)


def _pose_result(*, point_count: int = COCO_KEYPOINT_COUNT) -> _Result:
    points = tuple((float(index) + 0.9, float(index) + 10.1) for index in range(point_count))
    confidences = tuple(0.5 + index / 100 for index in range(point_count))
    return _Result(
        boxes=_Boxes(((10.9, 20.1, 110.7, 220.3),), (0.83,), (COCO_PERSON_CLASS_ID,)),
        keypoints=_Keypoints((points,), (confidences,)),
    )


def _person_result() -> _Result:
    return _Result(
        boxes=_Boxes(
            ((10.9, 20.1, 110.7, 220.3), (1, 2, 3, 4), (30, 40, 130, 240)),
            (0.83, 0.91, 0.20),
            (COCO_PERSON_CLASS_ID, COCO_BED_CLASS_ID, COCO_PERSON_CLASS_ID),
        )
    )


def _bed_result() -> _Result:
    return _Result(
        boxes=_Boxes(((5.2, 10.7, 105.9, 210.4),), (0.88,), (COCO_BED_CLASS_ID,)),
        masks=_Masks((np.asarray(((5.2, 10.7), (55.4, 9.8), (106.0, 211.0)), dtype=np.float64),)),
    )


def test_pose_predict_passes_exact_args_and_returns_coco17_keypoints() -> None:
    model = _Model(_pose_result())
    runner = YoloPoseRunner(confidence=0.05, device="mps", model=model)

    output = runner.run(np.zeros((2, 3, 3), dtype=np.uint8))

    assert model.calls == [((2, 3, 3), np.dtype(np.uint8), 0.05, False, "mps")]
    assert output.kind == "pose"
    assert len(output.poses) == 1
    assert len(output.poses[0]) == COCO_KEYPOINT_COUNT
    assert output.poses[0][0] == (0, 10, 0.5)
    assert output.boxes == ((10, 20, 110, 220, 0.83),)


def test_person_predict_passes_exact_args_and_filters_class_and_confidence() -> None:
    model = _Model(_person_result())
    runner = YoloPersonRunner(confidence=0.25, device="cpu", model=model)

    output = runner.run(np.zeros((4, 5, 3), dtype=np.uint8))

    assert model.calls == [((4, 5, 3), np.dtype(np.uint8), 0.25, False, "cpu")]
    assert output.kind == "person"
    assert output.boxes == ((10, 20, 110, 220, 0.83),)


def test_bed_predict_passes_exact_args_and_returns_mask_polygon() -> None:
    model = _Model(_bed_result(), names={COCO_BED_CLASS_ID: "bed"})
    runner = YoloBedSegRunner(confidence=0.25, device="cuda:0", model=model)

    output = runner.run(np.zeros((6, 7, 3), dtype=np.uint8))

    assert model.calls == [((6, 7, 3), np.dtype(np.uint8), 0.25, False, "cuda:0")]
    assert output.kind == "bed"
    assert tuple(output.boxes) == ((5, 10, 105, 210, 0.88, ((5, 11), (55, 10), (106, 211))),)


@pytest.mark.parametrize("task", ["pose", "person", "bed"])
def test_each_yolo_warmup_runs_one_forward_on_configured_rgb_frame(
    task: Literal["pose", "person", "bed"],
) -> None:
    frame_spec = WarmupFrameSpec(width=320, height=192)
    match task:
        case "pose":
            model = _Model(_pose_result())
            runner = YoloPoseRunner(device="cpu", warmup_frame=frame_spec, model=model)
        case "person":
            model = _Model(_person_result())
            runner = YoloPersonRunner(device="cpu", warmup_frame=frame_spec, model=model)
        case "bed":
            model = _Model(_bed_result(), names={COCO_BED_CLASS_ID: "bed"})
            runner = YoloBedSegRunner(device="cpu", warmup_frame=frame_spec, model=model)
        case unreachable:
            assert_never(unreachable)

    runner.warmup()

    assert len(model.calls) == 1
    assert model.calls[0][0:2] == ((192, 320, 3), np.dtype(np.uint8))


def test_missing_yolo_artifact_blocks_readiness(tmp_path: Path) -> None:
    runner = YoloPersonRunner(model_path=str(tmp_path / "missing.pt"))

    with pytest.raises(YoloArtifactError, match="person.*missing"):
        _ = warmup_to_ready(runner, device="cpu")


def test_load_yolo_model_times_out_instead_of_hanging_forever(tmp_path: Path) -> None:
    """Issue #111: a stalled model construction (e.g. ultralytics' import-time
    connectivity self-check blocking on unreachable DNS) must fail loud with a
    bounded, diagnosable error rather than hang the boot thread forever."""
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"not a real checkpoint")
    started = threading.Event()
    release = threading.Event()

    def _hang_forever(_path: Path) -> YoloModel:
        started.set()
        release.wait()  # never released within the test -- simulates a stuck import
        raise AssertionError("should have timed out before reaching this point")

    with pytest.raises(YoloLoadError, match="pose.*model.pt"):
        _ = load_yolo_model(
            artifact, "pose", timeout_seconds=0.05, construct=_hang_forever
        )
    assert started.is_set()
    release.set()  # let the leaked thread exit cleanly instead of leaking past the test


def test_load_yolo_model_returns_promptly_when_construction_is_fast(tmp_path: Path) -> None:
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"not a real checkpoint")
    sentinel = _Model(_pose_result())

    def _fast_construct(_path: Path) -> YoloModel:
        return sentinel

    model = load_yolo_model(artifact, "pose", timeout_seconds=5.0, construct=_fast_construct)

    assert model is sentinel


def test_forward_failure_blocks_readiness_before_cuda_sync() -> None:
    model = _Model(_pose_result(), failure=RuntimeError("kernel launch failed"))
    runner = YoloPoseRunner(device="cuda:0", model=model)
    synchronized: list[str] = []

    with pytest.raises(YoloForwardError, match="pose.*forward"):
        _ = warmup_to_ready(
            runner,
            device="cuda:0",
            synchronize=lambda: synchronized.append("sync"),
        )

    assert len(model.calls) == 1
    assert synchronized == []


def test_malformed_pose_output_raises_typed_adapter_error() -> None:
    model = _Model(_pose_result(point_count=COCO_KEYPOINT_COUNT - 1))
    runner = YoloPoseRunner(model=model)

    with pytest.raises(YoloOutputError, match="pose.*17"):
        _ = runner.run(np.zeros((2, 2, 3), dtype=np.uint8))
