from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    assert_never,
    final,
    runtime_checkable,
)

import numpy as np
import torch
from numpy.typing import NDArray
from typing_extensions import override

from contracts.runner import Image
from worker.adapters.model.errors import FatalAcceleratorError

if TYPE_CHECKING:
    from ultralytics import YOLO

# Issue #111: importing ultralytics runs a module-level connectivity
# self-check (ultralytics.utils.is_online(), gated on this exact env var --
# verified against the ultralytics version this repo's lockfile pins) that
# calls bare, timeout-less socket.getaddrinfo() against two fixed probe
# hosts. On a machine where DNS to those hosts is blackholed rather than
# refused, that call blocks forever, and the worker never logs another line
# past model construction -- the silent boot hang reported in #111. Every
# model in this module loads from a local artifact_dir and never uses
# ultralytics' online/HUB features (grep the repo for "ultralytics": this
# file is the only importer), so the self-check is disabled outright.
#
# This must be set here, at module import time, rather than immediately
# before the lazy `from ultralytics import YOLO` below: ultralytics'
# ONLINE = is_online() runs exactly once, at the first import of the
# ultralytics package anywhere in the process, so the guard has to be in
# place before that import can possibly happen. This module is the sole
# importer of ultralytics in the codebase, and this line runs the moment
# any of worker/adapters/model/{yolo_pose,yolo_person,yolo_bed_seg}.py --
# reachable from worker/__main__.py's very first imports -- pull this
# module in, long before model_backend_init_stage ever runs.
os.environ.setdefault("YOLO_OFFLINE", "true")

YoloTask: TypeAlias = Literal["pose", "person", "bed"]


@dataclass(slots=True)
class YoloArtifactError(RuntimeError):
    task: YoloTask
    path: Path

    @override
    def __str__(self) -> str:
        return f"{self.task} model artifact is missing: {self.path}"


@dataclass(slots=True)
class YoloLoadError(RuntimeError):
    task: YoloTask
    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        return f"{self.task} model load failed for {self.path}: {self.reason}"


@dataclass(slots=True)
class YoloForwardError(RuntimeError):
    task: YoloTask
    reason: str

    @override
    def __str__(self) -> str:
        return f"{self.task} model forward failed: {self.reason}"


@dataclass(slots=True)
class YoloOutputError(RuntimeError):
    task: YoloTask
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.task} model output is malformed: {self.detail}"


@runtime_checkable
class YoloTensor(Protocol):
    def cpu(self) -> YoloTensor: ...

    def numpy(self) -> NDArray[np.float32] | NDArray[np.float64]: ...


YoloArray: TypeAlias = torch.Tensor | NDArray[np.float32] | NDArray[np.float64] | YoloTensor


class YoloBoxes(Protocol):
    @property
    def xyxy(self) -> YoloArray: ...

    @property
    def conf(self) -> YoloArray: ...

    @property
    def cls(self) -> YoloArray: ...

    def __len__(self) -> int: ...


class YoloKeypoints(Protocol):
    @property
    def xy(self) -> YoloArray | None: ...

    @property
    def conf(self) -> YoloArray | None: ...


class YoloMasks(Protocol):
    @property
    def xy(self) -> Sequence[NDArray[np.float32] | NDArray[np.float64]]: ...


class YoloResult(Protocol):
    @property
    def boxes(self) -> YoloBoxes | None: ...

    @property
    def keypoints(self) -> YoloKeypoints | None: ...

    @property
    def masks(self) -> YoloMasks | None: ...


class YoloModel(Protocol):
    @property
    def names(self) -> Mapping[int, str]: ...

    def predict(
        self,
        *,
        source: Image | Sequence[Image],
        conf: float,
        verbose: bool,
        device: str,
    ) -> Sequence[YoloResult]: ...


@dataclass(frozen=True, slots=True)
class YoloPredictOptions:
    task: YoloTask
    confidence: float
    device: str


@final
class _UltralyticsModel:
    def __init__(self, model: YOLO) -> None:
        self._model: YOLO = model

    @property
    def names(self) -> Mapping[int, str]:
        return self._model.names

    def predict(
        self,
        *,
        source: Image | Sequence[Image],
        conf: float,
        verbose: bool,
        device: str,
    ) -> Sequence[YoloResult]:
        return self._model.predict(
            source=source,
            conf=conf,
            verbose=verbose,
            device=device,
        )


_YOLO_LOAD_TIMEOUT_SECONDS: Final = 30.0


def load_yolo_model(
    path: Path,
    task: YoloTask,
    *,
    timeout_seconds: float = _YOLO_LOAD_TIMEOUT_SECONDS,
    construct: Callable[[Path], YoloModel] | None = None,
) -> YoloModel:
    if not path.is_file():
        raise YoloArtifactError(task=task, path=path)
    try:
        model = _construct_with_timeout(
            construct or _construct_ultralytics_model, path, timeout_seconds
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise YoloLoadError(task=task, path=path, reason=str(exc)) from exc
    return model


def _construct_ultralytics_model(path: Path) -> YoloModel:
    """Load a local weights file into an ultralytics ``YOLO`` instance.

    See the ``YOLO_OFFLINE`` guard at the top of this module (issue #111)
    for why this import is safe from ultralytics' import-time connectivity
    self-check by the time this function ever runs.
    """
    from ultralytics import YOLO

    return _UltralyticsModel(YOLO(str(path)))


def _construct_with_timeout(
    construct: Callable[[Path], YoloModel], path: Path, timeout_seconds: float
) -> YoloModel:
    """Bound model construction so an unforeseen stall fails loud instead of hanging.

    ADR-0002: even where the ``YOLO_OFFLINE`` fix above turns out to be
    incomplete -- a future ultralytics release adding another network-touching
    path, or some other environment-specific stall -- a silent infinite wait
    at boot is itself a defect. This converts any such stall into a bounded,
    actionable ``TimeoutError`` (a ``OSError`` subclass, so it is wrapped into
    ``YoloLoadError`` like any other load failure) rather than leaving the
    worker hung with no further logs.

    Deliberately a raw ``threading.Thread(daemon=True)`` rather than
    ``ThreadPoolExecutor``: this process exits via ``sys.exit()`` (not
    ``os._exit()``), and CPython's ``concurrent.futures.thread`` module
    registers an ``atexit`` hook (``_python_exit()``) that joins every
    *non-daemon* worker thread it ever created, with no timeout of its own.
    A genuinely stuck ``construct`` call would therefore still hang the
    process at interpreter shutdown -- silently, with this function's own
    ``TimeoutError`` already raised and handled -- reintroducing exactly the
    defect this fix exists to close. A daemon thread is never joined by the
    interpreter at exit, so a permanently stuck thread can only be abandoned,
    never block shutdown; that abandonment is acceptable because a timeout
    here is fatal to the calling boot stage, which exits the process anyway.
    """
    done = threading.Event()
    outcome: list[YoloModel | Exception] = []

    def _run() -> None:
        try:
            outcome.append(construct(path))
        except Exception as exc:  # noqa: BLE001 -- re-raised verbatim below, in the caller thread
            outcome.append(exc)
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True).start()
    if not done.wait(timeout=timeout_seconds):
        raise TimeoutError(
            f"model construction did not finish within {timeout_seconds}s "
            f"(path={path}); check network/DNS reachability"
        ) from None
    result = outcome[0]
    if isinstance(result, Exception):
        raise result
    return result


def to_float_array(tensor: YoloArray) -> NDArray[np.float64]:
    match tensor:
        case np.ndarray():
            values = tensor
        case torch.Tensor():
            values = tensor.detach().cpu().numpy()
        case YoloTensor():
            values = tensor.cpu().numpy()
        case unreachable:
            assert_never(unreachable)
    return np.asarray(values, dtype=np.float64)


def predict_one(
    model: YoloModel,
    frame: Image,
    options: YoloPredictOptions,
    *,
    camera_id: str = "",
) -> YoloResult:
    try:
        results = model.predict(
            source=frame,
            conf=options.confidence,
            verbose=False,
            device=options.device,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        classify_forward_error(exc, task=options.task, camera_id=camera_id)
    if len(results) != 1:
        raise YoloOutputError(
            task=options.task,
            detail=f"expected one result, received {len(results)}",
        )
    return results[0]


_CUDA_KEYWORDS: tuple[str, ...] = (
    "cuda",
    "device-side assert",
    "illegal memory access",
    "device lost",
    "CUDA error",
    "cuLaunch",
    "cuDNN",
)


def classify_forward_error(exc: Exception, *, task: str, camera_id: str) -> None:
    """Public batch-adapter entry point for existing forward classification."""
    _classify_or_reraise(exc, task=task, camera_id=camera_id)


def _classify_or_reraise(exc: Exception, *, task: str, camera_id: str) -> None:
    """Re-raise exc as FatalAcceleratorError when it is a CUDA fault.

    Non-CUDA exceptions are wrapped in YoloForwardError (ordinary per-camera
    processing failure — isolated, never fatal).
    """
    msg = str(exc).lower()
    if any(kw.lower() in msg for kw in _CUDA_KEYWORDS):
        raise FatalAcceleratorError(
            f"{task} model CUDA fault: {exc}",
            camera_id=camera_id,
            task=task,
        ) from exc
    raise YoloForwardError(task=task, reason=str(exc)) from exc  # type: ignore[name-defined]


__all__ = [
    "YoloArray",
    "YoloArtifactError",
    "YoloBoxes",
    "YoloForwardError",
    "YoloKeypoints",
    "YoloLoadError",
    "YoloMasks",
    "YoloModel",
    "YoloOutputError",
    "YoloPredictOptions",
    "YoloResult",
    "YoloTask",
    "YoloTensor",
    "classify_forward_error",
    "load_yolo_model",
    "predict_one",
    "to_float_array",
]
