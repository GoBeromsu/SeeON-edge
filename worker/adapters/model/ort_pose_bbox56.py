"""CPU ONNX Runtime runner for verified pose+bbox56 fall bundles."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

import numpy as np

from contracts.model_selection import POSE_BBOX56_PREPROCESSING_IDENTITY, ModelSelection
from shared.detection_policies import FALL_POLICY_V2_DEFAULT
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.pose_bbox56_bundle_support import (
    member_digest,
    read_json,
    verify_bundle,
)
from worker.interfaces.fall_model import FallV2Probabilities
from worker.types import FallModelInput

_SHAPE: Final = (30, 56)
#: The output contract this runner implements: one logit, read as a binary
#: fall-transition score. The bundle calibration's class_order has two entries.
OUTPUT_CLASS_COUNT: Final = 2
OUTPUT_CLASS_ORDER: Final = ("non_fall", "fall_transition_proxy")
_CPU_PROVIDER: Final = ("CPUExecutionProvider",)


class _OrtSession(Protocol):
    def run(
        self, output_names: Sequence[str] | None, input_feed: dict[str, np.ndarray]
    ) -> Sequence[object]: ...


SessionFactory = Callable[[str, list[str]], _OrtSession]


class _AdmittedBundleProof(Protocol):
    observed: Mapping[str, object]


@dataclass(frozen=True)
class PackagedFallBundle:
    """The verified CPU runner and its published identity fields."""

    runner: OrtPoseBbox56Runner
    published_weights_digest: str
    preprocessing_identity: str


@dataclass(frozen=True)
class PoseBbox56Conformance:
    """Bundle-published input contract, parsed without importing its consumer."""

    relative_path: str
    preprocessing_identity: str
    vector_length: int
    tail_indices: Mapping[str, int]
    keypoint_order: tuple[str, ...]
    confidence_gate: float
    coordinate_system: Mapping[str, object]
    window_frames: int
    stride_frames: int
    fps: float
    document: Mapping[str, object]


class OrtPoseBbox56Runner:
    """Verified ONNX pose+bbox56 binary proxy running exclusively on CPU."""

    device: Final[str] = "cpu"

    def __init__(
        self,
        session: _OrtSession,
        temperature: float,
        receipt_threshold: float | None,
        receipt_transition_votes: int,
        receipt_transition_window: int,
        promotion_eligible: bool,
        artifact_digest: str,
        preprocessing_identity: str,
        conformance: PoseBbox56Conformance,
    ) -> None:
        self._session = session
        self._temperature = temperature
        self.receipt_threshold = receipt_threshold
        self.receipt_transition_votes = receipt_transition_votes
        self.receipt_transition_window = receipt_transition_window
        self.promotion_eligible = promotion_eligible
        self.artifact_digest = artifact_digest
        self.preprocessing_identity = preprocessing_identity
        self.conformance = conformance

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        device: str = "cpu",
        *,
        providers: Sequence[str] = _CPU_PROVIDER,
        session_factory: SessionFactory | None = None,
    ) -> OrtPoseBbox56Runner:
        if device != "cpu":
            raise ModelLoadError("pose-bbox56 ONNX bundle is pinned to cpu")
        if tuple(providers) != _CPU_PROVIDER:
            raise ModelLoadError("pose-bbox56 ONNX bundle requires CPUExecutionProvider only")
        root = Path(artifact_dir).expanduser().resolve()
        manifest = read_json(root / "bundle-manifest.json")
        verify_bundle(root, manifest)
        artifact_digest = member_digest(manifest, "model.onnx")
        conformance = _packaged_conformance(root, manifest)
        _validate_runner_conformance(conformance)
        (
            temperature,
            receipt_threshold,
            receipt_transition_votes,
            receipt_transition_window,
            promotion_eligible,
        ) = _bundle_metadata(root, conformance.preprocessing_identity)
        if session_factory is None:
            session_factory = _onnxruntime_session_factory
        try:
            session = session_factory(str(root / "model.onnx"), list(_CPU_PROVIDER))
        except Exception as exc:
            raise ModelLoadError(f"cannot load pose-bbox56 ONNX model: {exc}") from exc
        runner = cls(
            session,
            temperature,
            receipt_threshold,
            receipt_transition_votes,
            receipt_transition_window,
            promotion_eligible,
            artifact_digest,
            conformance.preprocessing_identity,
            conformance,
        )
        runner.warmup()
        return runner

    @classmethod
    def from_admitted_bundle(
        cls,
        artifact_dir: str | Path,
        proof: _AdmittedBundleProof,
        selection: ModelSelection,
        *,
        session_factory: SessionFactory | None = None,
    ) -> OrtPoseBbox56Runner:
        """Construct the selected ONNX runner from the already-admitted bundle."""
        # The selection declares an output contract; this runner implements
        # exactly one - a single logit read as a binary fall-transition score.
        # A replacement that emits a different class count is a different
        # structure, and it must refuse here rather than have its logit read
        # as if it were this one.
        if selection.output_class_count != OUTPUT_CLASS_COUNT:
            raise ModelLoadError(
                f"selected bundle declares output_class_count={selection.output_class_count}, "
                f"but this runner implements {OUTPUT_CLASS_COUNT} (a single fall-transition "
                "logit); a model with another output contract needs its own runner"
            )
        member_digests = proof.observed.get("member_digests")
        if not isinstance(member_digests, Mapping):
            raise ModelLoadError("admitted bundle proof has no member digests")
        model_digest = member_digests.get("model.onnx")
        calibration_digest = member_digests.get("calibration.json")
        if not isinstance(model_digest, str) or not isinstance(calibration_digest, str):
            raise ModelLoadError(
                "admitted bundle must contain model.onnx and calibration.json members"
            )
        threshold = selection.transition_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ModelLoadError("selected receipt threshold must be a probability")
        if selection.threshold_source not in {"default", "receipt"}:
            raise ModelLoadError("selected threshold_source must be default or receipt")
        # A selection that says its threshold is the image default must declare
        # the image default. Otherwise the document contradicts itself, and the
        # policy would honour the source word and run at 0.5 while the declared
        # number - the one the owner read - is silently discarded. Refuse.
        if selection.threshold_source == "default" and not math.isclose(
            float(threshold), FALL_POLICY_V2_DEFAULT.transition_threshold
        ):
            raise ModelLoadError(
                f"selected threshold_source is 'default' but transition_threshold is "
                f"{float(threshold):g}, not the image default "
                f"{FALL_POLICY_V2_DEFAULT.transition_threshold:g}; declare threshold_source "
                "'receipt' to run at the declared number"
            )
        root = Path(artifact_dir).expanduser().resolve()
        _verify_admitted_member(root, "model.onnx", model_digest)
        _verify_admitted_member(root, "calibration.json", calibration_digest)
        conformance = _admitted_conformance(root, member_digests, selection.conformance_digest)
        if conformance.preprocessing_identity != selection.preprocessing_identity:
            raise ModelLoadError(
                "selected preprocessing_identity differs from bundle conformance: "
                f"selection {selection.preprocessing_identity!r}, "
                f"conformance {conformance.preprocessing_identity!r}"
            )
        _validate_runner_conformance(conformance)
        calibration = read_json(root / "calibration.json")
        temperature = _calibration_temperature(root, calibration, selection.preprocessing_identity)
        receipt_transition_votes, receipt_transition_window = _calibration_temporal_rule(
            calibration
        )
        if session_factory is None:
            session_factory = _onnxruntime_session_factory
        try:
            session = session_factory(str(root / "model.onnx"), list(_CPU_PROVIDER))
        except Exception as exc:
            raise ModelLoadError(f"cannot load pose-bbox56 ONNX model: {exc}") from exc
        runner = cls(
            session,
            temperature,
            float(threshold),
            receipt_transition_votes,
            receipt_transition_window,
            selection.threshold_source == "receipt",
            model_digest,
            conformance.preprocessing_identity,
            conformance,
        )
        runner.warmup()
        return runner

    def predict(self, features: FallModelInput) -> FallV2Probabilities:
        values = np.asarray(features, dtype=np.float32)
        if values.shape != _SHAPE or not np.isfinite(values).all():
            raise ModelLoadError("pose-bbox56 input must be finite shape (30, 56)")
        try:
            outputs = self._session.run(None, {"window": values[np.newaxis, ...]})
        except Exception as exc:
            raise ModelLoadError(f"cannot run pose-bbox56 ONNX model: {exc}") from exc
        if len(outputs) != 1:
            raise ModelLoadError("pose-bbox56 ONNX model must return one output")
        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.shape != (1, 1) or not np.isfinite(logits).all():
            raise ModelLoadError("pose-bbox56 ONNX model must return finite shape (1, 1)")
        fall_transition = float(1.0 / (1.0 + np.exp(-logits[0, 0] / self._temperature)))
        return FallV2Probabilities(
            background=1.0 - fall_transition,
            fall_transition=fall_transition,
            fallen=0.0,
        )

    def warmup(self) -> None:
        self.predict(np.zeros(_SHAPE, dtype=np.float32))


def load_packaged_fall_bundle(artifact_dir: Path) -> PackagedFallBundle:
    """Load a packaged fall bundle and its published-weights identity."""
    root = artifact_dir.expanduser().resolve()
    runner = OrtPoseBbox56Runner.from_artifact_dir(root, device="cpu")
    manifest = read_json(root / "bundle-manifest.json")
    return PackagedFallBundle(
        runner,
        member_digest(manifest, "model.pt"),
        runner.preprocessing_identity,
    )


def _onnxruntime_session_factory(model_path: str, providers: list[str]) -> _OrtSession:
    try:
        import onnxruntime
    except ImportError as exc:
        raise ModelLoadError("onnxruntime is required for pose-bbox56 ONNX bundles") from exc
    return onnxruntime.InferenceSession(model_path, providers=providers)


def _bundle_metadata(
    root: Path, preprocessing_identity: str
) -> tuple[float, float | None, int, int, bool]:
    calibration = read_json(root / "calibration.json")
    receipt = read_json(root / "evaluation-receipt.json")
    if not isinstance(calibration, dict):
        raise ModelLoadError("invalid calibration.json")
    parsed_temperature = _calibration_temperature(root, calibration, preprocessing_identity)
    transition_votes, transition_window = _calibration_temporal_rule(calibration)
    if not isinstance(receipt, dict):
        raise ModelLoadError("invalid evaluation-receipt.json")
    candidate_threshold = receipt.get("threshold", calibration.get("threshold"))
    receipt_threshold = (
        float(candidate_threshold)
        if isinstance(candidate_threshold, (int, float))
        and not isinstance(candidate_threshold, bool)
        and 0.0 <= float(candidate_threshold) <= 1.0
        else None
    )
    promotion_eligible = receipt.get("promotion_eligible", calibration.get("promotion_eligible"))
    if not isinstance(promotion_eligible, bool):
        raise ModelLoadError("receipt promotion_eligible must be boolean")
    return (
        parsed_temperature,
        receipt_threshold,
        transition_votes,
        transition_window,
        promotion_eligible,
    )


def _calibration_temperature(
    root: Path,
    calibration: object | None = None,
    preprocessing_identity: str = POSE_BBOX56_PREPROCESSING_IDENTITY,
) -> float:
    document = read_json(root / "calibration.json") if calibration is None else calibration
    if not isinstance(document, dict):
        raise ModelLoadError("invalid calibration.json")
    _validate_calibration_class_order(document)
    declared_digest = document.get("preprocessing_identity_digest")
    expected_digest = hashlib.sha256(preprocessing_identity.encode("utf-8")).hexdigest()
    if declared_digest != expected_digest:
        raise ModelLoadError(
            "calibration preprocessing_identity_digest mismatch: "
            f"declared {declared_digest!r}, digest of preprocessing identity "
            f"{preprocessing_identity!r} is {expected_digest}"
        )
    temperature = document.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ModelLoadError("calibration temperature must be numeric")
    parsed_temperature = float(temperature)
    if not math.isfinite(parsed_temperature) or parsed_temperature <= 0:
        raise ModelLoadError("calibration temperature must be finite and positive")
    return parsed_temperature


def _validate_calibration_class_order(calibration: Mapping[str, object]) -> None:
    class_order = calibration.get("class_order")
    if not isinstance(class_order, list) or len(class_order) != OUTPUT_CLASS_COUNT:
        raise ModelLoadError(
            f"calibration class_order must contain exactly {OUTPUT_CLASS_COUNT} entries; "
            f"observed {class_order!r}"
        )
    observed_order = tuple(class_order)
    if observed_order != OUTPUT_CLASS_ORDER:
        raise ModelLoadError(
            f"calibration class_order {class_order!r} contradicts this runner's "
            f"implemented order {list(OUTPUT_CLASS_ORDER)!r}"
        )


def _calibration_temporal_rule(calibration: object) -> tuple[int, int]:
    if not isinstance(calibration, Mapping):
        raise ModelLoadError("invalid calibration.json")
    temporal_rule = calibration.get("temporal_rule")
    if not isinstance(temporal_rule, Mapping):
        raise ModelLoadError("calibration temporal_rule must declare integer m and n")
    votes = temporal_rule.get("m")
    window = temporal_rule.get("n")
    if (
        isinstance(votes, bool)
        or not isinstance(votes, int)
        or isinstance(window, bool)
        or not isinstance(window, int)
        or votes < 1
        or window < votes
    ):
        raise ModelLoadError(
            "calibration temporal_rule must declare integers satisfying 1 <= m <= n"
        )
    return votes, window


def _verify_admitted_member(root: Path, relative_path: str, expected_digest: str) -> None:
    try:
        actual_digest = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ModelLoadError(f"cannot read admitted bundle member {relative_path}") from exc
    if actual_digest != expected_digest:
        raise ModelLoadError(f"admitted bundle member changed: {relative_path}")


def _packaged_conformance(root: Path, manifest: object) -> PoseBbox56Conformance:
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("files"), list):
        raise ModelLoadError("invalid bundle-manifest.json")
    candidates = [
        item.get("relative_path")
        for item in manifest["files"]
        if isinstance(item, Mapping)
        and isinstance(item.get("relative_path"), str)
        and Path(item["relative_path"]).parent == Path("conformance")
    ]
    if len(candidates) != 1:
        raise ModelLoadError(
            f"bundle must contain exactly one conformance member; observed {candidates!r}"
        )
    return _parse_conformance(root, candidates[0])


def _admitted_conformance(
    root: Path, member_digests: Mapping[str, object], conformance_digest: str
) -> PoseBbox56Conformance:
    candidates = [path for path, digest in member_digests.items() if digest == conformance_digest]
    if len(candidates) != 1:
        raise ModelLoadError(
            f"conformance_digest {conformance_digest} names {len(candidates)} members "
            f"{candidates!r}, not exactly one"
        )
    relative_path = candidates[0]
    _verify_admitted_member(root, relative_path, conformance_digest)
    return _parse_conformance(root, relative_path)


def _parse_conformance(root: Path, relative_path: str) -> PoseBbox56Conformance:
    document = read_json(root / relative_path)
    if not isinstance(document, dict):
        raise ModelLoadError(f"invalid conformance document {relative_path}")
    vector = document.get("vector")
    confidence = document.get("confidence")
    temporal = document.get("temporal")
    coordinate_system = document.get("coordinate_system")
    keypoint_order = document.get("keypoint_order")
    if (
        not isinstance(vector, dict)
        or not isinstance(vector.get("tail_indices"), dict)
        or not isinstance(confidence, dict)
        or not isinstance(temporal, dict)
        or not isinstance(coordinate_system, dict)
        or not isinstance(keypoint_order, list)
        or any(not isinstance(value, str) for value in keypoint_order)
    ):
        raise ModelLoadError(f"invalid conformance document shape: {relative_path}")
    identity = document.get("preprocessing_identity")
    vector_length = vector.get("length")
    tail_indices = vector["tail_indices"]
    gate = confidence.get("gate")
    window_frames = temporal.get("window_frames")
    stride_frames = temporal.get("stride_frames")
    fps = temporal.get("fps")
    if not isinstance(identity, str) or not identity:
        raise ModelLoadError("conformance preprocessing_identity must be a non-empty string")
    if vector_length != _SHAPE[1]:
        raise ModelLoadError(
            f"conformance vector length {vector_length!r} differs from runner {_SHAPE[1]}"
        )
    if window_frames != _SHAPE[0]:
        raise ModelLoadError(
            f"conformance temporal.window_frames {window_frames!r} differs from runner {_SHAPE[0]}"
        )
    if (
        isinstance(gate, bool)
        or not isinstance(gate, (int, float))
        or isinstance(stride_frames, bool)
        or not isinstance(stride_frames, int)
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
    ):
        raise ModelLoadError("conformance confidence gate, stride_frames, and fps must be numeric")
    parsed_tail: dict[str, int] = {}
    for key, value in tail_indices.items():
        if not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool):
            raise ModelLoadError("conformance vector.tail_indices must map strings to integers")
        parsed_tail[key] = value
    return PoseBbox56Conformance(
        relative_path=relative_path,
        preprocessing_identity=identity,
        vector_length=vector_length,
        tail_indices=MappingProxyType(parsed_tail),
        keypoint_order=tuple(keypoint_order),
        confidence_gate=float(gate),
        coordinate_system=MappingProxyType(dict(coordinate_system)),
        window_frames=window_frames,
        stride_frames=stride_frames,
        fps=float(fps),
        document=MappingProxyType(document),
    )


def _validate_runner_conformance(conformance: PoseBbox56Conformance) -> None:
    if conformance.preprocessing_identity != POSE_BBOX56_PREPROCESSING_IDENTITY:
        raise ModelLoadError(
            "bundle conformance preprocessing_identity differs from runner contract: "
            f"bundle {conformance.preprocessing_identity!r}, "
            f"runner {POSE_BBOX56_PREPROCESSING_IDENTITY!r}"
        )
    expected_tail = {"x1": 51, "y1": 52, "x2": 53, "y2": 54, "valid": 55}
    if dict(conformance.tail_indices) != expected_tail:
        raise ModelLoadError(
            "bundle conformance vector.tail_indices differs from runner contract: "
            f"bundle {dict(conformance.tail_indices)!r}, runner {expected_tail!r}"
        )


__all__ = [
    "OrtPoseBbox56Runner",
    "PackagedFallBundle",
    "PoseBbox56Conformance",
    "SessionFactory",
    "load_packaged_fall_bundle",
]
