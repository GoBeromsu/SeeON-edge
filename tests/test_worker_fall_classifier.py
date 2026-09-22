from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from worker.domains.fall.classifier import FallWindowClassifier
from worker.interfaces.fall_model import FallProbabilities
from worker.types.trace import DecisionTraceMissingReason


@dataclass(slots=True)
class _Model:
    prediction: object = FallProbabilities(0.2, 0.7, 0.1)
    inputs: list[tuple[tuple[float, ...], ...]] = field(default_factory=list)

    def predict(self, features: tuple[tuple[float, ...], ...]) -> object:
        self.inputs.append(features)
        return self.prediction


def _row(value: float = 0.0) -> tuple[float, ...]:
    return (value,) * 56


def test_classifier_windows_exactly_30_pose_bbox56_rows_on_five_frame_stride() -> None:
    model = _Model()
    classifier = FallWindowClassifier(model)

    for _ in range(29):
        assert classifier.update({7: _row(0.25)}, (7,)) == {}
    due = classifier.update({7: _row(0.75)}, (7,))

    assert due == {7: FallProbabilities(0.2, 0.7, 0.1)}
    assert len(model.inputs) == 1
    assert len(model.inputs[0]) == 30
    assert all(len(row) == 56 for row in model.inputs[0])
    assert model.inputs[0][0] == _row(0.25)
    assert model.inputs[0][-1] == _row(0.75)


def test_classifier_missing_score_reason_is_current_call_not_cached_probability() -> None:
    model = _Model()
    classifier = FallWindowClassifier(model)

    for _ in range(4):
        assert classifier.update({7: _row()}, (7,)) == {}
        assert dict(classifier.current_call_missing_score_reasons) == {
            7: DecisionTraceMissingReason.CLASSIFIER_STRIDE_NOT_DUE
        }

    assert classifier.update({7: _row()}, (7,)) == {}
    assert dict(classifier.current_call_missing_score_reasons) == {
        7: DecisionTraceMissingReason.CLASSIFIER_WARMUP
    }

    for _ in range(24):
        classifier.update({7: _row()}, (7,))
    expected = FallProbabilities(0.2, 0.7, 0.1)
    assert classifier.update({7: _row()}, (7,)) == {7: expected}
    assert dict(classifier.current_call_missing_score_reasons) == {}
    assert classifier.probabilities_for(7) == expected
    assert len(model.inputs) == 1

    assert classifier.update({7: _row()}, (7,)) == {}
    assert dict(classifier.current_call_missing_score_reasons) == {
        7: DecisionTraceMissingReason.CLASSIFIER_STRIDE_NOT_DUE
    }
    assert classifier.probabilities_for(7) == expected
    assert len(model.inputs) == 1

    assert classifier.update({}, ()) == {}
    assert dict(classifier.current_call_missing_score_reasons) == {}


def test_classifier_missing_score_reasons_are_isolated_per_track() -> None:
    model = _Model()
    classifier = FallWindowClassifier(model)

    for _ in range(29):
        classifier.update({1: _row(0.1)}, (1,))
    due = classifier.update({1: _row(0.2), 2: _row(0.3)}, (1, 2))

    assert due == {1: FallProbabilities(0.2, 0.7, 0.1)}
    assert dict(classifier.current_call_missing_score_reasons) == {
        2: DecisionTraceMissingReason.CLASSIFIER_WARMUP
    }
    assert classifier.probabilities_for(1) is not None
    assert classifier.probabilities_for(2) is None

    assert classifier.update({1: _row(), 2: _row()}, (1, 2)) == {}
    assert dict(classifier.current_call_missing_score_reasons) == {
        1: DecisionTraceMissingReason.CLASSIFIER_STRIDE_NOT_DUE,
        2: DecisionTraceMissingReason.CLASSIFIER_STRIDE_NOT_DUE,
    }


def test_classifier_coasts_missing_rows_with_last_valid_pose_bbox_row() -> None:
    model = _Model()
    classifier = FallWindowClassifier(model)

    for _ in range(29):
        classifier.update({3: _row(0.4)}, (3,))
    classifier.update({3: None}, (3,))
    for _ in range(4):
        classifier.update({3: None}, (3,))
    classifier.update({3: None}, (3,))

    assert model.inputs[0][-1] == _row(0.4)


@pytest.mark.parametrize(
    "prediction",
    [
        (0.5, 0.5),
        (0.2, 0.7, float("nan")),
        (-0.2, 0.7, 0.2),
    ],
)
def test_classifier_rejects_invalid_model_probabilities(prediction: object) -> None:
    classifier = FallWindowClassifier(_Model(prediction))

    for _ in range(29):
        classifier.update({4: _row()}, (4,))
    with pytest.raises(ValueError, match="three finite probabilities"):
        classifier.update({4: _row()}, (4,))
