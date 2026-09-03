from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from worker.domains.fall.classifier_v2 import FallWindowClassifierV2
from worker.interfaces.fall_model import FallV2Probabilities


@dataclass(slots=True)
class _Model:
    prediction: object = FallV2Probabilities(0.2, 0.7, 0.1)
    inputs: list[tuple[tuple[float, ...], ...]] = field(default_factory=list)

    def predict(self, features: tuple[tuple[float, ...], ...]) -> object:
        self.inputs.append(features)
        return self.prediction


def _row(value: float = 0.0) -> tuple[float, ...]:
    return (value,) * 56


def test_classifier_windows_exactly_30_pose_bbox56_rows_on_five_frame_stride() -> None:
    model = _Model()
    classifier = FallWindowClassifierV2(model)

    for _ in range(29):
        assert classifier.update({7: _row(0.25)}, (7,)) == {}
    due = classifier.update({7: _row(0.75)}, (7,))

    assert due == {7: FallV2Probabilities(0.2, 0.7, 0.1)}
    assert len(model.inputs) == 1
    assert len(model.inputs[0]) == 30
    assert all(len(row) == 56 for row in model.inputs[0])
    assert model.inputs[0][0] == _row(0.25)
    assert model.inputs[0][-1] == _row(0.75)


def test_classifier_coasts_missing_rows_with_last_valid_pose_bbox_row() -> None:
    model = _Model()
    classifier = FallWindowClassifierV2(model)

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
    classifier = FallWindowClassifierV2(_Model(prediction))

    for _ in range(29):
        classifier.update({4: _row()}, (4,))
    with pytest.raises(ValueError, match="three finite probabilities"):
        classifier.update({4: _row()}, (4,))
