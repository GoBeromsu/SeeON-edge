"""Tests for worker.pipeline.perception.features.window_features.extract_window_features.

Verifies shape, dtype, NaN/inf-freedom for degenerate inputs.
All fixtures are synthetic; no real data is loaded.

Ported from the deleted tests/test_training_features.py (training/config.py
no longer lives here — training moved to eldercare-dataset-ops),
then re-ported off the legacy edge features module onto the worker replacement
(todo 30c gap #10: worker/pipeline/perception/features/window_features.py was
algorithmically identical to the legacy module but had zero direct worker-side
coverage before this port).
"""

from __future__ import annotations

import numpy as np

from contracts.model import DEFAULT_FALL_CONFIDENCE_THRESHOLD
from worker.pipeline.perception.features.window_features import extract_window_features

# Source of truth for the feature dimension: features.window_features._D
# (module docstring: "do not derive D from anywhere else"). T_WINDOW=30 is a
# locked ADR parameter (docs/rules/ml-training.md); extract_window_features
# itself is window-length agnostic, so any T works here — 30 just matches the
# real pipeline's window geometry.
_FEATURE_DIM = 45
_T_WINDOW = 30

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_zero_window() -> np.ndarray:
    """float32[T, 17, 3] — all coords and confidences are zero."""
    return np.zeros((_T_WINDOW, 17, 3), dtype=np.float32)


def _partial_conf_window(seed: int = 0) -> np.ndarray:
    """float32[T, 17, 3] — random coords; half the keypoints have conf=0 (missing)."""
    rng = np.random.default_rng(seed)
    window = rng.random((_T_WINDOW, 17, 3)).astype(np.float32)
    # Zero out the confidence channel for every other keypoint
    window[:, ::2, 2] = 0.0
    return window


def _above_threshold_window(seed: int = 1) -> np.ndarray:
    """float32[T, 17, 3] — all keypoints above DEFAULT_FALL_CONFIDENCE_THRESHOLD."""
    rng = np.random.default_rng(seed)
    window = rng.random((_T_WINDOW, 17, 3)).astype(np.float32)
    window[:, :, 2] = DEFAULT_FALL_CONFIDENCE_THRESHOLD + 0.1  # well above threshold
    return window


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractWindowFeaturesShape:
    def test_output_shape_is_feature_dim(self) -> None:
        feats = extract_window_features(_all_zero_window())
        assert feats.shape == (_FEATURE_DIM,)

    def test_output_dtype_is_float32(self) -> None:
        feats = extract_window_features(_all_zero_window())
        assert feats.dtype == np.float32

    def test_feature_dim_matches_config_constant(self) -> None:
        """_FEATURE_DIM here must equal the actual output length."""
        feats = extract_window_features(_above_threshold_window())
        assert len(feats) == _FEATURE_DIM


class TestExtractWindowFeaturesNumerical:
    def test_no_nan_inf_on_all_zero_window(self) -> None:
        """All-zero window (all conf=0, no valid keypoints) must produce finite features."""
        feats = extract_window_features(_all_zero_window())
        assert np.all(np.isfinite(feats)), (
            f"NaN or inf in features for all-zero window: indices "
            f"{np.where(~np.isfinite(feats))[0].tolist()}"
        )

    def test_no_nan_inf_on_partial_conf_window(self) -> None:
        """Half-missing keypoints (conf=0) must not produce NaN or inf."""
        feats = extract_window_features(_partial_conf_window())
        assert np.all(np.isfinite(feats)), (
            f"NaN or inf in features for partial-conf window: indices "
            f"{np.where(~np.isfinite(feats))[0].tolist()}"
        )

    def test_all_zero_window_produces_all_zero_features(self) -> None:
        """With no valid keypoints the feature vector must be all zeros."""
        feats = extract_window_features(_all_zero_window())
        np.testing.assert_array_equal(feats, 0.0)

    def test_active_window_features_are_non_negative(self) -> None:
        """All hand-crafted features are magnitudes or averages — must be >= 0."""
        feats = extract_window_features(_above_threshold_window())
        assert np.all(feats >= 0.0), (
            f"Negative feature values at indices: {np.where(feats < 0)[0].tolist()}"
        )
