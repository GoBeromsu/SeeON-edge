//! Logit to probabilities for the fall glue, ported from the NumPy float32
//! sigmoid in `ort_pose_bbox56.py` L260, widened to `f64` for
//! `FallProbabilities`.

use seeon_worker::fall::FallProbabilities;

/// `float(1.0 / (1.0 + np.exp(-logit / temperature)))` over float32;
/// `None` for a non-finite logit. `f32::exp` may differ from NumPy's float32
/// `exp` by one ulp before the division.
pub fn transition_probability(logit: f32, temperature: f32) -> Option<f64> {
    if !logit.is_finite() {
        return None;
    }
    Some(f64::from(
        1.0_f32 / (1.0_f32 + (-logit / temperature).exp()),
    ))
}

/// `FallProbabilities(1 - p, p, 0.0)` of `transition_probability`.
pub(super) fn probabilities(logit: f32, temperature: f32) -> Option<FallProbabilities> {
    let transition = transition_probability(logit, temperature)?;
    FallProbabilities::new(1.0 - transition, transition, 0.0).ok()
}
