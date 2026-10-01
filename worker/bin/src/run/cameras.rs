//! One fall stage per already admitted runtime camera.
//! Precedence is `worker/domains/registry.py` `_effective_transition_threshold`.

use std::fmt;

use seeon_deepstream_native::MEDIA_MAX_SOURCES;
use seeon_worker::fall::{
    FallCapacities, FallFailure, FallPolicy, FallPolicyDecider, FallPolicyParameters,
};

use super::calibration::Calibration;
use super::pump::CameraPolicy;
use crate::config::pull::PulledConfig;
use crate::policy::fall::{FallStage, FallStageError};
use crate::relay::cameras::RuntimeCamera;
use crate::relay::cameras::policies::{EffectivePolicy, PolicyBundle, PolicySource, PolicyValues};

const FALL_MODULE: &str = "fall";
const FALL_MODULE_VERSION: i128 = 2;

/// Where an applied fall number came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyNumberSource {
    /// Promoted calibration receipt.
    Receipt,
    /// Image-default policy or the canonical fall.v2 default.
    Default,
}

/// Applied numbers plus the receipt values that were not authoritative.
/// The root can use `resolve_fall_policy` to report unapplied settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedFallPolicy {
    pub transition_threshold: f64,
    pub threshold_source: PolicyNumberSource,
    pub receipt_threshold: Option<f64>,
    pub unapplied_policy_threshold: Option<f64>,
    pub transition_votes: usize,
    pub transition_window: usize,
    pub confirmation_rule_source: PolicyNumberSource,
    pub receipt_transition_votes: Option<usize>,
    pub receipt_transition_window: Option<usize>,
    pub unapplied_transition_votes: Option<usize>,
    pub unapplied_transition_window: Option<usize>,
    pub temperature: f64,
}

/// A roster entry cannot become a fall stage. No document text is retained.
#[derive(Clone, Debug, PartialEq)]
pub enum CameraPolicyError {
    /// The roster index does not fit `source_id` or `MEDIA_MAX_SOURCES`.
    SourceId,
    /// Calibration votes or window do not fit the policy `usize` fields.
    CalibrationBound,
    /// The admitted bundle has no fall.v2 policy for this camera.
    PolicyMissing,
    /// The admitted fall policy is not the fall.v2 value shape.
    PolicyShape,
    /// `FallPolicy::new` refused the assembled parameters.
    Policy(FallFailure),
    /// `FallPolicyDecider::new` refused identity, epoch, or capacities.
    Decider(FallFailure),
    /// `FallStage::new` refused the calibrated temperature.
    Stage(FallStageError),
}

impl fmt::Display for CameraPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceId => formatter.write_str("roster index is not a fall source id"),
            Self::CalibrationBound => {
                formatter.write_str("calibration votes or window do not fit fall policy")
            }
            Self::PolicyMissing => formatter.write_str("admitted fall.v2 policy is missing"),
            Self::PolicyShape => formatter.write_str("admitted fall policy is not fall.v2"),
            Self::Policy(error) => write!(formatter, "fall policy refused: {error}"),
            Self::Decider(error) => write!(formatter, "fall decider refused: {error}"),
            Self::Stage(_) => formatter.write_str("fall stage refused calibrated temperature"),
        }
    }
}

impl std::error::Error for CameraPolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Policy(error) | Self::Decider(error) => Some(error),
            // FallStageError is a typed refusal but not an std::error::Error.
            Self::Stage(_)
            | Self::SourceId
            | Self::CalibrationBound
            | Self::PolicyMissing
            | Self::PolicyShape => None,
        }
    }
}

/// Roster order is the source order. An empty admitted roster is an empty list.
pub fn camera_policies(
    config: &PulledConfig,
    calibration: &Calibration,
    boot_id: &str,
    stream_epoch: &str,
    source_generation: u64,
    capacities: FallCapacities,
) -> Result<Vec<CameraPolicy>, CameraPolicyError> {
    config
        .cameras
        .iter()
        .enumerate()
        .map(|(index, camera)| {
            let resolved = resolve_fall_policy(&config.policies, &camera.camera_id, calibration)?;
            stage_for(
                index,
                camera,
                boot_id,
                stream_epoch,
                source_generation,
                capacities,
                &resolved,
            )
        })
        .collect()
}

/// Python `_effective_transition_threshold` for one admitted camera.
pub fn resolve_fall_policy(
    policies: &PolicyBundle,
    camera_id: &str,
    calibration: &Calibration,
) -> Result<ResolvedFallPolicy, CameraPolicyError> {
    let selected = admitted_fall(policies, camera_id)?;
    let PolicyValues::FallV2 {
        transition_threshold: operator,
    } = selected.values
    else {
        return Err(CameraPolicyError::PolicyShape);
    };
    let canonical = FallPolicyParameters::default();
    let receipt_votes = fit(calibration.transition_votes)?;
    let receipt_window = fit(calibration.transition_window)?;
    let (threshold, threshold_source, unapplied_threshold) = match (
        calibration.promotion_eligible,
        calibration.receipt_threshold,
        selected.source,
    ) {
        (true, Some(receipt), _) => (receipt, PolicyNumberSource::Receipt, None),
        (_, _, PolicySource::ImageDefault) => (operator, PolicyNumberSource::Default, None),
        (_, _, _) => (
            canonical.transition_threshold,
            PolicyNumberSource::Default,
            Some(operator),
        ),
    };
    let promoted_rule = calibration.promotion_eligible;
    let (votes, window, confirmation_source, unapplied_votes, unapplied_window) = if promoted_rule {
        (
            receipt_votes,
            receipt_window,
            PolicyNumberSource::Receipt,
            None,
            None,
        )
    } else {
        (
            canonical.transition_votes,
            canonical.transition_window,
            PolicyNumberSource::Default,
            Some(receipt_votes),
            Some(receipt_window),
        )
    };
    Ok(ResolvedFallPolicy {
        transition_threshold: threshold,
        threshold_source,
        receipt_threshold: calibration.receipt_threshold,
        unapplied_policy_threshold: unapplied_threshold,
        transition_votes: votes,
        transition_window: window,
        confirmation_rule_source: confirmation_source,
        receipt_transition_votes: Some(receipt_votes),
        receipt_transition_window: Some(receipt_window),
        unapplied_transition_votes: unapplied_votes,
        unapplied_transition_window: unapplied_window,
        temperature: calibration.temperature,
    })
}

fn stage_for(
    index: usize,
    camera: &RuntimeCamera,
    boot_id: &str,
    stream_epoch: &str,
    source_generation: u64,
    capacities: FallCapacities,
    resolved: &ResolvedFallPolicy,
) -> Result<CameraPolicy, CameraPolicyError> {
    if index >= MEDIA_MAX_SOURCES {
        return Err(CameraPolicyError::SourceId);
    }
    let source_id = u32::try_from(index).map_err(|_| CameraPolicyError::SourceId)?;
    let parameters = FallPolicyParameters {
        transition_threshold: resolved.transition_threshold,
        transition_votes: resolved.transition_votes,
        transition_window: resolved.transition_window,
        ..FallPolicyParameters::default()
    };
    let policy = FallPolicy::new(parameters).map_err(CameraPolicyError::Policy)?;
    let decider = FallPolicyDecider::new(
        &camera.camera_id,
        &camera.facility_id,
        boot_id,
        stream_epoch,
        source_generation,
        policy,
        capacities,
    )
    .map_err(CameraPolicyError::Decider)?;
    let stage = FallStage::new(decider, resolved.temperature).map_err(CameraPolicyError::Stage)?;
    Ok(CameraPolicy { source_id, stage })
}

fn admitted_fall<'a>(
    policies: &'a PolicyBundle,
    camera_id: &str,
) -> Result<&'a EffectivePolicy, CameraPolicyError> {
    let selected = policies
        .cameras
        .get(camera_id)
        .and_then(|camera| camera.get(FALL_MODULE))
        .or_else(|| policies.defaults.get(FALL_MODULE))
        .ok_or(CameraPolicyError::PolicyMissing)?;
    if selected.module_version != FALL_MODULE_VERSION {
        return Err(CameraPolicyError::PolicyShape);
    }
    Ok(selected)
}

fn fit(value: i128) -> Result<usize, CameraPolicyError> {
    usize::try_from(value).map_err(|_| CameraPolicyError::CalibrationBound)
}
