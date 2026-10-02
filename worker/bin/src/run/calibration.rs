//! Pure parsing of already admitted calibration data; no artifact or selection authority.

use std::fmt;

use crate::json::Json;
use crate::records::id::sha256_hex;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    pub temperature: f64,
    pub receipt_threshold: Option<f64>,
    pub transition_votes: i128,
    pub transition_window: i128,
    pub promotion_eligible: bool,
}

/// Existing runner authority, not an inferred promotion from file contents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalibrationSource {
    Packaged,
    SelectedDefault(f64),
    SelectedReceipt(f64),
}

/// Static refusals never carry document values or preprocessing identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationError {
    InvalidDocument,
    ClassOrder,
    PreprocessingIdentity,
    TemperatureType,
    TemperatureValue,
    TemporalRule,
    PromotionEligibility,
    ReceiptIneligible,
    ReceiptGrant,
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidDocument => "calibration must be an object",
            Self::ClassOrder => "calibration class order does not match the runner",
            Self::PreprocessingIdentity => "calibration preprocessing binding does not match",
            Self::TemperatureType => "calibration temperature must be numeric",
            Self::TemperatureValue => "calibration temperature must be finite and positive",
            Self::TemporalRule => "calibration temporal rule must satisfy integer 1 <= m <= n",
            Self::PromotionEligibility => "calibration promotion eligibility must be boolean",
            Self::ReceiptIneligible => "calibration does not authorize receipt selection",
            Self::ReceiptGrant => "calibration does not grant the declared receipt threshold",
        })
    }
}

impl std::error::Error for CalibrationError {}

/// Mirrors packaged/selected `ort_pose_bbox56` metadata. Selection admission
/// owns declared threshold bounds; receipt selection still requires its grant.
pub fn parse(
    document: &Json,
    preprocessing_identity: &str,
    source: CalibrationSource,
) -> Result<Calibration, CalibrationError> {
    let Json::Object(members) = document else {
        return Err(CalibrationError::InvalidDocument);
    };
    match member(members, "class_order") {
        Some(Json::Array(order)) => match order.as_slice() {
            [Json::Str(first), Json::Str(second)]
                if first == "non_fall" && second == "fall_transition_proxy" => {}
            _ => return Err(CalibrationError::ClassOrder),
        },
        _ => return Err(CalibrationError::ClassOrder),
    }
    let expected_digest = sha256_hex(preprocessing_identity.as_bytes());
    if !matches!(member(members, "preprocessing_identity_digest"),
        Some(Json::Str(digest)) if digest == &expected_digest)
    {
        return Err(CalibrationError::PreprocessingIdentity);
    }
    let temperature =
        number(member(members, "temperature")).ok_or(CalibrationError::TemperatureType)?;
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(CalibrationError::TemperatureValue);
    }
    let Some(Json::Object(temporal)) = member(members, "temporal_rule") else {
        return Err(CalibrationError::TemporalRule);
    };
    let (Some(Json::Int(votes)), Some(Json::Int(window))) =
        (member(temporal, "m"), member(temporal, "n"))
    else {
        return Err(CalibrationError::TemporalRule);
    };
    if *votes < 1 || window < votes {
        return Err(CalibrationError::TemporalRule);
    }
    let candidate_threshold = number(member(members, "threshold"));
    let (receipt_threshold, promotion_eligible) = match source {
        CalibrationSource::Packaged => {
            let Some(Json::Bool(eligible)) = member(members, "promotion_eligible") else {
                return Err(CalibrationError::PromotionEligibility);
            };
            (
                candidate_threshold.filter(|value| (0.0..=1.0).contains(value)),
                *eligible,
            )
        }
        CalibrationSource::SelectedDefault(declared) => (Some(declared), false),
        CalibrationSource::SelectedReceipt(declared) => {
            if member(members, "promotion_eligible") != Some(&Json::Bool(true)) {
                return Err(CalibrationError::ReceiptIneligible);
            }
            if !candidate_threshold.is_some_and(|granted| isclose(granted, declared)) {
                return Err(CalibrationError::ReceiptGrant);
            }
            (Some(declared), true)
        }
    };
    Ok(Calibration {
        temperature,
        receipt_threshold,
        transition_votes: *votes,
        transition_window: *window,
        promotion_eligible,
    })
}

fn member<'a>(members: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    // A Python JSON object's last occurrence supplies its dictionary value.
    members
        .iter()
        .rev()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

fn number(value: Option<&Json>) -> Option<f64> {
    match value {
        Some(Json::Int(value)) => Some(*value as f64),
        Some(Json::Float(value)) => Some(*value),
        _ => None,
    }
}

/// Python `math.isclose` defaults: rel_tol=1e-9, abs_tol=0.0. Equality must
/// precede the infinity refusal; neither NaN nor opposite infinities is close.
pub(crate) fn isclose(left: f64, right: f64) -> bool {
    if left == right {
        return true;
    }
    if left.is_infinite() || right.is_infinite() {
        return false;
    }
    let difference = (right - left).abs();
    difference <= (1e-9 * right).abs() || difference <= (1e-9 * left).abs() || difference <= 0.0
}
