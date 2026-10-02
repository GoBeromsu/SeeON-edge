//! Step-3 fall metadata admission, before CUDA and owner creation.
//! ONNX/weights identities come from verified publisher members, never from
//! TensorRT engine hashes. Engine-to-ONNX build linkage is a separate gate.

use std::collections::BTreeMap;
use std::path::Path;

use seeon_worker::fall::FallPolicyParameters;
use seeon_worker::pose_bbox56::POSE_BBOX56_PREPROCESSING_IDENTITY;

use super::IdentityError;
use super::calibration::{self, Calibration, CalibrationError, CalibrationSource};
use crate::config::Checked;
use crate::config::model_bundle::AdmissionKind;
use crate::config::model_bundle::bundle::BundleProof;
use crate::config::model_bundle::conformance::{self, ConformanceError};
use crate::config::model_bundle::packaged::{
    PACKAGED_FALL_ROOT, PackagedProof, admit_packaged_bundle,
};
use crate::config::parse_json;
use crate::config::selection::ModelSelection;

// Python local_env / FallModelConfig authority. MODELS_ROOT_ENV is selected-only.
const OUTPUT_CLASS_COUNT: i128 = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct FallEvidence {
    pub calibration: Calibration,
    pub model_version: String,
    pub published_weights_digest: String,
    pub calibration_digest: String,
    pub preprocessing_identity: String,
}

pub(crate) fn admit(checked: &Checked) -> Result<FallEvidence, IdentityError> {
    match &checked.selection {
        Some((selection, proof)) => selected(selection, proof),
        None => {
            let proof = admit_packaged_bundle(Path::new(PACKAGED_FALL_ROOT))
                .map_err(|error| IdentityError::Packaged(error.kind))?;
            packaged(&proof)
        }
    }
}

/// Selection parsing and bundle file admission have already succeeded.
pub fn selected(
    selection: &ModelSelection,
    proof: &BundleProof,
) -> Result<FallEvidence, IdentityError> {
    if selection.output_class_count != OUTPUT_CLASS_COUNT {
        return Err(IdentityError::OutputClassCount);
    }
    let threshold = selection.transition_threshold;
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(IdentityError::SelectionThreshold);
    }
    let source = match selection.threshold_source.as_str() {
        "default"
            if calibration::isclose(
                threshold,
                FallPolicyParameters::default().transition_threshold,
            ) =>
        {
            CalibrationSource::SelectedDefault(threshold)
        }
        "default" => return Err(IdentityError::SelectionThreshold),
        "receipt" => CalibrationSource::SelectedReceipt(threshold),
        _ => return Err(IdentityError::Selection),
    };
    metadata(
        &proof.member_digests,
        &proof.calibration,
        &proof.conformance.1,
        &selection.preprocessing_identity,
        source,
    )
}

pub fn packaged(proof: &PackagedProof) -> Result<FallEvidence, IdentityError> {
    metadata(
        &proof.member_digests,
        &proof.calibration,
        &proof.conformance.1,
        POSE_BBOX56_PREPROCESSING_IDENTITY,
        CalibrationSource::Packaged,
    )
}

fn metadata(
    digests: &BTreeMap<String, String>,
    calibration: &[u8],
    conformance: &[u8],
    preprocessing: &str,
    source: CalibrationSource,
) -> Result<FallEvidence, IdentityError> {
    let conformance = parse_json(conformance).ok_or(IdentityError::Conformance(
        ConformanceError::Shape("document"),
    ))?;
    conformance::validate(&conformance).map_err(IdentityError::Conformance)?;
    if preprocessing != POSE_BBOX56_PREPROCESSING_IDENTITY {
        return Err(IdentityError::Conformance(ConformanceError::Mismatch(
            "selection preprocessing_identity",
        )));
    }
    let document = parse_json(calibration).ok_or(IdentityError::Calibration(
        CalibrationError::InvalidDocument,
    ))?;
    let calibration =
        calibration::parse(&document, preprocessing, source).map_err(IdentityError::Calibration)?;
    let member = |name| {
        digests
            .get(name)
            .cloned()
            .ok_or(IdentityError::Bundle(AdmissionKind::MemberUnlisted))
    };
    Ok(FallEvidence {
        calibration,
        model_version: member("model.onnx")?,
        published_weights_digest: member("model.pt")?,
        calibration_digest: member("calibration.json")?,
        preprocessing_identity: preprocessing.to_owned(),
    })
}
