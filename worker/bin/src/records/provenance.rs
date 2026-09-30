//! Wire provenance from the execution identities resolved at composition
//! (`worker/pipeline/diagnostics/provenance.py` `build_wire_provenance`).
//! A missing identity refuses the whole composition: the worker must not
//! start exporting records it cannot attribute.

use std::fmt;

use crate::records::id::ContractError;
use crate::records::wire::Provenance;

/// The six execution identities the process resolves at start-up. The
/// seventh member, `config_digest`, is passed beside them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Identities {
    pub build_revision: Option<String>,
    pub image_digest: Option<String>,
    pub model_digest: Option<String>,
    pub calibration_digest: Option<String>,
    pub preprocessing_identity: Option<String>,
    pub policy_identity: Option<String>,
}

/// Python `ExecutionRecordProvenanceError` raised by `build_wire_provenance`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProvenanceError {
    /// Wire member names whose identity is absent or empty, in wire order.
    Missing(Vec<&'static str>),
    /// Every identity is present but one breaks the `WireProvenance` contract.
    Contract(ContractError),
}

impl fmt::Display for ProvenanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(names) => write!(
                formatter,
                "execution-record provenance missing: {}",
                names.join(", ")
            ),
            Self::Contract(_) => formatter.write_str("execution-record provenance is invalid"),
        }
    }
}

impl std::error::Error for ProvenanceError {}

/// `build_wire_provenance`: absent or empty members are refused together,
/// then the assembled provenance must satisfy the wire contract.
pub fn build_provenance(
    identities: &Identities,
    config_digest: &str,
) -> Result<Provenance, ProvenanceError> {
    let members: [(&'static str, Option<&str>); 7] = [
        (
            "worker_build_revision",
            identities.build_revision.as_deref(),
        ),
        ("worker_image_digest", identities.image_digest.as_deref()),
        ("model_digest", identities.model_digest.as_deref()),
        (
            "calibration_digest",
            identities.calibration_digest.as_deref(),
        ),
        (
            "preprocessing_identity",
            identities.preprocessing_identity.as_deref(),
        ),
        ("config_digest", Some(config_digest)),
        ("policy_identity", identities.policy_identity.as_deref()),
    ];
    let missing: Vec<&'static str> = members
        .iter()
        .filter(|(_, value)| value.is_none_or(str::is_empty))
        .map(|(name, _)| *name)
        .collect();
    if !missing.is_empty() {
        return Err(ProvenanceError::Missing(missing));
    }
    let [
        build,
        image,
        model,
        calibration,
        preprocessing,
        config,
        policy,
    ] = members.map(|(_, value)| value.unwrap_or_default().to_owned());
    let provenance = Provenance {
        worker_build_revision: build,
        worker_image_digest: image,
        model_digest: model,
        calibration_digest: calibration,
        preprocessing_identity: preprocessing,
        config_digest: config,
        policy_identity: policy,
    };
    provenance.validate().map_err(ProvenanceError::Contract)?;
    Ok(provenance)
}
