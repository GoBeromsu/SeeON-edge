//! Model bundle admission (`worker/runtime/provenance/model_bundle.py`),
//! engine identity (`worker/runtime/flow/cold_start.py`) and the models
//! layout (`worker/tools/fetch_models/manifest.py`). check-config maps every
//! refusal from here to `Exit::RefuseToStart` (3).

pub mod bundle;
pub mod composition;
pub mod conformance;
pub mod flow_boot;
pub mod identity;
pub mod layout;
pub mod manifest;
pub mod onnx_shape;
pub mod packaged;
mod receipt;
mod tree;

/// Which `ModelBundleAdmissionError` message refused the bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionKind {
    /// `{label} is unavailable`.
    Unavailable,
    /// `{label} is not a regular directory`.
    NotRegularDirectory,
    /// `{label} is not a regular file`.
    NotRegularFile,
    /// `bundle path escapes its root`.
    PathEscapes,
    /// `bundle contains symlink path`.
    SymlinkPath,
    /// `bundle path is unavailable`.
    PathUnavailable,
    /// `bundle manifest is invalid JSON`.
    ManifestInvalidJson,
    /// `bundle manifest is not canonical`.
    ManifestNotCanonical,
    /// `bundle manifest contains non-JSON values`.
    ManifestNonJson,
    /// `bundle manifest schema mismatch`.
    SchemaMismatch,
    /// `bundle identity mismatch`.
    IdentityMismatch,
    /// `bundle runtime format mismatch`.
    RuntimeFormat,
    /// `bundle manifest shape mismatch`.
    ShapeMismatch,
    /// `bundle identities missing`.
    IdentitiesMissing,
    /// `{field} identity mismatch`.
    FieldIdentity,
    /// `bundle identities contain unknown fields`.
    UnknownIdentityFields,
    /// `bundle member is invalid`.
    MemberInvalid,
    /// A required packaged model/evidence member is not listed.
    MemberUnlisted,
    /// `bundle content identity mismatch`.
    ContentIdentity,
    /// `member mismatch: {path}`.
    MemberMismatch,
    /// `bundle members missing`.
    MembersMissing,
    /// `calibration.json digest mismatch: ...`.
    CalibrationDigest,
    /// `conformance_digest ... must name exactly one`.
    ConformanceDigest,
    /// `selected bundle has no bundle-manifest.json member for bundle_format`.
    NoBundleManifestMember,
    /// `{member} is invalid JSON` for bundle-manifest.json or calibration.json.
    MemberInvalidJson,
    /// `bundle format mismatch: ...`.
    BundleFormat,
    /// `policy_digest mismatch: ...`.
    PolicyDigest,
    /// `bundle manifest must declare payload members and two receipts`.
    ReceiptsMissing,
    /// `bundle receipt is invalid`.
    ReceiptInvalid,
    /// `{path} is not canonical JSON`.
    ReceiptNotCanonical,
    /// `bundle receipt identities are duplicated`.
    ReceiptDuplicated,
    /// `{path} is not a valid desired-bound receipt`.
    ReceiptNotValid,
    /// `bundle contains unsafe path: {relative}`.
    UnsafePath,
    /// `bundle tree contains missing or extra filesystem nodes`.
    TreeMismatch,
}

/// A refusal: the check and what the Python message names (a label, an
/// identity field or a member path; empty when it names nothing).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmissionError {
    pub kind: AdmissionKind,
    pub subject: String,
}

pub type Admission<T> = Result<T, AdmissionError>;

fn refuse<T>(kind: AdmissionKind, subject: &str) -> Admission<T> {
    Err(AdmissionError {
        kind,
        subject: subject.to_owned(),
    })
}
