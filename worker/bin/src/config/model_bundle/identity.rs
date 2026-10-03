//! Canonical provider-specific identity admission. TensorRT uses the unchanged
//! four-engine schema1; hybrid ORT CPU uses schema2 with only the live GPU engine.
//! Runtime reads this artifact and never builds engines or accepts legacy files.

mod admission;
mod environment;
mod infer;
mod schema;

pub use admission::{EnginePaths, IdentityInputs, fingerprint, verify_aggregate};
pub(crate) use environment::verify_environment;
pub(crate) use infer::engine_only as engine_only_config;
pub(crate) use schema::{SchemaError, native_profile, validate_entries, validate_hybrid_entries};

/// Explicit auxiliary provider selection; the live pose remains TensorRT.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AuxiliaryRuntime {
    #[default]
    TensorRt,
    OnnxRuntimeCpu,
}

impl AuxiliaryRuntime {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "tensorrt" => Some(Self::TensorRt),
            "onnxruntime-cpu" => Some(Self::OnnxRuntimeCpu),
            _ => None,
        }
    }
}

/// Exact source hashes compared by successful CPU identity admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuModelHashes {
    pub stored_pose: String,
    pub bed: String,
    pub fall: String,
}

pub(crate) struct VerifiedEnvironment {
    pub flow: std::collections::BTreeMap<String, String>,
    pub cpu_model_hashes: Option<CpuModelHashes>,
}

/// Caller-declared deployment authority, not a measured container identity.
pub fn deployment_image_digest(reference: &str) -> Option<&str> {
    let digest = reference
        .rsplit_once('@')
        .map_or(reference, |(_, digest)| digest);
    digest
        .strip_prefix("sha256:")
        .is_some_and(|value| crate::config::is_hex(value, 64))
        .then_some(digest)
}

/// Flow files retained in the aggregate's `flow` member. Pose ONNX identity
/// belongs to the live receipt and stored-pose source, not a duplicate Flow field.
pub const FLOW_IDENTITY_FILES: [(&str, &str); 4] = [
    ("infer_config_sha256", "ML_WORKER_FLOW_INFER_CONFIG"),
    ("tracker_config_sha256", "ML_WORKER_FLOW_TRACKER_CONFIG"),
    ("tracker_library_sha256", "ML_WORKER_FLOW_TRACKER_LIBRARY"),
    ("parser_lib_sha256", "ML_WORKER_FLOW_PARSER_LIBRARY"),
];

pub(crate) const OBSERVER_LIBRARY: &str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so";
/// Image-owned bed source specified by the approved engine catalog.
pub(crate) const BED_ONNX: &str = "models/bed/yolo26l-seg.onnx";

/// Compare already-measured native facts without entering CUDA here.
pub(crate) fn hardware_matches(
    identity: &std::collections::BTreeMap<String, String>,
    device: i32,
    actual: &seeon_deepstream_native::GpuHardwareIdentity,
) -> bool {
    let integer = |key| {
        identity
            .get(key)
            .and_then(|value| value.parse::<i32>().ok())
    };
    integer("device") == Some(device)
        && integer("trt_version") == Some(actual.trt_version)
        && identity.get("device_name") == Some(&actual.device_name)
        && identity.get("compute_capability")
            == Some(&format!(
                "{}.{}",
                actual.compute_major, actual.compute_minor
            ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentityKind {
    EngineAbsent,
    IdentityAbsent,
    IdentityUnreadable,
    IdentityNotUtf8,
    NotObject,
    BatchSize,
    NegativeDeployedBatch,
    BatchNotCovering,
    DigestInvalid,
    ArtifactAbsent,
    ArtifactUnreadable,
    DigestMismatch,
    ImageDigest,
    Schema,
    Hardware,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentityError {
    pub kind: IdentityKind,
    pub subject: String,
}

fn refuse<T>(kind: IdentityKind, subject: &str) -> Result<T, IdentityError> {
    Err(IdentityError {
        kind,
        subject: subject.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::AuxiliaryRuntime;

    #[test]
    fn auxiliary_runtime_accepts_only_explicit_cli_spellings() {
        assert_eq!(
            AuxiliaryRuntime::parse("tensorrt"),
            Some(AuxiliaryRuntime::TensorRt)
        );
        assert_eq!(
            AuxiliaryRuntime::parse("onnxruntime-cpu"),
            Some(AuxiliaryRuntime::OnnxRuntimeCpu)
        );
        for value in [
            "",
            "cpu",
            "onnxruntime",
            "TensorRt",
            "ONNXRUNTIME-CPU",
            " tensorrt",
            "tensorrt ",
            "onnxruntime-cpu\n",
            "tensorrt=onnxruntime-cpu",
            "tensorrt\0",
        ] {
            assert_eq!(AuxiliaryRuntime::parse(value), None, "{value:?}");
        }
    }
}
