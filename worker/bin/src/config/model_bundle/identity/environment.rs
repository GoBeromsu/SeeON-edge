//! Existing deployment paths and admitted fall selection feed the pure file gate.

use std::path::{Path, PathBuf};

use super::{
    AuxiliaryRuntime, BED_ONNX, CpuModelHashes, EnginePaths, FLOW_IDENTITY_FILES, IdentityError,
    IdentityInputs, IdentityKind, OBSERVER_LIBRARY, VerifiedEnvironment, fingerprint, refuse,
    verify_aggregate,
};
use crate::config::env::Env;
use crate::config::model_bundle::bundle::BundleProof;
use crate::config::model_bundle::packaged::{PACKAGED_FALL_ROOT, admit_packaged_bundle};
use crate::config::selection::ModelSelection;

pub(crate) fn verify_environment(
    env: &Env,
    selection: Option<&(ModelSelection, BundleProof)>,
    deployed_batch: Option<i128>,
    auxiliary_runtime: AuxiliaryRuntime,
) -> Result<VerifiedEnvironment, IdentityError> {
    let path = |key: &str| -> Result<PathBuf, IdentityError> {
        env.get(key)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .ok_or_else(|| IdentityError {
                kind: IdentityKind::ArtifactAbsent,
                subject: key.to_owned(),
            })
    };
    let live_pose = path("ML_WORKER_FLOW_ENGINE_PATH")?;
    let auxiliary_paths = match auxiliary_runtime {
        AuxiliaryRuntime::TensorRt => Some([
            path("ML_WORKER_STORED_POSE_ENGINE_PATH")?,
            path("ML_WORKER_BED_ENGINE_PATH")?,
            path("ML_WORKER_FALL_ENGINE_PATH")?,
        ]),
        AuxiliaryRuntime::OnnxRuntimeCpu => None,
    };
    let identity = path("ML_WORKER_FLOW_ENGINE_IDENTITY_PATH")?;
    let pose_onnx = path("ML_WORKER_FLOW_ONNX_PATH")?;
    let pose_sha = fingerprint(&pose_onnx, "pose.onnx")?;
    let bed_sha = fingerprint(Path::new(BED_ONNX), "bed.onnx")?;
    let packaged;
    let fall_members = match selection {
        Some((_, proof)) => &proof.member_digests,
        None => {
            packaged = admit_packaged_bundle(Path::new(PACKAGED_FALL_ROOT)).map_err(|_| {
                IdentityError {
                    kind: IdentityKind::DigestMismatch,
                    subject: "fall.bundle".to_owned(),
                }
            })?;
            &packaged.member_digests
        }
    };
    let Some(fall_sha) = fall_members.get("model.onnx") else {
        return refuse(IdentityKind::DigestInvalid, "fall.onnx");
    };
    let image = env.get("ML_WORKER_IMAGE").ok_or_else(|| IdentityError {
        kind: IdentityKind::ImageDigest,
        subject: "image_digest".to_owned(),
    })?;
    let configured_batch = env
        .get("ML_WORKER_FLOW_BATCH_SIZE")
        .and_then(|text| crate::config::model_bundle::flow_boot::positive_int(text))
        .and_then(|text| text.parse::<u32>().ok())
        .ok_or_else(|| IdentityError {
            kind: IdentityKind::BatchSize,
            subject: "batch_size".to_owned(),
        })?;
    let flow = FLOW_IDENTITY_FILES
        .iter()
        .map(|(key, name)| Ok((*key, path(name)?)))
        .collect::<Result<Vec<_>, IdentityError>>()?;
    let engines = match &auxiliary_paths {
        Some([stored_pose, bed, fall]) => EnginePaths::TensorRt {
            live_pose: &live_pose,
            stored_pose,
            bed,
            fall,
        },
        None => EnginePaths::OnnxRuntimeCpu {
            live_pose: &live_pose,
        },
    };
    let flow = verify_aggregate(
        &identity,
        IdentityInputs {
            engines,
            pose_onnx_sha256: &pose_sha,
            bed_onnx_sha256: &bed_sha,
            fall_onnx_sha256: fall_sha,
            flow: &flow,
            observer_library: Path::new(OBSERVER_LIBRARY),
            image_digest: image,
            configured_batch: Some(configured_batch),
            deployed_batch,
        },
    )?;
    let cpu_model_hashes = match auxiliary_runtime {
        AuxiliaryRuntime::TensorRt => None,
        AuxiliaryRuntime::OnnxRuntimeCpu => Some(CpuModelHashes {
            stored_pose: pose_sha,
            bed: bed_sha,
            fall: fall_sha.clone(),
        }),
    };
    Ok(VerifiedEnvironment {
        flow,
        cpu_model_hashes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refusal(env: &Env, runtime: AuxiliaryRuntime, kind: IdentityKind, subject: &str) {
        let error = verify_environment(env, None, None, runtime)
            .err()
            .expect("incomplete environment must refuse");
        assert_eq!((error.kind, error.subject), (kind, subject.to_owned()));
    }

    #[test]
    fn tensor_rt_keeps_all_required_engine_paths_in_existing_order() {
        let mut env = Env::from([(
            "ML_WORKER_FLOW_ENGINE_PATH".to_owned(),
            "/unused".to_owned(),
        )]);
        for key in [
            "ML_WORKER_STORED_POSE_ENGINE_PATH",
            "ML_WORKER_BED_ENGINE_PATH",
            "ML_WORKER_FALL_ENGINE_PATH",
            "ML_WORKER_FLOW_ENGINE_IDENTITY_PATH",
            "ML_WORKER_FLOW_ONNX_PATH",
        ] {
            refusal(
                &env,
                AuxiliaryRuntime::TensorRt,
                IdentityKind::ArtifactAbsent,
                key,
            );
            env.insert(key.to_owned(), "/unused".to_owned());
        }
    }

    #[test]
    fn cpu_skips_auxiliary_gpu_paths_but_requires_live_identity_and_source() {
        for value in [None, Some(""), Some("/unused"), Some("\0")] {
            let mut env = Env::new();
            if let Some(value) = value {
                for key in [
                    "ML_WORKER_STORED_POSE_ENGINE_PATH",
                    "ML_WORKER_BED_ENGINE_PATH",
                    "ML_WORKER_FALL_ENGINE_PATH",
                ] {
                    env.insert(key.to_owned(), value.to_owned());
                }
            }
            for key in [
                "ML_WORKER_FLOW_ENGINE_PATH",
                "ML_WORKER_FLOW_ENGINE_IDENTITY_PATH",
                "ML_WORKER_FLOW_ONNX_PATH",
            ] {
                refusal(
                    &env,
                    AuxiliaryRuntime::OnnxRuntimeCpu,
                    IdentityKind::ArtifactAbsent,
                    key,
                );
                env.insert(key.to_owned(), "/unused".to_owned());
            }
            // A directory refuses before the fixed image bed and SDK observer files.
            env.insert(
                "ML_WORKER_FLOW_ONNX_PATH".to_owned(),
                env!("CARGO_MANIFEST_DIR").to_owned(),
            );
            refusal(
                &env,
                AuxiliaryRuntime::OnnxRuntimeCpu,
                IdentityKind::ArtifactUnreadable,
                "pose.onnx",
            );
        }
    }
}
