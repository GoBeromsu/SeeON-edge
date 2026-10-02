//! Existing deployment paths and admitted fall selection feed the pure file gate.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::{
    BED_ONNX, EnginePaths, FLOW_IDENTITY_FILES, IdentityError, IdentityInputs, IdentityKind,
    OBSERVER_LIBRARY, fingerprint, refuse, verify_aggregate,
};
use crate::config::env::Env;
use crate::config::model_bundle::bundle::BundleProof;
use crate::config::model_bundle::packaged::{PACKAGED_FALL_ROOT, admit_packaged_bundle};
use crate::config::selection::ModelSelection;

pub(crate) fn verify_environment(
    env: &Env,
    selection: Option<&(ModelSelection, BundleProof)>,
    deployed_batch: Option<i128>,
) -> Result<BTreeMap<String, String>, IdentityError> {
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
    let stored_pose = path("ML_WORKER_STORED_POSE_ENGINE_PATH")?;
    let bed = path("ML_WORKER_BED_ENGINE_PATH")?;
    let fall = path("ML_WORKER_FALL_ENGINE_PATH")?;
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
    verify_aggregate(
        &identity,
        IdentityInputs {
            engines: EnginePaths {
                live_pose: &live_pose,
                stored_pose: &stored_pose,
                bed: &bed,
                fall: &fall,
            },
            pose_onnx_sha256: &pose_sha,
            bed_onnx_sha256: &bed_sha,
            fall_onnx_sha256: fall_sha,
            flow: &flow,
            observer_library: Path::new(OBSERVER_LIBRARY),
            image_digest: image,
            configured_batch: Some(configured_batch),
            deployed_batch,
        },
    )
}
