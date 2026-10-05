//! Bind immutable CPU model bytes to the successful aggregate admission proof.
//! No selection reread, model execution, or provider fallback occurs here.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::fall_evidence::FallEvidence;
use super::model_sources::capture_member;
use super::models::CpuModels;
use super::{FlowSettings, IdentityError, ModelRole};
use crate::config::Checked;
use crate::config::env::Env;
use crate::config::model_bundle::identity::{BED_ONNX, CpuModelHashes, IdentityKind};
use crate::config::model_bundle::packaged::PACKAGED_FALL_ROOT;

// Dockerfile.edge extracts the locked API-29 library into the native runtime path.
// This is a library location, not a claimed observation of the loaded runtime.
const RUNTIME_LIBRARY: &str = "/opt/seeon/lib/libonnxruntime.so.1.29.0";

pub(super) fn admit(
    env: &Env,
    checked: &Checked,
    fall: &FallEvidence,
    flow: &FlowSettings,
) -> Result<CpuModels, IdentityError> {
    let hashes = checked
        .cpu_model_hashes
        .as_ref()
        .ok_or(IdentityError::Engine(IdentityKind::Schema))?;
    if hashes.fall != fall.model_version {
        return Err(IdentityError::CpuSource(ModelRole::Fall));
    }
    let selected = checked.selection.is_some();
    let root = match &checked.selection {
        Some((selection, _)) => crate::config::models_root(env)
            .join("bundles")
            .join(&selection.model_publication.content),
        None => PathBuf::from(PACKAGED_FALL_ROOT),
    };
    capture(
        &flow.onnx,
        Path::new(BED_ONNX),
        &root.join("model.onnx"),
        selected,
        hashes,
    )
}

fn capture(
    pose: &Path,
    bed: &Path,
    fall: &Path,
    selected_fall: bool,
    hashes: &CpuModelHashes,
) -> Result<CpuModels, IdentityError> {
    let bound = |path: &Path, hash: &str, nofollow: bool, role| {
        capture_member(path, hash, nofollow)
            .map(|source| Arc::from(source.bytes))
            .map_err(|_| IdentityError::CpuSource(role))
    };
    // Match the existing fall, bed, stored-pose admission/readiness order.
    let fall = bound(fall, &hashes.fall, selected_fall, ModelRole::Fall)?;
    let bed = bound(bed, &hashes.bed, false, ModelRole::Bed)?;
    let stored_pose = bound(pose, &hashes.stored_pose, false, ModelRole::StoredPose)?;
    Ok(CpuModels {
        runtime_library: PathBuf::from(RUNTIME_LIBRARY),
        fall,
        bed,
        stored_pose,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::records::id::sha256_hex;
    use crate::seam::{IdSource, RandomIds};
    use std::fs;
    use std::os::unix::fs::symlink;

    struct Sources(PathBuf);

    impl Sources {
        fn new() -> Self {
            let root =
                std::env::temp_dir().join(format!("cpu-admission-{}", RandomIds.uuid4().unwrap()));
            fs::create_dir(&root).unwrap();
            for (name, bytes) in [("pose", b"pose"), ("bed", b"bed!"), ("fall", b"fall")] {
                fs::write(root.join(name), bytes).unwrap();
            }
            Self(root)
        }

        fn hashes(&self) -> CpuModelHashes {
            CpuModelHashes {
                stored_pose: sha256_hex(b"pose"),
                bed: sha256_hex(b"bed!"),
                fall: sha256_hex(b"fall"),
            }
        }

        fn capture(&self, selected: bool) -> Result<CpuModels, IdentityError> {
            capture(
                &self.0.join("pose"),
                &self.0.join("bed"),
                &self.0.join("fall"),
                selected,
                &self.hashes(),
            )
        }
    }

    impl Drop for Sources {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn captured_bytes_survive_replacement_and_removal_without_reopening() {
        let source = Sources::new();
        let models = source.capture(true).unwrap();
        fs::write(source.0.join("pose"), b"replacement").unwrap();
        fs::remove_file(source.0.join("bed")).unwrap();
        fs::remove_file(source.0.join("fall")).unwrap();
        assert_eq!(models.stored_pose.as_ref(), b"pose");
        assert_eq!(models.bed.as_ref(), b"bed!");
        assert_eq!(models.fall.as_ref(), b"fall");
    }

    #[test]
    fn changed_or_absent_sources_refuse_the_exact_role_before_owner_creation() {
        for (name, role) in [
            ("fall", ModelRole::Fall),
            ("bed", ModelRole::Bed),
            ("pose", ModelRole::StoredPose),
        ] {
            let source = Sources::new();
            fs::write(source.0.join(name), b"tampered").unwrap();
            assert!(
                matches!(source.capture(false), Err(IdentityError::CpuSource(actual)) if actual == role)
            );
            fs::remove_file(source.0.join(name)).unwrap();
            assert!(
                matches!(source.capture(false), Err(IdentityError::CpuSource(actual)) if actual == role)
            );
        }
    }

    #[test]
    fn selected_fall_refuses_symlinks_while_packaged_sources_follow_them() {
        let source = Sources::new();
        for name in ["pose", "bed", "fall"] {
            let target = source.0.join(format!("real-{name}"));
            fs::rename(source.0.join(name), &target).unwrap();
            symlink(target, source.0.join(name)).unwrap();
        }
        assert!(matches!(
            source.capture(true),
            Err(IdentityError::CpuSource(ModelRole::Fall))
        ));
        let captured = source.capture(false).unwrap();
        assert_eq!(captured.fall.as_ref(), b"fall");
        assert_eq!(captured.bed.as_ref(), b"bed!");
        assert_eq!(captured.stored_pose.as_ref(), b"pose");
    }

    #[test]
    fn malformed_proof_and_wrong_role_hashes_are_refused() {
        let source = Sources::new();
        let mut hashes = source.hashes();
        for bad in ["", "not-a-digest", hashes.bed.as_str()] {
            let mut invalid = hashes.clone();
            invalid.fall = bad.to_owned();
            assert!(matches!(
                capture(
                    &source.0.join("pose"),
                    &source.0.join("bed"),
                    &source.0.join("fall"),
                    false,
                    &invalid
                ),
                Err(IdentityError::CpuSource(ModelRole::Fall))
            ));
        }
        hashes.stored_pose = hashes.bed.clone();
        assert!(matches!(
            capture(
                &source.0.join("pose"),
                &source.0.join("bed"),
                &source.0.join("fall"),
                false,
                &hashes
            ),
            Err(IdentityError::CpuSource(ModelRole::StoredPose))
        ));
    }
}
