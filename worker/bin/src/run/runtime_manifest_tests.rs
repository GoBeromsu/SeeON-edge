use super::*;
use crate::seam::{IdSource, RandomIds};
use std::fs;
use std::os::unix::fs::PermissionsExt;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(RandomIds.uuid4().unwrap());
        fs::create_dir(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn content() -> Json {
    Json::Object(vec![
        ("manifest_schema_version".into(), Json::Int(SCHEMA_VERSION)),
        ("camera_id".into(), Json::Str("복도-α".into())),
        ("temperature".into(), Json::Float(0.37)),
        ("small".into(), Json::Float(1e-7)),
        ("negative_zero".into(), Json::Float(-0.0)),
        ("generation".into(), Json::Int(i128::from(u64::MAX))),
    ])
}

#[test]
fn manifest_is_immutable_reusable_and_verified_before_reference() {
    let root = Scratch::new();
    let manifest = Manifest::freeze(&content()).unwrap();
    let path = manifest.persist(&root.0).unwrap();
    assert_eq!(fs::read(&path).unwrap(), manifest.canonical().as_bytes());
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(
        path.parent().unwrap().file_name().unwrap(),
        manifest.sha256()
    );
    assert_eq!(path.file_name().unwrap(), "manifest.json");
    for directory in [
        path.parent().unwrap(),
        path.parent().unwrap().parent().unwrap(),
    ] {
        assert_eq!(
            fs::metadata(directory).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }
    assert_eq!(manifest.persist(&root.0).unwrap(), path);
    fs::write(&path, b"corrupted retained evidence").unwrap();
    assert_eq!(manifest.persist(&root.0), Err(ManifestError::Conflict));
    assert_eq!(fs::read(&path).unwrap(), b"corrupted retained evidence");
}

#[test]
fn restrictive_umask_cannot_make_a_new_manifest_conflict_with_itself() {
    // umask is process-global: isolate it from every parallel test.
    let output = std::process::Command::new("/bin/sh")
        .args([
            "-c",
            "umask 0277; exec \"$1\" --exact run::runtime_manifest::tests::manifest_is_immutable_reusable_and_verified_before_reference --nocapture",
            "manifest-umask",
        ])
        .arg(std::env::current_exe().unwrap())
        .output()
        .unwrap();
    assert!(
        output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed"),
        "isolated permission regression failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn manifest_rejects_paths_and_oversize_before_any_persistence() {
    for unsafe_value in [
        "rtsp://camera/secret",
        "/private/model",
        "C:\\private\\model",
        "bad\nvalue",
    ] {
        assert_eq!(
            Manifest::freeze(&Json::Str(unsafe_value.into())),
            Err(ManifestError::Canonical(JsonError::UnsafeValue))
        );
    }
    assert_eq!(
        Manifest::freeze(&Json::Str("x".repeat(MAX_MANIFEST_BYTES))),
        Err(ManifestError::Capacity)
    );
}

#[test]
fn manifest_store_refuses_symlink_redirection() {
    let root = Scratch::new();
    let elsewhere = root.0.join("elsewhere");
    fs::create_dir(&elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, root.0.join("runtime-provenance")).unwrap();
    let manifest = Manifest::freeze(&content()).unwrap();
    assert_eq!(manifest.persist(&root.0), Err(ManifestError::Conflict));
    assert_eq!(fs::read_dir(elsewhere).unwrap().count(), 0);
}

#[test]
fn existing_manifest_with_public_permissions_is_not_blessed() {
    let root = Scratch::new();
    let manifest = Manifest::freeze(&content()).unwrap();
    let path = manifest.persist(&root.0).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    assert_eq!(manifest.persist(&root.0), Err(ManifestError::Conflict));
    assert_eq!(fs::read(&path).unwrap(), manifest.canonical().as_bytes());
}

#[test]
fn capacity_never_prunes_referenced_manifest_history() {
    let root = Scratch::new();
    let first = Manifest::freeze(&content()).unwrap();
    let path = first.persist(&root.0).unwrap();
    let directory = root.0.join("runtime-provenance");
    for index in 1..MAX_MANIFESTS {
        fs::create_dir(directory.join(format!("{index:064x}"))).unwrap();
    }
    let another = Manifest::freeze(&Json::Object(vec![(
        "manifest_schema_version".into(),
        Json::Int(SCHEMA_VERSION),
    )]))
    .unwrap();
    assert_eq!(another.persist(&root.0), Err(ManifestError::Capacity));
    assert_eq!(first.persist(&root.0).unwrap(), path);
    assert_eq!(fs::read_dir(directory).unwrap().count(), MAX_MANIFESTS);
}

#[test]
fn prior_boot_records_do_not_consume_content_capacity_or_get_pruned() {
    let root = Scratch::new();
    let directory = root.0.join("runtime-provenance");
    fs::create_dir(&directory).unwrap();
    for index in 0..MAX_MANIFESTS {
        fs::write(
            directory.join(format!("{index:064x}.json")),
            b"prior runtime record",
        )
        .unwrap();
    }
    let manifest = Manifest::freeze(&content()).unwrap();
    let path = manifest.persist(&root.0).unwrap();
    assert_eq!(fs::read(path).unwrap(), manifest.canonical().as_bytes());
    for index in 0..MAX_MANIFESTS {
        assert_eq!(
            fs::read(directory.join(format!("{index:064x}.json"))).unwrap(),
            b"prior runtime record"
        );
    }
    assert_eq!(fs::read_dir(directory).unwrap().count(), MAX_MANIFESTS + 1);
}

#[test]
fn matching_manifest_reclaims_only_owned_interrupted_temporary_files() {
    let root = Scratch::new();
    let manifest = Manifest::freeze(&content()).unwrap();
    let path = manifest.persist(&root.0).unwrap();
    let directory = path.parent().unwrap();
    fs::hard_link(&path, directory.join(".manifest.tmp")).unwrap();
    fs::write(
        directory.join(".manifest.tmp.tmp"),
        b"partial temporary write",
    )
    .unwrap();
    fs::write(directory.join("unowned-file"), b"preserve").unwrap();
    assert_eq!(manifest.persist(&root.0).unwrap(), path);
    assert!(!directory.join(".manifest.tmp").exists());
    assert!(!directory.join(".manifest.tmp.tmp").exists());
    assert_eq!(
        fs::read(directory.join("unowned-file")).unwrap(),
        b"preserve"
    );
    assert_eq!(fs::read(&path).unwrap(), manifest.canonical().as_bytes());
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with canonical provenance implementation"]
fn python_accepts_actual_canonical_bytes_and_digest_without_rewriting() {
    let manifest = Manifest::freeze(&content()).unwrap();
    assert_python_accepts(&manifest);
}

pub(super) fn assert_python_accepts(manifest: &Manifest) {
    let python = std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON");
    let output = std::process::Command::new(python)
        .args([
            "-c",
            r#"
import sys
from worker.runtime.provenance.models import AppliedRuntimeManifest
value = AppliedRuntimeManifest.parse(sys.argv[1], sys.argv[2])
assert value.schema_version == 2
print(value.canonical_json, end='')
"#,
        ])
        .arg(manifest.canonical())
        .arg(manifest.sha256())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, manifest.canonical().as_bytes());
}
