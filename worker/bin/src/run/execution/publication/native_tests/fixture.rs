//! Explicit native-test environment. Release code does not reach this module.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process;

use seeon_deepstream_native::{MediaBinding, MediaConfig, SourceConfig};

use super::support::SOURCE_ID;

pub(super) fn prepared_record_dir() -> PathBuf {
    let record_dir = absolute_dir(Path::new(&required("SEEON_TEST_MEDIA_RECORD_DIR")));
    assert!(
        fs::read_dir(&record_dir)
            .expect("record directory")
            .next()
            .is_none(),
        "record directory must be new and empty"
    );
    probe_writable(&record_dir);
    record_dir
}

pub(super) fn evidence_dir(name: &str) -> PathBuf {
    let path = PathBuf::from(required("SEEON_TEST_MEDIA_RECORD_DIR"))
        .parent()
        .expect("owned evidence parent")
        .join(format!("seeon-{name}-{}", process::id()));
    fs::create_dir(&path).expect("owned directory");
    path
}

pub(super) fn media_config(record_directory: PathBuf, binding: MediaBinding) -> MediaConfig {
    let uri = required("SEEON_TEST_RTSP_URI")
        .into_string()
        .unwrap_or_else(|_| panic!("RTSP test URI must be UTF-8"));
    assert!(uri.starts_with("rtsp://") && uri.len() > "rtsp://".len());
    let infer = PathBuf::from(required("SEEON_TEST_MEDIA_INFER"));
    let tracker = PathBuf::from(required("SEEON_TEST_MEDIA_TRACKER"));
    for path in [&infer, &tracker] {
        assert!(fs::metadata(path).expect("config").is_file());
    }
    MediaConfig {
        sources: vec![SourceConfig {
            source_id: SOURCE_ID,
            binding,
            uri,
            record_prefix: "synthetic".into(),
        }],
        infer_config_path: infer,
        tracker_config_path: tracker,
        tracker_library_path:
            "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so".into(),
        record_directory,
        record_cache_seconds: 30,
        record_capacity: 4,
        mux_width: 640,
        mux_height: 360,
        mux_batch_timeout_us: 40_000,
        mux_live_source: true,
        tracker_width: 960,
        tracker_height: 544,
        queue_max_buffers: 4,
        preview_enabled: false,
        max_preview_bytes: 0,
        allow_file_uris: false,
        rtsp_reconnect_interval_sec: 5,
    }
}

fn probe_writable(root: &Path) {
    let probe = root.join(".seeon-rust-write-probe");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .expect("record directory must be writable");
    file.write_all(b"writable\n").expect("write probe");
    drop(file);
    fs::remove_file(&probe).expect("remove probe");
}

fn required(name: &str) -> std::ffi::OsString {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| panic!("missing explicit media test configuration: {name}"))
}

fn absolute_dir(path: &Path) -> PathBuf {
    assert!(path.is_absolute() && path.parent().is_some());
    let mut walked = PathBuf::new();
    for component in path.components() {
        assert!(matches!(
            component,
            Component::RootDir | Component::Normal(_)
        ));
        walked.push(component.as_os_str());
        assert!(fs::symlink_metadata(&walked).expect("metadata").is_dir());
    }
    fs::canonicalize(path).expect("canonical record directory")
}
