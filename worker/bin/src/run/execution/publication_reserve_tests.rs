//! Issue671 CPU composition with an actually full, bounded clip filesystem.
//! Explicit opt-in: SEEON_TEST_ENOSPC_DIR (normally a dedicated 8 MiB tmpfs)
//! and real ffprobe. State/recording must be on another device. A byte-full
//! tmpfs can still create directories; neither case claims mkdir failed.

use super::save_tests::failed_codec_probe_continuation;
use crate::clips::publish::{MANIFEST_FILE, TERMINAL_MARKER};
use crate::seam::{IdSource, RandomIds};
use std::fs::{self, File};
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

const MAX_VOLUME_BYTES: u64 = 64 * 1024 * 1024;
static ENOSPC_VOLUME: Mutex<()> = Mutex::new(());

struct FullStore {
    root: PathBuf,
    _serial: MutexGuard<'static, ()>,
}

impl FullStore {
    fn new() -> Self {
        let volume = PathBuf::from(
            std::env::var_os("SEEON_TEST_ENOSPC_DIR").expect("SEEON_TEST_ENOSPC_DIR is required"),
        );
        let serial = ENOSPC_VOLUME
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let stats = rustix::fs::statvfs(&volume).expect("explicit clip filesystem must exist");
        let capacity = stats.f_blocks.checked_mul(stats.f_frsize).unwrap();
        assert!(
            capacity > 0 && capacity <= MAX_VOLUME_BYTES,
            "refuse to fill an unbounded filesystem: {capacity} bytes"
        );
        let root = volume.join(format!(
            "publication-reserve-{}",
            RandomIds.uuid4().unwrap()
        ));
        fs::create_dir(&root).expect("owned clip fixture directory");
        Self {
            root,
            _serial: serial,
        }
    }
}

impl Drop for FullStore {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("owned full clip filesystem cleanup");
    }
}

fn fill(directory: &Path) {
    let mut filler = File::create(directory.join("filler")).unwrap();
    let mut written = 0_u64;
    for block in [vec![0_u8; 64 * 1024], vec![0_u8; 4096], vec![0_u8; 1]] {
        loop {
            match filler.write_all(&block) {
                Ok(()) => {
                    written += u64::try_from(block.len()).unwrap();
                    assert!(
                        written <= MAX_VOLUME_BYTES,
                        "clip filesystem grew beyond its bound"
                    );
                }
                Err(error) => {
                    assert_eq!(
                        error.raw_os_error(),
                        Some(rustix::io::Errno::NOSPC.raw_os_error())
                    );
                    break;
                }
            }
        }
    }
    filler.sync_all().unwrap();
    assert_eq!(rustix::fs::statvfs(directory).unwrap().f_bavail, 0);
}

fn full_store_case(existing_directories: bool) {
    let store = FullStore::new();
    failed_codec_probe_continuation(Some(&store.root), |output, clip_id| {
        let clip_device = fs::metadata(output.store.root()).unwrap().dev();
        for path in [output.queue.directory(), output.record_dir.as_path()] {
            assert_ne!(
                fs::metadata(path).unwrap().dev(),
                clip_device,
                "state/recording need a separate volume"
            );
        }
        let directory = output.store.clip_dir(clip_id);
        let probe = if existing_directories {
            output.store.reserve(&output.cameras[0].2, clip_id).unwrap();
            fs::create_dir(&directory).unwrap();
            crate::clips::durable::temp_path(&directory.join(MANIFEST_FILE)).unwrap()
        } else {
            assert!(!output.store.root().join("clips").exists());
            assert!(!output.store.staging_dir(clip_id).exists());
            assert!(!directory.exists());
            output.store.root().join(".full-write-probe")
        };
        assert!(!directory.join(MANIFEST_FILE).exists());
        assert!(!directory.join(TERMINAL_MARKER).exists());
        let mut probe_file = File::create(&probe).unwrap();
        fill(output.store.root());
        let error = probe_file
            .write_all(b"x")
            .expect_err("clip write must fail on the genuine full filesystem");
        assert_eq!(
            error.raw_os_error(),
            Some(rustix::io::Errno::NOSPC.raw_os_error())
        );
        drop(probe_file);
        fs::remove_file(probe).unwrap();
        assert_eq!(
            rustix::fs::statvfs(output.store.root()).unwrap().f_bavail,
            0
        );
        output.reserve.available() - 1
    });
}

#[test]
#[ignore = "requires SEEON_TEST_ENOSPC_DIR (separate bounded clip filesystem) and ffprobe"]
fn cpu_composition_failed_codec_probe_full_store_without_clip_directories_recovers_reserve() {
    full_store_case(false);
}

#[test]
#[ignore = "requires SEEON_TEST_ENOSPC_DIR (separate bounded clip filesystem) and ffprobe"]
fn cpu_composition_failed_codec_probe_full_store_manifest_write_recovers_reserve() {
    full_store_case(true);
}
