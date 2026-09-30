//! Repo-wide advisory GPU lease (plan row 20, design §2.4 step 1).
//!
//! One non-blocking `flock` on `<state_dir>/.gpu.lease`, the same file and
//! lock kind as the Python `worker.runtime.lease.GpuLease`, so a Rust worker
//! and a Python smoke or replay command exclude each other. Contention is a
//! refusal, never a wait. The file is never unlinked: a fresh inode would let
//! a second process lock it while the first holder still owns the old one.

use std::fs::{self, File};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use rustix::fs::{FlockOperation, Mode, OFlags};
use rustix::io::Errno;

/// Lease file name under the state directory, shared with the Python lease.
pub const GPU_LEASE_FILENAME: &str = ".gpu.lease";

/// Why the lease was not taken.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LeaseError {
    /// Another process holds the lease (`EWOULDBLOCK`/`EAGAIN` or `EACCES`).
    Unavailable { lease_path: PathBuf },
    /// The state directory could not be created.
    CreateDir {
        state_dir: PathBuf,
        kind: io::ErrorKind,
    },
    /// The lease file could not be opened or created.
    Open { lease_path: PathBuf, errno: Errno },
    /// `flock` failed for a reason other than contention.
    Lock { lease_path: PathBuf, errno: Errno },
}

/// A held lease. Dropping it unlocks and closes the file.
#[derive(Debug)]
pub struct GpuLease {
    file: File,
    lease_path: PathBuf,
}

impl GpuLease {
    pub fn lease_path(&self) -> &Path {
        &self.lease_path
    }
}

impl Drop for GpuLease {
    fn drop(&mut self) {
        // Unlock first; the descriptor closes when `file` drops. Never unlinked.
        let _ = rustix::fs::flock(&self.file, FlockOperation::Unlock);
    }
}

/// Takes the lease under `state_dir` without blocking.
pub fn acquire(state_dir: &Path) -> Result<GpuLease, LeaseError> {
    fs::create_dir_all(state_dir).map_err(|error| LeaseError::CreateDir {
        state_dir: state_dir.to_path_buf(),
        kind: error.kind(),
    })?;
    let lease_path = state_dir.join(GPU_LEASE_FILENAME);
    let fd = rustix::fs::open(
        &lease_path,
        OFlags::CLOEXEC | OFlags::CREATE | OFlags::RDWR,
        Mode::RUSR | Mode::WUSR,
    );
    let fd = match fd {
        Ok(fd) => fd,
        Err(errno) => return Err(LeaseError::Open { lease_path, errno }),
    };
    // On any lock failure `fd` drops here, closing the descriptor.
    if let Err(errno) = rustix::fs::flock(&fd, FlockOperation::NonBlockingLockExclusive) {
        // EWOULDBLOCK is EAGAIN on Linux; both mean another holder, as does EACCES.
        return Err(if errno == Errno::AGAIN || errno == Errno::ACCESS {
            LeaseError::Unavailable { lease_path }
        } else {
            LeaseError::Lock { lease_path, errno }
        });
    }
    let mut file = File::from(fd);
    // Diagnostics only: a failed stamp never invalidates a lease already held.
    let _ = record_owner(&mut file);
    Ok(GpuLease { file, lease_path })
}

/// Stamps the holder PID for operators, replacing any older stamp.
fn record_owner(file: &mut File) -> io::Result<()> {
    file.seek(SeekFrom::Start(0))?;
    file.set_len(0)?;
    file.write_all(format!("{}\n", std::process::id()).as_bytes())?;
    file.flush()
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    fn fresh_dir(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("seeon-ml-worker-{test}"));
        match fs::remove_dir_all(&dir) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => panic!("clear {}: {error}", dir.display()),
        }
        dir
    }

    #[test]
    fn a_second_acquire_is_unavailable_until_the_first_drops() {
        let state_dir = fresh_dir("lease-contention").join("nested").join("state");
        let lease_path = state_dir.join(".gpu.lease");

        let first = acquire(&state_dir).expect("first acquire");
        assert_eq!(
            acquire(&state_dir).unwrap_err(),
            LeaseError::Unavailable {
                lease_path: lease_path.clone()
            }
        );
        drop(first);

        let again = acquire(&state_dir).expect("reacquire after drop");
        assert_eq!(again.lease_path(), lease_path);
    }

    #[test]
    fn the_lease_file_holds_one_pid_stamp_and_outlives_the_lease() {
        let state_dir = fresh_dir("lease-file");
        let lease_path = state_dir.join(".gpu.lease");
        let stamp = format!("{}\n", std::process::id());

        drop(acquire(&state_dir).expect("acquire on a fresh dir"));
        let mode = fs::metadata(&lease_path)
            .expect("file kept")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);

        fs::write(&lease_path, "an older holder stamp, longer than any pid\n").unwrap();
        let lease = acquire(&state_dir).expect("acquire over an older stamp");
        assert_eq!(fs::read_to_string(&lease_path).unwrap(), stamp);
        drop(lease);
        assert_eq!(fs::read_to_string(&lease_path).unwrap(), stamp);
    }
}
