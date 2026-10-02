//! Immutable identity publication. The destination parent must already exist.
//! A late failure may leave the destination present; only the owned temp is
//! unlinked, and only when its inode still matches the file this call created.
//! Cleanup failure is returned even after a successful hard link. The published
//! destination is never deleted.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use rustix::fs::{self, AtFlags, FileType, Mode, OFlags};
use rustix::io::Errno;
use sha2::{Digest, Sha256};

use super::{FlowArtifacts, IdentityError, IdentityRequest};
use crate::seam::{IdSource, RandomIds};

const MAX_FLOW_BYTES: u64 = 512 * 1024 * 1024;
const FILE_MODE: Mode = Mode::from_bits_truncate(0o644);

pub fn publish_identity(request: IdentityRequest<'_>) -> Result<(), IdentityError> {
    let bytes = super::document(request)?;
    publish(request.destination, &bytes)
}

pub(super) fn flow_fingerprints(
    flow: FlowArtifacts<'_>,
) -> Result<serde_json::Value, IdentityError> {
    Ok(serde_json::json!({
        "parser_lib_sha256": fingerprint(flow.parser_lib)?,
        "infer_config_sha256": fingerprint(flow.infer_config)?,
        "tracker_config_sha256": fingerprint(flow.tracker_config)?,
        "tracker_library_sha256": fingerprint(flow.tracker_library)?,
    }))
}

fn fingerprint(path: &Path) -> Result<String, IdentityError> {
    let file = File::from(
        fs::open(
            path,
            OFlags::CLOEXEC | OFlags::RDONLY | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(flow_open)?,
    );
    let info = fs::fstat(&file).map_err(|_| IdentityError::Flow)?;
    if FileType::from_raw_mode(info.st_mode) != FileType::RegularFile {
        return Err(IdentityError::Flow);
    }
    let declared = u64::try_from(info.st_size).map_err(|_| IdentityError::Flow)?;
    if declared == 0 || declared > MAX_FLOW_BYTES {
        return Err(IdentityError::Flow);
    }
    let mut hasher = Sha256::new();
    let mut seen = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    let mut reader = file.take(declared + 1);
    loop {
        let count = reader.read(&mut buffer).map_err(|_| IdentityError::Flow)?;
        if count == 0 {
            break;
        }
        seen = seen
            .checked_add(u64::try_from(count).map_err(|_| IdentityError::Flow)?)
            .ok_or(IdentityError::Flow)?;
        if seen > declared {
            return Err(IdentityError::Flow);
        }
        hasher.update(&buffer[..count]);
    }
    if seen != declared {
        return Err(IdentityError::Flow);
    }
    encode_hex(hasher.finalize().as_slice())
}

fn encode_hex(bytes: &[u8]) -> Result<String, IdentityError> {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(encoded, "{byte:02x}").map_err(|_| IdentityError::Flow)?;
    }
    Ok(encoded)
}

fn flow_open(error: rustix::io::Errno) -> IdentityError {
    match error {
        Errno::NOENT
        | Errno::INVAL
        | Errno::ISDIR
        | Errno::NOTDIR
        | Errno::ACCESS
        | Errno::LOOP
        | Errno::NAMETOOLONG => IdentityError::Flow,
        _ => IdentityError::Io,
    }
}

fn publish(destination: &Path, bytes: &[u8]) -> Result<(), IdentityError> {
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = destination
        .file_name()
        .filter(|name| !name.is_empty() && !name.as_encoded_bytes().contains(&0))
        .ok_or(IdentityError::Io)?;
    let directory = fs::open(
        parent,
        OFlags::CLOEXEC | OFlags::RDONLY | OFlags::DIRECTORY,
        Mode::empty(),
    )
    .map_err(|_| IdentityError::Io)?;
    match fs::statat(&directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Err(Errno::NOENT) => {}
        Ok(_) | Err(_) => return Err(IdentityError::Io),
    }
    let token = RandomIds.uuid4().map_err(|_| IdentityError::Io)?;
    let temp_name = format!(".identity-{token}.tmp");
    let temp = fs::openat(
        &directory,
        temp_name.as_str(),
        OFlags::CLOEXEC | OFlags::CREATE | OFlags::EXCL | OFlags::WRONLY | OFlags::NOFOLLOW,
        FILE_MODE,
    )
    .map_err(|_| IdentityError::Io)?;
    let created = fs::fstat(&temp).map_err(|_| IdentityError::Io)?;
    let mut file = File::from(temp);
    let written = file
        .write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| IdentityError::Io);
    let linked = written.and_then(|()| {
        fs::linkat(
            &directory,
            temp_name.as_str(),
            &directory,
            name,
            AtFlags::empty(),
        )
        .map_err(|_| IdentityError::Io)
    });
    let published = linked.and_then(|()| fs::fsync(&directory).map_err(|_| IdentityError::Io));
    let cleanup = cleanup_temp(&directory, &temp_name, &created);
    published?;
    cleanup
}

fn cleanup_temp(
    directory: &impl std::os::fd::AsFd,
    name: &str,
    created: &fs::Stat,
) -> Result<(), IdentityError> {
    let info = match fs::statat(directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(info) => info,
        Err(Errno::NOENT) => return Ok(()),
        Err(_) => return Err(IdentityError::Io),
    };
    if info.st_dev != created.st_dev || info.st_ino != created.st_ino {
        return Err(IdentityError::Io);
    }
    fs::unlinkat(directory, name, AtFlags::empty()).map_err(|_| IdentityError::Io)?;
    fs::fsync(directory).map_err(|_| IdentityError::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Owned(std::path::PathBuf);

    impl Owned {
        fn new() -> Self {
            let path = std::env::temp_dir()
                .join(format!("identity-cleanup-{}", RandomIds.uuid4().unwrap()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn directory(&self, flags: OFlags) -> rustix::fd::OwnedFd {
            fs::open(&self.0, flags | OFlags::DIRECTORY, Mode::empty()).unwrap()
        }
    }

    impl Drop for Owned {
        fn drop(&mut self) {
            let result = std::fs::remove_dir_all(&self.0);
            if !std::thread::panicking() {
                result.expect("remove owned cleanup fixture");
            } else if let Err(error) = result {
                eprintln!("owned cleanup fixture removal failed: {error}");
            }
        }
    }

    #[test]
    fn replacement_is_preserved_and_not_reported_clean() {
        let owned = Owned::new();
        let path = owned.0.join("temp");
        std::fs::write(&path, b"original").unwrap();
        let original = File::open(&path).unwrap();
        let identity = fs::fstat(&original).unwrap();
        std::fs::rename(&path, owned.0.join("parked")).unwrap();
        std::fs::write(&path, b"replacement").unwrap();
        let directory = owned.directory(OFlags::RDONLY);
        assert!(cleanup_temp(&directory, "temp", &identity).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"replacement");
        assert_eq!(std::fs::read(owned.0.join("parked")).unwrap(), b"original");
        assert!(cleanup_temp(&original, "temp", &identity).is_err());
        std::fs::remove_file(&path).unwrap();
        assert!(cleanup_temp(&directory, "temp", &identity).is_ok());
    }

    #[test]
    fn cleanup_sync_failure_does_not_claim_retained_temp() {
        let owned = Owned::new();
        let path = owned.0.join("temp");
        std::fs::write(&path, b"owned").unwrap();
        let identity = fs::fstat(File::open(&path).unwrap()).unwrap();
        // O_PATH supports relative unlink/stat, but cannot fsync the directory.
        let directory = owned.directory(OFlags::PATH);
        assert!(cleanup_temp(&directory, "temp", &identity).is_err());
        assert!(
            !path.exists(),
            "unlink already committed before fsync failed"
        );
    }
}
