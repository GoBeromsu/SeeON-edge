use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use super::{MAX_MANIFEST_BYTES, MAX_MANIFESTS, Manifest, ManifestError};
use crate::clips::durable::{self, Existing, PRIVATE_FILE};

const DIRECTORY: &str = "runtime-provenance";
const BODY: &str = "manifest.json";

fn directory(path: &Path) -> Result<bool, ManifestError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(false),
        Ok(_) => Err(ManifestError::Conflict),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            durable::create_dir(path, 0o700)?;
            fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
            durable::fsync_dir(path)?;
            Ok(true)
        }
        Err(error) => Err(error.into()),
    }
}

fn matching(path: &Path, manifest: &Manifest) -> Result<bool, ManifestError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.permissions().mode() & 0o777 == 0o600 => {}
        Ok(_) => return Err(ManifestError::Conflict),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    }
    match durable::read_bounded(path, MAX_MANIFEST_BYTES as u64)? {
        Existing::Missing => Ok(false),
        Existing::Bytes(bytes) if bytes == manifest.canonical.as_bytes() => Ok(true),
        _ => Err(ManifestError::Conflict),
    }
}

pub(super) fn persist(manifest: &Manifest, state: &Path) -> Result<PathBuf, ManifestError> {
    let root = state.join(DIRECTORY);
    directory(&root)?;
    let content_dir = root.join(&manifest.sha256);
    if !content_dir.exists() {
        let mut count = 0usize;
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            let name = entry.file_name();
            if !entry.file_type()?.is_dir()
                || !name.to_str().is_some_and(|name| {
                    name.len() == 64
                        && name
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
            {
                continue;
            }
            count += 1;
            if count >= MAX_MANIFESTS {
                return Err(ManifestError::Capacity);
            }
        }
    }
    let created = directory(&content_dir)?;
    let result = publish(manifest, &content_dir);
    if result.is_err() && created {
        match fs::remove_dir(&content_dir) {
            Ok(()) => durable::fsync_dir(&root)?,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::DirectoryNotEmpty
                ) => {}
            Err(error) => return Err(error.into()),
        }
    }
    result
}

fn publish(manifest: &Manifest, content_dir: &Path) -> Result<PathBuf, ManifestError> {
    let target = content_dir.join(BODY);
    if matching(&target, manifest)? {
        durable::remove_durable(&content_dir.join(".manifest.tmp"))?;
        durable::remove_durable(&content_dir.join(".manifest.tmp.tmp"))?;
        durable::fsync_dir(content_dir)?;
        return Ok(target);
    }
    // The lease is the writer fence. A hard link publishes once without ever
    // replacing an existing body, even if another writer violates that fence.
    let temporary = content_dir.join(".manifest.tmp");
    durable::write_durable(&temporary, manifest.canonical.as_bytes(), PRIVATE_FILE)?;
    // Creation modes are filtered by umask; seal the private inode before publication.
    fs::set_permissions(&temporary, fs::Permissions::from_mode(PRIVATE_FILE.into()))?;
    fs::File::open(&temporary)?.sync_all()?;
    let published = match fs::hard_link(&temporary, &target) {
        Ok(()) => durable::fsync_dir(content_dir).map_err(ManifestError::from),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            matching(&target, manifest).and_then(|matches| {
                if matches {
                    Ok(())
                } else {
                    Err(ManifestError::Conflict)
                }
            })
        }
        Err(error) => Err(error.into()),
    };
    let cleanup = durable::remove_durable(&temporary).map_err(ManifestError::from);
    published?;
    cleanup?;
    if !matching(&target, manifest)? {
        return Err(ManifestError::Conflict);
    }
    Ok(target)
}
