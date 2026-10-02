//! Filesystem helpers of `model_bundle.py`: `_require_directory`,
//! `_read_regular`, `_require_below` and `_verify_exact_tree`.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use rustix::fs::{Mode, OFlags};
use sha2::{Digest, Sha256};

use super::{Admission, AdmissionKind, refuse};

/// Symlink hops before a resolve is treated as a loop (Linux `MAXSYMLINKS`).
const MAX_HOPS: usize = 40;

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// `_require_directory`: lstat, so a symlink to a directory is refused.
pub(super) fn require_directory(path: &Path, label: &str) -> Admission<()> {
    match fs::symlink_metadata(path) {
        Err(_) => refuse(AdmissionKind::Unavailable, label),
        Ok(info) if !info.is_dir() => refuse(AdmissionKind::NotRegularDirectory, label),
        Ok(_) => Ok(()),
    }
}

/// `_read_regular`: `O_NOFOLLOW` open, then fstat; a final symlink fails the
/// open (ELOOP) and so reports "unavailable", as in Python.
pub(super) fn read_regular(path: &Path, label: &str) -> Admission<Vec<u8>> {
    let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    read_file(path, label, flags)
}

/// Packaged bundles retain Python's final-component symlink-following policy.
pub(super) fn read_following(path: &Path, label: &str) -> Admission<Vec<u8>> {
    read_file(path, label, OFlags::RDONLY | OFlags::CLOEXEC)
}

fn read_file(path: &Path, label: &str, flags: OFlags) -> Admission<Vec<u8>> {
    // Both modes require regular files; never wait for a FIFO writer before fstat.
    let Ok(descriptor) = rustix::fs::open(path, flags | OFlags::NONBLOCK, Mode::empty()) else {
        return refuse(AdmissionKind::Unavailable, label);
    };
    let mut file = File::from(descriptor);
    match file.metadata() {
        Err(_) => return refuse(AdmissionKind::Unavailable, label),
        Ok(info) if !info.is_file() => return refuse(AdmissionKind::NotRegularFile, label),
        Ok(_) => {}
    }
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return refuse(AdmissionKind::Unavailable, label);
    }
    Ok(bytes)
}

/// `Path.resolve(strict=False)`: follow every existing symlink, keep the
/// missing tail as written, and apply `..` to the resolved prefix.
pub(super) fn resolve_lenient(path: &Path) -> Admission<PathBuf> {
    let unavailable = || refuse(AdmissionKind::PathUnavailable, "");
    let Ok(absolute) = std::path::absolute(path) else {
        return unavailable();
    };
    let mut pending: Vec<PathBuf> = absolute
        .components()
        .rev()
        .map(|part| PathBuf::from(part.as_os_str()))
        .collect();
    let mut resolved = PathBuf::from("/");
    let mut hops = 0;
    while let Some(part) = pending.pop() {
        match part.components().next() {
            Some(Component::RootDir) => resolved = PathBuf::from("/"),
            Some(Component::ParentDir) => {
                resolved.pop();
            }
            Some(Component::Normal(name)) => {
                let candidate = resolved.join(name);
                match fs::read_link(&candidate) {
                    Ok(target) => {
                        hops += 1;
                        if hops > MAX_HOPS {
                            return unavailable();
                        }
                        let parts = target.components().rev();
                        pending.extend(parts.map(|part| PathBuf::from(part.as_os_str())));
                    }
                    Err(_) => resolved = candidate,
                }
            }
            Some(Component::CurDir | Component::Prefix(_)) | None => {}
        }
    }
    Ok(resolved)
}

/// `_require_below(root, path)`: the resolved path stays under the resolved
/// root, and no existing component below `root` is a symlink.
pub(super) fn require_below(root: &Path, path: &Path) -> Admission<()> {
    let Ok(resolved_root) = fs::canonicalize(root) else {
        return refuse(AdmissionKind::PathUnavailable, "");
    };
    if !resolve_lenient(path)?.starts_with(&resolved_root) {
        return refuse(AdmissionKind::PathEscapes, "");
    }
    let Ok(relative) = path.strip_prefix(root) else {
        return refuse(AdmissionKind::PathUnavailable, "");
    };
    let mut current = root.to_path_buf();
    for part in relative.components() {
        current.push(part.as_os_str());
        if fs::metadata(&current).is_err() {
            continue;
        }
        match fs::symlink_metadata(&current) {
            Err(_) => return refuse(AdmissionKind::PathUnavailable, ""),
            Ok(info) if info.is_symlink() => return refuse(AdmissionKind::SymlinkPath, ""),
            Ok(_) => {}
        }
    }
    Ok(())
}

#[derive(Default)]
struct Found {
    files: BTreeSet<String>,
    directories: BTreeSet<String>,
}

fn walk(root: &Path, directory: &Path, found: &mut Found) -> Admission<()> {
    let unavailable = || refuse(AdmissionKind::PathUnavailable, "");
    let Ok(entries) = fs::read_dir(directory) else {
        return unavailable();
    };
    for entry in entries {
        let Ok(entry) = entry else {
            return unavailable();
        };
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let relative = relative.to_string_lossy().into_owned();
        let Ok(info) = fs::symlink_metadata(&path) else {
            return unavailable();
        };
        if info.is_file() {
            found.files.insert(relative);
        } else if info.is_dir() {
            found.directories.insert(relative);
            walk(root, &path, found)?;
        } else {
            return refuse(AdmissionKind::UnsafePath, &relative);
        }
    }
    Ok(())
}

/// `_verify_exact_tree`: the files below `root` are exactly `expected` and
/// the directories exactly their proper parents; any symlink, device, FIFO or
/// socket is an unsafe path.
pub(super) fn verify_exact_tree(root: &Path, expected: &BTreeSet<String>) -> Admission<()> {
    let expected_directories: BTreeSet<String> = expected
        .iter()
        .flat_map(|path| path.match_indices('/').map(|(at, _)| path[..at].to_owned()))
        .collect();
    let mut found = Found::default();
    walk(root, root, &mut found)?;
    if found.files != *expected || found.directories != expected_directories {
        return refuse(AdmissionKind::TreeMismatch, "");
    }
    Ok(())
}
