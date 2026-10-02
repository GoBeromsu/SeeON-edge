//! Same-descriptor SDK library identity. Parent aliases resolve; the final
//! node is never followed. Map text is evidence, never a second open of another
//! inode.

use std::fs::File;
use std::io::Read;
use std::os::fd::AsFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use rustix::fs::{self, FileType, Mode, OFlags};
use rustix::process::Pid;
use sha2::{Digest, Sha256};

use crate::config::is_hex;
use crate::engine_build::live::LiveBuildError;

const JSON_CAP: u64 = 64 * 1024;
const LIBRARY_CAP: u64 = 512 * 1024 * 1024;
const MAPS_CAP: u64 = 8 * 1024 * 1024;
const READ_CHUNK: usize = 64 * 1024;

pub(in crate::engine_build::live) struct LibraryIdentity {
    pub(in crate::engine_build::live) sha256: String,
    path: PathBuf,
    dev: u64,
    ino: u64,
}

pub(in crate::engine_build::live) fn library_identity(
    result: &Path,
    library: &Path,
) -> Result<LibraryIdentity, LiveBuildError> {
    let expected = result_sha(result)?;
    let captured = capture(library)?;
    if captured.sha256 != expected {
        return Err(LiveBuildError::Observer);
    }
    Ok(LibraryIdentity {
        sha256: captured.sha256,
        path: captured.path,
        dev: captured.dev,
        ino: captured.ino,
    })
}

impl LibraryIdentity {
    pub(in crate::engine_build::live) fn mapped(&self, pid: Pid) -> Result<bool, LiveBuildError> {
        if pid.is_init() {
            return Err(LiveBuildError::Process);
        }
        let text = read_maps(pid)?;
        let matched = segments(&text, &self.path, self.dev, self.ino)?;
        if matched {
            let current = capture(&self.path)?;
            if (current.dev, current.ino) != (self.dev, self.ino) || current.sha256 != self.sha256 {
                return Err(LiveBuildError::Observer);
            }
        }
        Ok(matched)
    }
}

struct Captured {
    path: PathBuf,
    dev: u64,
    ino: u64,
    sha256: String,
}

fn capture(path: &Path) -> Result<Captured, LiveBuildError> {
    let resolved = resolve_parent(path)?;
    let file = open_regular(&resolved)?;
    let before = fstat_file(&file)?;
    let (dev, ino) = dev_ino(&before);
    let sha256 = hash_exact(&file, before.st_size, LIBRARY_CAP)?;
    let after = fstat_file(&file)?;
    if dev_ino(&after) != (dev, ino) || after.st_size != before.st_size {
        return Err(LiveBuildError::Observer);
    }
    Ok(Captured {
        path: resolved,
        dev,
        ino,
        sha256,
    })
}

fn resolve_parent(path: &Path) -> Result<PathBuf, LiveBuildError> {
    let parent = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or(Path::new("/"));
    let name = path.file_name().ok_or(LiveBuildError::Observer)?;
    if name.as_bytes().contains(&0) {
        return Err(LiveBuildError::Observer);
    }
    let resolved = parent
        .canonicalize()
        .map_err(|_| LiveBuildError::Observer)?
        .join(name);
    if resolved.as_os_str().as_bytes().contains(&0) {
        return Err(LiveBuildError::Observer);
    }
    Ok(resolved)
}

fn open_regular(path: &Path) -> Result<File, LiveBuildError> {
    let flags = OFlags::CLOEXEC | OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK;
    let file =
        File::from(fs::open(path, flags, Mode::empty()).map_err(|_| LiveBuildError::Observer)?);
    if FileType::from_raw_mode(fstat_file(&file)?.st_mode) != FileType::RegularFile {
        return Err(LiveBuildError::Observer);
    }
    Ok(file)
}

fn result_sha(path: &Path) -> Result<String, LiveBuildError> {
    let file = open_regular(path)?;
    let before = fstat_file(&file)?;
    let bytes = read_exact(&file, before.st_size, JSON_CAP)?;
    let after = fstat_file(&file)?;
    if after.st_size != before.st_size || dev_ino(&after) != dev_ino(&before) {
        return Err(LiveBuildError::Observer);
    }
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|_| LiveBuildError::Observer)?;
    value
        .get("output_sha256")
        .and_then(|item| item.as_str())
        .filter(|text| is_hex(text, 64))
        .map(str::to_owned)
        .ok_or(LiveBuildError::Observer)
}

fn hash_exact(file: &File, declared: i64, cap: u64) -> Result<String, LiveBuildError> {
    let mut digest = Sha256::new();
    stream_exact(file, declared, cap, |chunk| digest.update(chunk))?;
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn read_exact(file: &File, declared: i64, cap: u64) -> Result<Vec<u8>, LiveBuildError> {
    let mut bytes = Vec::new();
    stream_exact(file, declared, cap, |chunk| bytes.extend_from_slice(chunk))?;
    Ok(bytes)
}

fn stream_exact(
    file: &File,
    declared: i64,
    cap: u64,
    mut sink: impl FnMut(&[u8]),
) -> Result<(), LiveBuildError> {
    let size = u64::try_from(declared).map_err(|_| LiveBuildError::Observer)?;
    if size == 0 || size > cap {
        return Err(LiveBuildError::Observer);
    }
    let mut seen = 0_u64;
    let mut buffer = [0_u8; READ_CHUNK];
    let mut reader = file.take(size + 1);
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|_| LiveBuildError::Observer)?;
        if count == 0 {
            break;
        }
        seen = seen
            .checked_add(u64::try_from(count).map_err(|_| LiveBuildError::Observer)?)
            .ok_or(LiveBuildError::Observer)?;
        if seen > size {
            return Err(LiveBuildError::Observer);
        }
        sink(&buffer[..count]);
    }
    (seen == size).then_some(()).ok_or(LiveBuildError::Observer)
}

fn read_maps(pid: Pid) -> Result<String, LiveBuildError> {
    let path = format!("/proc/{pid}/maps");
    let flags = OFlags::CLOEXEC | OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK;
    let file = match fs::open(path, flags, Mode::empty()) {
        Err(rustix::io::Errno::NOENT) => return Ok(String::new()),
        other => File::from(other.map_err(|_| LiveBuildError::Observer)?),
    };
    let before = fstat_file(&file)?;
    if FileType::from_raw_mode(before.st_mode) != FileType::RegularFile {
        return Err(LiveBuildError::Observer);
    }
    // procfs advertises length zero even when the maps stream is nonempty.
    let mut bytes = Vec::new();
    file.take(MAPS_CAP + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| LiveBuildError::Observer)?;
    if bytes.len() as u64 > MAPS_CAP {
        return Err(LiveBuildError::Observer);
    }
    String::from_utf8(bytes).map_err(|_| LiveBuildError::Observer)
}

fn map_path(line: &str) -> Option<&str> {
    let mut remaining = line;
    for _ in 0..5 {
        remaining = remaining.trim_start();
        let boundary = remaining.find(char::is_whitespace)?;
        remaining = &remaining[boundary..];
    }
    let path = remaining.trim_start();
    path.starts_with('/').then_some(path)
}

fn map_identity(line: &str) -> Result<(u64, u64), LiveBuildError> {
    let mut parts = line.split_whitespace();
    let _address = parts.next().ok_or(LiveBuildError::Observer)?;
    let _perms = parts.next().ok_or(LiveBuildError::Observer)?;
    let _offset = parts.next().ok_or(LiveBuildError::Observer)?;
    let device = parts.next().ok_or(LiveBuildError::Observer)?;
    let inode = parts.next().ok_or(LiveBuildError::Observer)?;
    let (major, minor) = device.split_once(':').ok_or(LiveBuildError::Observer)?;
    let major = u32::from_str_radix(major, 16).map_err(|_| LiveBuildError::Observer)?;
    let minor = u32::from_str_radix(minor, 16).map_err(|_| LiveBuildError::Observer)?;
    let inode = inode.parse().map_err(|_| LiveBuildError::Observer)?;
    Ok((fs::makedev(major, minor), inode))
}

fn fstat_file(file: &File) -> Result<fs::Stat, LiveBuildError> {
    fs::fstat(file.as_fd()).map_err(|_| LiveBuildError::Observer)
}

fn dev_ino(stat: &fs::Stat) -> (u64, u64) {
    (stat.st_dev, stat.st_ino)
}

fn segments(text: &str, expected: &Path, dev: u64, ino: u64) -> Result<bool, LiveBuildError> {
    let mut matched = false;
    for line in text.lines() {
        let Some(path) = map_path(line) else { continue };
        let candidate = path.strip_suffix(" (deleted)").unwrap_or(path);
        if candidate.rsplit('/').next() != Some("libnvds_infer.so") {
            continue;
        }
        if candidate != path {
            return Err(LiveBuildError::Observer);
        }
        if resolve_parent(Path::new(path))? != expected || map_identity(line)? != (dev, ino) {
            return Err(LiveBuildError::Observer);
        }
        matched = true;
    }
    Ok(matched)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::Path;

    use rustix::fs::{FileType, Mode, major, minor};
    use sha2::{Digest, Sha256};

    use super::{capture, library_identity, segments};
    use crate::seam::IdSource;

    struct Owned(std::path::PathBuf);

    impl Owned {
        fn new() -> Self {
            let name = crate::seam::RandomIds.uuid4().expect("fixture id");
            let root = std::env::temp_dir().join(format!("live-library-{name}"));
            fs::create_dir(&root).expect("exclusive fixture");
            Self(root)
        }

        fn line(&self, path: &Path, dev: u64, ino: u64, offset: &str) -> String {
            format!(
                "7f{offset}-7f{offset}fff r-xp 00000000 {:x}:{:02x} {ino} {}",
                major(dev),
                minor(dev),
                path.display()
            )
        }
    }

    #[test]
    fn proc_maps_stream_is_read_despite_zero_advertised_length() {
        let text = super::read_maps(rustix::process::getpid()).unwrap();
        assert!(!text.is_empty());
        assert!(
            text.lines()
                .all(|line| line.split_whitespace().count() >= 5)
        );
    }

    impl Drop for Owned {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn digest(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    #[test]
    fn parent_alias_streams_and_refuses_followed_nodes() {
        let owned = Owned::new();
        let versioned = owned.0.join("deepstream-9.1");
        fs::create_dir(&versioned).unwrap();
        let bytes = b"sdk-library-bytes";
        let library = versioned.join("libnvds_infer.so");
        fs::write(&library, bytes).unwrap();
        symlink("deepstream-9.1", owned.0.join("deepstream")).unwrap();
        let alias = owned.0.join("deepstream/libnvds_infer.so");
        let captured = capture(&alias).unwrap();
        assert_eq!(captured.path, library);
        assert_eq!(captured.sha256, digest(bytes));
        symlink(&library, versioned.join("linked.so")).unwrap();
        assert!(capture(&versioned.join("linked.so")).is_err());
        let fifo = versioned.join("pipe.so");
        rustix::fs::mknodat(
            rustix::fs::CWD,
            &fifo,
            FileType::Fifo,
            Mode::from_raw_mode(0o600),
            0,
        )
        .unwrap();
        assert!(capture(&fifo).is_err());
        let json = owned.0.join("result.json");
        fs::write(&json, format!(r#"{{"output_sha256":"{}"}}"#, digest(bytes))).unwrap();
        assert!(library_identity(&json, &alias).is_ok());
        fs::write(&json, vec![b' '; 64 * 1024 + 1]).unwrap();
        assert!(library_identity(&json, &alias).is_err());
    }

    #[test]
    fn same_dso_segments_pass_and_foreign_maps_fail() {
        let owned = Owned::new();
        let versioned = owned.0.join("deepstream-9.1");
        fs::create_dir(&versioned).unwrap();
        let library = versioned.join("libnvds_infer.so");
        fs::write(&library, b"mapped-sdk").unwrap();
        symlink("deepstream-9.1", owned.0.join("deepstream")).unwrap();
        let alias = owned.0.join("deepstream/libnvds_infer.so");
        let captured = capture(&alias).unwrap();
        let first = owned.line(&alias, captured.dev, captured.ino, "000");
        let second = owned.line(&library, captured.dev, captured.ino, "100");
        let text = format!("{first}\n{second}\n");
        assert!(segments(&text, &captured.path, captured.dev, captured.ino).unwrap());
        let wrong = owned.line(&alias, captured.dev, captured.ino + 1, "200");
        assert!(segments(&wrong, &captured.path, captured.dev, captured.ino).is_err());
        let other_dir = owned.0.join("other sdk");
        fs::create_dir(&other_dir).unwrap();
        let other = other_dir.join("libnvds_infer.so");
        fs::write(&other, b"other-sdk").unwrap();
        let extra = owned.line(&other, captured.dev, captured.ino + 2, "300");
        assert!(
            segments(
                &format!("{text}{extra}"),
                &captured.path,
                captured.dev,
                captured.ino
            )
            .is_err()
        );
        assert!(
            segments(
                &format!("{first} (deleted)"),
                &captured.path,
                captured.dev,
                captured.ino
            )
            .is_err()
        );
        assert!(!segments("", &captured.path, captured.dev, captured.ino).unwrap());
        let spaced = Owned::new();
        let parent = spaced.0.join("deep stream");
        fs::create_dir(&parent).unwrap();
        let library = parent.join("libnvds_infer.so");
        fs::write(&library, b"spaced-sdk").unwrap();
        let captured = capture(&library).unwrap();
        assert!(captured.path.ends_with("deep stream/libnvds_infer.so"));
        let first = spaced.line(&captured.path, captured.dev, captured.ino, "400");
        let second = spaced.line(&captured.path, captured.dev, captured.ino, "500");
        assert!(
            segments(
                &format!("{first}\n{second}\n"),
                &captured.path,
                captured.dev,
                captured.ino,
            )
            .unwrap()
        );
    }
}
