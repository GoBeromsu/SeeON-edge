//! Canonical file identities for collision checks; bounded descriptor reads.
use super::CommandError;
use rustix::fs::{self, AtFlags, FileType, Mode, OFlags};
use rustix::io::Errno;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

pub(super) fn reject_collisions(
    targets: &[&Path],
    inputs: &[&Path],
    served_index: usize,
) -> Result<(), CommandError> {
    let outputs = targets
        .iter()
        .map(|path| target_identity(path))
        .collect::<Result<Vec<_>, _>>()?;
    for (index, left) in outputs.iter().enumerate() {
        if outputs[index + 1..]
            .iter()
            .any(|right| same_file(left, right))
        {
            return Err(CommandError::Collision);
        }
    }
    for (input_index, input) in inputs.iter().enumerate() {
        let resolved = existing_identity(input)?;
        for (output_index, output) in outputs.iter().enumerate() {
            // The infer template is the served file when output is omitted,
            // and is the sole input the explicit served-output flag may replace.
            if input_index == 4 && output_index == served_index {
                continue;
            }
            if same_file(&resolved, output) {
                return Err(CommandError::Collision);
            }
        }
    }
    Ok(())
}

pub(super) fn require_replaceable(path: &Path) -> Result<(), CommandError> {
    let parent = parent_of(path)?;
    let directory = fs::open(
        &parent,
        OFlags::CLOEXEC | OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map_err(|_| CommandError::Io)?;
    match fs::statat(&directory, basename(path)?, AtFlags::SYMLINK_NOFOLLOW) {
        Err(Errno::NOENT) => Ok(()),
        Ok(info) if FileType::from_raw_mode(info.st_mode).is_file() => Ok(()),
        Ok(_) => Err(CommandError::Path),
        Err(_) => Err(CommandError::Io),
    }
}

pub(super) fn parent_of(path: &Path) -> Result<PathBuf, CommandError> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    match std::fs::metadata(parent) {
        Ok(info) if info.is_dir() => {}
        Ok(_) => return Err(CommandError::Path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(parent).map_err(|_| CommandError::Io)?;
        }
        Err(_) => return Err(CommandError::Io),
    }
    parent.canonicalize().map_err(|_| CommandError::Path)
}
pub(super) fn basename(path: &Path) -> Result<&std::ffi::OsStr, CommandError> {
    path.file_name()
        .filter(|name| !name.is_empty())
        .ok_or(CommandError::Path)
}

pub(in crate::engine_build::command) fn absolute_text(path: &Path) -> Result<String, CommandError> {
    let absolute = parent_of(path)?.join(basename(path)?);
    absolute
        .to_str()
        .filter(|text| !text.contains(['\0', '\n', '\r']))
        .map(str::to_owned)
        .ok_or(CommandError::Path)
}
pub(in crate::engine_build::command) fn read_text(path: &Path) -> Result<String, CommandError> {
    String::from_utf8(read_bounded(path, 1024 * 1024)?).map_err(|_| CommandError::Config)
}
pub(super) fn sync_parent(path: &Path) -> Result<(), CommandError> {
    let parent = parent_of(path)?;
    let directory = fs::open(
        &parent,
        OFlags::CLOEXEC | OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map_err(|_| CommandError::Io)?;
    fs::fsync(directory).map_err(|_| CommandError::Io)
}

fn target_identity(path: &Path) -> Result<(PathBuf, u64, u64), CommandError> {
    let resolved = parent_of(path)?.join(basename(path)?);
    match std::fs::metadata(&resolved) {
        Ok(info) => Ok((resolved, info.dev(), info.ino())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok((resolved, 0, 0)),
        Err(_) => Err(CommandError::Io),
    }
}
fn existing_identity(path: &Path) -> Result<(PathBuf, u64, u64), CommandError> {
    let resolved = path.canonicalize().map_err(|_| CommandError::Path)?;
    let info = std::fs::metadata(&resolved).map_err(|_| CommandError::Io)?;
    Ok((resolved, info.dev(), info.ino()))
}
fn same_file(left: &(PathBuf, u64, u64), right: &(PathBuf, u64, u64)) -> bool {
    left.0 == right.0 || left.2 != 0 && (left.1, left.2) == (right.1, right.2)
}

pub(super) fn read_bounded(path: &Path, max: u64) -> Result<Vec<u8>, CommandError> {
    let file = File::from(
        fs::open(
            path,
            OFlags::CLOEXEC | OFlags::RDONLY | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(|_| CommandError::Io)?,
    );
    let info = file.metadata().map_err(|_| CommandError::Io)?;
    let declared = info.len();
    if !info.is_file() || declared == 0 || declared > max {
        return Err(CommandError::Io);
    }
    let mut bytes = Vec::with_capacity(declared as usize);
    file.take(declared + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| CommandError::Io)?;
    (bytes.len() as u64 == declared)
        .then_some(bytes)
        .ok_or(CommandError::Io)
}
