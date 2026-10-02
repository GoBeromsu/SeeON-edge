//! Same-filesystem replacement; identity is last, not a multi-file transaction.
use super::{CommandError, Layout, PRIVATE_DIR, paths};
use crate::seam::{IdSource, RandomIds};
use rustix::fs::{self, Mode, OFlags};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub(in crate::engine_build::command) fn write_exclusive(
    path: &Path,
    bytes: &[u8],
) -> Result<(), CommandError> {
    let flags = OFlags::CLOEXEC | OFlags::CREATE | OFlags::EXCL | OFlags::WRONLY | OFlags::NOFOLLOW;
    let mode = Mode::from_bits_truncate(0o644);
    let mut file = File::from(fs::open(path, flags, mode).map_err(|_| CommandError::Io)?);
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| CommandError::Io)?;
    paths::sync_parent(path)
}

pub(in crate::engine_build::command) fn identity_staging(
    final_path: &Path,
) -> Result<PathBuf, CommandError> {
    let parent = paths::parent_of(final_path)?;
    let token = RandomIds.uuid4().map_err(|_| CommandError::Io)?;
    let directory = parent.join(format!(".identity-stage-{token}"));
    fs::mkdir(&directory, PRIVATE_DIR).map_err(|_| CommandError::Io)?;
    Ok(directory.join(paths::basename(final_path)?))
}

pub(in crate::engine_build::command) fn commit(
    layout: &Layout,
    staged_identity: &Path,
    write_served: bool,
) -> Result<(), CommandError> {
    for (source, target) in [
        (&layout.live, &layout.final_live),
        (&layout.stored, &layout.final_stored),
        (&layout.bed, &layout.final_bed),
        (&layout.fall, &layout.final_fall),
    ] {
        rename_same(source, target)?;
        paths::sync_parent(target)?;
    }
    if write_served {
        rename_same(&layout.served, &layout.final_served)?;
        paths::sync_parent(&layout.final_served)?;
    }
    rename_same(staged_identity, &layout.identity)?;
    paths::sync_parent(&layout.identity)
}

fn rename_same(source: &Path, target: &Path) -> Result<(), CommandError> {
    paths::require_replaceable(target)?;
    let parent = paths::parent_of(target)?;
    let directory = fs::open(
        &parent,
        OFlags::CLOEXEC | OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map_err(|_| CommandError::Io)?;
    fs::renameat(
        fs::CWD,
        source,
        &directory,
        target.file_name().ok_or(CommandError::Path)?,
    )
    .map_err(|_| CommandError::Io)
}
