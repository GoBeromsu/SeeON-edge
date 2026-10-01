//! `thumbnail.jpg`: one JPEG frame, 640 pixels wide, from the clip media
//! (`worker/adapters/media/ffmpeg_thumbnail.py` arguments), published with
//! dot-temp, fsync, rename and parent fsync. A refusal never blocks the clip;
//! the caller publishes the clip either way.

use std::fs::{self, File};
use std::path::Path;
use std::process::Command;

use crate::clips::durable::{PUBLIC_FILE, fsync_dir};
use crate::clips::publish::MEDIA_FILE;

use super::probe::stream_fields;
use super::run::run;
use super::{RenditionError, Tools, remove_if_present};

pub const THUMBNAIL_FILE: &str = "thumbnail.jpg";
pub const THUMBNAIL_WIDTH: &str = "640";
/// `thumbnail_files.MAX_THUMBNAIL_BYTES`.
pub const MAX_THUMBNAIL_BYTES: u64 = 2 * 1024 * 1024;
/// The temp keeps the `.jpg` suffix so ffmpeg selects the JPEG muxer.
const TEMP_FILE: &str = ".thumbnail.tmp.jpg";

/// The seek offset the Python generator uses: half a second before the
/// end, within `[0, 15]` seconds, printed as Python prints a float.
pub fn seek_offset(duration_ms: i64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let seconds = (duration_ms as f64 / 1000.0 - 0.5).clamp(0.0, 15.0);
    format!("{seconds:?}")
}

/// Writes `thumbnail.jpg` beside the clip media. The JPEG is verified to be
/// one MJPEG picture 640 pixels wide and at most `MAX_THUMBNAIL_BYTES`;
/// any refusal leaves no temp behind.
pub fn write_thumbnail(
    tools: &Tools,
    clip_dir: &Path,
    duration_ms: i64,
) -> Result<(), RenditionError> {
    let temp = clip_dir.join(TEMP_FILE);
    remove_if_present(&temp)?;
    let result = generate(tools, clip_dir, duration_ms, &temp);
    if result.is_err() {
        let _ = remove_if_present(&temp);
    }
    result
}

fn generate(
    tools: &Tools,
    clip_dir: &Path,
    duration_ms: i64,
    temp: &Path,
) -> Result<(), RenditionError> {
    let mut command = Command::new(&tools.ffmpeg);
    command
        .args(["-nostdin", "-loglevel", "error", "-y", "-ss"])
        .arg(seek_offset(duration_ms))
        .arg("-i")
        .arg(clip_dir.join(MEDIA_FILE))
        .args(["-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "3"])
        .arg(temp);
    run(command, tools.thumbnail_deadline)?;
    let size = match fs::symlink_metadata(temp) {
        Ok(metadata) if metadata.is_file() => metadata.len(),
        _ => return Err(RenditionError::Probe),
    };
    if size == 0 || size > MAX_THUMBNAIL_BYTES {
        return Err(RenditionError::Mismatch);
    }
    match stream_fields(tools, temp, "codec_name,width")?.as_slice() {
        [codec, width] if codec == "mjpeg" && width == THUMBNAIL_WIDTH => {}
        _ => return Err(RenditionError::Mismatch),
    }
    rustix::fs::chmod(temp, PUBLIC_FILE).map_err(std::io::Error::from)?;
    File::open(temp)?.sync_all()?;
    fs::rename(temp, clip_dir.join(THUMBNAIL_FILE))?;
    fsync_dir(clip_dir)?;
    Ok(())
}
