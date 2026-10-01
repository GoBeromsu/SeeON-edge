//! Playback renditions (`worker/runtime/clips/playback_rendition_publish.py`).
//!
//! A clip whose first video stream is not H.264 gets an H.264 copy beside it,
//! `clip.playback-h264.<sha256[:16]>.mp4`, and the attestation
//! `clip.playback-h264.json` the backend verifies before serving it. The copy
//! is verified before it is published: H.264, yuv420p, the same time base and
//! frame timestamps as the source, and `moov` before `mdat`. Every tool runs
//! under a deadline and is killed when it expires; refusals are typed.

mod attestation;
mod probe;
mod run;
pub mod thumbnail;

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use super::durable::{
    Existing, PUBLIC_FILE, fsync_dir, read_bounded, remove_durable, sha256_file, write_durable,
};
use super::manifest::MAX_MANIFEST_BYTES;
use super::publish::{MANIFEST_FILE, MEDIA_FILE};

pub use attestation::Attestation;
pub use probe::Timing;
pub use thumbnail::{THUMBNAIL_FILE, write_thumbnail};

pub const ATTESTATION_FILE: &str = "clip.playback-h264.json";
pub const RENDITION_PREFIX: &str = "clip.playback-h264.";
/// Sources already playable everywhere get no rendition.
pub const SKIP_CODECS: [&str; 2] = ["avc1", "h264"];
/// The temp keeps the `.mp4` suffix so ffmpeg selects the MP4 muxer.
pub const TEMP_FILE: &str = ".playback-rendition.tmp.mp4";

/// Tool binaries and deadlines. Product uses `Tools::default()`; tests take
/// the binaries from `SEEON_TEST_FFMPEG` and `SEEON_TEST_FFPROBE`. There is
/// no `PATH` lookup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub transcode_deadline: Duration,
    pub probe_deadline: Duration,
    pub thumbnail_deadline: Duration,
}

impl Default for Tools {
    fn default() -> Self {
        Self {
            ffmpeg: PathBuf::from("/usr/bin/ffmpeg"),
            ffprobe: PathBuf::from("/usr/bin/ffprobe"),
            transcode_deadline: Duration::from_secs(120),
            probe_deadline: Duration::from_secs(10),
            thumbnail_deadline: Duration::from_secs(30),
        }
    }
}

/// Why no rendition (or thumbnail) was published.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenditionError {
    /// A tool failed, exited non-zero or printed an unusable answer.
    Probe,
    /// The output, or the source digest, does not verify.
    Mismatch,
    /// A tool outlived its deadline and was killed.
    Deadline,
    Io(io::ErrorKind),
}

impl From<io::Error> for RenditionError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.kind())
    }
}

/// Publishes the playback rendition of `clip_dir/clip.mp4`. An H.264 source
/// returns `Ok(None)` and writes nothing. Any refusal leaves no temp and no
/// newly published rendition behind.
pub fn write_playback_rendition(
    tools: &Tools,
    clip_dir: &Path,
) -> Result<Option<Attestation>, RenditionError> {
    let source = clip_dir.join(MEDIA_FILE);
    let codec = probe::video_codec(tools, &source)?;
    if SKIP_CODECS.contains(&codec.as_str()) {
        return Ok(None);
    }
    let source_timing = probe::timing(tools, &source)?;
    let source_sha256 = verified_source_digest(clip_dir, &source)?;
    let temp = clip_dir.join(TEMP_FILE);
    remove_if_present(&temp)?;
    let result = transcode_and_publish(tools, clip_dir, &source_timing, source_sha256, &temp);
    if result.is_err() {
        let _ = remove_if_present(&temp);
    }
    result.map(Some)
}

/// The manifest's `sha256`, after checking it names the media on disk.
fn verified_source_digest(clip_dir: &Path, source: &Path) -> Result<String, RenditionError> {
    let limit = u64::try_from(MAX_MANIFEST_BYTES).unwrap_or(u64::MAX);
    let Existing::Bytes(bytes) = read_bounded(&clip_dir.join(MANIFEST_FILE), limit)? else {
        return Err(RenditionError::Mismatch);
    };
    let manifest: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|_| RenditionError::Mismatch)?;
    let recorded = manifest["sha256"]
        .as_str()
        .ok_or(RenditionError::Mismatch)?;
    let well_formed = recorded.len() == 64
        && recorded
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    let (actual, _) = sha256_file(source)?;
    if !well_formed || recorded != actual {
        return Err(RenditionError::Mismatch);
    }
    Ok(actual)
}

fn transcode_and_publish(
    tools: &Tools,
    clip_dir: &Path,
    source_timing: &Timing,
    source_sha256: String,
    temp: &Path,
) -> Result<Attestation, RenditionError> {
    transcode(
        tools,
        &clip_dir.join(MEDIA_FILE),
        temp,
        source_timing.tb_den,
    )?;
    match probe::stream_fields(tools, temp, "codec_name,pix_fmt")?.as_slice() {
        [codec, pix_fmt] if codec == "h264" && pix_fmt == "yuv420p" => {}
        _ => return Err(RenditionError::Mismatch),
    }
    probe::require_moov_first(temp)?;
    let timing = probe::timing(tools, temp)?;
    if timing != *source_timing {
        return Err(RenditionError::Mismatch);
    }
    rustix::fs::chmod(temp, PUBLIC_FILE).map_err(io::Error::from)?;
    File::open(temp)?.sync_all()?;
    let (rendition_sha256, _) = sha256_file(temp)?;
    let rendition = format!("{RENDITION_PREFIX}{}.mp4", &rendition_sha256[..16]);
    let target = clip_dir.join(&rendition);
    let fresh = fs::symlink_metadata(&target).is_err();
    if fresh {
        fs::rename(temp, &target)?;
        fsync_dir(clip_dir)?;
    } else {
        fs::remove_file(temp)?;
    }
    let attestation = Attestation {
        rendition,
        rendition_sha256,
        source_sha256,
        pts_identical: true,
        time_base: timing.time_base(),
        frames: count(&timing),
        source_frames: count(source_timing),
    };
    let written = attestation.to_bytes().and_then(|bytes| {
        write_durable(&clip_dir.join(ATTESTATION_FILE), &bytes, PUBLIC_FILE).map_err(Into::into)
    });
    if let Err(error) = written {
        if fresh {
            let _ = remove_durable(&target);
        }
        return Err(error);
    }
    remove_stale_renditions(clip_dir, &attestation.rendition)?;
    Ok(attestation)
}

/// The Python transcode arguments, under the transcode deadline.
fn transcode(tools: &Tools, source: &Path, temp: &Path, tb_den: u64) -> Result<(), RenditionError> {
    let mut command = Command::new(&tools.ffmpeg);
    command
        .args(["-nostdin", "-loglevel", "error", "-y", "-i"])
        .arg(source)
        .args([
            "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
        ])
        .args([
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
        ])
        .args(["-enc_time_base", "-1", "-video_track_timescale"])
        .arg(tb_den.to_string())
        .args(["-movflags", "+faststart"])
        .arg(temp);
    run::run(command, tools.transcode_deadline)?;
    match fs::metadata(temp) {
        Ok(metadata) if metadata.is_file() && metadata.len() > 0 => Ok(()),
        _ => Err(RenditionError::Probe),
    }
}

fn count(timing: &Timing) -> u64 {
    u64::try_from(timing.pts.len()).unwrap_or(u64::MAX)
}

/// Older `clip.playback-h264.<16 hex>.mp4` copies superseded by `keep`.
fn remove_stale_renditions(clip_dir: &Path, keep: &str) -> Result<(), RenditionError> {
    for entry in fs::read_dir(clip_dir)? {
        let name = entry?.file_name();
        let Some(name) = name.to_str() else { continue };
        let digest = name
            .strip_prefix(RENDITION_PREFIX)
            .and_then(|rest| rest.strip_suffix(".mp4"));
        let is_rendition = digest.is_some_and(|hex| {
            hex.len() == 16
                && hex
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        });
        if is_rendition && name != keep {
            remove_durable(&clip_dir.join(name))?;
        }
    }
    Ok(())
}

fn remove_if_present(path: &Path) -> Result<(), RenditionError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}
