//! Readings of a media file: ffprobe for the first video stream (codec,
//! pixel format, width, time base, decoded frame timestamps) and a walk of
//! the top-level MP4 atoms for `moov` placement.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process::Command;

use super::run::run;
use super::{RenditionError, Tools};

/// Presentation timing of the first video stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Timing {
    pub tb_num: u64,
    pub tb_den: u64,
    pub pts: Vec<i64>,
}

impl Timing {
    pub fn time_base(&self) -> String {
        format!("{}/{}", self.tb_num, self.tb_den)
    }
}

/// `stream=<entries>` of the first video stream, one value per entry.
pub fn stream_fields(
    tools: &Tools,
    path: &Path,
    entries: &str,
) -> Result<Vec<String>, RenditionError> {
    let entries = format!("stream={entries}");
    let rows = ffprobe(tools, path, &["-show_entries", &entries])?;
    match rows.as_slice() {
        [row] => Ok(row.split(',').map(str::to_ascii_lowercase).collect()),
        _ => Err(RenditionError::Probe),
    }
}

/// The lowercase codec name of the first video stream.
pub fn video_codec(tools: &Tools, path: &Path) -> Result<String, RenditionError> {
    match stream_fields(tools, path, "codec_name")?.as_slice() {
        [codec] if !codec.is_empty() => Ok(codec.clone()),
        _ => Err(RenditionError::Probe),
    }
}

/// The time base and every decoded frame's timestamp, in decode output
/// order. A missing time base or timestamp, or no frames, is `Probe`.
pub fn timing(tools: &Tools, path: &Path) -> Result<Timing, RenditionError> {
    let (tb_num, tb_den) = match stream_fields(tools, path, "time_base")?.as_slice() {
        [time_base] => parse_time_base(time_base)?,
        _ => return Err(RenditionError::Probe),
    };
    let rows = ffprobe(tools, path, &["-show_entries", "frame=pts"])?;
    let pts = rows
        .iter()
        .map(|row| {
            // Frames with side data print empty trailing fields (`0,`).
            let pts = row.split(',').next().unwrap_or_default();
            pts.parse::<i64>().map_err(|_| RenditionError::Probe)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if pts.is_empty() {
        return Err(RenditionError::Probe);
    }
    Ok(Timing {
        tb_num,
        tb_den,
        pts,
    })
}

fn parse_time_base(text: &str) -> Result<(u64, u64), RenditionError> {
    let (num, den) = text.split_once('/').ok_or(RenditionError::Probe)?;
    let num = num.parse::<u64>().map_err(|_| RenditionError::Probe)?;
    let den = den.parse::<u64>().map_err(|_| RenditionError::Probe)?;
    if num == 0 || den == 0 {
        return Err(RenditionError::Probe);
    }
    Ok((num, den))
}

/// Non-empty output rows of `ffprobe -v error -select_streams v:0 <args>
/// -of csv=p=0 <path>`, bounded by the probe deadline.
fn ffprobe(tools: &Tools, path: &Path, args: &[&str]) -> Result<Vec<String>, RenditionError> {
    let mut command = Command::new(&tools.ffprobe);
    command
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(args)
        .args(["-of", "csv=p=0"])
        .arg(path);
    let output = run(command, tools.probe_deadline)?;
    let text = String::from_utf8(output).map_err(|_| RenditionError::Probe)?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|row| !row.is_empty())
        .map(str::to_owned)
        .collect())
}

/// `moov` precedes `mdat` among the top-level atoms (`+faststart`). An
/// unreadable atom chain or a file without both atoms is `Mismatch`.
pub fn require_moov_first(path: &Path) -> Result<(), RenditionError> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    let mut offset = 0_u64;
    while length.saturating_sub(offset) >= 8 {
        file.seek(SeekFrom::Start(offset))?;
        let mut header = [0_u8; 8];
        file.read_exact(&mut header)?;
        let declared = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        let size = match declared {
            0 => length - offset,
            1 => {
                let mut large = [0_u8; 8];
                file.read_exact(&mut large)?;
                u64::from_be_bytes(large)
            }
            size => u64::from(size),
        };
        match &header[4..8] {
            b"moov" => return Ok(()),
            b"mdat" => return Err(RenditionError::Mismatch),
            _ if size < 8 => return Err(RenditionError::Mismatch),
            _ => offset = offset.checked_add(size).ok_or(RenditionError::Mismatch)?,
        }
    }
    Err(RenditionError::Mismatch)
}
