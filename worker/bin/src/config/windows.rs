//! Zone admission for pulled detection windows.
//!
//! Candidate order is the relay's single walk. This owner reads one explicit
//! zoneinfo root, compiles the captured TZif bytes, and never substitutes UTC.
//! A reportable drop is not a typed fault: missing, non-regular, bad magic,
//! a missing version byte, or a non-digit version fail open. IO after a
//! regular file, a valid header with a bad body, an oversized file, and an
//! invalid root stay typed. Startup strictness runs only after that walk.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{self, ErrorKind, Read};
use std::path::Path;

use rustix::fs::{Mode, OFlags};
use seeon_worker::detection_window::{
    DetectionWindow, DetectionWindowError, MAX_TZIF_BYTES, zoneinfo_path,
};

use crate::relay::cameras::{
    DetectionWindow as WindowDefinition, WindowCandidate, WorkerConfigPayload, window_candidates,
};

/// Pinned image zoneinfo tree. Not an environment knob.
pub const ZONEINFO_DIR: &str = "/usr/share/zoneinfo";

/// When the original two-character clock check runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionMode {
    /// After the canonical walk, every surviving bound must be two characters.
    Startup,
    /// Canonical drops only. Loose clocks that Python's pull keeps stay in.
    Poll,
}

/// A window that survived admission, with its original spelling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedWindow {
    pub definition: WindowDefinition,
    pub window: DetectionWindow,
}

/// A fault that refuses the whole admission. Display carries no path or document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowError {
    /// Startup found a surviving clock that is not exactly two characters per field.
    StrictClock,
    /// A regular file could not be opened or read.
    Io(ErrorKind),
    /// Magic and version were Python-valid; the body was not.
    CorruptTzif,
    /// Captured bytes exceed the primitive bound.
    TooLarge,
    /// A primitive refusal that is not a reportable drop.
    Invariant(DetectionWindowError),
}

/// Why one candidate failed open. The caller logs the domain and definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropReason {
    /// Not an object with exactly the three non-empty string fields.
    Shape,
    /// A bound is not `%H:%M`, or the parsed minutes are equal.
    Time,
    /// The zone key is not a bounded relative IANA path.
    ZoneName,
    /// Metadata failed, or the path is not a regular file.
    Missing,
    /// The four-byte magic, version byte, or version class is not TZif.
    Header,
}

/// One dropped input. `definition` is absent when the value never shaped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowDrop<'a> {
    pub domain: &'a str,
    pub definition: Option<&'a WindowDefinition>,
    pub reason: DropReason,
}

impl fmt::Display for WindowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StrictClock => formatter
                .write_str("startup window time must use exactly two characters per HH:MM field"),
            Self::Io(kind) => write!(formatter, "zone file io failed: {kind}"),
            Self::CorruptTzif => {
                formatter.write_str("zone file has a TZif header and a malformed body")
            }
            Self::TooLarge => formatter.write_str("zone file exceeds the byte bound"),
            Self::Invariant(error) => write!(formatter, "window invariant refused: {error}"),
        }
    }
}

impl fmt::Display for DropReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Shape => "window shape is not an object with start, end, and tz",
            Self::Time => "window time is malformed or names equal minutes",
            Self::ZoneName => "zone name is not a bounded relative IANA key",
            Self::Missing => "zone file is missing or not a regular file",
            Self::Header => "zone file magic or version is not TZif",
        })
    }
}

impl std::error::Error for WindowError {}

/// Admit every canonical candidate, then apply startup strictness.
///
/// The complete walk reports before any strict refusal. Bytes are read once
/// and compiled with `from_tzif_bytes`. A later mutation of the source file
/// cannot change membership.
pub fn admit_at(
    config: &WorkerConfigPayload,
    root: &Path,
    mode: AdmissionMode,
    report: &mut dyn FnMut(WindowDrop<'_>),
) -> Result<BTreeMap<String, AdmittedWindow>, WindowError> {
    let mut admitted = BTreeMap::new();
    for (domain, candidate) in window_candidates(config) {
        let WindowCandidate::Window(definition) = candidate else {
            if matches!(candidate, WindowCandidate::Invalid) {
                report(WindowDrop {
                    domain,
                    definition: None,
                    reason: DropReason::Shape,
                });
            }
            continue;
        };
        match admit_one(&definition, root)? {
            One::Keep(window) => {
                admitted.insert(domain.to_owned(), AdmittedWindow { definition, window });
            }
            One::Drop(reason) => report(WindowDrop {
                domain,
                definition: Some(&definition),
                reason,
            }),
        }
    }
    if mode == AdmissionMode::Startup
        && admitted.values().any(|admitted| {
            !exact_two(&admitted.definition.start) || !exact_two(&admitted.definition.end)
        })
    {
        return Err(WindowError::StrictClock);
    }
    Ok(admitted)
}

enum One {
    Keep(DetectionWindow),
    Drop(DropReason),
}

fn admit_one(definition: &WindowDefinition, root: &Path) -> Result<One, WindowError> {
    let Some((start, end)) = definition.parsed_minutes() else {
        return Ok(One::Drop(DropReason::Time));
    };
    let path = match zoneinfo_path(&definition.tz, root) {
        Ok(path) => path,
        Err(DetectionWindowError::InvalidZoneName) => return Ok(One::Drop(DropReason::ZoneName)),
        Err(error) => return Err(WindowError::Invariant(error)),
    };
    let bytes = match capture(&path)? {
        Captured::Missing => return Ok(One::Drop(DropReason::Missing)),
        Captured::Bytes(bytes) => bytes,
    };
    if bytes.len() > MAX_TZIF_BYTES {
        return Err(WindowError::TooLarge);
    }
    match header_class(&bytes) {
        Header::Drop => return Ok(One::Drop(DropReason::Header)),
        Header::Corrupt => return Err(WindowError::CorruptTzif),
        Header::Present => {}
    }
    let start_text = ascii_hhmm(start);
    let end_text = ascii_hhmm(end);
    match DetectionWindow::from_tzif_bytes(&start_text, &end_text, &definition.tz, root, &bytes) {
        Ok(window) => Ok(One::Keep(window)),
        Err(DetectionWindowError::InvalidZoneData) => Err(WindowError::CorruptTzif),
        Err(DetectionWindowError::ZoneDataTooLarge) => Err(WindowError::TooLarge),
        Err(error) => Err(WindowError::Invariant(error)),
    }
}

enum Captured {
    Missing,
    Bytes(Vec<u8>),
}

fn capture(path: &Path) -> Result<Captured, WindowError> {
    // Follow symlinks, as Python `os.path.isfile` does. Once that boundary
    // says regular, a failed open or read is IO, never a missing drop.
    if !path.is_file() {
        return Ok(Captured::Missing);
    }
    let flags = OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC;
    let descriptor = rustix::fs::open(path, flags, Mode::empty()).map_err(io::Error::from)?;
    let file = File::from(descriptor);
    if !file.metadata()?.is_file() {
        return Ok(Captured::Missing);
    }
    let mut bytes = Vec::new();
    file.take(MAX_TZIF_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    Ok(Captured::Bytes(bytes))
}

enum Header {
    Drop,
    Corrupt,
    Present,
}

/// Python `_TZifHeader.from_file` classes that `ZoneInfo` turns into a miss.
///
/// Bad or short magic, a missing version byte, and a version that is neither
/// NUL nor an ASCII digit are `ValueError` and fail open. A valid version
/// whose 24-byte count words are absent is `struct.error`: typed corruption,
/// not ALWAYS. This does not claim the rest of Jiff and CPython agree.
fn header_class(bytes: &[u8]) -> Header {
    match single_header(bytes) {
        Header::Present => {}
        failure => return failure,
    }
    // Python treats NUL and the ASCII digit '1' as the 32-bit format.
    // Other numeric versions seek over that block and parse another header.
    if matches!(bytes[4], 0 | b'1') {
        return Header::Present;
    }
    let counts = bytes[20..44].chunks_exact(4);
    let widths = [1_i64, 1, 8, 5, 6, 1];
    let mut offset = 44_i64;
    for (count, width) in counts.zip(widths) {
        let count = i32::from_be_bytes(count.try_into().expect("four-byte count"));
        // Negative counts are not admitted as a missing-header fallback.
        // Their Python seek/unpack behavior is outside these measured classes.
        if count < 0 {
            return Header::Corrupt;
        }
        offset += i64::from(count) * width;
    }
    let remaining = usize::try_from(offset)
        .ok()
        .and_then(|offset| bytes.get(offset..));
    single_header(remaining.unwrap_or_default())
}

fn single_header(bytes: &[u8]) -> Header {
    if bytes.len() < 4 || &bytes[..4] != b"TZif" {
        return Header::Drop;
    }
    let Some(version) = bytes.get(4).copied() else {
        return Header::Drop;
    };
    if version != 0 && !version.is_ascii_digit() {
        return Header::Drop;
    }
    if bytes.len() < 44 {
        return Header::Corrupt;
    }
    Header::Present
}

fn exact_two(text: &str) -> bool {
    let Some((hour, minute)) = text.split_once(':') else {
        return false;
    };
    hour.chars().count() == 2 && minute.chars().count() == 2
}

fn ascii_hhmm(minutes: u16) -> String {
    format!("{:02}:{:02}", minutes / 60, minutes % 60)
}

impl From<io::Error> for WindowError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.kind())
    }
}
