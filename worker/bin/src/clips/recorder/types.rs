//! The values the recorder hands out: its state, the extension boundary,
//! what an admitted alert did, refusals, and the sealed recording.

use std::fmt;
use std::path::PathBuf;

use seeon_deepstream_native::{MediaResult, RecordTicket};

use super::PlaneRefusal;
use crate::clips::manifest::{Contributor, Extension};
use crate::clips::publish::PublishError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Idle,
    Recording,
    Stopping,
    Finalizing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    None,
    ExtensionBounded,
    ExtensionRaced,
}

impl Boundary {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ExtensionBounded => "extension_bounded",
            Self::ExtensionRaced => "extension_raced",
        }
    }
}

/// What one admitted alert did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Admit {
    Started(RecordTicket),
    Extended,
    /// The alert waits for the recording after the one being stopped.
    Queued,
    /// The media plane refused to start; the alert stays pending for `tick`.
    Refused(PlaneRefusal),
}

#[derive(Debug)]
pub enum RecorderError {
    BlankEventRef,
    PendingFull,
    WrongCamera { expected: u32, received: u32 },
    DuplicateSealed(u32),
    UnexpectedSession(u32),
    Save(PublishError),
}

impl fmt::Display for RecorderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "clip recorder refused: {self:?}")
    }
}
impl std::error::Error for RecorderError {}

/// A recording the media plane sealed, with the alerts it covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipSealed {
    pub ticket: RecordTicket,
    pub result: MediaResult,
    pub contains_video: bool,
    pub duration_ms: u64,
    pub boundary: Boundary,
    /// Sorted by `(detected_at, event_ref)`.
    pub contributors: Vec<Contributor>,
    pub path: PathBuf,
}

impl ClipSealed {
    pub fn extension(&self) -> Extension {
        Extension {
            boundary: self.boundary.as_str().to_owned(),
            contributors: self.contributors.clone(),
            duration_ms: i64::try_from(self.duration_ms).unwrap_or(i64::MAX),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub sequence: u64,
    pub extended: u64,
    pub raced: u64,
    pub refused: u64,
}
