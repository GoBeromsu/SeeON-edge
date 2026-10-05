use super::{SealedClip, SealedContributor};

/// A native recording that completed without a publishable ready clip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedUnavailable {
    pub clip_id: String,
    /// The locator actually reported by the media plane, when present.
    pub path: Option<String>,
    pub duration_ms: u64,
    pub boundary: String,
    pub contributors: Vec<SealedContributor>,
    pub native_result: i32,
    pub contains_video: bool,
}

/// The sealed observation persisted before any later media processing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SealedObservation {
    Ready(SealedClip),
    Unavailable(SealedUnavailable),
}

impl SealedObservation {
    /// The observation's safe clip identifier.
    pub fn clip_id(&self) -> &str {
        match self {
            Self::Ready(sealed) => &sealed.clip_id,
            Self::Unavailable(sealed) => &sealed.clip_id,
        }
    }

    /// The reported source locator, absent when no unavailable locator was reported.
    pub fn path(&self) -> Option<&str> {
        match self {
            Self::Ready(sealed) => Some(&sealed.path),
            Self::Unavailable(sealed) => sealed.path.as_deref(),
        }
    }

    /// The exact duration in milliseconds without narrowing unsigned observations.
    pub fn duration_ms(&self) -> i128 {
        match self {
            Self::Ready(sealed) => i128::from(sealed.duration_ms),
            Self::Unavailable(sealed) => i128::from(sealed.duration_ms),
        }
    }

    /// The recording boundary as reported by the recorder.
    pub fn boundary(&self) -> &str {
        match self {
            Self::Ready(sealed) => &sealed.boundary,
            Self::Unavailable(sealed) => &sealed.boundary,
        }
    }

    /// The contributors attributed to this recording.
    pub fn contributors(&self) -> &[SealedContributor] {
        match self {
            Self::Ready(sealed) => &sealed.contributors,
            Self::Unavailable(sealed) => &sealed.contributors,
        }
    }
}

pub(super) fn valid_clip_id(clip_id: &str) -> bool {
    !clip_id.is_empty() && !clip_id.starts_with('.') && !clip_id.contains(['/', '\0'])
}
