//! Owned media values. Configuration and recording paths deliberately have no Debug output.
use std::{ffi::OsString, path::PathBuf};

pub const MEDIA_ABI_VERSION: u32 = 2;
pub const MEDIA_MAX_SOURCES: usize = 16;
pub const MEDIA_POSE_ROWS: usize = 300;
pub const MEDIA_POSE_COLUMNS: usize = 57;
pub const MEDIA_MAX_OBJECTS: usize = 150;
pub const MEDIA_MAX_RECORDS: u32 = 256;
pub const MEDIA_PATH_BYTES: usize = 4096;
pub const MEDIA_MAX_PREVIEW_BYTES: u32 = 16 * 1024 * 1024;

/// Token identifies the caller's complete immutable source binding, not an SDK pointer.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MediaBinding {
    pub token: u64,
    pub generation: u64,
    pub epoch: u64,
}

pub struct SourceConfig {
    /// Must equal this source's immutable position in the roster.
    pub source_id: u32,
    pub binding: MediaBinding,
    pub uri: String,
    /// Unique safe filename component for RTSP; unused for file fixtures.
    pub record_prefix: String,
}

/// No implicit numeric defaults. Source count and inference batch equal sources.len().
/// The caller owns config/artifact immutability and preprocessing admission; native
/// open validates the existing engine/parser. It never builds a replacement engine.
pub struct MediaConfig {
    pub sources: Vec<SourceConfig>,
    pub infer_config_path: PathBuf,
    pub tracker_config_path: PathBuf,
    pub tracker_library_path: PathBuf,
    pub record_directory: PathBuf,
    pub record_cache_seconds: u32,
    /// Bounded reusable slots, not a lifetime session limit requiring owner rotation.
    /// Consumed slots become reusable only after native completion and callback
    /// retirement make reuse safe. Consumption does not seal or publish a clip
    /// and makes no crash-durability guarantee.
    pub record_capacity: u32,
    pub mux_width: u32,
    pub mux_height: u32,
    pub mux_batch_timeout_us: u32,
    pub mux_live_source: bool,
    pub tracker_width: u32,
    pub tracker_height: u32,
    pub queue_max_buffers: u32,
    pub preview_enabled: bool,
    pub max_preview_bytes: u32,
    pub allow_file_uris: bool,
    /// Required explicit nvurisrcbin interval in seconds, applied to both
    /// init-rtsp-reconnect-interval and rtsp-reconnect-interval. 0 disables
    /// reconnect; values above 86400 are refused. There is no implicit 5.
    pub rtsp_reconnect_interval_sec: u32,
}

/// Integer-only ABI value: pts_valid is 0 or 1. sequence is zero for previews;
/// preview correlation uses the actual SDK frame number and PTS instead.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FrameIdentity {
    pub binding: MediaBinding,
    pub sequence: u64,
    pub pts_ns: u64,
    pub frame_number: i64,
    pub pts_valid: u32,
    pub source_id: u32,
    pub batch_id: u32,
    pub pad_index: u32,
    pub source_width: u32,
    pub source_height: u32,
    pub analysis_width: u32,
    pub analysis_height: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct TrackedObject {
    /// UINT64_MAX remains the SDK's untracked sentinel.
    pub track_id: u64,
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
}

pub struct PosePacket {
    pub frame: FrameIdentity,
    pub tensor_present: bool,
    pub rows: Vec<[f32; MEDIA_POSE_COLUMNS]>,
    pub objects: Vec<TrackedObject>,
}

/// These Rust enums never cross FFI; unknown native integers remain representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaResult {
    Ok,
    Empty,
    Busy,
    Stale,
    TooSmall,
    Unsupported,
    Fatal,
    Unknown(i32),
}
impl From<i32> for MediaResult {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Ok,
            1 => Self::Empty,
            2 => Self::Busy,
            3 => Self::Stale,
            4 => Self::TooSmall,
            5 => Self::Unsupported,
            6 => Self::Fatal,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaState {
    Open,
    Starting,
    Running,
    Stopping,
    Stopped,
    Unknown(i32),
}
impl From<i32> for MediaState {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Open,
            1 => Self::Starting,
            2 => Self::Running,
            3 => Self::Stopping,
            4 => Self::Stopped,
            other => Self::Unknown(other),
        }
    }
}

/// Numeric, sanitized diagnostics only; no native error/debug text is retained.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MediaDiagnostic {
    pub severity: u32,
    /// SeeonMediaError numeric code, including unrecognized future values.
    pub code: u32,
    pub sdk_domain: u32,
    pub sdk_code: i32,
}

#[must_use = "check the native result; admission is not completion and stop failure is fatal"]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaCallStatus {
    pub result: MediaResult,
    /// Independent snapshots on non-OK calls, not necessarily caused by this call.
    /// None means no diagnostic snapshot; Some with code zero means no native fault.
    pub fatal: Option<MediaDiagnostic>,
    pub warning: Option<MediaDiagnostic>,
    /// Preview TOO_SMALL does not consume the pending descriptor/JPEG.
    pub required_bytes: Option<u64>,
}

/// Empty/Busy/etc. carry no observation; only Ready is an actual copied packet.
/// For preview/record, Ready means delivered, not successful: inspect packet.result.
#[must_use = "check the poll status and any delivered packet result"]
pub enum MediaPoll<T> {
    Ready(T),
    Status(MediaCallStatus),
}

pub struct PreviewPacket {
    pub request_id: u64,
    pub batch_pts_ns: u64,
    pub frame: FrameIdentity,
    pub result: MediaResult,
    /// Exact SeeonMediaError integer.
    pub error: i32,
    pub jpeg: Vec<u8>,
}

/// Coalesced tickets retain the ORIGINAL request/session, never the overlapping ID.
/// session_valid and coalesced are ABI flags (0 or 1).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RecordTicket {
    pub binding: MediaBinding,
    pub request_id: u64,
    pub source_id: u32,
    pub session_id: u32,
    pub session_valid: u32,
    pub coalesced: u32,
}

/// Vendor completion only, NOT a sealed or publication-ready recording. The higher
/// owner must verify contained files and seals before publishing an event. Paths
/// preserve Unix bytes; this adapter neither reads nor verifies the named files.
pub struct RecordReceipt {
    pub ticket: RecordTicket,
    pub result: MediaResult,
    pub error: i32,
    pub duration_ms: u64,
    pub width: u32,
    pub height: u32,
    pub contains_video: bool,
    pub contains_audio: bool,
    pub directory: PathBuf,
    pub filename: OsString,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SourceStatus {
    pub binding: MediaBinding,
    pub frames: u64,
    pub overwritten: u64,
    pub dropped: u64,
    pub malformed: u64,
    pub tensor_absent: u64,
    pub objects: u64,
    pub video_linked: u32,
    /// request_id == 0 means no active recording; a queued ticket has no session yet.
    pub active_record: RecordTicket,
}

#[must_use = "check result and fatal even when status data is populated"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaStatus {
    /// FATAL can accompany useful counters and diagnostics. It is not discarded.
    pub result: MediaResult,
    pub state: MediaState,
    pub fatal: MediaDiagnostic,
    pub warning: MediaDiagnostic,
    pub warnings: u64,
    pub capacity_refusals: u64,
    pub preview_dropped: u64,
    pub late_record_callbacks: u64,
    pub callbacks_active: u32,
    pub stop_timed_out: bool,
    pub records_reserved: u32,
    pub sources: Vec<SourceStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaArgument {
    Sources,
    SourceId,
    Binding,
    Uri,
    RecordPrefix,
    InferConfigPath,
    TrackerConfigPath,
    TrackerLibraryPath,
    RecordDirectory,
    RecordCache,
    RecordCapacity,
    MuxDimensions,
    MuxTimeout,
    TrackerDimensions,
    QueueCapacity,
    PreviewCapacity,
    ShutdownBudget,
    PreviewTimeout,
    RequestId,
    RecordingWindow,
    RecordSession,
    ReconnectInterval,
}

/// Errors contain field identifiers and numeric native diagnostics, never input text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaError {
    InvalidArgument(MediaArgument),
    Closed,
    NotStopped,
    NativeContract,
    Native(MediaCallStatus),
}
impl std::fmt::Display for MediaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "media adapter: {self:?}")
    }
}
impl std::error::Error for MediaError {}
