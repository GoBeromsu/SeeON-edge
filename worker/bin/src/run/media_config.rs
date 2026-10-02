//! Pure media configuration for one fixed admitted roster.
//!
//! Called after boot gates. It does not read the environment, files, the
//! network, or a GPU, and it does not open the media plane. An empty admitted
//! roster is [`MediaAssembly::Idle`]. A nonempty roster becomes one
//! [`MediaConfig`] whose every field is owned here.
//!
//! Image policy, not environment knobs:
//! - Python `service_maker._build_flow` overrides only frame width and height.
//!   SDK `Flow.batch_capture` then uses mux timeout 33000 µs, `live_source=false`,
//!   buffer pool 4, `batch-size` equal to the camera count, and `compute-hw=1`
//!   (GPU, not VIC).
//! - SDK `Flow.track` inserts `nvtrackerbin`. Installed `nvtrackerbin` and
//!   `nvtracker` both report tracker 960×544.
//! - Native queues use [`crate::msg::POSE_PER_CAMERA`] (4). Record slots use
//!   [`crate::msg::RECORD_CAPACITY`] (32 reusable slots, not a lifetime limit).
//!   Preview stays enabled at [`MEDIA_MAX_PREVIEW_BYTES`] (16 MiB). Production
//!   refuses `file://` URIs.

use std::fmt;
use std::path::Path;

use seeon_deepstream_native::{
    MEDIA_MAX_PREVIEW_BYTES, MEDIA_MAX_SOURCES, MediaBinding, MediaConfig, SourceConfig,
};

use super::FlowSettings;
use super::event_payload::is_uuid;
use crate::msg::{POSE_PER_CAMERA, RECORD_CAPACITY};
use crate::relay::cameras::RuntimeCamera;

/// SDK `Flow.DEFAULT_BATCH_PUSH_TIMEOUT`, applied by `Flow.batch_capture`.
pub const MUX_BATCH_TIMEOUT_US: u32 = 33_000;
/// SDK `Flow.batch_capture` default `live-source`.
pub const MUX_LIVE_SOURCE: bool = false;
/// SDK `Flow.batch_capture` sets video `compute-hw=1` (GPU, not VIC).
/// `MediaConfig` has no compute-hw field; native open sets mux `gpu-id` to 0.
/// Buffer pool 4 is likewise an SDK mux property this ABI does not carry.
/// Neither becomes an environment knob.
/// Installed `nvtrackerbin` and `nvtracker` `tracker-width`.
pub const TRACKER_WIDTH: u32 = 960;
/// Installed `nvtrackerbin` and `nvtracker` `tracker-height`.
pub const TRACKER_HEIGHT: u32 = 544;
/// Production media accepts RTSP only. `file://` is a fixture path.
pub const ALLOW_FILE_URIS: bool = false;

/// Fresh fixed-roster process. Not a Python `SourceBinding` generation.
const FRESH_GENERATION: u64 = 1;
/// Fresh fixed-roster process. Not a Python `SourceBinding` stream epoch.
const FRESH_EPOCH: u64 = 1;
/// Native refuses reconnect intervals above one day. There is no implicit 5.
const MAX_RECONNECT_INTERVAL_SEC: u32 = 86_400;

/// Idle only when the admitted roster is empty. Otherwise the owned plane config.
pub enum MediaAssembly {
    Idle,
    Configured(MediaConfig),
}

/// A typed refusal. Variants carry indexes and limits, never a URI or boot id.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaConfigError {
    /// `boot_id` is not the canonical lowercase hyphenated form `is_uuid` accepts.
    BootId,
    /// Roster longer than [`MEDIA_MAX_SOURCES`]. Not truncated or wrapped.
    RosterLimit { cameras: usize, limit: usize },
    /// Nonempty roster length is not the admitted `flow.batch_size`.
    BatchMismatch { cameras: usize, batch_size: u32 },
    /// Admitted frame width or height is zero.
    FrameGeometry,
    /// Admitted record cache is zero. Native open refuses that.
    RecordCache,
    /// Admitted reconnect interval is above one day.
    ReconnectInterval,
    /// Admitted infer, tracker, library, or record path is empty.
    Path,
    /// Camera URI is not `rtsp://` or does not fit the native path bound.
    Uri { index: usize },
}

impl fmt::Display for MediaConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BootId => formatter.write_str("boot id is not a canonical uuid"),
            Self::RosterLimit { cameras, limit } => {
                write!(formatter, "roster of {cameras} exceeds media limit {limit}")
            }
            Self::BatchMismatch {
                cameras,
                batch_size,
            } => write!(
                formatter,
                "roster of {cameras} does not match admitted batch {batch_size}"
            ),
            Self::FrameGeometry => formatter.write_str("admitted frame geometry is empty"),
            Self::RecordCache => formatter.write_str("admitted record cache is empty"),
            Self::ReconnectInterval => {
                formatter.write_str("admitted rtsp reconnect interval is outside 0..=86400")
            }
            Self::Path => formatter.write_str("admitted media path is empty"),
            Self::Uri { index } => {
                write!(formatter, "source {index} uri is not an admitted rtsp uri")
            }
        }
    }
}

impl std::error::Error for MediaConfigError {}

/// Assemble the fixed roster. Empty admitted cameras are [`MediaAssembly::Idle`].
///
/// `source_id` is the immutable vector position. The binding token is
/// `index + 1` and is unique and nonzero. Generation and epoch both start at 1
/// for this fresh process; they are not a Python `SourceBinding`. The record
/// prefix is the validated boot UUID, a hyphen, and the decimal index. It
/// contains no URI or camera credential.
pub fn assemble(
    flow: &FlowSettings,
    cameras: &[RuntimeCamera],
    boot_id: &str,
) -> Result<MediaAssembly, MediaConfigError> {
    if cameras.is_empty() {
        return Ok(MediaAssembly::Idle);
    }
    if !is_uuid(boot_id) {
        return Err(MediaConfigError::BootId);
    }
    if cameras.len() > MEDIA_MAX_SOURCES {
        return Err(MediaConfigError::RosterLimit {
            cameras: cameras.len(),
            limit: MEDIA_MAX_SOURCES,
        });
    }
    let Some(admitted_batch) = usize::try_from(flow.batch_size).ok() else {
        return Err(MediaConfigError::BatchMismatch {
            cameras: cameras.len(),
            batch_size: flow.batch_size,
        });
    };
    if admitted_batch != cameras.len() {
        return Err(MediaConfigError::BatchMismatch {
            cameras: cameras.len(),
            batch_size: flow.batch_size,
        });
    }
    if flow.frame_width == 0 || flow.frame_height == 0 {
        return Err(MediaConfigError::FrameGeometry);
    }
    if flow.record_cache_seconds == 0 {
        return Err(MediaConfigError::RecordCache);
    }
    if flow.rtsp_reconnect_interval_sec > MAX_RECONNECT_INTERVAL_SEC {
        return Err(MediaConfigError::ReconnectInterval);
    }
    if path_missing(&flow.infer_config)
        || path_missing(&flow.tracker_config)
        || path_missing(&flow.tracker_library)
        || path_missing(&flow.record_dir)
    {
        return Err(MediaConfigError::Path);
    }

    let mut sources = Vec::with_capacity(cameras.len());
    for (index, camera) in cameras.iter().enumerate() {
        sources.push(source(index, camera, boot_id)?);
    }
    // Image facts named above. Queue and record capacities are the existing
    // native bounds, not a second configuration surface. SDK buffer-pool-size
    // 4 and compute-hw 1 are documented above; this ABI has no field for them.
    Ok(MediaAssembly::Configured(MediaConfig {
        sources,
        infer_config_path: flow.infer_config.clone(),
        tracker_config_path: flow.tracker_config.clone(),
        tracker_library_path: flow.tracker_library.clone(),
        record_directory: flow.record_dir.clone(),
        record_cache_seconds: flow.record_cache_seconds,
        record_capacity: u32::try_from(RECORD_CAPACITY).expect("record capacity fits u32"),
        mux_width: flow.frame_width,
        mux_height: flow.frame_height,
        mux_batch_timeout_us: MUX_BATCH_TIMEOUT_US,
        mux_live_source: MUX_LIVE_SOURCE,
        tracker_width: TRACKER_WIDTH,
        tracker_height: TRACKER_HEIGHT,
        queue_max_buffers: u32::try_from(POSE_PER_CAMERA).expect("pose queue fits u32"),
        preview_enabled: true,
        max_preview_bytes: MEDIA_MAX_PREVIEW_BYTES,
        allow_file_uris: ALLOW_FILE_URIS,
        rtsp_reconnect_interval_sec: flow.rtsp_reconnect_interval_sec,
    }))
}

fn source(
    index: usize,
    camera: &RuntimeCamera,
    boot_id: &str,
) -> Result<SourceConfig, MediaConfigError> {
    let source_id = u32::try_from(index).expect("roster index is below MEDIA_MAX_SOURCES");
    let token = u64::try_from(index).expect("roster index") + 1;
    let record_prefix = format!("{boot_id}-{index}");
    let Some(uri) = native_rtsp(&camera.rtsp_url) else {
        return Err(MediaConfigError::Uri { index });
    };
    Ok(SourceConfig {
        source_id,
        binding: MediaBinding {
            token,
            generation: FRESH_GENERATION,
            epoch: FRESH_EPOCH,
        },
        uri,
        record_prefix,
    })
}

fn path_missing(path: &Path) -> bool {
    path.as_os_str().is_empty()
}

/// Native open accepts only a lowercase `rtsp://` scheme. Admission accepts any
/// ASCII case, so only those seven bytes are folded. Credentials, host, path,
/// and query stay byte-for-byte.
fn native_rtsp(uri: &str) -> Option<String> {
    let (scheme, rest) = uri.split_at_checked(7)?;
    if !scheme.eq_ignore_ascii_case("rtsp://") || rest.is_empty() || uri.contains('\0') {
        return None;
    }
    if uri.len() >= seeon_deepstream_native::MEDIA_PATH_BYTES {
        return None;
    }
    let mut native = String::with_capacity(uri.len());
    native.push_str("rtsp://");
    native.push_str(rest);
    Some(native)
}
