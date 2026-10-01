//! Private mirror of native/media_runtime.h. All C enums are raw i32, never Rust enums.
use crate::media_types::*;
use std::{
    ffi::{CString, c_char},
    mem::size_of,
    os::unix::ffi::OsStrExt,
};

#[repr(C)]
pub(super) struct Media {
    _opaque: [u8; 0],
}
#[repr(C)]
pub(super) struct Source {
    pub source_id: u32,
    pub binding: MediaBinding,
    pub uri: *const c_char,
    pub record_prefix: *const c_char,
}
#[repr(C)]
pub(super) struct Config {
    pub abi_version: u32,
    pub struct_size: u32,
    pub source_count: u32,
    pub sources: *const Source,
    pub infer_config_path: *const c_char,
    pub tracker_config_path: *const c_char,
    pub tracker_library_path: *const c_char,
    pub record_directory: *const c_char,
    pub record_cache_seconds: u32,
    pub record_capacity: u32,
    pub mux_width: u32,
    pub mux_height: u32,
    pub mux_batch_timeout_us: u32,
    pub mux_live_source: u32,
    pub tracker_width: u32,
    pub tracker_height: u32,
    pub queue_max_buffers: u32,
    pub preview_enabled: u32,
    pub max_preview_bytes: u32,
    pub allow_file_uris: u32,
    pub rtsp_reconnect_interval_sec: u32,
}
#[repr(C)]
pub(super) struct Pose {
    pub frame: FrameIdentity,
    pub tensor_present: u32,
    pub row_count: u32,
    pub object_count: u32,
    pub rows: [[f32; MEDIA_POSE_COLUMNS]; MEDIA_POSE_ROWS],
    pub objects: [TrackedObject; MEDIA_MAX_OBJECTS],
}
#[repr(C)]
#[derive(Default)]
pub(super) struct Preview {
    pub request_id: u64,
    pub jpeg_bytes: u64,
    pub batch_pts_ns: u64,
    pub frame: FrameIdentity,
    pub result: i32,
    pub error: i32,
}
#[repr(C)]
pub(super) struct Record {
    pub ticket: RecordTicket,
    pub result: i32,
    pub error: i32,
    pub duration_ms: u64,
    pub width: u32,
    pub height: u32,
    pub contains_video: u32,
    pub contains_audio: u32,
    pub directory: [c_char; MEDIA_PATH_BYTES],
    pub filename: [c_char; MEDIA_PATH_BYTES],
}
#[repr(C)]
pub(super) struct Status {
    pub state: i32,
    pub fatal: MediaDiagnostic,
    pub warning: MediaDiagnostic,
    pub warnings: u64,
    pub capacity_refusals: u64,
    pub preview_dropped: u64,
    pub late_record_callbacks: u64,
    pub source_count: u32,
    pub callbacks_active: u32,
    pub stop_timed_out: u32,
    pub records_reserved: u32,
    pub sources: [SourceStatus; MEDIA_MAX_SOURCES],
}

// Allocate the large, reusable mailboxes directly on the heap, not the callback
// stack. This is deliberately restricted to these three primitive-only layouts.
macro_rules! buffers {
    ($($ty:ty),+ $(,)?) => {$(
        impl $ty {
            pub(super) fn buffer() -> Box<Self> {
                let mut buffer = Box::<Self>::new_uninit();
                // SAFETY: these repr(C) types contain only integers, floats and
                // arrays thereof. Zero is valid for every field; no pointers,
                // Rust enums, references or Drop implementations are present.
                unsafe {
                    buffer.as_mut_ptr().write_bytes(0, 1);
                    buffer.assume_init()
                }
            }
        }
    )+};
}
buffers!(Pose, Record, Status);

unsafe extern "C" {
    pub(super) fn seeon_media_open(
        config: *const Config,
        out: *mut *mut Media,
        error: *mut MediaDiagnostic,
    ) -> i32;
    pub(super) fn seeon_media_start(media: *mut Media) -> i32;
    pub(super) fn seeon_media_try_read_pose(
        media: *mut Media,
        source_id: u32,
        out: *mut Pose,
        bytes: usize,
    ) -> i32;
    pub(super) fn seeon_media_request_preview(
        media: *mut Media,
        source_id: u32,
        binding: *const MediaBinding,
        request_id: u64,
        draw_objects: u32,
        timeout_ms: u32,
    ) -> i32;
    pub(super) fn seeon_media_try_read_preview(
        media: *mut Media,
        out: *mut Preview,
        descriptor_bytes: usize,
        jpeg: *mut u8,
        jpeg_capacity: usize,
    ) -> i32;
    pub(super) fn seeon_media_record_start(
        media: *mut Media,
        source_id: u32,
        binding: *const MediaBinding,
        request_id: u64,
        lookback_seconds: u32,
        forward_seconds: u32,
        out: *mut RecordTicket,
        bytes: usize,
    ) -> i32;
    pub(super) fn seeon_media_record_stop(
        media: *mut Media,
        source_id: u32,
        binding: *const MediaBinding,
        request_id: u64,
        session_id: u32,
    ) -> i32;
    pub(super) fn seeon_media_try_read_record(
        media: *mut Media,
        out: *mut Record,
        bytes: usize,
    ) -> i32;
    pub(super) fn seeon_media_read_status(media: *mut Media, out: *mut Status, bytes: usize)
    -> i32;
    pub(super) fn seeon_media_stop(media: *mut Media, deadline_ms: u32) -> i32;
    pub(super) fn seeon_media_destroy(media: *mut Media) -> i32;
}

pub(super) fn require(condition: bool, argument: MediaArgument) -> Result<(), MediaError> {
    if condition {
        Ok(())
    } else {
        Err(MediaError::InvalidArgument(argument))
    }
}

pub(super) fn text(
    bytes: &[u8],
    limit: usize,
    argument: MediaArgument,
    allow_empty: bool,
) -> Result<CString, MediaError> {
    require(
        (allow_empty || !bytes.is_empty()) && bytes.len() < limit,
        argument,
    )?;
    CString::new(bytes).map_err(|_| MediaError::InvalidArgument(argument))
}

pub(super) fn budget(value: u32, argument: MediaArgument) -> Result<(), MediaError> {
    require((1..=60000).contains(&value), argument)
}

pub(super) fn request(value: u64) -> Result<(), MediaError> {
    // Native owns strict monotonicity and exhaustion. Never increment a caller ID.
    require(value != 0, MediaArgument::RequestId)
}

pub(super) fn recording_window(cache: u32, lookback: u32, forward: u32) -> Result<(), MediaError> {
    // Native widens forward to u64 BEFORE adding its 30-second completion margin.
    require(
        lookback < cache && forward != 0,
        MediaArgument::RecordingWindow,
    )
}

fn rtsp(uri: &str) -> bool {
    uri.starts_with("rtsp://") || uri.starts_with("rtsps://")
}

/// All CStrings and the pointer-bearing roster outlive open. Native copies them;
/// no SDK callback borrows this storage and no pointer is exported to the caller.
pub(super) struct OpenConfig {
    pub raw: Config,
    _paths: [CString; 4],
    _uris: Vec<CString>,
    _prefixes: Vec<CString>,
    _sources: Vec<Source>,
}
impl OpenConfig {
    pub(super) fn new(config: &MediaConfig) -> Result<Self, MediaError> {
        use MediaArgument as A;
        require(
            (1..=MEDIA_MAX_SOURCES).contains(&config.sources.len()),
            A::Sources,
        )?;
        for (width, height, argument) in [
            (config.mux_width, config.mux_height, A::MuxDimensions),
            (
                config.tracker_width,
                config.tracker_height,
                A::TrackerDimensions,
            ),
        ] {
            require(
                (1..=i32::MAX as u32).contains(&width) && (1..=i32::MAX as u32).contains(&height),
                argument,
            )?;
        }
        require(
            (1..=i32::MAX as u32).contains(&config.mux_batch_timeout_us),
            A::MuxTimeout,
        )?;
        require(config.record_cache_seconds != 0, A::RecordCache)?;
        require(
            (1..=MEDIA_MAX_RECORDS).contains(&config.record_capacity),
            A::RecordCapacity,
        )?;
        require(
            config.rtsp_reconnect_interval_sec <= 86400,
            A::ReconnectInterval,
        )?;
        require(
            (1..=64).contains(&config.queue_max_buffers),
            A::QueueCapacity,
        )?;
        require(
            if config.preview_enabled {
                (4..=MEDIA_MAX_PREVIEW_BYTES).contains(&config.max_preview_bytes)
            } else {
                config.max_preview_bytes == 0
            },
            A::PreviewCapacity,
        )?;
        let paths = [
            (&config.infer_config_path, A::InferConfigPath),
            (&config.tracker_config_path, A::TrackerConfigPath),
            (&config.tracker_library_path, A::TrackerLibraryPath),
            (&config.record_directory, A::RecordDirectory),
        ]
        .map(|(path, argument)| {
            text(
                path.as_os_str().as_bytes(),
                MEDIA_PATH_BYTES,
                argument,
                false,
            )
        });
        let [infer, tracker, library, directory] = paths;
        let paths = [infer?, tracker?, library?, directory?];
        let mut uris = Vec::with_capacity(config.sources.len());
        let mut prefixes = Vec::with_capacity(config.sources.len());
        for (index, source) in config.sources.iter().enumerate() {
            require(source.source_id as usize == index, A::SourceId)?;
            require(
                source.binding.token != 0
                    && config.sources[..index]
                        .iter()
                        .all(|previous| previous.binding.token != source.binding.token),
                A::Binding,
            )?;
            uris.push(text(
                source.uri.as_bytes(),
                MEDIA_PATH_BYTES,
                A::Uri,
                false,
            )?);
            let is_rtsp = rtsp(&source.uri);
            require(
                is_rtsp || (config.allow_file_uris && source.uri.starts_with("file://")),
                A::Uri,
            )?;
            prefixes.push(text(
                source.record_prefix.as_bytes(),
                128,
                A::RecordPrefix,
                !is_rtsp,
            )?);
            if is_rtsp {
                require(
                    source.record_prefix != "."
                        && source.record_prefix != ".."
                        && source.record_prefix.bytes().all(|byte| {
                            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
                        })
                        && config.sources[..index].iter().all(|previous| {
                            !rtsp(&previous.uri) || previous.record_prefix != source.record_prefix
                        }),
                    A::RecordPrefix,
                )?;
            }
        }
        let sources: Vec<_> = config
            .sources
            .iter()
            .enumerate()
            .map(|(index, source)| Source {
                source_id: source.source_id,
                binding: source.binding,
                uri: uris[index].as_ptr(),
                record_prefix: prefixes[index].as_ptr(),
            })
            .collect();
        let raw = Config {
            abi_version: MEDIA_ABI_VERSION,
            struct_size: size_of::<Config>() as u32,
            source_count: sources.len() as u32,
            sources: sources.as_ptr(),
            infer_config_path: paths[0].as_ptr(),
            tracker_config_path: paths[1].as_ptr(),
            tracker_library_path: paths[2].as_ptr(),
            record_directory: paths[3].as_ptr(),
            record_cache_seconds: config.record_cache_seconds,
            record_capacity: config.record_capacity,
            mux_width: config.mux_width,
            mux_height: config.mux_height,
            mux_batch_timeout_us: config.mux_batch_timeout_us,
            mux_live_source: u32::from(config.mux_live_source),
            tracker_width: config.tracker_width,
            tracker_height: config.tracker_height,
            queue_max_buffers: config.queue_max_buffers,
            preview_enabled: u32::from(config.preview_enabled),
            max_preview_bytes: config.max_preview_bytes,
            allow_file_uris: u32::from(config.allow_file_uris),
            rtsp_reconnect_interval_sec: config.rtsp_reconnect_interval_sec,
        };
        Ok(Self {
            raw,
            _paths: paths,
            _uris: uris,
            _prefixes: prefixes,
            _sources: sources,
        })
    }
}
