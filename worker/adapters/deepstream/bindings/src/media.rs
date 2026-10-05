use crate::{media_ffi as ffi, media_types::*};
use std::{
    ffi::{OsString, c_char},
    io::Write,
    marker::PhantomData,
    mem::size_of,
    os::unix::ffi::OsStringExt,
    ptr::NonNull,
    rc::Rc,
};

/// Serialized, thread-confined owner of the real native media graph (!Send/!Sync).
/// SDK callbacks never call Rust or retain Rust packet storage. Reusable bounded
/// ABI buffers are allocated before open; only delivered packets allocate owned
/// copies, outside callbacks. EMPTY is never an empty-room observation.
///
/// stop returning OK is the ONLY proof of quiescence; status.state alone is not.
/// A timeout retains the handle for retry. Drop attempts the explicit constructor
/// budget, then destroys only after successful stop. On failure it intentionally
/// retains the foreign handle and emits a static fatal notice, never an unbounded
/// join or a free beneath SDK threads. Rust buffers remain safe to release because
/// all native borrows end at each C call.
///
/// This crate is for a Worker executable with linked DSOs, NOT an unloadable
/// plugin. A retained foreign handle requires the native DSO and its dependencies
/// to stay loaded until process exit. Retention is not process fencing: the root
/// must treat shutdown failure as fatal and terminate the child before unloading.
pub struct MediaOwner {
    handle: Option<NonNull<ffi::Media>>,
    quiescent: bool,
    shutdown_budget_ms: u32,
    source_count: u32,
    record_cache_seconds: u32,
    pose: Box<ffi::Pose>,
    preview: ffi::Preview,
    jpeg: Vec<u8>,
    record: Box<ffi::Record>,
    status: Box<ffi::Status>,
    _thread_confined: PhantomData<Rc<()>>,
}

impl MediaOwner {
    /// shutdown_budget_ms is required and must be 1..=60000. No fallback provider.
    pub fn open(config: &MediaConfig, shutdown_budget_ms: u32) -> Result<Self, MediaError> {
        ffi::budget(shutdown_budget_ms, MediaArgument::ShutdownBudget)?;
        let prepared = ffi::OpenConfig::new(config)?;
        let mut owner = Self {
            handle: None,
            quiescent: false,
            shutdown_budget_ms,
            source_count: prepared.raw.source_count,
            record_cache_seconds: config.record_cache_seconds,
            pose: ffi::Pose::buffer(),
            preview: ffi::Preview::default(),
            jpeg: vec![0; config.max_preview_bytes as usize],
            record: ffi::Record::buffer(),
            status: ffi::Status::buffer(),
            _thread_confined: PhantomData,
        };
        let mut handle = std::ptr::null_mut();
        let mut diagnostic = MediaDiagnostic::default();
        // SAFETY: prepared owns every C string and roster pointer until open
        // returns. Native copies them and owns cleanup of partial construction.
        let result = unsafe { ffi::seeon_media_open(&prepared.raw, &mut handle, &mut diagnostic) };
        if result != 0 {
            return Err(MediaError::Native(MediaCallStatus {
                result: result.into(),
                fatal: Some(diagnostic),
                warning: None,
                required_bytes: None,
            }));
        }
        owner.handle = Some(NonNull::new(handle).ok_or(MediaError::NativeContract)?);
        Ok(owner)
    }

    /// Admission, not a claim that the pipeline is PLAYING; read_status owns that observation.
    pub fn start(&mut self) -> Result<MediaCallStatus, MediaError> {
        let handle = self.handle()?;
        // SAFETY: this unique, thread-confined owner serializes every native call.
        let result = unsafe { ffi::seeon_media_start(handle.as_ptr()) };
        Ok(self.call_status(result))
    }

    /// FATAL still returns the populated status, including exact sanitized codes.
    pub fn read_status(&mut self) -> Result<MediaStatus, MediaError> {
        let handle = self.handle()?;
        // SAFETY: exact repr(C) writable storage, live handle, no retained borrow.
        let result = unsafe {
            ffi::seeon_media_read_status(
                handle.as_ptr(),
                &mut *self.status,
                size_of::<ffi::Status>(),
            )
        };
        if !matches!(result, 0 | 6) {
            return Err(MediaError::Native(self.call_status(result)));
        }
        status_packet(result.into(), &self.status)
    }

    pub fn poll_pose(&mut self, source_id: u32) -> Result<MediaPoll<PosePacket>, MediaError> {
        let handle = self.source_handle(source_id)?;
        // SAFETY: the private, reusable output has the exact header size/layout.
        let result = unsafe {
            ffi::seeon_media_try_read_pose(
                handle.as_ptr(),
                source_id,
                &mut *self.pose,
                size_of::<ffi::Pose>(),
            )
        };
        if result != 0 {
            return Ok(MediaPoll::Status(self.call_status(result)));
        }
        Ok(MediaPoll::Ready(pose_packet(&self.pose)?))
    }

    pub fn request_preview(
        &mut self,
        source_id: u32,
        binding: MediaBinding,
        request_id: u64,
        draw_objects: bool,
        timeout_ms: u32,
    ) -> Result<MediaCallStatus, MediaError> {
        let handle = self.source_handle(source_id)?;
        ffi::request(request_id)?;
        ffi::budget(timeout_ms, MediaArgument::PreviewTimeout)?;
        // SAFETY: binding is an integer-only borrowed value, copied during the
        // call. Native remains authority for identity, monotonic IDs and capacity.
        let result = unsafe {
            ffi::seeon_media_request_preview(
                handle.as_ptr(),
                source_id,
                &binding,
                request_id,
                u32::from(draw_objects),
                timeout_ms,
            )
        };
        Ok(self.call_status(result))
    }

    pub fn poll_preview(&mut self) -> Result<MediaPoll<PreviewPacket>, MediaError> {
        let handle = self.handle()?;
        self.preview = ffi::Preview::default();
        // SAFETY: buffers are disjoint and sized exactly; native only copies up
        // to jpeg.len(), which is the validated configured maximum, never a hint.
        let result = unsafe {
            ffi::seeon_media_try_read_preview(
                handle.as_ptr(),
                &mut self.preview,
                size_of::<ffi::Preview>(),
                self.jpeg.as_mut_ptr(),
                self.jpeg.len(),
            )
        };
        if result != 0 {
            let mut status = self.call_status(result);
            if result == 4 {
                status.required_bytes = Some(self.preview.jpeg_bytes);
            }
            return Ok(MediaPoll::Status(status));
        }
        Ok(MediaPoll::Ready(preview_packet(&self.preview, &self.jpeg)?))
    }

    /// A Ready ticket is admission, not completion. Coalescing returns the exact
    /// original ticket; native owns request IDs and bounded reusable recording
    /// slots. Consumed slots are reusable only after native completion and callback
    /// retirement make reuse safe, without owner rotation. Neither admission nor
    /// consumption seals, publishes, or guarantees crash durability of a recording.
    pub fn record_start(
        &mut self,
        source_id: u32,
        binding: MediaBinding,
        request_id: u64,
        lookback_seconds: u32,
        forward_seconds: u32,
    ) -> Result<MediaPoll<RecordTicket>, MediaError> {
        let handle = self.source_handle(source_id)?;
        ffi::request(request_id)?;
        ffi::recording_window(self.record_cache_seconds, lookback_seconds, forward_seconds)?;
        let mut ticket = RecordTicket::default();
        // SAFETY: the borrowed binding and exact output ticket live through the
        // call. Native retains its own session context, never this output address.
        let result = unsafe {
            ffi::seeon_media_record_start(
                handle.as_ptr(),
                source_id,
                &binding,
                request_id,
                lookback_seconds,
                forward_seconds,
                &mut ticket,
                size_of::<RecordTicket>(),
            )
        };
        if result != 0 {
            return Ok(MediaPoll::Status(self.call_status(result)));
        }
        Ok(MediaPoll::Ready(ticket))
    }

    /// Use a session-valid ticket (read_status exposes it after asynchronous start).
    /// In particular, use the original IDs of a coalesced ticket. Accepted stop is
    /// not completion or a seal; poll_record must still report vendor completion.
    pub fn record_stop(&mut self, ticket: &RecordTicket) -> Result<MediaCallStatus, MediaError> {
        let handle = self.source_handle(ticket.source_id)?;
        ffi::request(ticket.request_id)?;
        ffi::require(
            ticket.session_valid == 1 && ticket.session_id != u32::MAX,
            MediaArgument::RecordSession,
        )?;
        // SAFETY: only owned integer identities cross this boundary; native
        // compares them to its active session before emitting any SDK action.
        let result = unsafe {
            ffi::seeon_media_record_stop(
                handle.as_ptr(),
                ticket.source_id,
                &ticket.binding,
                ticket.request_id,
                ticket.session_id,
            )
        };
        Ok(self.call_status(result))
    }

    /// Completions remain drainable after a sticky native failure and after stop.
    pub fn poll_record(&mut self) -> Result<MediaPoll<RecordReceipt>, MediaError> {
        let handle = self.handle()?;
        // SAFETY: native copies the complete fixed-size record, retaining no pointer.
        let result = unsafe {
            ffi::seeon_media_try_read_record(
                handle.as_ptr(),
                &mut *self.record,
                size_of::<ffi::Record>(),
            )
        };
        if result != 0 {
            return Ok(MediaPoll::Status(self.call_status(result)));
        }
        Ok(MediaPoll::Ready(record_receipt(&self.record)?))
    }

    /// Relative monotonic deadline, in milliseconds (zero is a nonblocking try).
    /// Failure retains the handle for retry and MUST be treated as process-fatal
    /// by the composition root. A STOPPED status is insufficient proof of joining.
    pub fn stop(&mut self, deadline_ms: u32) -> Result<MediaCallStatus, MediaError> {
        let handle = self.handle()?;
        // SAFETY: native stop bounds its join attempt and retains every resource
        // on failure. Only OK proves callbacks and the native thread have exited.
        let result = unsafe { ffi::seeon_media_stop(handle.as_ptr(), deadline_ms) };
        if result == 0 {
            self.quiescent = true;
        }
        Ok(self.call_status(result))
    }

    /// Refuses until stop returned OK. Removal of the handle happens only after
    /// destroy OK; a refused destroy can be retried. Every method fails once closed.
    pub fn close(&mut self) -> Result<MediaCallStatus, MediaError> {
        let handle = self.handle()?;
        if !self.quiescent {
            return Err(MediaError::NotStopped);
        }
        // SAFETY: proven stop precedes destruction. No SDK thread or callback can
        // retain graph storage, and native also checks quiescence before deletion.
        let result = unsafe { ffi::seeon_media_destroy(handle.as_ptr()) };
        if result == 0 {
            self.handle = None;
        }
        Ok(self.call_status(result))
    }

    fn handle(&self) -> Result<NonNull<ffi::Media>, MediaError> {
        self.handle.ok_or(MediaError::Closed)
    }

    fn source_handle(&self, source_id: u32) -> Result<NonNull<ffi::Media>, MediaError> {
        let handle = self.handle()?;
        ffi::require(source_id < self.source_count, MediaArgument::SourceId)?;
        Ok(handle)
    }

    fn call_status(&mut self, raw: i32) -> MediaCallStatus {
        let mut result = MediaCallStatus {
            result: raw.into(),
            fatal: None,
            warning: None,
            required_bytes: None,
        };
        if raw != 0
            && let Some(handle) = self.handle
        {
            // SAFETY: non-OK native operations retain the live handle. This
            // integer-only snapshot is bounded and preserves FATAL diagnostics
            // even though read_status itself returns FATAL when populating them.
            let read = unsafe {
                ffi::seeon_media_read_status(
                    handle.as_ptr(),
                    &mut *self.status,
                    size_of::<ffi::Status>(),
                )
            };
            if matches!(read, 0 | 6) {
                result.fatal = Some(self.status.fatal);
                result.warning = Some(self.status.warning);
            }
        }
        result
    }
}

impl Drop for MediaOwner {
    fn drop(&mut self) {
        if self.handle.is_none() {
            return;
        }
        if self
            .stop(self.shutdown_budget_ms)
            .is_ok_and(|status| status.result == MediaResult::Ok)
            && self
                .close()
                .is_ok_and(|status| status.result == MediaResult::Ok)
        {
            return;
        }
        // NonNull has no destructor: intentionally retain the foreign allocation.
        // No panic on a failed diagnostic write, and no caller/native text is logged.
        let _ = std::io::stderr().write_all(b"fatal: native media retained after shutdown failure; terminate child before unloading native libraries\n");
    }
}

fn pose_packet(raw: &ffi::Pose) -> Result<PosePacket, MediaError> {
    if !matches!((raw.tensor_present, raw.row_count), (0, 0) | (1, 300)) {
        return Err(MediaError::NativeContract);
    }
    let rows = raw
        .rows
        .get(..raw.row_count as usize)
        .ok_or(MediaError::NativeContract)?;
    let objects = raw
        .objects
        .get(..raw.object_count as usize)
        .ok_or(MediaError::NativeContract)?;
    Ok(PosePacket {
        frame: raw.frame,
        tensor_present: raw.tensor_present == 1,
        rows: rows.to_vec(),
        objects: objects.to_vec(),
    })
}

fn preview_packet(raw: &ffi::Preview, jpeg: &[u8]) -> Result<PreviewPacket, MediaError> {
    let length = usize::try_from(raw.jpeg_bytes).map_err(|_| MediaError::NativeContract)?;
    let bytes = jpeg.get(..length).ok_or(MediaError::NativeContract)?;
    let result = raw.result.into();
    if result == MediaResult::Ok && raw.frame.sequence == 0 {
        return Err(MediaError::NativeContract);
    }
    Ok(PreviewPacket {
        request_id: raw.request_id,
        batch_pts_ns: raw.batch_pts_ns,
        frame: raw.frame,
        result,
        error: raw.error,
        jpeg: bytes.to_vec(),
    })
}

fn bounded_path(bytes: &[c_char; MEDIA_PATH_BYTES]) -> Result<OsString, MediaError> {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .ok_or(MediaError::NativeContract)?;
    Ok(OsString::from_vec(
        bytes[..end].iter().map(|&byte| byte as u8).collect(),
    ))
}

fn record_receipt(raw: &ffi::Record) -> Result<RecordReceipt, MediaError> {
    Ok(RecordReceipt {
        ticket: raw.ticket,
        result: raw.result.into(),
        error: raw.error,
        duration_ms: raw.duration_ms,
        width: raw.width,
        height: raw.height,
        contains_video: raw.contains_video != 0,
        contains_audio: raw.contains_audio != 0,
        directory: bounded_path(&raw.directory)?.into(),
        filename: bounded_path(&raw.filename)?,
    })
}

fn status_packet(result: MediaResult, raw: &ffi::Status) -> Result<MediaStatus, MediaError> {
    let sources = raw
        .sources
        .get(..raw.source_count as usize)
        .ok_or(MediaError::NativeContract)?;
    Ok(MediaStatus {
        result,
        state: raw.state.into(),
        fatal: raw.fatal,
        warning: raw.warning,
        warnings: raw.warnings,
        capacity_refusals: raw.capacity_refusals,
        preview_dropped: raw.preview_dropped,
        late_record_callbacks: raw.late_record_callbacks,
        callbacks_active: raw.callbacks_active,
        stop_timed_out: raw.stop_timed_out != 0,
        records_reserved: raw.records_reserved,
        sources: sources.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        ffi::CStr,
        fs,
        mem::{align_of, offset_of},
        os::unix::ffi::OsStrExt,
        path::{Component, Path, PathBuf},
        sync::{Mutex, MutexGuard, PoisonError},
        time::{Duration, Instant},
    };

    // Production runs one MediaOwner per process; a second actual pipeline
    // started concurrently fails its state change, so actual tests take turns.
    static ACTUAL_MEDIA: Mutex<()> = Mutex::new(());

    fn actual_media() -> MutexGuard<'static, ()> {
        ACTUAL_MEDIA.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn required(name: &str) -> OsString {
        std::env::var_os(name)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| panic!("missing explicit media test configuration: {name}"))
    }

    fn directory(path: &Path) -> PathBuf {
        assert!(
            path.is_absolute() && path.parent().is_some(),
            "record directory must be an explicit non-root absolute path"
        );
        let mut walked = PathBuf::new();
        for component in path.components() {
            assert!(
                matches!(component, Component::RootDir | Component::Normal(_)),
                "record directory traversal"
            );
            walked.push(component.as_os_str());
            assert!(
                fs::symlink_metadata(&walked)
                    .expect("record directory metadata")
                    .is_dir(),
                "record directory components must be existing nonsymlink directories"
            );
        }
        fs::canonicalize(path).expect("record directory canonicalization")
    }

    /// Native state has no wait API. The pause only spaces attempts; the
    /// deadline and the attempt's own result decide every outcome.
    fn poll_until<T>(deadline: Instant, what: &str, mut attempt: impl FnMut() -> Option<T>) -> T {
        const PAUSE: Duration = Duration::from_millis(1);
        loop {
            assert!(Instant::now() < deadline, "{what}");
            if let Some(value) = attempt() {
                return value;
            }
            std::thread::sleep(PAUSE);
        }
    }

    /// Owned per-test record directory; CPU tests never share a system path.
    struct RecordDirectory(PathBuf);

    impl RecordDirectory {
        fn new(test: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("seeon-media-{test}-{}", std::process::id()));
            fs::create_dir(&path).expect("create per-test record directory");
            Self(path)
        }
    }

    impl Drop for RecordDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir(&self.0);
        }
    }

    fn fixture_config(record_directory: PathBuf) -> MediaConfig {
        MediaConfig {
            sources: vec![SourceConfig {
                source_id: 0,
                binding: MediaBinding {
                    token: 73,
                    generation: 7,
                    epoch: 11,
                },
                uri: "file:///pinned-synthetic-fixture.mp4".into(),
                record_prefix: "synthetic".into(),
            }],
            infer_config_path: "admitted-inference.txt".into(),
            tracker_config_path: "tracker.yml".into(),
            tracker_library_path:
                "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so".into(),
            record_directory,
            record_cache_seconds: 30,
            record_capacity: 4,
            mux_width: 640,
            mux_height: 360,
            mux_batch_timeout_us: 40000,
            mux_live_source: false,
            tracker_width: 960,
            tracker_height: 544,
            queue_max_buffers: 4,
            preview_enabled: true,
            max_preview_bytes: 1024 * 1024,
            allow_file_uris: true,
            rtsp_reconnect_interval_sec: 5,
        }
    }

    #[test]
    fn abi_sizes_alignment_and_offsets_match_64_bit_header() {
        // DeepStream's supported Linux x86_64/aarch64 ABIs have 64-bit pointers.
        assert_eq!(size_of::<usize>(), 8);
        assert_eq!(align_of::<ffi::Pose>(), 8);
        assert_eq!(
            [
                size_of::<MediaBinding>(),
                size_of::<ffi::Source>(),
                size_of::<ffi::Config>(),
                size_of::<FrameIdentity>(),
                size_of::<TrackedObject>(),
                size_of::<ffi::Pose>(),
                size_of::<ffi::Preview>(),
                size_of::<RecordTicket>(),
                size_of::<ffi::Record>(),
                size_of::<MediaDiagnostic>(),
                size_of::<SourceStatus>(),
                size_of::<ffi::Status>()
            ],
            [24, 48, 112, 80, 32, 73296, 112, 48, 8272, 16, 128, 2136]
        );
        assert_eq!(
            [
                offset_of!(ffi::Source, source_id),
                offset_of!(ffi::Source, binding),
                offset_of!(ffi::Source, uri),
                offset_of!(ffi::Source, record_prefix)
            ],
            [0, 8, 32, 40]
        );
        assert_eq!(
            [
                offset_of!(ffi::Config, source_count),
                offset_of!(ffi::Config, sources),
                offset_of!(ffi::Config, infer_config_path),
                offset_of!(ffi::Config, tracker_config_path),
                offset_of!(ffi::Config, tracker_library_path),
                offset_of!(ffi::Config, record_directory),
                offset_of!(ffi::Config, record_cache_seconds),
                offset_of!(ffi::Config, record_capacity),
                offset_of!(ffi::Config, mux_width),
                offset_of!(ffi::Config, mux_height),
                offset_of!(ffi::Config, mux_batch_timeout_us),
                offset_of!(ffi::Config, mux_live_source),
                offset_of!(ffi::Config, tracker_width),
                offset_of!(ffi::Config, tracker_height),
                offset_of!(ffi::Config, queue_max_buffers),
                offset_of!(ffi::Config, preview_enabled),
                offset_of!(ffi::Config, max_preview_bytes),
                offset_of!(ffi::Config, allow_file_uris),
                offset_of!(ffi::Config, rtsp_reconnect_interval_sec)
            ],
            [
                8, 16, 24, 32, 40, 48, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104
            ]
        );
        assert_eq!(
            [
                offset_of!(FrameIdentity, sequence),
                offset_of!(FrameIdentity, pts_ns),
                offset_of!(FrameIdentity, frame_number),
                offset_of!(FrameIdentity, pts_valid),
                offset_of!(FrameIdentity, source_id),
                offset_of!(FrameIdentity, batch_id),
                offset_of!(FrameIdentity, pad_index),
                offset_of!(FrameIdentity, source_width),
                offset_of!(FrameIdentity, source_height),
                offset_of!(FrameIdentity, analysis_width),
                offset_of!(FrameIdentity, analysis_height)
            ],
            [24, 32, 40, 48, 52, 56, 60, 64, 68, 72, 76]
        );
        assert_eq!(
            [
                offset_of!(ffi::Pose, tensor_present),
                offset_of!(ffi::Pose, row_count),
                offset_of!(ffi::Pose, object_count),
                offset_of!(ffi::Pose, rows),
                offset_of!(ffi::Pose, objects)
            ],
            [80, 84, 88, 92, 68496]
        );
        assert_eq!(
            [
                offset_of!(ffi::Preview, frame),
                offset_of!(ffi::Preview, result),
                offset_of!(ffi::Preview, error)
            ],
            [24, 104, 108]
        );
        assert_eq!(
            [
                offset_of!(RecordTicket, request_id),
                offset_of!(RecordTicket, source_id),
                offset_of!(RecordTicket, session_id),
                offset_of!(RecordTicket, session_valid),
                offset_of!(RecordTicket, coalesced)
            ],
            [24, 32, 36, 40, 44]
        );
        assert_eq!(
            [
                offset_of!(ffi::Record, result),
                offset_of!(ffi::Record, error),
                offset_of!(ffi::Record, duration_ms),
                offset_of!(ffi::Record, width),
                offset_of!(ffi::Record, height),
                offset_of!(ffi::Record, contains_video),
                offset_of!(ffi::Record, contains_audio),
                offset_of!(ffi::Record, directory),
                offset_of!(ffi::Record, filename)
            ],
            [48, 52, 56, 64, 68, 72, 76, 80, 4176]
        );
        assert_eq!(
            [
                offset_of!(SourceStatus, frames),
                offset_of!(SourceStatus, video_linked),
                offset_of!(SourceStatus, active_record)
            ],
            [24, 72, 80]
        );
        assert_eq!(
            [
                offset_of!(ffi::Status, fatal),
                offset_of!(ffi::Status, warning),
                offset_of!(ffi::Status, warnings),
                offset_of!(ffi::Status, capacity_refusals),
                offset_of!(ffi::Status, preview_dropped),
                offset_of!(ffi::Status, late_record_callbacks),
                offset_of!(ffi::Status, source_count),
                offset_of!(ffi::Status, callbacks_active),
                offset_of!(ffi::Status, stop_timed_out),
                offset_of!(ffi::Status, records_reserved),
                offset_of!(ffi::Status, sources)
            ],
            [4, 20, 40, 48, 56, 64, 72, 76, 80, 84, 88]
        );
        assert_eq!(offset_of!(MediaDiagnostic, sdk_code), 12);
    }

    #[test]
    fn native_enum_integers_preserve_all_results_and_unknowns() {
        for (raw, expected) in [
            MediaResult::Ok,
            MediaResult::Empty,
            MediaResult::Busy,
            MediaResult::Stale,
            MediaResult::TooSmall,
            MediaResult::Unsupported,
            MediaResult::Fatal,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(MediaResult::from(raw as i32), expected);
        }
        for raw in [-1, 7, i32::MAX] {
            assert_eq!(MediaResult::from(raw), MediaResult::Unknown(raw));
        }
        for (raw, expected) in [
            MediaState::Open,
            MediaState::Starting,
            MediaState::Running,
            MediaState::Stopping,
            MediaState::Stopped,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(MediaState::from(raw as i32), expected);
        }
        assert_eq!(MediaState::from(-1), MediaState::Unknown(-1));
    }

    #[test]
    fn c_strings_check_byte_bounds_nul_and_preserve_unix_paths() {
        for bytes in [
            b"".as_slice(),
            b"credential-secret\0suffix",
            &[b'x'; MEDIA_PATH_BYTES],
        ] {
            assert_eq!(
                ffi::text(bytes, MEDIA_PATH_BYTES, MediaArgument::Uri, false).err(),
                Some(MediaError::InvalidArgument(MediaArgument::Uri))
            );
        }
        let boundary = vec![b'x'; MEDIA_PATH_BYTES - 1];
        assert_eq!(
            ffi::text(&boundary, MEDIA_PATH_BYTES, MediaArgument::Uri, false)
                .unwrap()
                .as_bytes(),
            boundary
        );
        let record_directory = RecordDirectory::new("c_strings");
        let mut config = fixture_config(record_directory.0.clone());
        config.infer_config_path =
            PathBuf::from(OsString::from_vec(b"/admitted/\xff.ini".to_vec()));
        let prepared = ffi::OpenConfig::new(&config).unwrap();
        drop(config);
        // SAFETY: prepared, not config, owns the live terminated string/roster.
        unsafe {
            assert_eq!(
                CStr::from_ptr(prepared.raw.infer_config_path).to_bytes(),
                b"/admitted/\xff.ini"
            );
            let source = &*prepared.raw.sources;
            assert_eq!(
                CStr::from_ptr(source.uri).to_bytes(),
                b"file:///pinned-synthetic-fixture.mp4"
            );
            assert_eq!(
                source.binding,
                MediaBinding {
                    token: 73,
                    generation: 7,
                    epoch: 11
                }
            );
        }
        let error = ffi::text(
            b"rtsp://credential-secret\0",
            MEDIA_PATH_BYTES,
            MediaArgument::Uri,
            false,
        )
        .unwrap_err();
        assert!(!format!("{error:?} {error}").contains("credential-secret"));
    }

    #[test]
    fn config_rejects_numeric_bounds_without_touching_sdk() {
        type InvalidConfigCase = (fn(&mut MediaConfig), MediaArgument);
        let cases: &[InvalidConfigCase] = &[
            (|c| c.mux_width = 0, MediaArgument::MuxDimensions),
            (|c| c.mux_height = u32::MAX, MediaArgument::MuxDimensions),
            (
                |c| c.tracker_width = u32::MAX,
                MediaArgument::TrackerDimensions,
            ),
            (|c| c.tracker_height = 0, MediaArgument::TrackerDimensions),
            (|c| c.mux_batch_timeout_us = 0, MediaArgument::MuxTimeout),
            (
                |c| c.mux_batch_timeout_us = u32::MAX,
                MediaArgument::MuxTimeout,
            ),
            (|c| c.record_cache_seconds = 0, MediaArgument::RecordCache),
            (|c| c.record_capacity = 0, MediaArgument::RecordCapacity),
            (|c| c.record_capacity = 257, MediaArgument::RecordCapacity),
            (|c| c.queue_max_buffers = 0, MediaArgument::QueueCapacity),
            (|c| c.queue_max_buffers = 65, MediaArgument::QueueCapacity),
            (|c| c.max_preview_bytes = 3, MediaArgument::PreviewCapacity),
            (
                |c| c.max_preview_bytes = MEDIA_MAX_PREVIEW_BYTES + 1,
                MediaArgument::PreviewCapacity,
            ),
            (
                |c| c.preview_enabled = false,
                MediaArgument::PreviewCapacity,
            ),
            (
                |c| c.rtsp_reconnect_interval_sec = 86401,
                MediaArgument::ReconnectInterval,
            ),
            (
                |c| c.rtsp_reconnect_interval_sec = u32::MAX,
                MediaArgument::ReconnectInterval,
            ),
        ];
        let record_directory = RecordDirectory::new("numeric_bounds");
        for (change, argument) in cases {
            let mut config = fixture_config(record_directory.0.clone());
            change(&mut config);
            assert_eq!(
                ffi::OpenConfig::new(&config).err(),
                Some(MediaError::InvalidArgument(*argument))
            );
        }
        let mut config = fixture_config(record_directory.0.clone());
        config.preview_enabled = false;
        config.max_preview_bytes = 0;
        config.record_capacity = 256;
        config.queue_max_buffers = 64;
        config.mux_batch_timeout_us = i32::MAX as u32;
        let prepared = ffi::OpenConfig::new(&config).unwrap();
        assert_eq!(prepared.raw.preview_enabled, 0);
        assert_eq!(prepared.raw.record_capacity, 256);
        assert_eq!(prepared.raw.queue_max_buffers, 64);
        assert_eq!(prepared.raw.mux_batch_timeout_us, i32::MAX as u32);
        for interval in [0, 7, 86400] {
            config.rtsp_reconnect_interval_sec = interval;
            assert_eq!(
                ffi::OpenConfig::new(&config)
                    .unwrap()
                    .raw
                    .rtsp_reconnect_interval_sec,
                interval
            );
        }
    }

    #[test]
    fn roster_binding_scheme_prefix_and_path_admission_are_explicit() {
        let record_directory = RecordDirectory::new("path_admission");
        let mut config = fixture_config(record_directory.0.clone());
        config.sources.clear();
        assert_eq!(
            ffi::OpenConfig::new(&config).err(),
            Some(MediaError::InvalidArgument(MediaArgument::Sources))
        );
        for index in 0..=MEDIA_MAX_SOURCES {
            config.sources.push(SourceConfig {
                source_id: index as u32,
                binding: MediaBinding {
                    token: index as u64 + 1,
                    generation: 7,
                    epoch: 11,
                },
                uri: "rtsp://test.invalid/source".into(),
                record_prefix: format!("source-{index}"),
            });
            if index < MEDIA_MAX_SOURCES {
                assert_eq!(
                    ffi::OpenConfig::new(&config).unwrap().raw.source_count,
                    index as u32 + 1
                );
            }
        }
        assert_eq!(
            ffi::OpenConfig::new(&config).err(),
            Some(MediaError::InvalidArgument(MediaArgument::Sources))
        );
        config.sources.pop();
        config.sources[1].binding.token = 1;
        assert_eq!(
            ffi::OpenConfig::new(&config).err(),
            Some(MediaError::InvalidArgument(MediaArgument::Binding))
        );
        config.sources[1].binding.token = 2;
        config.sources[1].record_prefix = "source-0".into();
        assert_eq!(
            ffi::OpenConfig::new(&config).err(),
            Some(MediaError::InvalidArgument(MediaArgument::RecordPrefix))
        );
        let mut config = fixture_config(record_directory.0.clone());
        config.allow_file_uris = false;
        assert_eq!(
            ffi::OpenConfig::new(&config).err(),
            Some(MediaError::InvalidArgument(MediaArgument::Uri))
        );
        config.sources[0].uri = "https://test.invalid/source".into();
        assert_eq!(
            ffi::OpenConfig::new(&config).err(),
            Some(MediaError::InvalidArgument(MediaArgument::Uri))
        );
        config.sources[0].uri = "rtsps://test.invalid/source".into();
        for prefix in [
            "".to_owned(),
            ".".into(),
            "..".into(),
            "bad/name".into(),
            "nul\0".into(),
            "x".repeat(128),
        ] {
            config.sources[0].record_prefix = prefix;
            assert_eq!(
                ffi::OpenConfig::new(&config).err(),
                Some(MediaError::InvalidArgument(MediaArgument::RecordPrefix))
            );
        }
        config.sources[0].record_prefix = "x".repeat(127);
        assert!(ffi::OpenConfig::new(&config).is_ok());
        config.sources[0].source_id = u32::MAX;
        assert_eq!(
            ffi::OpenConfig::new(&config).err(),
            Some(MediaError::InvalidArgument(MediaArgument::SourceId))
        );
        config.sources[0].source_id = 0;
        config.sources[0].binding.token = 0;
        assert_eq!(
            ffi::OpenConfig::new(&config).err(),
            Some(MediaError::InvalidArgument(MediaArgument::Binding))
        );
        config.sources[0].binding.token = 1;
        config.record_directory = OsString::from_vec(b"/private\0path".to_vec()).into();
        assert_eq!(
            ffi::OpenConfig::new(&config).err(),
            Some(MediaError::InvalidArgument(MediaArgument::RecordDirectory))
        );
    }

    #[test]
    fn request_deadline_and_recording_arithmetic_is_bounded() {
        assert_eq!(
            ffi::request(0),
            Err(MediaError::InvalidArgument(MediaArgument::RequestId))
        );
        assert!(ffi::request(u64::MAX).is_ok());
        for value in [0, 60001, u32::MAX] {
            assert!(ffi::budget(value, MediaArgument::ShutdownBudget).is_err());
            assert!(ffi::budget(value, MediaArgument::PreviewTimeout).is_err());
        }
        for value in [1, 60000] {
            assert!(ffi::budget(value, MediaArgument::ShutdownBudget).is_ok());
        }
        for (cache, back, forward) in [(0, 0, 1), (30, 30, 1), (30, u32::MAX, 1), (30, 2, 0)] {
            assert_eq!(
                ffi::recording_window(cache, back, forward),
                Err(MediaError::InvalidArgument(MediaArgument::RecordingWindow))
            );
        }
        assert!(ffi::recording_window(30, 29, u32::MAX).is_ok());
    }

    #[test]
    fn fatal_status_retains_counters_codes_state_and_stop_timeout() {
        let mut raw = ffi::Status::buffer();
        raw.state = 4;
        raw.source_count = 1;
        raw.stop_timed_out = 1;
        raw.sources[0].frames = 20;
        raw.sources[0].malformed = 3;
        raw.fatal = MediaDiagnostic {
            severity: 2,
            code: 18,
            sdk_domain: 4,
            sdk_code: -17,
        };
        raw.warning = MediaDiagnostic {
            severity: 1,
            code: 10,
            sdk_domain: 0,
            sdk_code: 0,
        };
        let status = status_packet(MediaResult::Fatal, &raw).unwrap();
        assert_eq!(status.result, MediaResult::Fatal);
        assert_eq!(status.state, MediaState::Stopped);
        assert_eq!(status.fatal, raw.fatal);
        assert_eq!(status.warning, raw.warning);
        assert_eq!(
            (status.sources[0].frames, status.sources[0].malformed),
            (20, 3)
        );
        assert!(status.stop_timed_out);
        raw.source_count = 17;
        assert_eq!(
            status_packet(MediaResult::Fatal, &raw).err(),
            Some(MediaError::NativeContract)
        );
    }

    #[test]
    fn pose_and_preview_copies_check_counts_before_slicing() {
        let mut raw = ffi::Pose::buffer();
        raw.tensor_present = 1;
        raw.row_count = 300;
        raw.object_count = 150;
        raw.rows[299][56] = 0.75;
        raw.objects[149].track_id = u64::MAX;
        let packet = pose_packet(&raw).unwrap();
        assert_eq!(packet.rows[299][56], 0.75);
        assert_eq!(packet.objects[149].track_id, u64::MAX);
        raw.object_count = 151;
        assert_eq!(pose_packet(&raw).err(), Some(MediaError::NativeContract));
        raw.object_count = 0;
        raw.row_count = 301;
        assert_eq!(pose_packet(&raw).err(), Some(MediaError::NativeContract));
        raw.row_count = 0;
        raw.tensor_present = 0;
        raw.object_count = 1;
        raw.objects[0].track_id = 53;
        let absent = pose_packet(&raw).unwrap();
        assert!(!absent.tensor_present);
        assert!(absent.rows.is_empty());
        assert_eq!(absent.objects[0].track_id, 53);
        raw.tensor_present = 2;
        assert_eq!(pose_packet(&raw).err(), Some(MediaError::NativeContract));
        let mut preview = ffi::Preview {
            jpeg_bytes: u64::MAX,
            ..Default::default()
        };
        assert_eq!(
            preview_packet(&preview, &[1, 2]).err(),
            Some(MediaError::NativeContract)
        );
        preview.jpeg_bytes = 3;
        assert_eq!(
            preview_packet(&preview, &[1, 2]).err(),
            Some(MediaError::NativeContract)
        );
        preview.jpeg_bytes = 2;
        assert_eq!(
            preview_packet(&preview, &[1, 2]).err(),
            Some(MediaError::NativeContract)
        );
        preview.frame.sequence = 17;
        let published = preview_packet(&preview, &[1, 2]).unwrap();
        assert_eq!(published.frame.sequence, 17);
        assert_eq!(published.jpeg, [1, 2]);
        preview.frame.sequence = 0;
        preview.jpeg_bytes = 0;
        preview.result = 6;
        preview.error = 11;
        preview.request_id = 91;
        let delivered_failure = preview_packet(&preview, &[]).unwrap();
        assert_eq!(
            (
                delivered_failure.result,
                delivered_failure.error,
                delivered_failure.request_id
            ),
            (MediaResult::Fatal, 11, 91)
        );
    }

    #[test]
    fn record_paths_are_bounded_bytes_and_original_tickets_are_not_rewritten() {
        let mut raw = ffi::Record::buffer();
        raw.ticket = RecordTicket {
            binding: MediaBinding {
                token: 73,
                generation: 7,
                epoch: 11,
            },
            request_id: 9,
            source_id: 0,
            session_id: 17,
            session_valid: 1,
            coalesced: 1,
        };
        raw.result = 6;
        raw.error = 17;
        raw.duration_ms = 3000;
        raw.directory[0] = b'/' as c_char;
        raw.filename[0] = 0xff_u8 as c_char;
        raw.filename[1] = b'.' as c_char;
        let receipt = record_receipt(&raw).unwrap();
        assert_eq!(receipt.ticket, raw.ticket);
        assert_eq!(
            (receipt.result, receipt.error, receipt.duration_ms),
            (MediaResult::Fatal, 17, 3000)
        );
        assert_eq!(receipt.filename.as_bytes(), b"\xff.");
        assert_eq!(receipt.directory.as_os_str().as_bytes(), b"/");
        raw.filename.fill(b'x' as c_char);
        assert_eq!(record_receipt(&raw).err(), Some(MediaError::NativeContract));
        raw.filename[MEDIA_PATH_BYTES - 1] = 0;
        assert_eq!(
            bounded_path(&raw.filename).unwrap().as_bytes().len(),
            MEDIA_PATH_BYTES - 1
        );
    }

    #[test]
    #[ignore = "requires actual GPU/SDK and pinned synthetic media fixture"]
    fn actual_media_fixture_pose_tracking_preview_and_shutdown() {
        let _actual = actual_media();
        // record_start is called, so the record directory is explicit too.
        let mut config = fixture_config(directory(Path::new(&required(
            "SEEON_TEST_MEDIA_RECORD_DIR",
        ))));
        config.sources[0].uri = required("SEEON_TEST_MEDIA_URI")
            .into_string()
            .unwrap_or_else(|_| panic!("media fixture URI must be UTF-8"));
        assert!(
            config.sources[0].uri.starts_with("file://"),
            "requires the pinned file fixture, not RTSP"
        );
        config.infer_config_path = required("SEEON_TEST_MEDIA_INFER").into();
        config.tracker_config_path = required("SEEON_TEST_MEDIA_TRACKER").into();
        let binding = config.sources[0].binding;
        let mut media = MediaOwner::open(&config, 5000).expect("actual media admission failed");
        drop(config); // Native must own every config string after open.
        assert_eq!(media.close().err(), Some(MediaError::NotStopped));
        assert_eq!(media.start().unwrap().result, MediaResult::Ok);
        let (mut poses, mut tensors, mut objects, mut sequence) = (0_u64, 0_u64, 0_usize, 0_u64);
        let (mut requested, mut preview_seen, mut file_refused) = (false, false, false);
        let mut preview_floor = 0;
        let deadline = Instant::now() + Duration::from_secs(25);
        poll_until(deadline, "media fixture observation deadline", || {
            let status = media.read_status().unwrap();
            if status.result == MediaResult::Fatal {
                assert_eq!(
                    status.fatal.code, 6,
                    "unexpected sanitized SDK failure: {status:?}"
                );
                return Some(());
            }
            assert_eq!(status.result, MediaResult::Ok);
            match media.poll_pose(0).unwrap() {
                MediaPoll::Ready(pose) => {
                    poses += 1;
                    assert_eq!(pose.frame.binding, binding);
                    assert_eq!((pose.frame.source_id, pose.frame.pts_valid), (0, 1));
                    assert_eq!(
                        (pose.frame.analysis_width, pose.frame.analysis_height),
                        (640, 360)
                    );
                    assert!(pose.frame.sequence > sequence);
                    sequence = pose.frame.sequence;
                    if pose.tensor_present {
                        tensors += 1;
                        assert_eq!(pose.rows.len(), 300);
                        assert!(pose.rows.iter().flatten().all(|value| value.is_finite()));
                    }
                    objects += pose
                        .objects
                        .iter()
                        .filter(|object| object.track_id != u64::MAX)
                        .count();
                    if !file_refused {
                        let stale = MediaBinding {
                            epoch: 12,
                            ..binding
                        };
                        assert_eq!(
                            media
                                .request_preview(0, stale, 1, false, 5000)
                                .unwrap()
                                .result,
                            MediaResult::Stale
                        );
                        match media.record_start(0, binding, 1, 2, 3).unwrap() {
                            MediaPoll::Status(status) => {
                                assert_eq!(status.result, MediaResult::Unsupported)
                            }
                            MediaPoll::Ready(_) => {
                                panic!("file fixture fabricated Smart Record support")
                            }
                        }
                        file_refused = true;
                    }
                    if !requested {
                        let before = media.read_status().unwrap();
                        assert_eq!(before.result, MediaResult::Ok);
                        let status = media.request_preview(0, binding, 1, false, 5000).unwrap();
                        assert!(
                            matches!(status.result, MediaResult::Ok | MediaResult::Busy),
                            "{status:?}"
                        );
                        requested = status.result == MediaResult::Ok;
                        if requested {
                            preview_floor = before.sources[0].frames;
                        }
                    }
                }
                MediaPoll::Status(status) => assert!(
                    matches!(status.result, MediaResult::Empty | MediaResult::Busy),
                    "{status:?}"
                ),
            }
            if requested && !preview_seen {
                match media.poll_preview().unwrap() {
                    MediaPoll::Ready(preview) => {
                        assert_eq!(
                            (preview.result, preview.error, preview.request_id),
                            (MediaResult::Ok, 0, 1)
                        );
                        assert_eq!(preview.frame.binding, binding);
                        assert_eq!((preview.frame.source_id, preview.frame.pts_valid), (0, 1));
                        assert!(preview.frame.sequence > preview_floor);
                        assert_ne!(preview.batch_pts_ns, u64::MAX);
                        assert!(preview.jpeg.len() >= 4);
                        assert!(
                            preview.jpeg.starts_with(&[0xff, 0xd8])
                                && preview.jpeg.ends_with(&[0xff, 0xd9])
                        );
                        preview_seen = true;
                    }
                    MediaPoll::Status(status) => assert!(
                        matches!(status.result, MediaResult::Empty | MediaResult::Busy),
                        "{status:?}"
                    ),
                }
            }
            (poses >= 20 && tensors >= 20 && preview_seen).then_some(())
        });
        assert_eq!(media.stop(5000).unwrap().result, MediaResult::Ok);
        assert_eq!(media.read_status().unwrap().state, MediaState::Stopped);
        assert_eq!(media.close().unwrap().result, MediaResult::Ok);
        assert_eq!(media.start().err(), Some(MediaError::Closed));
        assert_eq!(media.read_status().err(), Some(MediaError::Closed));
        assert_eq!(media.poll_pose(0).err(), Some(MediaError::Closed));
        assert_eq!(
            media.request_preview(0, binding, 2, false, 5000).err(),
            Some(MediaError::Closed)
        );
        assert_eq!(media.poll_preview().err(), Some(MediaError::Closed));
        assert_eq!(
            media.record_start(0, binding, 2, 2, 3).err(),
            Some(MediaError::Closed)
        );
        assert_eq!(
            media.record_stop(&RecordTicket::default()).err(),
            Some(MediaError::Closed)
        );
        assert_eq!(media.poll_record().err(), Some(MediaError::Closed));
        assert_eq!(media.stop(5000).err(), Some(MediaError::Closed));
        assert_eq!(media.close().err(), Some(MediaError::Closed));
        assert!(
            poses >= 20 && tensors >= 20 && objects > 0 && preview_seen && file_refused,
            "incomplete real observations: poses={poses} tensors={tensors} tracked_objects={objects} preview={preview_seen}"
        );
    }

    #[test]
    #[ignore = "requires actual GPU/SDK, isolated synthetic RTSP and a new owned output directory"]
    fn actual_rtsp_smart_record_reuse_and_shutdown() {
        use std::{
            fs::{File, OpenOptions},
            io::{Read, Write},
            os::unix::fs::MetadataExt,
            sync::mpsc::{self, RecvTimeoutError},
        };
        let _actual = actual_media();

        const SESSIONS: usize = 6;
        const STOP_MS: u32 = 5000;

        fn check_call(status: MediaCallStatus, expected: MediaResult) {
            assert_eq!(status.result, expected, "{status:?}");
            assert!(
                status.fatal.is_none_or(|fatal| fatal.code == 0),
                "{status:?}"
            );
        }

        fn pending(status: MediaCallStatus) {
            assert!(
                matches!(status.result, MediaResult::Empty | MediaResult::Busy),
                "{status:?}"
            );
            assert!(
                status.fatal.is_none_or(|fatal| fatal.code == 0),
                "{status:?}"
            );
        }

        fn live_status(
            media: &mut MediaOwner,
            binding: MediaBinding,
            capacity: u32,
        ) -> MediaStatus {
            let status = media.read_status().expect("real RTSP status");
            assert_eq!(status.result, MediaResult::Ok, "{status:?}");
            assert_eq!(status.fatal.code, 0, "{status:?}");
            assert!(
                matches!(
                    status.state,
                    MediaState::Open | MediaState::Starting | MediaState::Running
                ),
                "{status:?}"
            );
            assert_eq!(status.sources.len(), 1, "{status:?}");
            assert_eq!(status.sources[0].binding, binding, "{status:?}");
            assert!(status.records_reserved <= capacity, "{status:?}");
            assert_eq!(status.capacity_refusals, 0, "{status:?}");
            assert!(!status.stop_timed_out, "{status:?}");
            status
        }

        fn check_frame(frame: FrameIdentity, binding: MediaBinding) {
            assert_eq!(frame.binding, binding);
            assert_eq!(
                (frame.source_id, frame.batch_id, frame.pad_index),
                (0, 0, 0)
            );
            assert_eq!(frame.pts_valid, 1);
            assert_ne!(frame.pts_ns, u64::MAX);
            assert!(frame.frame_number >= 0);
            assert!(frame.source_width > 0 && frame.source_height > 0);
            assert_eq!((frame.analysis_width, frame.analysis_height), (640, 360));
        }

        fn observe_ticket(
            ticket: RecordTicket,
            binding: MediaBinding,
            request_id: u64,
            session: &mut Option<u32>,
        ) {
            assert_eq!(ticket.binding, binding);
            assert_eq!((ticket.source_id, ticket.request_id), (0, request_id));
            assert_eq!(ticket.coalesced, 0, "serial recordings must not coalesce");
            assert!(ticket.session_valid <= 1);
            if ticket.session_valid == 1 {
                // Zero is a valid SDK session; only the explicit flag admits it.
                assert_ne!(ticket.session_id, u32::MAX);
                if let Some(known) = *session {
                    assert_eq!(ticket.session_id, known, "SDK session changed");
                }
                *session = Some(ticket.session_id);
            }
        }

        fn no_record_receipt(media: &mut MediaOwner) {
            match media.poll_record().expect("poll consumed recording") {
                MediaPoll::Ready(receipt) => {
                    panic!(
                        "duplicate/unrequested recording receipt: {:?}",
                        receipt.ticket
                    )
                }
                MediaPoll::Status(status) => pending(status),
            }
        }

        fn verify_mp4(receipt: &RecordReceipt, root: &Path) -> (PathBuf, u64) {
            assert!(
                directory(&receipt.directory) == root,
                "record directory escaped the explicit output root"
            );
            let leaf = Path::new(&receipt.filename);
            assert!(
                leaf.components().count() == 1
                    && matches!(leaf.components().next(), Some(Component::Normal(_)))
                    && leaf.file_name() == Some(receipt.filename.as_os_str())
                    && leaf.extension().is_some_and(|extension| extension == "mp4"),
                "record filename is not a single MP4 leaf"
            );
            let path = root.join(leaf);
            let metadata = fs::symlink_metadata(&path).expect("record file metadata");
            assert!(
                metadata.is_file(),
                "record is not a regular nonsymlink file"
            );
            let canonical = fs::canonicalize(&path).expect("record file canonicalization");
            assert!(
                canonical == path && canonical.parent() == Some(root),
                "record file escaped the explicit output root"
            );
            let mut file = File::open(&path).expect("record file open");
            let opened = file.metadata().expect("opened record metadata");
            assert!(opened.is_file() && opened.len() >= 16);
            assert_eq!(
                (opened.dev(), opened.ino()),
                (metadata.dev(), metadata.ino())
            );
            // Bounded closed-file smoke check only: not a seal, publication,
            // full MP4 validation or a claim of crash durability.
            let mut prefix = [0_u8; 12];
            file.read_exact(&mut prefix).expect("record MP4 prefix");
            assert_eq!(&prefix[4..8], b"ftyp");
            let box_bytes = u64::from(u32::from_be_bytes(prefix[..4].try_into().unwrap()));
            assert!(
                (16..=opened.len()).contains(&box_bytes),
                "invalid ftyp size"
            );
            (canonical, opened.len())
        }

        let overall_deadline = Instant::now() + Duration::from_secs(60);
        let operation_deadline = overall_deadline - Duration::from_millis(u64::from(STOP_MS));
        std::thread::scope(|scope| {
            let (finished, completion) = mpsc::channel();
            // A scoped, finite watchdog also bounds a synchronous SDK/file call.
            // It never owns media state. On unwind, the sender drops after the
            // MediaOwner's bounded Drop; the scoped thread is always joined.
            scope.spawn(move || {
                match completion
                    .recv_timeout(overall_deadline.saturating_duration_since(Instant::now()))
                {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => {}
                    Err(RecvTimeoutError::Timeout) => {
                        eprintln!("actual_rtsp_smart_record_reuse_and_shutdown: overall deadline");
                        std::process::exit(1);
                    }
                }
            });

            let root = directory(Path::new(&required("SEEON_TEST_MEDIA_RECORD_DIR")));
            assert!(
                fs::read_dir(&root)
                    .expect("read explicit record directory")
                    .next()
                    .is_none(),
                "record directory must be new and empty"
            );
            let probe_path = root.join(".seeon-rust-write-probe");
            let mut probe = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&probe_path)
                .expect("record directory must be writable");
            probe
                .write_all(b"writable\n")
                .expect("write record directory probe");
            drop(probe);
            fs::remove_file(&probe_path).expect("remove record directory probe");
            let mut config = fixture_config(root.clone());
            config.sources[0].uri = required("SEEON_TEST_RTSP_URI")
                .into_string()
                .unwrap_or_else(|_| panic!("RTSP test URI must be UTF-8"));
            assert!(
                config.sources[0].uri.starts_with("rtsp://")
                    && config.sources[0].uri.len() > "rtsp://".len(),
                "requires an explicit rtsp:// synthetic fixture URI"
            );
            config.infer_config_path = required("SEEON_TEST_MEDIA_INFER").into();
            config.tracker_config_path = required("SEEON_TEST_MEDIA_TRACKER").into();
            for path in [&config.infer_config_path, &config.tracker_config_path] {
                assert!(
                    fs::metadata(path)
                        .expect("existing RTSP test inference/tracker config")
                        .is_file(),
                    "RTSP inference/tracker config must be an existing file"
                );
            }
            config.mux_live_source = true;
            config.allow_file_uris = false;
            let binding = config.sources[0].binding;
            let capacity = config.record_capacity;
            assert_eq!((capacity, config.record_cache_seconds), (4, 30));
            let max_preview_bytes = config.max_preview_bytes as usize;
            let mut media = MediaOwner::open(&config, STOP_MS).expect("actual RTSP admission");
            drop(config); // Every string must be owned natively after open.
            assert_eq!(media.close().err(), Some(MediaError::NotStopped));
            check_call(media.start().expect("start actual RTSP"), MediaResult::Ok);

            let (mut poses, mut tensors, mut tracked_objects, mut tracked_tensor_frames) =
                (0_u64, 0_u64, 0_u64, 0_u64);
            let mut last_frame: Option<FrameIdentity> = None;
            let mut preview_frame: Option<FrameIdentity> = None;
            let (mut stale_checked, mut preview_requested) = (false, false);
            let mut preview_floor = 0;
            let acquisition_deadline =
                operation_deadline.min(Instant::now() + Duration::from_secs(10));
            poll_until(acquisition_deadline, "RTSP acquisition deadline", || {
                drop(live_status(&mut media, binding, capacity));
                match media.poll_pose(0).expect("poll actual RTSP pose") {
                    MediaPoll::Ready(pose) => {
                        check_frame(pose.frame, binding);
                        assert!(pose.frame.sequence > 0);
                        if let Some(previous) = last_frame {
                            assert!(pose.frame.sequence > previous.sequence);
                            assert!(pose.frame.frame_number > previous.frame_number);
                            assert!(pose.frame.pts_ns > previous.pts_ns);
                            assert_eq!(
                                (pose.frame.source_width, pose.frame.source_height),
                                (previous.source_width, previous.source_height)
                            );
                        }
                        last_frame = Some(pose.frame);
                        poses += 1;
                        if pose.tensor_present {
                            assert_eq!(pose.rows.len(), MEDIA_POSE_ROWS);
                            assert!(pose.rows.iter().flatten().all(|value| value.is_finite()));
                            tensors += 1;
                        } else {
                            assert!(pose.rows.is_empty());
                        }
                        assert!(pose.objects.len() <= MEDIA_MAX_OBJECTS);
                        let mut tracked = 0;
                        for object in &pose.objects {
                            assert!(
                                [
                                    object.left,
                                    object.top,
                                    object.width,
                                    object.height,
                                    object.confidence
                                ]
                                .iter()
                                .all(|value| value.is_finite())
                            );
                            assert!(object.width > 0.0 && object.height > 0.0);
                            if object.track_id != u64::MAX {
                                tracked += 1;
                            }
                        }
                        tracked_objects += tracked;
                        if pose.tensor_present
                            && tracked > 0
                            && pose.rows.iter().flatten().any(|&value| value != 0.0)
                        {
                            tracked_tensor_frames += 1;
                        }
                    }
                    MediaPoll::Status(status) => pending(status),
                }
                if last_frame.is_some() && !stale_checked {
                    let stale = MediaBinding {
                        epoch: binding.epoch + 1,
                        ..binding
                    };
                    check_call(
                        media
                            .request_preview(0, stale, 1, false, STOP_MS)
                            .expect("stale epoch preview request"),
                        MediaResult::Stale,
                    );
                    match media
                        .record_start(0, stale, 1, 0, 1)
                        .expect("stale epoch recording")
                    {
                        MediaPoll::Status(status) => check_call(status, MediaResult::Stale),
                        MediaPoll::Ready(_) => panic!("stale epoch recording was admitted"),
                    }
                    let status = live_status(&mut media, binding, capacity);
                    assert_eq!(status.records_reserved, 0, "{status:?}");
                    assert_eq!(status.sources[0].active_record.request_id, 0, "{status:?}");
                    no_record_receipt(&mut media);
                    stale_checked = true;
                }
                if stale_checked && !preview_requested {
                    let before = live_status(&mut media, binding, capacity);
                    let status = media
                        .request_preview(0, binding, 2, false, STOP_MS)
                        .expect("actual RTSP preview request");
                    if status.result == MediaResult::Busy {
                        check_call(status, MediaResult::Busy);
                    } else {
                        check_call(status, MediaResult::Ok);
                        preview_requested = true;
                        preview_floor = before.sources[0].frames;
                    }
                }
                if preview_requested && preview_frame.is_none() {
                    match media.poll_preview().expect("poll actual RTSP preview") {
                        MediaPoll::Ready(preview) => {
                            assert_eq!(
                                (preview.result, preview.error, preview.request_id),
                                (MediaResult::Ok, 0, 2)
                            );
                            check_frame(preview.frame, binding);
                            assert!(preview.frame.sequence > preview_floor);
                            assert_ne!(preview.batch_pts_ns, u64::MAX);
                            assert!((4..=max_preview_bytes).contains(&preview.jpeg.len()));
                            assert!(preview.jpeg.starts_with(&[0xff, 0xd8]));
                            assert!(preview.jpeg.ends_with(&[0xff, 0xd9]));
                            preview_frame = Some(preview.frame);
                            eprintln!(
                                "rtsp_preview bytes={} frame={:?}",
                                preview.jpeg.len(),
                                preview.frame
                            );
                        }
                        MediaPoll::Status(status) => pending(status),
                    }
                }
                (poses >= 20
                    && tensors > 0
                    && tracked_tensor_frames > 0
                    && last_frame
                        .zip(preview_frame)
                        .is_some_and(|(pose, preview)| pose.frame_number >= preview.frame_number))
                .then_some(())
            });
            assert!(
                poses >= 20 && tensors > 0 && tracked_tensor_frames > 0 && stale_checked,
                "RTSP acquisition deadline: poses={poses} tensors={tensors} tracked={tracked_objects} tracked_tensor_frames={tracked_tensor_frames}"
            );
            let pose_frame = last_frame.expect("actual RTSP pose missing");
            let preview = preview_frame.expect("actual RTSP JPEG missing");
            assert!(pose_frame.frame_number >= preview.frame_number);
            assert!(pose_frame.pts_ns >= preview.pts_ns);
            if pose_frame.frame_number == preview.frame_number {
                assert_eq!(pose_frame.pts_ns, preview.pts_ns);
            }
            assert_eq!(
                (pose_frame.source_width, pose_frame.source_height),
                (preview.source_width, preview.source_height)
            );
            eprintln!(
                "rtsp_acquired poses={poses} tensors={tensors} tracked={tracked_objects} tracked_tensor_frames={tracked_tensor_frames} status={:?}",
                live_status(&mut media, binding, capacity)
            );

            let mut files = Vec::with_capacity(SESSIONS);
            let mut last_consumed = 0;
            let mut next_request = 2_u64; // Stale request 1 was never admitted.
            for _ in 0..SESSIONS {
                let admission_deadline =
                    operation_deadline.min(Instant::now() + Duration::from_secs(4));
                let admitted = poll_until(admission_deadline, "record admission deadline", || {
                    drop(live_status(&mut media, binding, capacity));
                    // Opaque caller IDs increase even on a BUSY retry. SDK
                    // session IDs come only from tickets/status/receipts.
                    let request_id = next_request;
                    next_request += 1;
                    match media
                        .record_start(0, binding, request_id, 0, 1)
                        .expect("actual RTSP record start")
                    {
                        MediaPoll::Ready(ticket) => {
                            assert_eq!(ticket.request_id, request_id);
                            Some(ticket)
                        }
                        MediaPoll::Status(status) => {
                            check_call(status, MediaResult::Busy);
                            drop(live_status(&mut media, binding, capacity));
                            None
                        }
                    }
                });
                let mut session = None;
                observe_ticket(admitted, binding, admitted.request_id, &mut session);
                assert!(admitted.request_id > last_consumed);
                eprintln!("rtsp_record_admitted ticket={admitted:?}");
                let receipt_deadline =
                    operation_deadline.min(Instant::now() + Duration::from_secs(4));
                let receipt = poll_until(receipt_deadline, "record completion deadline", || {
                    let status = live_status(&mut media, binding, capacity);
                    let active = status.sources[0].active_record;
                    if active.request_id != 0 {
                        observe_ticket(active, binding, admitted.request_id, &mut session);
                    }
                    match media.poll_record().expect("poll actual RTSP recording") {
                        MediaPoll::Ready(receipt) => Some(receipt),
                        MediaPoll::Status(status) => {
                            pending(status);
                            None
                        }
                    }
                });
                assert_eq!((receipt.result, receipt.error), (MediaResult::Ok, 0));
                assert_eq!(receipt.ticket.session_valid, 1);
                observe_ticket(receipt.ticket, binding, admitted.request_id, &mut session);
                assert!(receipt.duration_ms > 0 && receipt.contains_video);
                assert_eq!(
                    (receipt.width, receipt.height),
                    (pose_frame.source_width, pose_frame.source_height)
                );
                let (path, bytes) = verify_mp4(&receipt, &root);
                assert!(!files.contains(&path), "record output path was reused");
                files.push(path);
                last_consumed = receipt.ticket.request_id;
                eprintln!(
                    "rtsp_record_consumed ticket={:?} duration_ms={} file_bytes={bytes} completed={} capacity={capacity} status={:?}",
                    receipt.ticket,
                    receipt.duration_ms,
                    files.len(),
                    live_status(&mut media, binding, capacity)
                );
                no_record_receipt(&mut media);
            }
            assert_eq!(files.len(), SESSIONS);
            let final_status = live_status(&mut media, binding, capacity);
            eprintln!("rtsp_six_consumed capacity={capacity} status={final_status:?}");
            assert_eq!(final_status.state, MediaState::Running);
            assert_eq!(final_status.sources[0].video_linked, 1);
            assert!(final_status.sources[0].frames >= poses);
            assert!(final_status.sources[0].objects > 0);
            assert_eq!(final_status.sources[0].malformed, 0);
            assert_eq!(final_status.sources[0].active_record.request_id, 0);
            assert_eq!(final_status.preview_dropped, 0);
            assert_eq!(final_status.late_record_callbacks, 0);
            assert_eq!(media.close().err(), Some(MediaError::NotStopped));
            assert!(
                Instant::now() < operation_deadline,
                "RTSP operation deadline"
            );
            let stopped = media.stop(STOP_MS).expect("stop actual RTSP");
            eprintln!("rtsp_stop {stopped:?}");
            check_call(stopped, MediaResult::Ok);
            let status = media.read_status().expect("stopped RTSP status");
            eprintln!("rtsp_stopped completed={} status={status:?}", files.len());
            assert_eq!(status.result, MediaResult::Ok, "{status:?}");
            assert_eq!(status.fatal.code, 0, "{status:?}");
            assert_eq!(status.state, MediaState::Stopped);
            assert_eq!(status.callbacks_active, 0);
            assert!(!status.stop_timed_out);
            assert!(status.records_reserved <= capacity);
            assert_eq!(status.capacity_refusals, 0);
            assert_eq!(status.late_record_callbacks, 0);
            assert_eq!(status.sources.len(), 1);
            assert_eq!(status.sources[0].binding, binding);
            assert_eq!(status.sources[0].active_record.request_id, 0);
            match media.poll_record().expect("drain after stop") {
                MediaPoll::Status(status) => check_call(status, MediaResult::Empty),
                MediaPoll::Ready(_) => panic!("duplicate recording receipt after stop"),
            }
            match media.poll_preview().expect("drain preview after stop") {
                MediaPoll::Status(status) => check_call(status, MediaResult::Empty),
                MediaPoll::Ready(_) => panic!("duplicate preview receipt after stop"),
            }
            let closed = media.close().expect("close actual RTSP");
            eprintln!("rtsp_close {closed:?}");
            check_call(closed, MediaResult::Ok);
            assert_eq!(media.read_status().err(), Some(MediaError::Closed));
            assert!(Instant::now() < overall_deadline, "RTSP overall deadline");
            finished
                .send(())
                .expect("RTSP watchdog exited unexpectedly");
        });
    }
}
