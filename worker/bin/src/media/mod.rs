//! Media thread (design §2.3): the one thread that owns the `!Send` Pass 1
//! `MediaOwner`, and the counters it publishes.
//!
//! `msg.rs` has no media command type, so the commands other threads send to
//! the media thread live here. They travel on a bounded
//! `sync_channel(COMMAND_CAPACITY)` that the media loop drains with
//! `try_recv`; each reply is a `sync_channel(ONESHOT_CAPACITY)` answered with
//! `try_send`.

pub mod diagnostics;
pub mod owner;
mod record_stop;
mod release;
pub mod shutdown;

use std::sync::mpsc::SyncSender;

use seeon_deepstream_native::{MediaBinding, MediaCallStatus, MediaError, MediaPoll, RecordTicket};

/// Commands queued for the media thread before it refuses more.
pub const COMMAND_CAPACITY: usize = 8;

pub type RecordStartReply = Result<MediaPoll<RecordTicket>, MediaError>;
pub type CallReply = Result<MediaCallStatus, MediaError>;

pub enum Command {
    /// `MediaOwner::record_start`; a `Ready` ticket is what a later
    /// `RecordStop` names.
    RecordStart {
        source_id: u32,
        binding: MediaBinding,
        request_id: u64,
        lookback_seconds: u32,
        forward_seconds: u32,
        reply: SyncSender<RecordStartReply>,
    },
    /// `MediaOwner::record_stop` for a ticket from `RecordStart`.
    RecordStop {
        ticket: RecordTicket,
        reply: SyncSender<CallReply>,
    },
    /// `MediaOwner::request_preview`; the JPEG arrives on `preview_tx`.
    Preview {
        source_id: u32,
        binding: MediaBinding,
        request_id: u64,
        draw_objects: bool,
        timeout_ms: u32,
        reply: SyncSender<CallReply>,
    },
}
