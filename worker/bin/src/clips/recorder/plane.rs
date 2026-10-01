//! The recorder's view of the media plane, and its implementation over the
//! media thread's command channel (`media::Command`).

use std::sync::mpsc::{RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::time::Duration;

use seeon_deepstream_native::{
    MediaBinding, MediaCallStatus, MediaError, MediaPoll, MediaResult, RecordTicket,
};

use crate::media::Command;
use crate::msg::ONESHOT_CAPACITY;

/// Why the media plane did not start or stop a recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneRefusal {
    /// The command channel is full.
    Busy,
    /// The media thread is gone.
    Closed,
    /// No reply within the deadline.
    Timeout,
    Status(MediaCallStatus),
    Media(MediaError),
}

pub trait RecordPlane {
    fn start(
        &mut self,
        lookback_seconds: u32,
        forward_seconds: u32,
    ) -> Result<RecordTicket, PlaneRefusal>;

    /// Stops a recording early; a refusal means the cap will seal it.
    fn stop(&mut self, ticket: &RecordTicket) -> Result<(), PlaneRefusal>;
}

/// One camera's recordings, requested from the media thread.
pub struct CommandPlane {
    commands: SyncSender<Command>,
    source_id: u32,
    binding: MediaBinding,
    next_request: u64,
    deadline: Duration,
}

impl CommandPlane {
    pub fn new(
        commands: SyncSender<Command>,
        source_id: u32,
        binding: MediaBinding,
        deadline: Duration,
    ) -> Self {
        Self {
            commands,
            source_id,
            binding,
            next_request: 0,
            deadline,
        }
    }

    fn send(&self, command: Command) -> Result<(), PlaneRefusal> {
        self.commands
            .try_send(command)
            .map_err(|error| match error {
                TrySendError::Full(_) => PlaneRefusal::Busy,
                TrySendError::Disconnected(_) => PlaneRefusal::Closed,
            })
    }
}

fn waited(error: RecvTimeoutError) -> PlaneRefusal {
    match error {
        RecvTimeoutError::Timeout => PlaneRefusal::Timeout,
        RecvTimeoutError::Disconnected => PlaneRefusal::Closed,
    }
}

impl RecordPlane for CommandPlane {
    fn start(
        &mut self,
        lookback_seconds: u32,
        forward_seconds: u32,
    ) -> Result<RecordTicket, PlaneRefusal> {
        self.next_request += 1;
        let (reply, answer) = sync_channel(ONESHOT_CAPACITY);
        self.send(Command::RecordStart {
            source_id: self.source_id,
            binding: self.binding,
            request_id: self.next_request,
            lookback_seconds,
            forward_seconds,
            reply,
        })?;
        match answer.recv_timeout(self.deadline).map_err(waited)? {
            Ok(MediaPoll::Ready(ticket)) => Ok(ticket),
            Ok(MediaPoll::Status(status)) => Err(PlaneRefusal::Status(status)),
            Err(error) => Err(PlaneRefusal::Media(error)),
        }
    }

    fn stop(&mut self, ticket: &RecordTicket) -> Result<(), PlaneRefusal> {
        let (reply, answer) = sync_channel(ONESHOT_CAPACITY);
        self.send(Command::RecordStop {
            ticket: *ticket,
            reply,
        })?;
        match answer.recv_timeout(self.deadline).map_err(waited)? {
            Ok(status) if status.result == MediaResult::Ok => Ok(()),
            Ok(status) => Err(PlaneRefusal::Status(status)),
            Err(error) => Err(PlaneRefusal::Media(error)),
        }
    }
}
