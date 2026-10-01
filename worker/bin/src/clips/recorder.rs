//! The per-camera smart-record state machine, ported from `SmartRecordActor`.
//!
//! Recording is always on: there is no toggle. Alerts start a recording or
//! extend the running one up to `CAP_SECONDS`; alerts that race a stop wait
//! for the next recording. A sealed recording is handed to a save callback
//! and the recorder returns to idle whatever the save returned (X18: the
//! Python actor stays in FINALIZING when its sink raises).

pub mod plane;
pub mod types;

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use seeon_deepstream_native::RecordTicket;

use super::entry::FLOW_LOOKBACK_MILLIS;
use super::manifest::Contributor;
use super::publish::PublishError;
use super::reserve::SaveOutcome;
use super::time::Utc;
use crate::msg::RecordReceipt;
use crate::seam::Clock;

pub use plane::{CommandPlane, PlaneRefusal, RecordPlane};
pub use types::{Admit, Boundary, ClipSealed, Counters, RecorderError, State};

/// The recording length the media plane is asked for, and the hard deadline.
pub const CAP_SECONDS: u32 = 120;
pub const EXTENSION_SECONDS: u32 = 45;
pub const LOOKBACK_SECONDS: u32 = (FLOW_LOOKBACK_MILLIS / 1000) as u32;
pub const MAX_PENDING_ALERTS: usize = 128;
const SEALED_MEMORY: usize = 64;

pub struct Recorder<P: RecordPlane> {
    source_id: u32,
    plane: P,
    clock: Arc<dyn Clock>,
    state: State,
    boundary: Boundary,
    pending: Vec<Contributor>,
    contributors: Vec<Contributor>,
    ticket: Option<RecordTicket>,
    hard_deadline: Duration,
    stop_due: Duration,
    sealed: VecDeque<u32>,
    counters: Counters,
}

impl<P: RecordPlane> Recorder<P> {
    pub fn new(source_id: u32, plane: P, clock: Arc<dyn Clock>) -> Self {
        Self {
            source_id,
            plane,
            clock,
            state: State::Idle,
            boundary: Boundary::None,
            pending: Vec::new(),
            contributors: Vec::new(),
            ticket: None,
            hard_deadline: Duration::ZERO,
            stop_due: Duration::ZERO,
            sealed: VecDeque::new(),
            counters: Counters::default(),
        }
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn boundary(&self) -> Boundary {
        self.boundary
    }

    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    pub fn counters(&self) -> Counters {
        self.counters
    }

    pub fn admit(&mut self, event_ref: &str, detected_at: Utc) -> Result<Admit, RecorderError> {
        if event_ref.trim().is_empty() {
            return Err(RecorderError::BlankEventRef);
        }
        let alert = Contributor {
            event_ref: event_ref.to_owned(),
            detected_at,
        };
        match self.state {
            State::Recording => {
                self.counters.sequence += 1;
                self.extend(alert);
                Ok(Admit::Extended)
            }
            State::Stopping | State::Finalizing => {
                self.enqueue(alert)?;
                self.boundary = Boundary::ExtensionRaced;
                self.counters.raced += 1;
                Ok(Admit::Queued)
            }
            State::Idle => {
                self.enqueue(alert)?;
                Ok(self.start_pending())
            }
        }
    }

    /// Starts waiting alerts when idle and stops a recording whose extension
    /// window closed before the cap.
    pub fn tick(&mut self) -> State {
        match self.state {
            State::Idle if !self.pending.is_empty() => {
                self.start_pending();
            }
            State::Recording if self.clock.monotonic() >= self.stop_due => {
                if self.stop_due >= self.hard_deadline {
                    self.boundary = Boundary::ExtensionBounded;
                } else if let Some(ticket) = self.ticket {
                    self.state = State::Stopping;
                    if self.plane.stop(&ticket).is_err() {
                        // Early stop unsupported: the cap seals the recording.
                        self.state = State::Recording;
                        self.boundary = Boundary::ExtensionBounded;
                        self.stop_due = self.hard_deadline;
                    }
                }
            }
            _ => {}
        }
        self.state
    }

    /// Hands a sealed recording to `save`, then returns to idle and starts
    /// any alerts that waited, whatever `save` returned.
    pub fn on_receipt(
        &mut self,
        receipt: &RecordReceipt,
        save: impl FnOnce(&ClipSealed) -> Result<SaveOutcome, PublishError>,
    ) -> Result<SaveOutcome, RecorderError> {
        let session = receipt.ticket.session_id;
        if receipt.ticket.source_id != self.source_id {
            return Err(RecorderError::WrongCamera {
                expected: self.source_id,
                received: receipt.ticket.source_id,
            });
        }
        if self.sealed.contains(&session) {
            return Err(RecorderError::DuplicateSealed(session));
        }
        let Some(ticket) = self.ticket.filter(|t| t.session_id == session) else {
            return Err(RecorderError::UnexpectedSession(session));
        };
        if self.state == State::Recording {
            self.boundary = Boundary::ExtensionBounded;
        }
        self.state = State::Finalizing;
        let mut contributors = std::mem::take(&mut self.contributors);
        contributors
            .sort_by(|a, b| (a.detected_at, &a.event_ref).cmp(&(b.detected_at, &b.event_ref)));
        let sealed = ClipSealed {
            ticket,
            result: receipt.result,
            contains_video: receipt.contains_video,
            duration_ms: receipt.duration_ms,
            boundary: self.boundary,
            contributors,
            path: receipt.directory.join(&receipt.filename),
        };
        let saved = save(&sealed);
        if self.sealed.len() == SEALED_MEMORY {
            self.sealed.pop_front();
        }
        self.sealed.push_back(session);
        self.ticket = None;
        self.boundary = Boundary::None;
        self.state = State::Idle;
        if !self.pending.is_empty() {
            self.start_pending();
        }
        saved.map_err(RecorderError::Save)
    }

    fn enqueue(&mut self, alert: Contributor) -> Result<(), RecorderError> {
        if self.pending.len() >= MAX_PENDING_ALERTS {
            return Err(RecorderError::PendingFull);
        }
        self.counters.sequence += 1;
        self.pending.push(alert);
        Ok(())
    }

    fn extend(&mut self, alert: Contributor) {
        self.contributors.push(alert);
        let desired = self.clock.monotonic() + seconds(EXTENSION_SECONDS);
        self.stop_due = desired.min(self.hard_deadline);
        if desired >= self.hard_deadline {
            self.boundary = Boundary::ExtensionBounded;
        }
        self.counters.extended += 1;
    }

    fn start_pending(&mut self) -> Admit {
        let ticket = match self.plane.start(LOOKBACK_SECONDS, CAP_SECONDS) {
            Ok(ticket) => ticket,
            Err(refusal) => {
                self.counters.refused += 1;
                return Admit::Refused(refusal);
            }
        };
        let now = self.clock.monotonic();
        self.contributors.append(&mut self.pending);
        self.ticket = Some(ticket);
        self.boundary = Boundary::None;
        self.hard_deadline = now + seconds(CAP_SECONDS);
        self.stop_due = (now + seconds(EXTENSION_SECONDS)).min(self.hard_deadline);
        self.state = State::Recording;
        Admit::Started(ticket)
    }
}

fn seconds(value: u32) -> Duration {
    Duration::from_secs(u64::from(value))
}
