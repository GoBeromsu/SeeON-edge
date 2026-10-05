//! The per-camera smart-record state machine, ported from `SmartRecordActor`.
//!
//! Recording is always on until `quiesce`: there is no toggle and no restart.
//! Alerts start a recording with a nominal `CAP_SECONDS` content budget
//! (lookback plus forward window), or extend the running one only through its
//! forward window. SDK timer/frame overshoot means this is not a guarantee of
//! measured playback duration. Alerts that race a stop or reach the content
//! boundary wait for the next recording. A missing receipt latches a failure
//! at the independent `CAP_SECONDS` plus native completion grace deadline
//! after acknowledgement, without stopping or forgetting native work. A
//! matching receipt still saves once, but cannot clear that failure. If even
//! FINALIZE_FAILED could not publish, its seal stays owned in FINALIZING and
//! blocks further admission.
//! Other save outcomes return to idle (X18: the Python actor stays in FINALIZING
//! when its sink raises).
//!
//! `quiesce` is a one-way admission guard for process shutdown. It does not
//! stop, finalize, or close the native recording: the media owner still does
//! that and delivers the receipt. After quiescence, no path starts or extends
//! a recording. Alerts that never started stay pending for `take_unstarted`.

pub mod plane;
pub mod types;

#[cfg(test)]
mod failed_seal_tests;

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use seeon_deepstream_native::{MediaResult, RecordTicket};

use super::entry::FLOW_LOOKBACK_MILLIS;
use super::manifest::Contributor;
use super::publish::PublishError;
use super::reserve::SaveOutcome;
use super::time::Utc;
use crate::msg::RecordReceipt;
use crate::seam::Clock;

pub use plane::{CommandPlane, PlaneRefusal, RecordPlane};
pub use types::{Admit, Boundary, ClipSealed, Counters, RecorderError, State};

/// Total nominal requested content budget: lookback plus forward window.
/// This does not guarantee the measured SDK playback duration.
pub const CAP_SECONDS: u32 = 120;
pub const LOOKBACK_SECONDS: u32 = (FLOW_LOOKBACK_MILLIS / 1000) as u32;
/// Native forward request derived from the total nominal content budget.
pub const FORWARD_SECONDS: u32 = CAP_SECONDS - LOOKBACK_SECONDS;
/// Additional receipt-delivery grace beyond `CAP_SECONDS`, independent of content stop.
pub const NATIVE_COMPLETION_GRACE_SECONDS: u32 = 30;
pub const EXTENSION_SECONDS: u32 = 45;
/// The admission budget for each of the active and pending contributor sets.
pub const MAX_PENDING_ALERTS: usize = 128;
const SEALED_MEMORY: usize = 64;
/// Native uses this sentinel for an unassigned SDK session. It is never a session.
const NO_SESSION: u32 = u32::MAX;

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
    receipt_deadline: Duration,
    receipt_overdue: Option<RecordTicket>,
    stop_due: Duration,
    sealed: VecDeque<RecordTicket>,
    /// One bounded in-memory refusal owner, not crash durability.
    failed_seal: Option<ClipSealed>,
    counters: Counters,
    quiesced: bool,
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
            receipt_deadline: Duration::ZERO,
            receipt_overdue: None,
            stop_due: Duration::ZERO,
            sealed: VecDeque::new(),
            failed_seal: None,
            counters: Counters::default(),
            quiesced: false,
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

    pub fn is_quiesced(&self) -> bool {
        self.quiesced
    }

    /// One-way shutdown admission guard. Idempotent. Does not stop, finalize,
    /// or close native recording, and does not clear accepted alerts.
    pub fn quiesce(&mut self) {
        self.quiesced = true;
    }

    /// Drains alerts that never started. Only legal after `quiesce`; does not
    /// steal contributors of a recording the media owner still has to seal.
    pub fn take_unstarted(&mut self) -> Result<Vec<Contributor>, RecorderError> {
        if !self.quiesced {
            return Err(RecorderError::NotQuiesced);
        }
        Ok(std::mem::take(&mut self.pending))
    }

    /// Bounds active and pending admissions before retaining an alert. Drain
    /// actual receipts first; neither alerts nor refusals renew receipt delivery.
    pub fn admit(&mut self, event_ref: &str, detected_at: Utc) -> Result<Admit, RecorderError> {
        self.admit_observed(event_ref, detected_at, None)
    }

    /// Uses the queue owner's cutoff sampled before draining receipts. Content
    /// scheduling still uses current time, not that observation cutoff.
    pub(crate) fn admit_with_receipt_observation_cutoff(
        &mut self,
        event_ref: &str,
        detected_at: Utc,
        receipt_observation_cutoff: Duration,
    ) -> Result<Admit, RecorderError> {
        self.admit_observed(event_ref, detected_at, Some(receipt_observation_cutoff))
    }

    fn admit_observed(
        &mut self,
        event_ref: &str,
        detected_at: Utc,
        receipt_observation_cutoff: Option<Duration>,
    ) -> Result<Admit, RecorderError> {
        if let Some(error) = self.latched_failure() {
            return Err(error);
        }
        if event_ref.trim().is_empty() {
            return Err(RecorderError::BlankEventRef);
        }
        let now = self.clock.monotonic();
        self.check_receipt_deadline(receipt_observation_cutoff.unwrap_or(now))?;
        if self.quiesced {
            // Late alerts stay pending. They are not attributed to media that
            // is already stopping, and they are not started.
            self.enqueue(event_ref, detected_at)?;
            return Ok(Admit::Queued);
        }
        match self.state {
            State::Recording if now < self.hard_deadline => {
                if self.contributors.len() >= MAX_PENDING_ALERTS {
                    return Err(RecorderError::ContributorsFull);
                }
                let alert = Contributor {
                    event_ref: event_ref.to_owned(),
                    detected_at,
                };
                self.counters.sequence += 1;
                self.extend(alert, now);
                Ok(Admit::Extended)
            }
            State::Recording | State::Stopping | State::Finalizing => {
                self.enqueue(event_ref, detected_at)?;
                self.boundary = Boundary::ExtensionRaced;
                self.counters.raced += 1;
                Ok(Admit::Queued)
            }
            State::Idle => {
                self.enqueue(event_ref, detected_at)?;
                Ok(self.start_pending())
            }
        }
    }

    /// Starts waiting alerts when idle and stops a recording whose extension
    /// window closed before the cap. Drain actual receipts before ticking.
    /// Receipt expiry latches a failure even after `quiesce`, without changing
    /// native ownership. After quiescence or failure, starts nothing.
    pub fn tick(&mut self) -> Result<State, RecorderError> {
        self.tick_observed(None)
    }

    /// Keeps receipt expiry at the queue owner's pre-drain cutoff throughout a
    /// sweep, even when another recorder's native work delays this tick.
    pub(crate) fn tick_with_receipt_observation_cutoff(
        &mut self,
        receipt_observation_cutoff: Duration,
    ) -> Result<State, RecorderError> {
        self.tick_observed(Some(receipt_observation_cutoff))
    }

    fn tick_observed(
        &mut self,
        receipt_observation_cutoff: Option<Duration>,
    ) -> Result<State, RecorderError> {
        if let Some(error) = self.latched_failure() {
            return Err(error);
        }
        let now = self.clock.monotonic();
        self.check_receipt_deadline(receipt_observation_cutoff.unwrap_or(now))?;
        if self.quiesced {
            return Ok(self.state);
        }
        match self.state {
            State::Idle if !self.pending.is_empty() => {
                self.start_pending();
            }
            State::Recording if now >= self.stop_due => {
                if now >= self.hard_deadline {
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
        Ok(self.state)
    }

    /// Hands a sealed recording to `save` once. Failed persistence or an
    /// unpublished FINALIZE_FAILED retains the seal in Finalizing.
    /// Waiting alerts cannot start after quiescence
    /// or a latched failure. An earlier receipt failure keeps its priority.
    pub fn on_receipt(
        &mut self,
        receipt: &RecordReceipt,
        save: impl FnOnce(&ClipSealed) -> Result<SaveOutcome, PublishError>,
    ) -> Result<SaveOutcome, RecorderError> {
        let ticket = receipt_identity(self.source_id, &self.sealed, self.ticket, receipt)?;
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
            path: (!receipt.filename.is_empty()).then(|| receipt.directory.join(&receipt.filename)),
        };
        let saved = save(&sealed);
        if self.sealed.len() == SEALED_MEMORY {
            self.sealed.pop_front();
        }
        self.sealed.push_back(ticket);
        self.ticket = None;
        if saved.is_err() || matches!(saved, Ok(SaveOutcome::FinalizeFailed(None))) {
            self.failed_seal = Some(sealed);
            return Err(match self.receipt_overdue {
                Some(ticket) => RecorderError::ReceiptOverdue { ticket },
                None => match saved {
                    Err(error) => RecorderError::Save(error),
                    Ok(_) => RecorderError::Unpublished { ticket },
                },
            });
        }
        self.boundary = Boundary::None;
        self.state = State::Idle;
        if saved.is_ok()
            && !self.quiesced
            && self.receipt_overdue.is_none()
            && !self.pending.is_empty()
        {
            self.start_pending();
        }
        saved.map_err(RecorderError::Save)
    }

    fn latched_failure(&self) -> Option<RecorderError> {
        if let Some(ticket) = self.receipt_overdue {
            return Some(RecorderError::ReceiptOverdue { ticket });
        }
        self.failed_seal
            .as_ref()
            .map(|sealed| RecorderError::Unpublished {
                ticket: sealed.ticket,
            })
    }

    fn check_receipt_deadline(
        &mut self,
        receipt_observation_cutoff: Duration,
    ) -> Result<(), RecorderError> {
        if let Some(ticket) = self.receipt_overdue {
            return Err(RecorderError::ReceiptOverdue { ticket });
        }
        if let Some(ticket) = self.ticket
            && receipt_observation_cutoff >= self.receipt_deadline
        {
            self.receipt_overdue = Some(ticket);
            return Err(RecorderError::ReceiptOverdue { ticket });
        }
        Ok(())
    }

    fn enqueue(&mut self, event_ref: &str, detected_at: Utc) -> Result<(), RecorderError> {
        if self.pending.len() >= MAX_PENDING_ALERTS {
            return Err(RecorderError::PendingFull);
        }
        let alert = Contributor {
            event_ref: event_ref.to_owned(),
            detected_at,
        };
        self.counters.sequence += 1;
        self.pending.push(alert);
        Ok(())
    }

    fn extend(&mut self, alert: Contributor, now: Duration) {
        self.contributors.push(alert);
        let desired = now + seconds(EXTENSION_SECONDS);
        self.stop_due = desired.min(self.hard_deadline);
        if desired >= self.hard_deadline {
            self.boundary = Boundary::ExtensionBounded;
        }
        self.counters.extended += 1;
    }

    fn start_pending(&mut self) -> Admit {
        if self.quiesced || self.receipt_overdue.is_some() || self.failed_seal.is_some() {
            return Admit::Queued;
        }
        let ticket = match self.plane.start(LOOKBACK_SECONDS, FORWARD_SECONDS) {
            Ok(ticket) => ticket,
            Err(refusal) => {
                self.counters.refused += 1;
                return Admit::Refused(refusal);
            }
        };
        // Sample after the successful acknowledgement, never before native admission.
        let now = self.clock.monotonic();
        self.contributors.append(&mut self.pending);
        self.ticket = Some(ticket);
        self.boundary = Boundary::None;
        self.hard_deadline = now + seconds(FORWARD_SECONDS);
        // Receipt delivery keeps the total-budget watchdog, not the earlier content stop.
        self.receipt_deadline = now + seconds(CAP_SECONDS + NATIVE_COMPLETION_GRACE_SECONDS);
        self.stop_due = (now + seconds(EXTENSION_SECONDS)).min(self.hard_deadline);
        self.state = State::Recording;
        Admit::Started(ticket)
    }
}

fn seconds(value: u32) -> Duration {
    Duration::from_secs(u64::from(value))
}

fn same_recording(left: &RecordTicket, right: &RecordTicket) -> bool {
    left.source_id == right.source_id
        && left.binding == right.binding
        && left.request_id == right.request_id
}

fn receipt_identity(
    source_id: u32,
    sealed: &VecDeque<RecordTicket>,
    admitted: Option<RecordTicket>,
    receipt: &RecordReceipt,
) -> Result<RecordTicket, RecorderError> {
    let observed = receipt.ticket;
    if observed.source_id != source_id {
        return Err(RecorderError::WrongSource {
            expected: source_id,
            received: observed.source_id,
        });
    }
    if let Some(previous) = sealed
        .iter()
        .find(|previous| same_recording(previous, &observed))
    {
        return Err(RecorderError::DuplicateRequest(previous.request_id));
    }
    let Some(admitted) = admitted.filter(|admitted| {
        admitted.source_id == observed.source_id && admitted.request_id == observed.request_id
    }) else {
        return Err(RecorderError::UnexpectedRequest(observed.request_id));
    };
    if observed.binding.generation != admitted.binding.generation {
        return Err(RecorderError::WrongGeneration {
            expected: admitted.binding.generation,
            received: observed.binding.generation,
        });
    }
    if observed.binding.epoch != admitted.binding.epoch {
        return Err(RecorderError::WrongEpoch {
            expected: admitted.binding.epoch,
            received: observed.binding.epoch,
        });
    }
    if observed.binding.token != admitted.binding.token {
        return Err(RecorderError::WrongBinding {
            expected: admitted.binding,
            received: observed.binding,
        });
    }
    if observed.session_valid > 1 || admitted.session_valid > 1 {
        return Err(RecorderError::InvalidReceiptTicket);
    }
    if (observed.session_valid == 1 && observed.session_id == NO_SESSION)
        || (admitted.session_valid == 1 && admitted.session_id == NO_SESSION)
    {
        return Err(RecorderError::InvalidReceiptTicket);
    }
    if admitted.session_valid == 1
        && (observed.session_valid != 1 || observed.session_id != admitted.session_id)
    {
        return Err(
            if observed.session_valid == 1 && observed.session_id != admitted.session_id {
                RecorderError::SessionContradiction {
                    admitted: admitted.session_id,
                    received: observed.session_id,
                }
            } else {
                RecorderError::InvalidReceiptTicket
            },
        );
    }
    let omitted_session = observed.session_valid == 0;
    let successful = receipt.result == MediaResult::Ok;
    if successful && observed.session_valid != 1 {
        return Err(RecorderError::InvalidReceiptTicket);
    }
    let negative_without_session =
        matches!(receipt.result, MediaResult::Stale | MediaResult::Fatal) && omitted_session;
    if !successful && !negative_without_session && observed.session_valid != 1 {
        return Err(RecorderError::InvalidReceiptTicket);
    }
    Ok(observed)
}
