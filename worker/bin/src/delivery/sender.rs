//! One sender pass over the delivery queue: Python `EvidenceSender.run_once`
//! (`evidence_sender.py`) called repeatedly while each call acknowledges,
//! exactly as `EvidenceExportRuntime._run_sender` does before it waits.
//! EVENT entries POST to `/alerts`, CLIP entries PUT `clips/{clip_id}`, and
//! snapshot entries POST through [`crate::delivery::snapshot`].

mod base64;
pub mod body;
pub mod outcome;
pub mod state;
mod step;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::Value;

pub use body::{BodyError, ClipRequest, clip_request, event_body};
pub use outcome::{DrainStop, DrainSummary, EntryOutcome, InvalidEntry};
pub use state::{MAX_ENTRY_ATTEMPTS, SenderState};

use crate::delivery::{DeliveryQueue, MAX_ACCEPTED_ENTRIES};
use crate::relay::RelayClient;
use crate::seam::Clock;
use crate::shutdown::ShutdownDeadline;
use step::Pass;

/// Dead-letter status for an entry retired after `MAX_ENTRY_ATTEMPTS`.
pub const EXHAUSTED_STATUS: u16 = 599;
/// Python `_OPERATOR_BLOCKED_RETRY_SECONDS` for CAMERA_MAPPING_MISSING clips.
pub const OPERATOR_BLOCKED_RETRY: Duration = Duration::from_secs(300);
/// Python `_run_sender`'s wait after any step that did not acknowledge.
pub const SENDER_IDLE_WAIT: Duration = Duration::from_secs(1);

/// Run `run_once` steps until one does not acknowledge, nothing is due,
/// shutdown has cut off, the owner stop flag is set, or
/// `MAX_ACCEPTED_ENTRIES` steps have run. Never blocks on the relay beyond
/// the client's timeout and never panics.
///
/// Cutoff is cooperative: a deadline observed before a step starts, or
/// returned by a relay call, stops the pass without acknowledging or
/// rewriting sender state. A filesystem move that has already committed
/// still finishes its bookkeeping. `stop` is owner control, not cutoff: it
/// is read before the next entry starts, and an entry already in progress
/// finishes within the shared deadline.
pub fn drain_pass(
    queue: &DeliveryQueue,
    client: &RelayClient,
    state: &mut SenderState,
    clock: &dyn Clock,
    deadline: &ShutdownDeadline,
    owner_stop: &AtomicBool,
    clip_export_enabled: bool,
) -> DrainSummary {
    let view = client.with_deadline(clock, deadline);
    let pass = Pass {
        queue,
        client: &view,
        clock,
        clip_export_enabled,
    };
    let mut outcomes = Vec::new();
    let stop = loop {
        if owner_stop.load(Ordering::Acquire) {
            break DrainStop::Stopped;
        }
        if outcomes.len() == MAX_ACCEPTED_ENTRIES {
            break DrainStop::StepLimit;
        }
        if view.check().is_err() {
            break DrainStop::Cutoff;
        }
        let entries = match queue.entries() {
            Ok(entries) => entries,
            Err(error) => break DrainStop::Queue(error),
        };
        if owner_stop.load(Ordering::Acquire) {
            break DrainStop::Stopped;
        }
        if view.check().is_err() {
            break DrainStop::Cutoff;
        }
        // The queue's own publish always writes a string `entry_id` (files
        // are named after it). This guards foreign or hand-placed files
        // without one: Python's `_select` would raise `KeyError` on them,
        // and sending one here would end in an acknowledgement the queue
        // refuses, so the same entry would be resent on every pass.
        let addressable: Vec<&Value> = entries
            .iter()
            .filter(|entry| entry.get("entry_id").is_some_and(Value::is_string))
            .collect();
        // Sample before reading the deadline, matching the relay view: a
        // publication made during the clock read is visible to selection.
        let now = clock.monotonic();
        if view.check().is_err() {
            break DrainStop::Cutoff;
        }
        if owner_stop.load(Ordering::Acquire) {
            break DrainStop::Stopped;
        }
        let Some(selected) = state.select(&addressable, now) else {
            break DrainStop::Idle;
        };
        if owner_stop.load(Ordering::Acquire) {
            break DrainStop::Stopped;
        }
        if view.check().is_err() {
            break DrainStop::Cutoff;
        }
        let entry_id = state::entry_id(selected.entry).to_owned();
        let outcome = pass.step(state, selected.entry, &entry_id, selected.reset_deferred);
        outcomes.push((entry_id, outcome));
        if matches!(
            outcomes.last().map(|(_, outcome)| outcome),
            Some(EntryOutcome::Cutoff)
        ) {
            break DrainStop::Cutoff;
        }
        if owner_stop.load(Ordering::Acquire) {
            break DrainStop::Stopped;
        }
        let acknowledged = outcomes
            .last()
            .is_some_and(|(_, outcome)| outcome.is_acknowledged());
        if !acknowledged {
            break DrainStop::Unacknowledged;
        }
    };
    DrainSummary { outcomes, stop }
}
