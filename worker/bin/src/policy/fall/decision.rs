//! Borrowed completion receipts tied to the frame actually decided.

use std::collections::{BTreeMap, BTreeSet};

use seeon_deepstream_native::FrameIdentity;
use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::{FallPolicyDecider, FallProbabilities};
use seeon_worker::trace::{DecisionTraceMissingReason as Reason, DecisionTraceSnapshot};

use super::{FallStage, FallStageError};

/// One successful policy update, including zero events or an empty snapshot batch.
///
/// Observers run synchronously immediately after the update, before any later
/// update or coast. They must be receipt-only and non-reentrant; their signature
/// is non-fallible. Borrowed slices are valid only during the callback.
/// Already observed successes remain valid if the enclosing call later fails.
/// Callers must not double-deliver events through both observation and the method
/// return. Failed updates, stale/partial responses, empty flushes and coast do not
/// notify.
pub struct DecisionUpdate<'a> {
    pub frame: FrameIdentity,
    pub frame_index: i64,
    pub time_sec: f64,
    pub snapshots: &'a [DecisionTraceSnapshot],
    pub events: &'a [BusinessEvent],
    decider: &'a FallPolicyDecider,
}

impl DecisionUpdate<'_> {
    /// Read generation before a later update can evict or replace this track.
    pub fn generation_for(&self, track_id: u64) -> Option<u64> {
        self.decider.generation_for(track_id)
    }
}

/// A decision waiting for scores, visible only to the parent fall stage.
pub(super) struct Pending {
    pub(super) frame: FrameIdentity,
    pub(super) frame_index: i64,
    pub(super) time_sec: f64,
    pub(super) live: Vec<u64>,
    pub(super) awaited: BTreeSet<u64>,
    pub(super) probabilities: BTreeMap<u64, FallProbabilities>,
    pub(super) reasons: BTreeMap<u64, Reason>,
}

impl FallStage {
    pub(super) fn decide(
        &mut self,
        pending: Pending,
        observer: &mut dyn FnMut(DecisionUpdate<'_>),
    ) -> Result<Vec<BusinessEvent>, FallStageError> {
        let events = self.decider.update(
            pending.frame_index,
            pending.time_sec,
            &pending.probabilities,
            pending.live.iter().copied(),
            Some(&pending.reasons),
        )?;
        observer(DecisionUpdate {
            frame: pending.frame,
            frame_index: pending.frame_index,
            time_sec: pending.time_sec,
            snapshots: self.decider.last_trace_snapshots(),
            events: &events,
            decider: &self.decider,
        });
        Ok(events)
    }
}
