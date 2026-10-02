//! Fall glue per camera, ported from `FallDomainDecider.update` (`policy.py`
//! L348-397): PTS resampling and rollback reset, the classifier windows, and
//! one `FallPolicyDecider::update` per accepted frame. Scoring is the gpu-fall
//! owner's (`[1,30,56]` to `[1,1]`): due windows go out with `try_send`, and
//! the decision waits for their responses. A request `fall_req_tx` does not
//! take, or a score that fails, is a counted missing observation with reason
//! `AdapterReturnedNoData`, never a low score.

mod decision;
pub mod score;
pub mod window;
#[cfg(test)]
#[path = "fall/window_gate_tests.rs"]
mod window_gate_tests;

pub use decision::DecisionUpdate;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc::SyncSender;

use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::{FallError, FallPolicyDecider};
use seeon_worker::pose_bbox56::{PoseBbox56Row, ZERO_ROW};
use seeon_worker::temporal::{PtsGapTooLargeError, PtsResampler};
use seeon_worker::trace::DecisionTraceMissingReason as Reason;

use super::ingest::Frame;
use crate::msg::{FallRequest, FallResponse};
use decision::Pending;
use score::probabilities;
use window::Windows;

#[derive(Clone, Debug, PartialEq)]
pub enum FallStageError {
    /// The calibration temperature is not finite and positive.
    Temperature,
    /// Python raises `PtsGapTooLargeError` out of the update.
    Gap(PtsGapTooLargeError),
    Policy(FallError),
}

impl From<FallError> for FallStageError {
    fn from(error: FallError) -> Self {
        Self::Policy(error)
    }
}

/// In-process counters; they never reach the wire.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FallCounters {
    /// Windows `fall_req_tx` refused, scores that failed, and scores still
    /// awaited when the next frame arrived.
    pub missing_observations: u64,
    /// Python `resample_gap_rows_total`.
    pub resample_gap_rows: u64,
    /// Responses for no awaited request.
    pub stale_responses: u64,
}

pub struct FallStage {
    decider: FallPolicyDecider,
    temperature: f32,
    resampler: PtsResampler,
    last_pts_ns: Option<i64>,
    windows: Windows,
    pending: Option<Pending>,
    counters: FallCounters,
}

impl FallStage {
    /// `temperature` is the model's calibration temperature, applied as the
    /// NumPy float32 arithmetic of `ort_pose_bbox56.py` does.
    pub fn new(decider: FallPolicyDecider, temperature: f64) -> Result<Self, FallStageError> {
        let narrowed = temperature as f32;
        if !temperature.is_finite() || !narrowed.is_finite() || narrowed <= 0.0 {
            return Err(FallStageError::Temperature);
        }
        Ok(Self {
            decider,
            temperature: narrowed,
            resampler: PtsResampler::default(),
            last_pts_ns: None,
            windows: Windows::default(),
            pending: None,
            counters: FallCounters::default(),
        })
    }

    /// Takes one frame. Events of a decision still pending from the previous
    /// frame come first; its unanswered tracks are missing observations.
    /// Each successful update notifies `observer` under the [`DecisionUpdate`]
    /// contract, even if a later update in this call fails.
    pub fn observe(
        &mut self,
        frame: &Frame,
        requests: &SyncSender<FallRequest>,
        observer: &mut dyn FnMut(DecisionUpdate<'_>),
    ) -> Result<Vec<BusinessEvent>, FallStageError> {
        let seconds = frame.time_sec.unwrap_or(0.0);
        let pts_ns = (seconds * 1e9) as i64;
        if self.last_pts_ns.is_some_and(|last| pts_ns < last) {
            self.resampler = PtsResampler::default();
            self.windows.clear();
        }
        self.last_pts_ns = Some(pts_ns);
        let resampled = self
            .resampler
            .push(pts_ns, &frame.rows)
            .map_err(FallStageError::Gap)?;
        let mut events = self.flush(observer)?;
        if resampled.is_empty() {
            events.extend(self.decider.coast()?);
            return Ok(events);
        }
        let mut pending = Pending {
            frame: frame.identity,
            frame_index: frame.frame_index,
            time_sec: seconds,
            live: frame.live_track_ids.clone(),
            awaited: BTreeSet::new(),
            probabilities: BTreeMap::new(),
            reasons: BTreeMap::new(),
        };
        for row in resampled {
            let Some(rows) = row.value.filter(|_| row.valid != 0) else {
                let zero: BTreeMap<u64, PoseBbox56Row> = frame
                    .live_track_ids
                    .iter()
                    .map(|&id| (id, ZERO_ROW))
                    .collect();
                self.windows.update(&zero, &frame.live_track_ids);
                self.counters.resample_gap_rows += 1;
                continue;
            };
            // Only the last resampled row is valid, so these are the frame's.
            let outcome = self.windows.update(rows, &frame.live_track_ids);
            pending.reasons = outcome.reasons;
            for (track_id, window) in outcome.due {
                let request = FallRequest {
                    frame: frame.identity,
                    track_id,
                    window,
                };
                if requests.try_send(request).is_ok() {
                    pending.awaited.insert(track_id);
                } else {
                    self.counters.missing_observations += 1;
                    pending
                        .reasons
                        .insert(track_id, Reason::AdapterReturnedNoData);
                }
            }
        }
        if pending.awaited.is_empty() {
            events.extend(self.decide(pending, observer)?);
        } else {
            self.pending = Some(pending);
        }
        Ok(events)
    }

    /// Whether this response is for the current pending frame and an awaited track.
    /// Does not consume the response or change counters.
    pub fn expects_response(&self, response: &FallResponse) -> bool {
        self.pending.as_ref().is_some_and(|pending| {
            pending.frame == response.frame && pending.awaited.contains(&response.track_id)
        })
    }

    /// Takes one gpu-fall response; the pending decision is made once its
    /// last awaited score arrives. See [`DecisionUpdate`] for observer duties.
    pub fn consume(
        &mut self,
        response: FallResponse,
        observer: &mut dyn FnMut(DecisionUpdate<'_>),
    ) -> Result<Vec<BusinessEvent>, FallStageError> {
        let temperature = self.temperature;
        let Some(pending) = self.pending.as_mut() else {
            self.counters.stale_responses += 1;
            return Ok(Vec::new());
        };
        if pending.frame != response.frame || !pending.awaited.remove(&response.track_id) {
            self.counters.stale_responses += 1;
            return Ok(Vec::new());
        }
        let probabilities = response
            .score
            .ok()
            .and_then(|score| probabilities(score.logit, temperature));
        match probabilities {
            Some(probabilities) => {
                pending
                    .probabilities
                    .insert(response.track_id, probabilities);
            }
            None => {
                self.counters.missing_observations += 1;
                pending
                    .reasons
                    .insert(response.track_id, Reason::AdapterReturnedNoData);
            }
        }
        if !pending.awaited.is_empty() {
            return Ok(Vec::new());
        }
        match self.pending.take() {
            Some(pending) => self.decide(pending, observer),
            None => Ok(Vec::new()),
        }
    }

    /// Decides a pending frame now, its unanswered tracks as missing.
    /// See [`DecisionUpdate`] for observer duties; an empty flush never notifies.
    pub fn flush(
        &mut self,
        observer: &mut dyn FnMut(DecisionUpdate<'_>),
    ) -> Result<Vec<BusinessEvent>, FallStageError> {
        let Some(mut pending) = self.pending.take() else {
            return Ok(Vec::new());
        };
        for track_id in std::mem::take(&mut pending.awaited) {
            self.counters.missing_observations += 1;
            pending
                .reasons
                .insert(track_id, Reason::AdapterReturnedNoData);
        }
        self.decide(pending, observer)
    }

    /// Python `release_onset`; `false` when the onset was not held.
    pub fn release_onset(&mut self, event: &BusinessEvent) -> Result<bool, FallStageError> {
        Ok(self.decider.release_onset(event)?)
    }

    pub fn counters(&self) -> FallCounters {
        self.counters
    }

    /// Actual outstanding scores for the current pending decision.
    pub fn pending_scores(&self) -> usize {
        self.pending
            .as_ref()
            .map_or(0, |pending| pending.awaited.len())
    }

    /// The wrapped decider, for the Python `FallDomainDecider` read-outs
    /// (`last_trace_snapshots`, `last_update_evaluated`, the switch total).
    pub fn decider(&self) -> &FallPolicyDecider {
        &self.decider
    }
}
