//! Fall glue per camera, ported from `FallDomainDecider.update` (`policy.py`
//! L348-397): PTS resampling and rollback reset, the classifier windows, and
//! one `FallPolicyDecider::update` per accepted frame. Scoring is the gpu-fall
//! owner's (`[1,30,56]` to `[1,1]`): due windows go out with `try_send`, and
//! the decision waits for their responses. A request `fall_req_tx` does not
//! take, or a score that fails, is a counted missing observation with reason
//! `AdapterReturnedNoData`, never a low score.

pub mod score;
pub mod window;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc::SyncSender;

use seeon_deepstream_native::FrameIdentity;
use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::{FallError, FallPolicyDecider, FallProbabilities};
use seeon_worker::pose_bbox56::{PoseBbox56Row, ZERO_ROW};
use seeon_worker::temporal::{PtsGapTooLargeError, PtsResampler};
use seeon_worker::trace::DecisionTraceMissingReason as Reason;

use super::ingest::Frame;
use crate::msg::{FallRequest, FallResponse};
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

/// A decision waiting for scores.
struct Pending {
    frame: FrameIdentity,
    frame_index: i64,
    time_sec: f64,
    live: Vec<u64>,
    awaited: BTreeSet<u64>,
    probabilities: BTreeMap<u64, FallProbabilities>,
    reasons: BTreeMap<u64, Reason>,
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
    pub fn observe(
        &mut self,
        frame: &Frame,
        requests: &SyncSender<FallRequest>,
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
        let mut events = self.flush()?;
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
            events.extend(self.decide(pending)?);
        } else {
            self.pending = Some(pending);
        }
        Ok(events)
    }

    /// Takes one gpu-fall response; the pending decision is made once its
    /// last awaited score arrives.
    pub fn consume(
        &mut self,
        response: FallResponse,
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
            Some(pending) => self.decide(pending),
            None => Ok(Vec::new()),
        }
    }

    /// Decides a pending frame now, its unanswered tracks as missing.
    pub fn flush(&mut self) -> Result<Vec<BusinessEvent>, FallStageError> {
        let Some(mut pending) = self.pending.take() else {
            return Ok(Vec::new());
        };
        for track_id in std::mem::take(&mut pending.awaited) {
            self.counters.missing_observations += 1;
            pending
                .reasons
                .insert(track_id, Reason::AdapterReturnedNoData);
        }
        self.decide(pending)
    }

    /// Python `release_onset`; `false` when the onset was not held.
    pub fn release_onset(&mut self, event: &BusinessEvent) -> Result<bool, FallStageError> {
        Ok(self.decider.release_onset(event)?)
    }

    pub fn counters(&self) -> FallCounters {
        self.counters
    }

    /// The wrapped decider, for the Python `FallDomainDecider` read-outs
    /// (`last_trace_snapshots`, `last_update_evaluated`, the switch total).
    pub fn decider(&self) -> &FallPolicyDecider {
        &self.decider
    }

    fn decide(&mut self, pending: Pending) -> Result<Vec<BusinessEvent>, FallStageError> {
        let events = self.decider.update(
            pending.frame_index,
            pending.time_sec,
            &pending.probabilities,
            pending.live.iter().copied(),
            Some(&pending.reasons),
        )?;
        Ok(events)
    }
}
