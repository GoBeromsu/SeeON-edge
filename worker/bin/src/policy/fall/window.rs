//! Per-track 30-row windows, ported from `FallWindowClassifier.update`
//! (`domains/fall/classifier.py`): TTL coasting and eviction, reconnect
//! padding and the stride gate. The model call is not here; each due window
//! goes to the gpu-fall owner instead.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, PoseBbox56History, PoseBbox56Row, ZERO_ROW};
use seeon_worker::trace::DecisionTraceMissingReason as Reason;

/// Python `_TRACK_TTL_FRAMES`.
pub(super) const TRACK_TTL_FRAMES: u64 = 45;
/// Python `FALL_STRIDE_FRAMES`: a window is due every fifth update.
pub const FALL_STRIDE_FRAMES: u64 = 5;
/// Evicted ids retained for reconnect padding; overflow refuses the eviction.
pub(super) const RECONNECT_CAPACITY: usize = 4096;

pub type Window = Box<[PoseBbox56Row; FALL_WINDOW_FRAMES]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ReconnectCapacityError;

#[derive(Debug, Default)]
struct Track {
    history: PoseBbox56History,
    last_seen: u64,
}

#[derive(Debug, Default)]
pub struct Windows {
    counter: u64,
    tracks: BTreeMap<u64, Track>,
    reconnect: VecDeque<u64>,
}

/// One update: the windows due for scoring in track order, and why each
/// other live track has no score this update.
#[derive(Debug, Default)]
pub struct Outcome {
    pub due: Vec<(u64, Window)>,
    pub reasons: BTreeMap<u64, Reason>,
}

impl Windows {
    /// Check the entire resampled batch before flushing a pending decision.
    /// All updates in one batch share the same live track identities.
    pub(super) fn preflight(
        &self,
        live: &[u64],
        updates: usize,
    ) -> Result<(), ReconnectCapacityError> {
        self.check_capacity(&live.iter().copied().collect(), updates)
    }

    fn check_capacity(
        &self,
        live: &BTreeSet<u64>,
        updates: usize,
    ) -> Result<(), ReconnectCapacityError> {
        if updates == 0 {
            return Ok(());
        }
        let reconnecting = live
            .iter()
            .filter(|id| !self.tracks.contains_key(id) && self.reconnect.contains(id))
            .count();
        let horizon = u128::from(self.counter) + updates as u128;
        let expiring = self
            .tracks
            .iter()
            .filter(|(id, track)| {
                !live.contains(id)
                    && horizon - u128::from(track.last_seen) >= u128::from(TRACK_TTL_FRAMES)
            })
            .count();
        if expiring > RECONNECT_CAPACITY - (self.reconnect.len() - reconnecting) {
            return Err(ReconnectCapacityError);
        }
        Ok(())
    }

    /// Appends one row per live track (a missing row coasts) and one coasted
    /// row per absent track until its TTL evicts it. An expired track stays
    /// live if its reconnect marker cannot be retained.
    pub(super) fn update(
        &mut self,
        rows: &BTreeMap<u64, PoseBbox56Row>,
        live: &[u64],
    ) -> Result<Outcome, ReconnectCapacityError> {
        let live: BTreeSet<u64> = live.iter().copied().collect();
        self.check_capacity(&live, 1)?;
        self.counter += 1;
        for &track_id in &live {
            let reconnected = !self.tracks.contains_key(&track_id) && self.forget(track_id);
            let track = self.tracks.entry(track_id).or_default();
            if reconnected {
                for _ in 1..FALL_WINDOW_FRAMES {
                    track.history.push(Some(ZERO_ROW));
                }
            }
            track.history.push(rows.get(&track_id).copied());
            track.last_seen = self.counter;
        }
        let absent: Vec<u64> = self
            .tracks
            .keys()
            .copied()
            .filter(|track_id| !live.contains(track_id))
            .collect();
        for track_id in absent {
            let Some(expired) = self
                .tracks
                .get(&track_id)
                .map(|track| self.counter - track.last_seen >= TRACK_TTL_FRAMES)
            else {
                continue;
            };
            if expired {
                self.remember(track_id)?;
                self.tracks.remove(&track_id);
            } else if let Some(track) = self.tracks.get_mut(&track_id) {
                track.history.push(None);
            }
        }
        let mut outcome = Outcome::default();
        if !self.counter.is_multiple_of(FALL_STRIDE_FRAMES) {
            outcome.reasons = live
                .iter()
                .map(|&track_id| (track_id, Reason::ClassifierStrideNotDue))
                .collect();
            return Ok(outcome);
        }
        for track_id in live {
            let rows = self.tracks.get(&track_id).map(|track| track.history.rows());
            match rows {
                Some(rows) if rows.len() == FALL_WINDOW_FRAMES => {
                    let mut window: Window = Box::new([ZERO_ROW; FALL_WINDOW_FRAMES]);
                    for (slot, row) in window.iter_mut().zip(rows) {
                        *slot = *row;
                    }
                    outcome.due.push((track_id, window));
                }
                _ => {
                    outcome.reasons.insert(track_id, Reason::ClassifierWarmup);
                }
            }
        }
        Ok(outcome)
    }
    /// The stream-epoch reset of a PTS rollback: a fresh classifier.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    fn remember(&mut self, track_id: u64) -> Result<(), ReconnectCapacityError> {
        if self.reconnect.contains(&track_id) {
            return Ok(());
        }
        if self.reconnect.len() >= RECONNECT_CAPACITY {
            return Err(ReconnectCapacityError);
        }
        self.reconnect.push_back(track_id);
        Ok(())
    }

    /// Whether `track_id` was evicted before; it is forgotten either way.
    fn forget(&mut self, track_id: u64) -> bool {
        let position = self.reconnect.iter().position(|&id| id == track_id);
        position
            .and_then(|position| self.reconnect.remove(position))
            .is_some()
    }
}
