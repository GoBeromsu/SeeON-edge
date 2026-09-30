//! Per-track 30-row windows, ported from `FallWindowClassifier.update`
//! (`domains/fall/classifier.py`): TTL coasting and eviction, reconnect
//! padding and the stride gate. The model call is not here; each due window
//! goes to the gpu-fall owner instead.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, PoseBbox56History, PoseBbox56Row, ZERO_ROW};
use seeon_worker::trace::DecisionTraceMissingReason as Reason;

/// Python `_TRACK_TTL_FRAMES`.
const TRACK_TTL_FRAMES: u64 = 45;
/// Python `FALL_STRIDE_FRAMES`: a window is due every fifth update.
pub const FALL_STRIDE_FRAMES: u64 = 5;
/// Evicted ids remembered for reconnect padding; Python's set is unbounded,
/// here the oldest id is forgotten first.
const RECONNECT_CAPACITY: usize = 4096;

pub type Window = Box<[PoseBbox56Row; FALL_WINDOW_FRAMES]>;

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
    /// Appends one row per live track (a missing row coasts) and one coasted
    /// row per absent track until its TTL evicts it.
    pub fn update(&mut self, rows: &BTreeMap<u64, PoseBbox56Row>, live: &[u64]) -> Outcome {
        self.counter += 1;
        let live: BTreeSet<u64> = live.iter().copied().collect();
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
            let Some(track) = self.tracks.get_mut(&track_id) else {
                continue;
            };
            if self.counter - track.last_seen >= TRACK_TTL_FRAMES {
                self.tracks.remove(&track_id);
                self.remember(track_id);
            } else {
                track.history.push(None);
            }
        }
        let mut outcome = Outcome::default();
        if !self.counter.is_multiple_of(FALL_STRIDE_FRAMES) {
            outcome.reasons = live
                .iter()
                .map(|&track_id| (track_id, Reason::ClassifierStrideNotDue))
                .collect();
            return outcome;
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
        outcome
    }

    /// The stream-epoch reset of a PTS rollback: a fresh classifier.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    fn remember(&mut self, track_id: u64) {
        if self.reconnect.contains(&track_id) {
            return;
        }
        if self.reconnect.len() == RECONNECT_CAPACITY {
            self.reconnect.pop_front();
        }
        self.reconnect.push_back(track_id);
    }

    /// Whether `track_id` was evicted before; it is forgotten either way.
    fn forget(&mut self, track_id: u64) -> bool {
        let position = self.reconnect.iter().position(|&id| id == track_id);
        position
            .and_then(|position| self.reconnect.remove(position))
            .is_some()
    }
}
