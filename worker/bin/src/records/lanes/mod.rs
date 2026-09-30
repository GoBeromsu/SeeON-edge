//! Bounded per-camera execution-record lanes
//! (`worker/pipeline/diagnostics/lanes.py`). One lane per camera, worker
//! boot and producer. A full lane drops the incoming record but still
//! consumes its producer sequence, so the loss reaches the backend as a
//! `lane-overflow` gap on the next drained batch.

mod gaps;
mod state;

use std::fmt;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

pub use gaps::{account_unsendable_records, gaps_for_records, single_record_gap};

use crate::poll::poll_until;
use crate::records::wire::{Gap, Record, RecordBody};
use crate::seam::Clock;
use state::State;

pub const LANE_OVERFLOW: &str = "lane-overflow";
pub const EXPORT_FAILED: &str = "export-failed";
pub const RECORD_INVALID: &str = "record-invalid";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanesError {
    /// `lane_capacity` below 1.
    Capacity,
    /// A drain limit below 1.
    DrainLimit,
    /// An unsendable record that the drained batch does not hold.
    UnknownRecord,
}

impl fmt::Display for LanesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Capacity => "lane_capacity must be at least 1",
            Self::DrainLimit => "drain limit must be at least 1",
            Self::UnknownRecord => "unsendable records must belong to the drained batch",
        })
    }
}

impl std::error::Error for LanesError {}

/// Records and gaps taken from the lanes of one camera and worker boot.
#[derive(Clone, Debug, PartialEq)]
pub struct Drained {
    pub camera_id: String,
    pub worker_boot_id: String,
    pub records: Vec<Record>,
    pub gaps: Vec<Gap>,
}

pub struct Lanes {
    capacity: usize,
    state: Mutex<State>,
}

impl Lanes {
    pub fn new(lane_capacity: u64) -> Result<Self, LanesError> {
        if lane_capacity < 1 {
            return Err(LanesError::Capacity);
        }
        Ok(Self {
            capacity: usize::try_from(lane_capacity).unwrap_or(usize::MAX),
            state: Mutex::new(State::default()),
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Appends or drops without blocking on delivery. The record takes the
    /// lane's next sequence whether it is kept, dropped as overflow, or
    /// refused as invalid; only a kept record returns true.
    pub fn try_emit(&self, record: Record) -> bool {
        let mut state = self.lock();
        let lane = state.lane_mut(record.body());
        let sequence = lane.next_sequence;
        lane.next_sequence = sequence.saturating_add(1);
        let sequenced = Record::new(RecordBody {
            producer_sequence: sequence,
            ..record.body().clone()
        });
        match sequenced {
            Err(_) => {
                lane.invalid
                    .push(single_record_gap(&record, RECORD_INVALID, sequence));
                false
            }
            Ok(sequenced) if lane.records.len() >= self.capacity => {
                lane.overflow.push(sequenced);
                false
            }
            Ok(sequenced) => {
                lane.records.push_back(sequenced);
                true
            }
        }
    }

    /// Returns true once some camera has `batch_max` records or any lane has
    /// work, false when `timeout` of monotonic time passes first.
    pub fn wait_for_work(&self, clock: &dyn Clock, timeout: Duration, batch_max: u64) -> bool {
        let deadline = clock.monotonic().saturating_add(timeout);
        poll_until(clock, deadline, "execution-record work", || {
            let state = self.lock();
            state.batch_ready(batch_max) || state.has_work()
        })
        .is_ok()
    }

    /// Cameras and boots with records, overflow, invalid or export-failed
    /// gaps, deduplicated in first-seen order.
    pub fn cameras_with_work(&self) -> Vec<(String, String)> {
        self.lock().cameras_with_work()
    }

    pub fn queued(&self) -> usize {
        self.lock().queued()
    }

    pub fn has_work(&self) -> bool {
        self.lock().has_work()
    }

    /// Up to `limit` records of one camera and boot, lane by lane, plus every
    /// pending overflow, invalid and export-failed gap; `None` when empty.
    pub fn drain_for(
        &self,
        camera_id: &str,
        worker_boot_id: &str,
        limit: u64,
    ) -> Result<Option<Drained>, LanesError> {
        if limit < 1 {
            return Err(LanesError::DrainLimit);
        }
        let limit = usize::try_from(limit).unwrap_or(usize::MAX);
        let (records, gaps) = self.lock().drain(camera_id, worker_boot_id, limit);
        if records.is_empty() && gaps.is_empty() {
            return Ok(None);
        }
        Ok(Some(Drained {
            camera_id: camera_id.to_owned(),
            worker_boot_id: worker_boot_id.to_owned(),
            records,
            gaps,
        }))
    }

    /// Puts never-attempted records back at the front of their lanes; a
    /// newer record they displace becomes overflow. Unattempted gaps wait
    /// as export-failed gaps of the drained camera.
    pub fn restore_unattempted(&self, drained: Drained) {
        let mut state = self.lock();
        let mut grouped: Vec<Vec<Record>> = Vec::new();
        for record in drained.records {
            let same_lane = |group: &Vec<Record>| {
                group.first().is_some_and(|first| {
                    let (a, b) = (first.body(), record.body());
                    (&a.camera_id, &a.worker_boot_id, &a.producer)
                        == (&b.camera_id, &b.worker_boot_id, &b.producer)
                })
            };
            match grouped.iter().position(same_lane) {
                Some(index) => grouped[index].push(record),
                None => grouped.push(vec![record]),
            }
        }
        for records in grouped {
            state.restore(records, self.capacity);
        }
        if !drained.gaps.is_empty() {
            state.export_failed(&drained.camera_id, &drained.worker_boot_id, drained.gaps);
        }
    }

    /// Records of an attempted chunk that failed become export-failed gaps,
    /// ahead of the chunk's own gaps.
    pub fn note_export_failure(&self, drained: &Drained) {
        let mut gaps = gaps_for_records(&drained.records, EXPORT_FAILED);
        gaps.extend(drained.gaps.iter().cloned());
        if !gaps.is_empty() {
            let mut state = self.lock();
            state.export_failed(&drained.camera_id, &drained.worker_boot_id, gaps);
        }
    }
}
