//! Lane bookkeeping behind the `Lanes` mutex: lanes and export-failed gaps
//! kept in insertion order, as the Python dicts are.

use std::collections::VecDeque;

use crate::records::lanes::{LANE_OVERFLOW, gaps_for_records};
use crate::records::wire::{Gap, Record, RecordBody};

pub(super) struct Lane {
    camera_id: String,
    worker_boot_id: String,
    producer: String,
    pub(super) records: VecDeque<Record>,
    pub(super) next_sequence: u64,
    pub(super) overflow: Vec<Record>,
    pub(super) invalid: Vec<Gap>,
}

impl Lane {
    fn is_for(&self, camera_id: &str, worker_boot_id: &str) -> bool {
        self.camera_id == camera_id && self.worker_boot_id == worker_boot_id
    }

    fn has_work(&self) -> bool {
        !(self.records.is_empty() && self.overflow.is_empty() && self.invalid.is_empty())
    }
}

#[derive(Default)]
pub(super) struct State {
    lanes: Vec<Lane>,
    export_failed: Vec<(String, String, Vec<Gap>)>,
}

impl State {
    pub(super) fn lane_mut(&mut self, body: &RecordBody) -> &mut Lane {
        let found = self.lanes.iter().position(|lane| {
            lane.is_for(&body.camera_id, &body.worker_boot_id) && lane.producer == body.producer
        });
        let index = found.unwrap_or_else(|| {
            self.lanes.push(Lane {
                camera_id: body.camera_id.clone(),
                worker_boot_id: body.worker_boot_id.clone(),
                producer: body.producer.clone(),
                records: VecDeque::new(),
                next_sequence: 0,
                overflow: Vec::new(),
                invalid: Vec::new(),
            });
            self.lanes.len() - 1
        });
        &mut self.lanes[index]
    }

    pub(super) fn export_failed(&mut self, camera_id: &str, worker_boot_id: &str, gaps: Vec<Gap>) {
        let found = self
            .export_failed
            .iter_mut()
            .find(|(camera, boot, _)| camera == camera_id && boot == worker_boot_id);
        match found {
            Some((_, _, existing)) => existing.extend(gaps),
            None => {
                let key = (camera_id.to_owned(), worker_boot_id.to_owned());
                self.export_failed.push((key.0, key.1, gaps));
            }
        }
    }

    pub(super) fn has_work(&self) -> bool {
        !self.export_failed.is_empty() || self.lanes.iter().any(Lane::has_work)
    }

    /// True once one camera and boot holds `batch_max` queued records.
    pub(super) fn batch_ready(&self, batch_max: u64) -> bool {
        let mut counts: Vec<(&str, &str, u64)> = Vec::new();
        for lane in &self.lanes {
            let queued = u64::try_from(lane.records.len()).unwrap_or(u64::MAX);
            let key = (lane.camera_id.as_str(), lane.worker_boot_id.as_str());
            let total = match counts.iter_mut().find(|(c, b, _)| (*c, *b) == key) {
                Some(entry) => {
                    entry.2 = entry.2.saturating_add(queued);
                    entry.2
                }
                None => {
                    counts.push((key.0, key.1, queued));
                    queued
                }
            };
            if queued > 0 && total >= batch_max {
                return true;
            }
        }
        false
    }

    pub(super) fn cameras_with_work(&self) -> Vec<(String, String)> {
        let lanes = self.lanes.iter().filter(|lane| lane.has_work());
        let keys = lanes
            .map(|lane| (&lane.camera_id, &lane.worker_boot_id))
            .chain(self.export_failed.iter().map(|(c, b, _)| (c, b)));
        let mut unique: Vec<(String, String)> = Vec::new();
        for (camera, boot) in keys {
            if !unique.iter().any(|(c, b)| c == camera && b == boot) {
                unique.push((camera.clone(), boot.clone()));
            }
        }
        unique
    }

    pub(super) fn queued(&self) -> usize {
        self.lanes.iter().map(|lane| lane.records.len()).sum()
    }

    /// Up to `limit` records lane by lane, then each lane's overflow and
    /// invalid gaps, then the camera's export-failed gaps.
    pub(super) fn drain(
        &mut self,
        camera_id: &str,
        worker_boot_id: &str,
        limit: usize,
    ) -> (Vec<Record>, Vec<Gap>) {
        let mut records = Vec::new();
        let mut gaps = Vec::new();
        for lane in &mut self.lanes {
            if !lane.is_for(camera_id, worker_boot_id) {
                continue;
            }
            let take = lane.records.len().min(limit.saturating_sub(records.len()));
            records.extend(lane.records.drain(..take));
            gaps.extend(gaps_for_records(&lane.overflow, LANE_OVERFLOW));
            lane.overflow.clear();
            gaps.append(&mut lane.invalid);
        }
        let failed = self
            .export_failed
            .iter()
            .position(|(camera, boot, _)| camera == camera_id && boot == worker_boot_id);
        if let Some(index) = failed {
            gaps.extend(self.export_failed.remove(index).2);
        }
        (records, gaps)
    }

    /// Puts one lane's records back at its front; newer queued records they
    /// displace move to the front of that lane's overflow.
    pub(super) fn restore(&mut self, records: Vec<Record>, capacity: usize) {
        let Some(first) = records.first() else {
            return;
        };
        let lane = self.lane_mut(&first.body().clone());
        let mut evicted = Vec::new();
        while lane.records.len() + records.len() > capacity {
            match lane.records.pop_back() {
                Some(record) => evicted.push(record),
                None => break,
            }
        }
        for record in records.into_iter().rev() {
            let next = record.body().producer_sequence.saturating_add(1);
            lane.next_sequence = lane.next_sequence.max(next);
            lane.records.push_front(record);
        }
        evicted.reverse();
        lane.overflow.splice(0..0, evicted);
    }
}
