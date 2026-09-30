//! The execution-record relay exporter
//! (`worker/pipeline/diagnostics/exporter.py`). One flush pass drains every
//! camera's lanes, packs gaps then records into bodies of at most
//! `MAX_BODY_BYTES`, and posts them. A failed chunk becomes explicit
//! `export-failed` loss and is never replayed; never-attempted members go
//! back to the lanes. Nothing here blocks a producer or panics.

mod send;

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::records::batch::{MAX_BODY_BYTES, Receipt};
use crate::records::lanes::{Drained, Lanes};
use crate::records::wire::{Gap, Provenance, Record};
use crate::relay::RelayClient;
use crate::relay::wire::DeliveryFailure;
use crate::seam::Clock;

use send::{chunk, gap_bytes, item_bytes, permanent};

/// Python `ExecutionRecordsClient.post_batch` `timeout_sec`.
pub const RELAY_TIMEOUT: Duration = Duration::from_secs(2);
/// Receipts and failures kept for inspection, newest last.
pub const EXPORT_HISTORY_LIMIT: usize = 256;
/// Python `_FAILURE_BACKOFF_SEC`.
pub const FAILURE_BACKOFF: Duration = Duration::from_millis(50);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExporterError {
    /// `batch_max` or `flush_ms` below 1.
    Settings,
}

impl fmt::Display for ExporterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("batch_max and flush_ms must be positive")
    }
}

impl std::error::Error for ExporterError {}

/// One member of a drained batch, in posting order.
enum Item {
    Gap(Gap),
    Record(Record),
}

pub struct Exporter {
    lanes: Arc<Lanes>,
    client: RelayClient,
    provenance: Provenance,
    batch_max: u64,
    flush: Duration,
    receipts: VecDeque<Receipt>,
    failures: VecDeque<DeliveryFailure>,
    had_failure: bool,
}

impl Exporter {
    pub fn new(
        lanes: Arc<Lanes>,
        client: RelayClient,
        provenance: Provenance,
        batch_max: u64,
        flush_ms: u64,
    ) -> Result<Self, ExporterError> {
        if batch_max < 1 || flush_ms < 1 {
            return Err(ExporterError::Settings);
        }
        Ok(Self {
            lanes,
            client,
            provenance,
            batch_max,
            flush: Duration::from_millis(flush_ms),
            receipts: VecDeque::new(),
            failures: VecDeque::new(),
            had_failure: false,
        })
    }

    pub fn receipts(&self) -> Vec<Receipt> {
        self.receipts.iter().cloned().collect()
    }

    pub fn failures(&self) -> Vec<DeliveryFailure> {
        self.failures.iter().cloned().collect()
    }

    /// Waits on the injected clock until some lane has work or `flush_ms`
    /// passes; true when work is ready.
    pub fn wait_for_work(&self, clock: &dyn Clock) -> bool {
        self.lanes.wait_for_work(clock, self.flush, self.batch_max)
    }

    /// The pause before the next pass: `max(50 ms, flush_ms)` after a pass
    /// with a failed chunk, since a gap alone wakes the lanes at once.
    pub fn failure_backoff(&self) -> Option<Duration> {
        self.had_failure.then(|| self.flush.max(FAILURE_BACKOFF))
    }

    /// One pass: drain up to `batch_max` records per camera and post them.
    pub fn flush_once(&mut self, clock: &dyn Clock) {
        self.had_failure = false;
        for (camera_id, worker_boot_id) in self.lanes.cameras_with_work() {
            let drained = self
                .lanes
                .drain_for(&camera_id, &worker_boot_id, self.batch_max);
            if let Ok(Some(drained)) = drained {
                self.post(clock, drained);
            }
        }
    }

    /// Python `_post`: size each member once and send a chunk before it
    /// would exceed the byte cap.
    fn post(&mut self, clock: &dyn Clock, drained: Drained) {
        let Some(envelope) = self.envelope_bytes(&drained) else {
            let failed = chunk(&drained, Vec::new(), drained.gaps.clone());
            self.failed(failed, permanent("ENCODING_ERROR"));
            let records = drained.records.clone();
            self.lanes
                .restore_unattempted(chunk(&drained, records, Vec::new()));
            return;
        };
        let mut items: VecDeque<Item> = drained.gaps.iter().cloned().map(Item::Gap).collect();
        items.extend(drained.records.iter().cloned().map(Item::Record));
        let (mut records, mut gaps) = (Vec::new(), Vec::new());
        let mut encoded = envelope;
        while let Some(item) = items.pop_front() {
            let (item, size) = match item_bytes(&item) {
                Some(size) => (item, size),
                None => match item {
                    Item::Gap(gap) => {
                        self.failed(
                            chunk(&drained, Vec::new(), vec![gap]),
                            permanent("ENCODING_ERROR"),
                        );
                        self.restore(&drained, gaps, records, items);
                        return;
                    }
                    Item::Record(record) => {
                        let gap = self.invalid_record(&drained, record, "ENCODING_ERROR");
                        let size = gap_bytes(&gap);
                        (Item::Gap(gap), size)
                    }
                },
            };
            let (item, size) = match item {
                Item::Record(record) if envelope + size > MAX_BODY_BYTES => {
                    let gap = self.invalid_record(&drained, record, "OVERSIZE");
                    let size = gap_bytes(&gap);
                    (Item::Gap(gap), size)
                }
                other => (other, size),
            };
            if envelope + size > MAX_BODY_BYTES {
                let single = match item {
                    Item::Gap(gap) => chunk(&drained, Vec::new(), vec![gap]),
                    Item::Record(record) => chunk(&drained, vec![record], Vec::new()),
                };
                self.failed(single, permanent("OVERSIZE"));
                self.restore(&drained, gaps, records, items);
                return;
            }
            let mut comma = usize::from(match item {
                Item::Record(_) => !records.is_empty(),
                Item::Gap(_) => !gaps.is_empty(),
            });
            if encoded + size + comma > MAX_BODY_BYTES {
                let chunk = chunk(
                    &drained,
                    std::mem::take(&mut records),
                    std::mem::take(&mut gaps),
                );
                if !self.send(clock, chunk) {
                    items.push_front(item);
                    self.restore(&drained, Vec::new(), Vec::new(), items);
                    return;
                }
                encoded = envelope;
                comma = 0;
            }
            match item {
                Item::Record(record) => records.push(record),
                Item::Gap(gap) => gaps.push(gap),
            }
            encoded += size + comma;
        }
        if !records.is_empty() || !gaps.is_empty() {
            self.send(clock, chunk(&drained, records, gaps));
        }
    }
}
