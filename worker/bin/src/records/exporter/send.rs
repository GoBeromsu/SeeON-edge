//! Python `_send`, `_failed`, `_invalid_record`, `_restore_unattempted` and
//! the envelope size probe of `_post`.

use std::collections::VecDeque;

use serde_json::Value;

use super::{EXPORT_HISTORY_LIMIT, Exporter, Item};
use crate::json::Json;
use crate::records::batch::{Batch, MAX_BODY_BYTES, RELAY_PATH, Receipt, StorageState};
use crate::records::id::canonical;
use crate::records::lanes::{
    Drained, EXPORT_FAILED, RECORD_INVALID, account_unsendable_records, single_record_gap,
};
use crate::records::wire::{Gap, Record};
use crate::relay::wire::{
    DeliveryDisposition, DeliveryFailure, classify_http_failure, parse_json_object,
};
use crate::seam::Clock;

impl Exporter {
    /// Python `_send` plus `ExecutionRecordsClient.post_batch`: an encoded
    /// body over `MAX_BODY_BYTES` is PERMANENT OVERSIZE and never posted.
    pub(super) fn send(&mut self, clock: &dyn Clock, drained: Drained) -> bool {
        let batch = Batch::new(
            &drained.camera_id,
            &drained.worker_boot_id,
            self.provenance.clone(),
            drained.records.clone(),
            drained.gaps.clone(),
        );
        let (batch, body) = match batch.and_then(|batch| batch.encode().map(|body| (batch, body))) {
            Ok(encoded) => encoded,
            Err(_) => {
                self.failed(drained, permanent("ENCODING_ERROR"));
                return false;
            }
        };
        if body.len() > MAX_BODY_BYTES {
            self.failed(drained, permanent("OVERSIZE"));
            return false;
        }
        let receipt = match self.client.post_json(RELAY_PATH, &body) {
            Err(error) => Err(DeliveryFailure::from(error)),
            Ok(response) if (200..300).contains(&response.status) => {
                let object = Value::Object(parse_json_object(&response.body));
                match Receipt::from_json(&object) {
                    Ok(receipt) if receipt.batch_id == batch.batch_id() => Ok(receipt),
                    _ => Err(DeliveryFailure::malformed_receipt()),
                }
            }
            Ok(response) => Err(classify_http_failure(
                response.status,
                &response.headers,
                Some(&response.body),
                clock,
            )),
        };
        match receipt {
            Err(failure) => {
                self.failed(drained, failure);
                false
            }
            Ok(receipt) if receipt.storage_state != StorageState::Committed => {
                let unavailable = failure(DeliveryDisposition::Retry, "STORAGE_UNAVAILABLE");
                self.failed(drained, unavailable);
                false
            }
            Ok(receipt) => {
                self.receipts.push_back(receipt);
                while self.receipts.len() > EXPORT_HISTORY_LIMIT {
                    self.receipts.pop_front();
                }
                true
            }
        }
    }

    /// Python `_failed`: the chunk becomes explicit `export-failed` loss.
    pub(super) fn failed(&mut self, drained: Drained, failure: DeliveryFailure) {
        self.had_failure = true;
        self.note(failure);
        self.lanes.note_export_failure(&drained);
    }

    /// Python `_invalid_record`: the record is replaced by its
    /// `record-invalid` gap; the pass itself has not failed.
    pub(super) fn invalid_record(&mut self, drained: &Drained, record: Record, code: &str) -> Gap {
        self.note(permanent(code));
        let fallback = single_record_gap(&record, RECORD_INVALID, record.body().producer_sequence);
        let single = chunk(drained, vec![record], Vec::new());
        let unsendable = single.records.clone();
        account_unsendable_records(single, &unsendable)
            .ok()
            .and_then(|accounted| accounted.gaps.into_iter().next())
            .unwrap_or(fallback)
    }

    fn note(&mut self, failure: DeliveryFailure) {
        self.failures.push_back(failure);
        while self.failures.len() > EXPORT_HISTORY_LIMIT {
            self.failures.pop_front();
        }
    }

    /// The canonical size of a batch with empty member arrays, derived from
    /// the batch owner rather than a second wire definition.
    pub(super) fn envelope_bytes(&self, drained: &Drained) -> Option<usize> {
        let probe = Gap {
            producer: "exporter".to_owned(),
            from_sequence: 0,
            to_sequence: 0,
            from_ns: 0,
            to_ns: 0,
            record_count: 1,
            cause: EXPORT_FAILED.to_owned(),
            scope: None,
        };
        let batch = Batch::new(
            &drained.camera_id,
            &drained.worker_boot_id,
            self.provenance.clone(),
            Vec::new(),
            vec![probe],
        )
        .ok()?;
        let Json::Object(mut members) = batch.to_json() else {
            return None;
        };
        for (key, value) in &mut members {
            if key == "records" || key == "gaps" {
                *value = Json::Array(Vec::new());
            }
        }
        canonical(&Json::Object(members))
            .ok()
            .map(|text| text.len())
    }

    /// Python `_restore_unattempted` over `gaps`, `records`, then `items`.
    pub(super) fn restore(
        &self,
        drained: &Drained,
        gaps: Vec<Gap>,
        records: Vec<Record>,
        items: VecDeque<Item>,
    ) {
        let mut unattempted = chunk(drained, records, gaps);
        for item in items {
            match item {
                Item::Record(record) => unattempted.records.push(record),
                Item::Gap(gap) => unattempted.gaps.push(gap),
            }
        }
        self.lanes.restore_unattempted(unattempted);
    }
}

pub(super) fn permanent(code: &str) -> DeliveryFailure {
    failure(DeliveryDisposition::Permanent, code)
}

/// `DeliveryFailure(disposition, code)` without HTTP details.
fn failure(disposition: DeliveryDisposition, code: &str) -> DeliveryFailure {
    DeliveryFailure {
        disposition,
        code: code.to_owned(),
        status_code: None,
        retry_after_seconds: None,
        transport_error: None,
    }
}

pub(super) fn item_bytes(item: &Item) -> Option<usize> {
    let json = match item {
        Item::Gap(gap) => gap.to_json(),
        Item::Record(record) => record.to_json(),
    };
    canonical(&json).ok().map(|text| text.len())
}

pub(super) fn gap_bytes(gap: &Gap) -> usize {
    canonical(&gap.to_json()).map_or(usize::MAX, |text| text.len())
}

/// A lane of the drained camera and boot holding only `records` and `gaps`.
pub(super) fn chunk(drained: &Drained, records: Vec<Record>, gaps: Vec<Gap>) -> Drained {
    Drained {
        camera_id: drained.camera_id.clone(),
        worker_boot_id: drained.worker_boot_id.clone(),
        records,
        gaps,
    }
}
