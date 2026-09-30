//! Gap accounting shared by the lanes and the exporter
//! (`lanes.py` `_gaps_for_records`, `_single_record_gap` and
//! `account_unsendable_records`).

use crate::records::lanes::{Drained, LanesError, RECORD_INVALID};
use crate::records::wire::{Gap, Record};

/// Producer, source generation and stream epoch: the unit a gap covers.
type GapKey<'a> = (&'a str, u64, u64);
/// Runs of consecutive sequences for one key, in first-seen order.
type Runs<'a> = Vec<Vec<&'a Record>>;

/// One gap per run of consecutive sequences, grouped by producer, source
/// generation and stream epoch in first-seen order. The time range is the
/// min/max because dropped items need not be time-ordered.
pub fn gaps_for_records(records: &[Record], cause: &str) -> Vec<Gap> {
    let mut grouped: Vec<(GapKey<'_>, Runs<'_>)> = Vec::new();
    for record in records {
        let body = record.body();
        let key = (
            body.producer.as_str(),
            body.source_generation,
            body.stream_epoch,
        );
        let index = match grouped.iter().position(|(existing, _)| *existing == key) {
            Some(index) => index,
            None => {
                grouped.push((key, Vec::new()));
                grouped.len() - 1
            }
        };
        let runs = &mut grouped[index].1;
        let continues = runs.last().and_then(|run| run.last()).is_some_and(|last| {
            last.body().producer_sequence.checked_add(1) == Some(body.producer_sequence)
        });
        match runs.last_mut() {
            Some(run) if continues => run.push(record),
            _ => runs.push(vec![record]),
        }
    }
    grouped
        .into_iter()
        .flat_map(|((producer, generation, epoch), runs)| {
            runs.into_iter().filter_map(move |run| {
                let first = run.first()?.body();
                let last = run.last()?.body();
                let times = run.iter().map(|item| item.body().observed_at_ns);
                Some(Gap {
                    producer: producer.to_owned(),
                    from_sequence: first.producer_sequence,
                    to_sequence: last.producer_sequence,
                    from_ns: times.clone().min()?,
                    to_ns: times.max()?,
                    record_count: u64::try_from(run.len()).unwrap_or(u64::MAX),
                    cause: cause.to_owned(),
                    scope: Some((generation, epoch)),
                })
            })
        })
        .collect()
}

/// A one-record gap at `sequence`, scoped to the record's stream.
pub fn single_record_gap(record: &Record, cause: &str, sequence: u64) -> Gap {
    let body = record.body();
    Gap {
        producer: body.producer.clone(),
        from_sequence: sequence,
        to_sequence: sequence,
        from_ns: body.observed_at_ns,
        to_ns: body.observed_at_ns,
        record_count: 1,
        cause: cause.to_owned(),
        scope: Some((body.source_generation, body.stream_epoch)),
    }
}

/// Drops `unsendable` records and appends one `record-invalid` gap each,
/// after the existing gaps; the kept records keep their order. A record
/// that is not in the batch is `LanesError::UnknownRecord`.
pub fn account_unsendable_records(
    drained: Drained,
    unsendable: &[Record],
) -> Result<Drained, LanesError> {
    if unsendable.is_empty() {
        return Ok(drained);
    }
    let mut pending: Vec<&str> = unsendable.iter().map(Record::record_id).collect();
    pending.sort_unstable();
    pending.dedup();
    let Drained {
        camera_id,
        worker_boot_id,
        records,
        mut gaps,
    } = drained;
    let mut kept = Vec::with_capacity(records.len());
    for record in records {
        match pending.iter().position(|id| *id == record.record_id()) {
            Some(index) => {
                pending.swap_remove(index);
                let sequence = record.body().producer_sequence;
                gaps.push(single_record_gap(&record, RECORD_INVALID, sequence));
            }
            None => kept.push(record),
        }
    }
    if !pending.is_empty() {
        return Err(LanesError::UnknownRecord);
    }
    Ok(Drained {
        camera_id,
        worker_boot_id,
        records: kept,
        gaps,
    })
}
