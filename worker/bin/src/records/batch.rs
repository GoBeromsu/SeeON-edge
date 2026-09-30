//! `WireBatch` and `WireBatchReceipt` (`shared/events/execution_records.py`).
//! The batch id covers camera, boot, provenance, sorted record ids and gaps.

use serde_json::Value;

use crate::json::Json;
use crate::records::id::{ContractError, canonical, content_id, identity, sha256_hex_field, text};
use crate::records::wire::{Gap, Provenance, Record};

/// Relay path of the ingest endpoint, relative to the relay base URL.
pub const RELAY_PATH: &str = "api/v1/relay/execution-records";

/// `MAX_EXECUTION_RECORD_BODY_BYTES`: a larger body is never sent.
pub const MAX_BODY_BYTES: usize = 1024 * 1024;

/// One camera/boot batch with its derived `batch_id`.
#[derive(Clone, Debug, PartialEq)]
pub struct Batch {
    camera_id: String,
    worker_boot_id: String,
    provenance: Provenance,
    records: Vec<Record>,
    gaps: Vec<Gap>,
    batch_id: String,
}

impl Batch {
    /// Validates as `WireBatch.__post_init__` and derives `batch_id`.
    pub fn new(
        camera_id: &str,
        worker_boot_id: &str,
        provenance: Provenance,
        records: Vec<Record>,
        gaps: Vec<Gap>,
    ) -> Result<Self, ContractError> {
        identity(camera_id, "camera_id")?;
        identity(worker_boot_id, "worker_boot_id")?;
        provenance.validate()?;
        gaps.iter().try_for_each(Gap::validate)?;
        if records.is_empty() && gaps.is_empty() {
            return Err(ContractError::EmptyBatch);
        }
        let foreign = |record: &Record| {
            record.body().camera_id != camera_id || record.body().worker_boot_id != worker_boot_id
        };
        if records.iter().any(foreign) {
            return Err(ContractError::BatchMismatch);
        }
        let mut record_ids: Vec<&str> = records.iter().map(Record::record_id).collect();
        record_ids.sort_unstable();
        let identity_json = Json::Object(vec![
            text("camera_id", camera_id),
            text("worker_boot_id", worker_boot_id),
            ("provenance".to_owned(), provenance.to_json()),
            (
                "record_ids".to_owned(),
                Json::Array(
                    record_ids
                        .into_iter()
                        .map(|id| Json::Str(id.to_owned()))
                        .collect(),
                ),
            ),
            (
                "gaps".to_owned(),
                Json::Array(gaps.iter().map(Gap::to_json).collect()),
            ),
        ]);
        let batch_id = content_id(&identity_json)?;
        Ok(Self {
            camera_id: camera_id.to_owned(),
            worker_boot_id: worker_boot_id.to_owned(),
            provenance,
            records,
            gaps,
            batch_id,
        })
    }

    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    pub fn camera_id(&self) -> &str {
        &self.camera_id
    }

    pub fn records(&self) -> &[Record] {
        &self.records
    }

    pub fn gaps(&self) -> &[Gap] {
        &self.gaps
    }

    pub fn to_json(&self) -> Json {
        Json::Object(vec![
            text("batch_id", &self.batch_id),
            text("camera_id", &self.camera_id),
            text("worker_boot_id", &self.worker_boot_id),
            ("provenance".to_owned(), self.provenance.to_json()),
            (
                "records".to_owned(),
                Json::Array(self.records.iter().map(Record::to_json).collect()),
            ),
            (
                "gaps".to_owned(),
                Json::Array(self.gaps.iter().map(Gap::to_json).collect()),
            ),
        ])
    }

    /// `WireBatch.encode`: the canonical UTF-8 body posted to the relay.
    pub fn encode(&self) -> Result<Vec<u8>, ContractError> {
        canonical(&self.to_json()).map(String::into_bytes)
    }
}

/// `STORAGE_STATES`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageState {
    Committed,
    StorageUnavailable,
}

/// `WireBatchReceipt`: the Backend's answer. `Committed` is a diagnostics
/// commit, never a Hub acceptance or event receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    pub batch_id: String,
    pub accepted: u64,
    pub duplicates: u64,
    pub rejected: Vec<(String, String)>,
    pub storage_state: StorageState,
    pub committed_at_ns: u64,
}

impl Receipt {
    /// `WireBatchReceipt.from_json`. A missing `rejected` is empty; each
    /// rejected entry is a `[record_id, why]` pair of strings.
    pub fn from_json(value: &Value) -> Result<Self, ContractError> {
        let object = value.as_object().ok_or(ContractError::Receipt("object"))?;
        let member = |name: &'static str| object.get(name).ok_or(ContractError::Receipt(name));
        let count = |name: &'static str| {
            member(name)?
                .as_u64()
                .ok_or(ContractError::NonNegative(name))
        };
        let batch_id = member("batch_id")?
            .as_str()
            .ok_or(ContractError::Sha256("batch_id"))?;
        sha256_hex_field(batch_id, "batch_id")?;
        let storage_state = match member("storage_state")?.as_str() {
            Some("committed") => StorageState::Committed,
            Some("STORAGE_UNAVAILABLE") => StorageState::StorageUnavailable,
            _ => return Err(ContractError::StorageState),
        };
        let rejected = match object.get("rejected") {
            None => Vec::new(),
            Some(Value::Array(items)) => {
                items.iter().map(rejected_pair).collect::<Result<_, _>>()?
            }
            Some(_) => return Err(ContractError::Receipt("rejected")),
        };
        Ok(Self {
            batch_id: batch_id.to_owned(),
            accepted: count("accepted")?,
            duplicates: count("duplicates")?,
            rejected,
            storage_state,
            committed_at_ns: count("committed_at_ns")?,
        })
    }
}

fn rejected_pair(item: &Value) -> Result<(String, String), ContractError> {
    match item.as_array().map(Vec::as_slice) {
        Some([Value::String(record_id), Value::String(why)]) => {
            Ok((record_id.clone(), why.clone()))
        }
        _ => Err(ContractError::Receipt("rejected")),
    }
}
