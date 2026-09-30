//! Execution records: canonical wire shapes and content ids, typed record
//! builders, bounded per-camera lanes, the relay exporter and its
//! composition (design §2.6; Python `worker/pipeline/diagnostics/`).

pub mod batch;
pub mod builder;
pub mod compose;
pub mod exporter;
pub mod id;
pub mod lanes;
pub mod provenance;
pub mod wire;

pub use batch::{Batch, MAX_BODY_BYTES, RELAY_PATH, Receipt, StorageState};
pub use compose::{ComposeError, Composed, compose};
pub use exporter::{EXPORT_HISTORY_LIMIT, Exporter, ExporterError, RELAY_TIMEOUT};
pub use id::ContractError;
pub use lanes::{Drained, EXPORT_FAILED, LANE_OVERFLOW, Lanes, LanesError, RECORD_INVALID};
pub use provenance::{Identities, ProvenanceError, build_provenance};
pub use wire::{Gap, PROCESS_SCOPE, Provenance, Record, RecordBody, RecordKind, TimeQuality};
