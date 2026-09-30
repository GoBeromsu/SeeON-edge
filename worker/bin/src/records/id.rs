//! Canonical JSON and content ids of execution records
//! (`shared/events/execution_records.py` `canonical_json`, `_sha256`, and the
//! field validators). Ids are the sha256 hex of the Python-canonical text.

use std::fmt;

use sha2::{Digest, Sha256};

use crate::json::{Json, JsonError, Serialiser};

/// Longest identity string the wire accepts (`_IDENTITY_MAX`).
pub const IDENTITY_MAX: usize = 128;

/// Why a record, gap, batch or receipt was refused
/// (Python raises `ExecutionRecordContractError`). The field name is the
/// Python attribute the refusal names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContractError {
    /// Empty, longer than [`IDENTITY_MAX`] characters, or holds a NUL.
    Identity(&'static str),
    /// Not a non-negative integer.
    NonNegative(&'static str),
    /// Not 64 lowercase hex characters.
    Sha256(&'static str),
    /// A float that is NaN or infinite; the record would not round-trip.
    NonFinite,
    /// A payload object holds a duplicate key.
    DuplicateKey,
    /// A process-scoped kind names a source generation or stream epoch.
    ProcessScope,
    /// A gap ends before it starts.
    GapRange,
    /// A record's camera or boot differs from its batch.
    BatchMismatch,
    /// A batch carries neither records nor gaps.
    EmptyBatch,
    /// A receipt member is missing or has the wrong JSON type.
    Receipt(&'static str),
    /// A receipt `storage_state` outside `STORAGE_STATES`.
    StorageState,
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity(what) | Self::NonNegative(what) | Self::Sha256(what) => {
                write!(f, "invalid {what}")
            }
            Self::NonFinite => f.write_str("payload holds a non-finite float"),
            Self::DuplicateKey => f.write_str("payload holds a duplicate key"),
            Self::ProcessScope => f.write_str("process-scoped kind must use PROCESS_SCOPE"),
            Self::GapRange => f.write_str("invalid gap range"),
            Self::BatchMismatch => f.write_str("record camera/boot does not match batch"),
            Self::EmptyBatch => f.write_str("batch has no records and no gaps"),
            Self::Receipt(what) => write!(f, "receipt invalid: {what}"),
            Self::StorageState => f.write_str("invalid storage_state"),
        }
    }
}

impl std::error::Error for ContractError {}

/// `canonical_json`: sorted keys, `(",", ":")` separators, raw UTF-8 and the
/// Python float repr. A non-finite float is refused before serialising.
pub fn canonical(value: &Json) -> Result<String, ContractError> {
    refuse_non_finite(value)?;
    Serialiser::ExecutionRecords
        .canonical(value)
        .map_err(|error| match error {
            JsonError::NonFinite => ContractError::NonFinite,
            JsonError::DuplicateKey | JsonError::UnsafeValue => ContractError::DuplicateKey,
        })
}

/// sha256 hex of the canonical text of `value` (`_sha256(canonical_json(value))`).
pub fn content_id(value: &Json) -> Result<String, ContractError> {
    canonical(value).map(|text| sha256_hex(text.as_bytes()))
}

/// Lowercase hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest.iter() {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

/// `_identity`: a non-empty string of at most [`IDENTITY_MAX`] characters
/// without NUL.
pub fn identity(value: &str, what: &'static str) -> Result<(), ContractError> {
    let length = value.chars().count();
    if length == 0 || length > IDENTITY_MAX || value.contains('\0') {
        return Err(ContractError::Identity(what));
    }
    Ok(())
}

/// `_sha256_hex`: exactly 64 lowercase hex characters.
pub fn sha256_hex_field(value: &str, what: &'static str) -> Result<(), ContractError> {
    let valid = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid {
        Ok(())
    } else {
        Err(ContractError::Sha256(what))
    }
}

fn refuse_non_finite(value: &Json) -> Result<(), ContractError> {
    match value {
        Json::Float(number) if !number.is_finite() => Err(ContractError::NonFinite),
        Json::Array(items) => items.iter().try_for_each(refuse_non_finite),
        Json::Object(members) => members
            .iter()
            .try_for_each(|(_, member)| refuse_non_finite(member)),
        _ => Ok(()),
    }
}

/// A `Json` string member.
pub(crate) fn text(key: &str, value: &str) -> (String, Json) {
    (key.to_owned(), Json::Str(value.to_owned()))
}

/// A `Json` integer member.
pub(crate) fn int(key: &str, value: impl Into<i128>) -> (String, Json) {
    (key.to_owned(), Json::Int(value.into()))
}

/// A `Json` member that is `null` when `value` is `None`.
pub(crate) fn optional(key: &str, value: Option<Json>) -> (String, Json) {
    (key.to_owned(), value.unwrap_or(Json::Null))
}
