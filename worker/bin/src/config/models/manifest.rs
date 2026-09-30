//! The entries of `worker/tools/fetch_models/manifest.py`: `Source`,
//! `Artifact` and `Bundle` with their `_parse_*` functions and the helpers
//! `parse_manifest` (in `layout`) shares, each in the Python check order.

use std::collections::{BTreeMap, BTreeSet};

use super::layout::{LayoutError, Parsed};
use super::tree::sha256_hex;
use crate::config::{is_hex, is_segment, lookup};
use crate::json::{Json, Serialiser};

type Members = [(String, Json)];

/// The two supported `Source.kind` values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceKind {
    HuggingFace,
    GithubRelease,
}

/// `Source`: `reference` is the Hugging Face revision or the release tag.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Source {
    pub name: String,
    pub kind: SourceKind,
    pub source_locator: String,
    pub reference: String,
}

/// `Artifact`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Artifact {
    pub path: String,
    pub source: Source,
    pub remote_path: String,
    pub size: i128,
    pub sha256: String,
}

impl Artifact {
    /// `Artifact.url` (`Source.url_for(remote_path)`).
    pub fn url(&self) -> String {
        let Source {
            kind,
            source_locator: locator,
            reference,
            ..
        } = &self.source;
        let remote = &self.remote_path;
        match kind {
            SourceKind::HuggingFace => {
                format!("https://huggingface.co/{locator}/resolve/{reference}/{remote}")
            }
            SourceKind::GithubRelease => {
                format!("https://github.com/{locator}/releases/download/{reference}/{remote}")
            }
        }
    }
}

/// `Bundle`.
#[derive(Clone, Debug, PartialEq)]
pub struct Bundle {
    pub sha256: String,
    pub members: Vec<Artifact>,
    pub payload: Json,
    pub receipts: Vec<Artifact>,
    pub runtime_format: Option<String>,
}

pub(super) fn object(value: &Json, error: LayoutError) -> Parsed<&Members> {
    match value {
        Json::Object(members) => Ok(members),
        _ => Err(error),
    }
}

/// `_require`: presence only; a JSON `null` is present.
pub(super) fn require<'a>(members: &'a Members, key: &str) -> Parsed<&'a Json> {
    lookup(members, key).ok_or_else(|| LayoutError::Missing(key.to_owned()))
}

/// `raw.get(key, [])`, which must be a list.
pub(super) fn optional_list<'a>(
    members: &'a Members,
    key: &str,
    error: LayoutError,
) -> Parsed<&'a [Json]> {
    match lookup(members, key) {
        None => Ok(&[]),
        Some(Json::Array(items)) => Ok(items),
        Some(_) => Err(error),
    }
}

/// `_relative_path`.
pub(super) fn relative_path(value: &Json) -> Parsed<String> {
    match value {
        Json::Str(path) if path.split('/').all(is_segment) => Ok(path.clone()),
        _ => Err(LayoutError::Path),
    }
}

fn sha256(members: &Members) -> Parsed<String> {
    match require(members, "sha256")? {
        Json::Str(digest) if is_hex(digest, 64) => Ok(digest.clone()),
        _ => Err(LayoutError::Sha256),
    }
}

/// `_parse_source`.
pub(super) fn parse_source(name: &str, raw: &Json) -> Parsed<Source> {
    let raw = object(raw, LayoutError::EntryNotObject)?;
    let kind = require(raw, "kind")?;
    let locator = require(raw, "source_locator")?;
    let (kind, reference) = match kind {
        Json::Str(kind) if kind == "huggingface" => match require(raw, "revision")? {
            Json::Str(revision) if is_hex(revision, 40) => (SourceKind::HuggingFace, revision),
            _ => return Err(LayoutError::Revision),
        },
        Json::Str(kind) if kind == "github-release" => match require(raw, "tag")? {
            Json::Str(tag) if !tag.is_empty() => (SourceKind::GithubRelease, tag),
            _ => return Err(LayoutError::Tag),
        },
        _ => return Err(LayoutError::Kind),
    };
    match locator {
        Json::Str(locator) if locator.matches('/').count() == 1 && !locator.starts_with('/') => {
            Ok(Source {
                name: name.to_owned(),
                kind,
                source_locator: locator.clone(),
                reference: reference.clone(),
            })
        }
        _ => Err(LayoutError::SourceLocator),
    }
}

/// `_parse_artifact` over a list, in order.
pub(super) fn parse_artifacts(
    raw: &[Json],
    sources: &BTreeMap<String, Source>,
) -> Parsed<Vec<Artifact>> {
    let parse = |raw: &Json| {
        let raw = object(raw, LayoutError::EntryNotObject)?;
        let path = relative_path(require(raw, "path")?)?;
        let source = match require(raw, "source")? {
            Json::Str(name) => sources.get(name).ok_or(LayoutError::UnknownSource)?,
            _ => return Err(LayoutError::UnknownSource),
        };
        let remote_path = relative_path(require(raw, "remote_path")?)?;
        let size = match require(raw, "size")? {
            Json::Int(size) if *size > 0 => *size,
            _ => return Err(LayoutError::Size),
        };
        let sha256 = sha256(raw)?;
        let source = source.clone();
        Ok(Artifact {
            path,
            source,
            remote_path,
            size,
            sha256,
        })
    };
    raw.iter().map(parse).collect()
}

/// `_parse_bundle`, ending with `sha256(canonical_json({members, payload}))`.
pub(super) fn parse_bundle(raw: &Json, sources: &BTreeMap<String, Source>) -> Parsed<Bundle> {
    let raw = object(raw, LayoutError::EntryNotObject)?;
    let sha256 = sha256(raw)?;
    let members = match require(raw, "members")? {
        Json::Array(items) if !items.is_empty() => parse_artifacts(items, sources)?,
        _ => return Err(LayoutError::Members),
    };
    let receipts = match optional_list(raw, "receipts", LayoutError::ReceiptsNotList)? {
        [] if lookup(raw, "receipts").is_some() => return Err(LayoutError::ReceiptsEmpty),
        items => parse_artifacts(items, sources)?,
    };
    let paths = |list: &[Artifact]| -> BTreeSet<String> {
        list.iter().map(|artifact| artifact.path.clone()).collect()
    };
    let (member_paths, receipt_paths) = (paths(&members), paths(&receipts));
    if member_paths.len() != members.len() || receipt_paths.len() != receipts.len() {
        return Err(LayoutError::DuplicateMembers);
    }
    if !member_paths.is_disjoint(&receipt_paths) {
        return Err(LayoutError::ReceiptOverlap);
    }
    let payload = require(raw, "payload")?;
    object(payload, LayoutError::Payload)?;
    let runtime_format = match lookup(raw, "runtime_format") {
        None | Some(Json::Null) => None,
        Some(Json::Str(format)) if !format.is_empty() => Some(format.clone()),
        Some(_) => return Err(LayoutError::RuntimeFormat),
    };
    let entry = |member: &Artifact| {
        Json::Object(vec![
            ("path".to_owned(), Json::Str(member.path.clone())),
            ("sha256".to_owned(), Json::Str(member.sha256.clone())),
            ("size".to_owned(), Json::Int(member.size)),
        ])
    };
    let identity = Json::Object(vec![
        (
            "members".to_owned(),
            Json::Array(members.iter().map(entry).collect()),
        ),
        ("payload".to_owned(), payload.clone()),
    ]);
    let canonical = Serialiser::FetchModelsManifest.canonical(&identity);
    let canonical = canonical.map_err(|_| LayoutError::Payload)?;
    if sha256_hex(canonical.as_bytes()) != sha256 {
        return Err(LayoutError::BundleDigest);
    }
    let payload = payload.clone();
    Ok(Bundle {
        sha256,
        members,
        payload,
        receipts,
        runtime_format,
    })
}
