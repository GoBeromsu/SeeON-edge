//! `parse_manifest` of `worker/tools/fetch_models/manifest.py`, in the Python
//! check order, and the flat models layout `fetcher.fetch_all` leaves behind:
//! every artifact at `<models root>/<path>`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use super::manifest::{
    Artifact, Bundle, Source, object, optional_list, parse_artifacts, parse_bundle, parse_source,
    relative_path, require,
};
use super::tree::sha256_hex;
use crate::config::lookup;
use crate::json::Json;

/// Which `ManifestError` refused the manifest, or which artifact is not in
/// place.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayoutError {
    /// `manifest must be a JSON object`.
    NotObject,
    /// `manifest: published_bundles is obsolete; ...`.
    PublishedBundles,
    /// `{where}: missing {key!r}`, naming the key.
    Missing(String),
    /// `manifest schema_version {v!r} is not 1`.
    SchemaVersion,
    /// `manifest: sources must be a non-empty object`.
    Sources,
    /// `{where}: must be an object` for a source, artifact or bundle.
    EntryNotObject,
    /// `{where}: revision must be a 40-hex commit`.
    Revision,
    /// `{where}: tag must be a non-empty string`.
    Tag,
    /// `{where}: unsupported kind`.
    Kind,
    /// `{where}: source_locator must be 'owner/name'`.
    SourceLocator,
    /// `manifest: artifacts must be a non-empty list`.
    Artifacts,
    /// `{where}: path must be a plain relative path`.
    Path,
    /// `{where}: unknown source`.
    UnknownSource,
    /// `{where}: size must be a positive integer`.
    Size,
    /// `{where}: sha256 must be 64 lowercase hex characters`.
    Sha256,
    /// `manifest: sidecars must be a list`.
    Sidecars,
    /// `manifest: duplicate destination paths`.
    DuplicateDestinations,
    /// `manifest: bundles must be a list`.
    Bundles,
    /// `{where}: members must be a non-empty list`.
    Members,
    /// `{where}: receipts must be a list`.
    ReceiptsNotList,
    /// `{where}: receipts must be non-empty when present`.
    ReceiptsEmpty,
    /// `{where}: duplicate member paths`.
    DuplicateMembers,
    /// `{where}: receipt paths overlap payload members`.
    ReceiptOverlap,
    /// `{where}: payload must be an object` (or `must be JSON data`).
    Payload,
    /// `{where}: runtime_format must be a non-empty string`.
    RuntimeFormat,
    /// `{where}: sha256 does not match canonical members and payload`.
    BundleDigest,
    /// `manifest: duplicate bundle identities`.
    DuplicateBundles,
    /// The artifact at this path is missing, not a regular file, or not the
    /// recorded size and SHA-256 (`fetcher._matches` is false).
    Absent(String),
}

pub type Parsed<T> = Result<T, LayoutError>;

/// `Manifest`.
#[derive(Clone, Debug, PartialEq)]
pub struct Manifest {
    pub sources: BTreeMap<String, Source>,
    pub artifacts: Vec<Artifact>,
    pub sidecars: Vec<String>,
    pub bundles: Vec<Bundle>,
}

/// `parse_manifest`. Like Python's `version != 1`, a `schema_version` of
/// `1.0` or `true` is accepted.
pub fn parse_manifest(raw: &Json) -> Parsed<Manifest> {
    let raw = object(raw, LayoutError::NotObject)?;
    if lookup(raw, "published_bundles").is_some() {
        return Err(LayoutError::PublishedBundles);
    }
    let supported = match require(raw, "schema_version")? {
        Json::Int(version) => *version == 1,
        Json::Float(version) => *version == 1.0,
        Json::Bool(version) => *version,
        _ => false,
    };
    if !supported {
        return Err(LayoutError::SchemaVersion);
    }
    let sources = match require(raw, "sources")? {
        Json::Object(entries) if !entries.is_empty() => entries,
        _ => return Err(LayoutError::Sources),
    };
    let sources = sources
        .iter()
        .map(|(name, value)| Ok((name.clone(), parse_source(name, value)?)))
        .collect::<Parsed<BTreeMap<_, _>>>()?;
    let artifacts = match require(raw, "artifacts")? {
        Json::Array(items) if !items.is_empty() => parse_artifacts(items, &sources)?,
        _ => return Err(LayoutError::Artifacts),
    };
    let sidecars = optional_list(raw, "sidecars", LayoutError::Sidecars)?;
    let sidecars = sidecars
        .iter()
        .map(relative_path)
        .collect::<Parsed<Vec<_>>>()?;
    let mut destinations = BTreeSet::new();
    let paths = artifacts.iter().map(|artifact| &artifact.path);
    if !paths.chain(&sidecars).all(|path| destinations.insert(path)) {
        return Err(LayoutError::DuplicateDestinations);
    }
    let bundles = optional_list(raw, "bundles", LayoutError::Bundles)?;
    let bundles = bundles
        .iter()
        .map(|bundle| parse_bundle(bundle, &sources))
        .collect::<Parsed<Vec<_>>>()?;
    let mut identities = BTreeSet::new();
    if !bundles
        .iter()
        .all(|bundle| identities.insert(&bundle.sha256))
    {
        return Err(LayoutError::DuplicateBundles);
    }
    Ok(Manifest {
        sources,
        artifacts,
        sidecars,
        bundles,
    })
}

/// The flat layout: each artifact at `<root>/<path>` matched as
/// `fetcher._matches` does (a regular file after symlinks, the recorded
/// size, then the recorded SHA-256; an unreadable file does not match).
/// Returns `(path, sha256, size)` sorted by path. Sidecars are not checked.
pub fn verify_layout(root: &Path, manifest: &Manifest) -> Parsed<Vec<(String, String, i128)>> {
    let mut present = Vec::new();
    for artifact in &manifest.artifacts {
        let path = root.join(&artifact.path);
        let sized = fs::metadata(&path)
            .is_ok_and(|info| info.is_file() && i128::from(info.len()) == artifact.size);
        if !sized || !fs::read(&path).is_ok_and(|body| sha256_hex(&body) == artifact.sha256) {
            return Err(LayoutError::Absent(artifact.path.clone()));
        }
        let (sha256, size) = (artifact.sha256.clone(), artifact.size);
        present.push((artifact.path.clone(), sha256, size));
    }
    present.sort();
    Ok(present)
}
