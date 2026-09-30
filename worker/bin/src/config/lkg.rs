//! `WorkerConfigLkgStore` of `worker/runtime/config/lkg_store.py`: the last
//! accepted relay worker config under `<state dir>/config-lkg`, as
//! `current.json` plus one file per directive in `revisions/`, written with
//! the same names and bytes so either implementation reads the other's store.

use std::fs::{self, DirBuilder, File, OpenOptions, Permissions};
use std::io::{self, ErrorKind, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rustix::fs::{FlockOperation, flock};

use super::{lookup, parse_json};
use crate::json::{Json, Serialiser};

/// `CONFIG_HISTORY_RETENTION_COUNT`.
const RETENTION: usize = 50;

static TEMPORARY: AtomicUsize = AtomicUsize::new(0);

/// `RestartDirective`: ordered by generation, then config version, then
/// registry version.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Directive {
    pub generation: i128,
    pub version: i128,
    pub registry: i128,
}

/// `StoredConfigPayload` (its `registry_version` is `directive.registry`).
#[derive(Clone, Debug, PartialEq)]
pub struct StoredConfig {
    pub payload: Json,
    pub directive: Directive,
}

/// Why the store could not be used; Python prints these and carries on.
#[derive(Debug)]
pub enum LkgError {
    /// A filesystem operation failed.
    Io(io::Error),
    /// The payload has no Python `json.dumps` form here (a non-finite float).
    Payload,
    /// `current.json` is not a valid record (`_decode_record`).
    Record,
}

impl From<io::Error> for LkgError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// The store rooted at `<state dir>/config-lkg`.
#[derive(Clone, Debug)]
pub struct LkgStore {
    directory: PathBuf,
}

impl LkgStore {
    pub fn new(state_dir: &Path) -> Self {
        let directory = state_dir.join("config-lkg");
        Self { directory }
    }

    /// `save`: `Ok(false)` when an older directive would replace a newer one.
    pub fn save(&self, payload: &Json, directive: Directive) -> Result<bool, LkgError> {
        let encoded = encode(payload, directive)?;
        let _lock = self.lock()?;
        if self
            .current()?
            .is_some_and(|current| directive < current.directive)
        {
            return Ok(false);
        }
        let revisions = self.directory.join("revisions");
        let Directive {
            generation,
            version,
            registry,
        } = directive;
        let name = format!("{generation:020}-{version:020}-{registry:020}.json");
        write_atomic(&revisions.join(name), encoded.as_bytes())?;
        write_atomic(&self.directory.join("current.json"), encoded.as_bytes())?;
        prune(&revisions)?;
        Ok(true)
    }

    /// `load`: `None` when no store or no `current.json` exists.
    pub fn load(&self) -> Result<Option<StoredConfig>, LkgError> {
        if !fs::metadata(&self.directory).is_ok_and(|info| info.is_dir()) {
            return Ok(None);
        }
        let _lock = self.lock()?;
        self.current()
    }

    /// `_locked`: the returned file holds the exclusive lock until dropped.
    fn lock(&self) -> io::Result<File> {
        make_private_dir(&self.directory)?;
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .mode(0o600)
            .open(self.directory.join(".lock"))?;
        flock(&file, FlockOperation::LockExclusive)?;
        Ok(file)
    }

    /// `_load_current` and `_decode_record`.
    fn current(&self) -> Result<Option<StoredConfig>, LkgError> {
        let raw = match fs::read(self.directory.join("current.json")) {
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            read => read?,
        };
        let Some(Json::Object(record)) = parse_json(&raw) else {
            return Err(LkgError::Record);
        };
        let integer = |key| match lookup(&record, key) {
            Some(Json::Int(value)) => Ok(*value),
            _ => Err(LkgError::Record),
        };
        let directive = Directive {
            generation: integer("generation")?,
            version: integer("config_version")?,
            registry: integer("registry_version")?,
        };
        match lookup(&record, "payload") {
            Some(payload @ Json::Object(_)) => Ok(Some(StoredConfig {
                payload: payload.clone(),
                directive,
            })),
            _ => Err(LkgError::Record),
        }
    }
}

/// `json.dumps(record, sort_keys=True, separators=(",", ":"))`.
fn encode(payload: &Json, directive: Directive) -> Result<String, LkgError> {
    let record = Json::Object(vec![
        ("config_version".to_owned(), Json::Int(directive.version)),
        ("generation".to_owned(), Json::Int(directive.generation)),
        ("payload".to_owned(), payload.clone()),
        ("registry_version".to_owned(), Json::Int(directive.registry)),
    ]);
    Serialiser::ModelSelection
        .canonical(&record)
        .map_err(|_| LkgError::Payload)
}

/// `Path.mkdir(mode=0o700, parents=True, exist_ok=True)`: only the leaf gets
/// the private mode.
fn make_private_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    match DirBuilder::new().mode(0o700).create(path) {
        Err(error) if error.kind() == ErrorKind::AlreadyExists && path.is_dir() => Ok(()),
        created => created,
    }
}

fn fsync_directory(directory: &Path) -> io::Result<()> {
    File::open(directory)?.sync_all()
}

/// `_write_atomic`: a private temporary file, fsynced, renamed over `path`,
/// then the directory fsynced; the temporary never outlives a failure.
fn write_atomic(path: &Path, contents: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    make_private_dir(parent)?;
    let serial = TEMPORARY.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".tmp-{}-{serial}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary)?;
    let written = file
        .set_permissions(Permissions::from_mode(0o600))
        .and_then(|()| file.write_all(contents))
        .and_then(|()| file.sync_all())
        .and_then(|()| fs::rename(&temporary, path));
    if written.is_err() {
        fs::remove_file(&temporary).ok();
    }
    written?;
    fsync_directory(parent)
}

/// `_prune_revisions`: keeps the newest `RETENTION` `*.json` names.
fn prune(revisions: &Path) -> io::Result<()> {
    let mut names = Vec::new();
    for entry in fs::read_dir(revisions)? {
        let name = entry?.file_name();
        if name.as_encoded_bytes().ends_with(b".json") {
            names.push(name);
        }
    }
    names.sort_unstable_by(|left, right| right.cmp(left));
    for name in names.iter().skip(RETENTION) {
        fs::remove_file(revisions.join(name))?;
    }
    fsync_directory(revisions)
}
