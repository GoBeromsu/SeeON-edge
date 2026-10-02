//! Captured ONNX bytes for an offline engine build. Selection and packaged
//! admission stay with their existing owners; this module only opens the
//! admitted `model.onnx` and compares its digest. No ORT, CUDA, or network.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};

use rustix::fs::{Mode, OFlags};
use sha2::{Digest, Sha256};

use crate::config::env::Env;
use crate::config::model_bundle::packaged::{PACKAGED_FALL_ROOT, admit_packaged_bundle};
use crate::config::{self, CheckConfigError};
use crate::run::fall_evidence;

const MAX_ONNX_BYTES: u64 = 512 * 1024 * 1024;
const ONNX_MEMBER: &str = "model.onnx";

/// Bytes captured from one descriptor, hashed once. The path is retained only
/// to prevent output/source collisions; builders never reopen it for inference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CapturedOnnx {
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub source_path: PathBuf,
}

/// Why an ONNX source was not captured. Display names the stage only.
#[derive(Debug)]
pub(crate) enum SourceError {
    Selection,
    Admission,
    Metadata,
    Capture,
    Mismatch,
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Selection => "model selection was refused",
            Self::Admission => "model bundle admission was refused",
            Self::Metadata => "admitted fall metadata was refused",
            Self::Capture => "ONNX source is not a bounded regular file",
            Self::Mismatch => "captured ONNX digest does not match the admitted member",
        })
    }
}

impl std::error::Error for SourceError {}

impl From<CheckConfigError> for SourceError {
    fn from(error: CheckConfigError) -> Self {
        match error {
            CheckConfigError::Selection(_) => Self::Selection,
            CheckConfigError::Admission(_) => Self::Admission,
            CheckConfigError::Env(_) | CheckConfigError::Identity(_) => Self::Admission,
        }
    }
}

/// Open an image ONNX through ordinary symlinks, then capture that descriptor.
pub(crate) fn capture_image_onnx(path: &Path) -> Result<CapturedOnnx, SourceError> {
    capture(path, false)
}

/// Admit the selected fall bundle when a selection document exists; otherwise
/// admit the packaged fall root. Either way, capture only the admitted member.
pub(crate) fn capture_fall(env: &Env) -> Result<CapturedOnnx, SourceError> {
    match config::admit_selected_bundle(env)? {
        Some((selection, proof)) => {
            fall_evidence::selected(&selection, &proof).map_err(|_| SourceError::Metadata)?;
            let digest = proof
                .member_digests
                .get(ONNX_MEMBER)
                .ok_or(SourceError::Admission)?;
            let root = config::models_root(env)
                .join("bundles")
                .join(&selection.model_publication.content);
            capture_member(&root.join(ONNX_MEMBER), digest, true)
        }
        None => {
            let proof = admit_packaged_bundle(Path::new(PACKAGED_FALL_ROOT))
                .map_err(|_| SourceError::Admission)?;
            fall_evidence::packaged(&proof).map_err(|_| SourceError::Metadata)?;
            let digest = proof
                .member_digests
                .get(ONNX_MEMBER)
                .ok_or(SourceError::Admission)?;
            let root = Path::new(PACKAGED_FALL_ROOT);
            capture_member(&root.join(ONNX_MEMBER), digest, false)
        }
    }
}

fn capture_member(
    path: &Path,
    expected: &str,
    nofollow: bool,
) -> Result<CapturedOnnx, SourceError> {
    if !config::is_hex(expected, 64) {
        return Err(SourceError::Admission);
    }
    let captured = capture(path, nofollow)?;
    if captured.sha256 != expected {
        return Err(SourceError::Mismatch);
    }
    Ok(captured)
}

fn capture(path: &Path, nofollow: bool) -> Result<CapturedOnnx, SourceError> {
    let mut flags = OFlags::CLOEXEC | OFlags::RDONLY | OFlags::NONBLOCK;
    if nofollow {
        flags |= OFlags::NOFOLLOW;
    }
    let file =
        File::from(rustix::fs::open(path, flags, Mode::empty()).map_err(|_| SourceError::Capture)?);
    let info = file.metadata().map_err(|_| SourceError::Capture)?;
    let declared = info.len();
    if !bounded_regular(&info, declared) {
        return Err(SourceError::Capture);
    }
    let mut hasher = Sha256::new();
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut actual = 0_u64;
    let mut reader = file.take(declared.saturating_add(1));
    loop {
        let count = reader.read(&mut buffer).map_err(|_| SourceError::Capture)?;
        if count == 0 {
            break;
        }
        let count_u64 = u64::try_from(count).map_err(|_| SourceError::Capture)?;
        actual = actual.checked_add(count_u64).ok_or(SourceError::Capture)?;
        if actual > declared {
            return Err(SourceError::Capture);
        }
        hasher.update(&buffer[..count]);
        bytes.extend_from_slice(&buffer[..count]);
    }
    if actual != declared {
        return Err(SourceError::Capture);
    }
    let mut sha256 = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(sha256, "{byte:02x}").map_err(|_| SourceError::Capture)?;
    }
    Ok(CapturedOnnx {
        bytes,
        sha256,
        source_path: path.to_path_buf(),
    })
}

fn bounded_regular(info: &std::fs::Metadata, declared: u64) -> bool {
    let kind = info.file_type();
    kind.is_file()
        && !kind.is_symlink()
        && !info.is_dir()
        && !kind.is_fifo()
        && !kind.is_socket()
        && !kind.is_block_device()
        && !kind.is_char_device()
        && (1..=MAX_ONNX_BYTES).contains(&declared)
}

#[cfg(test)]
mod tests {
    use super::{MAX_ONNX_BYTES, capture_image_onnx};
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::io::Write;
    use std::os::unix::fs::symlink;
    use std::path::Path;

    fn hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    struct Temp(std::path::PathBuf);

    impl Temp {
        fn new() -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("onnx-source-{}-{}", std::process::id(), nanos,));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self, name: &str) -> std::path::PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn bounded_image_capture_preserves_bytes_and_sha() {
        let dir = Temp::new();
        let path = dir.path("model.onnx");
        let payload = b"onnx-bytes-not-a-model".to_vec();
        fs::write(&path, &payload).unwrap();
        let captured = capture_image_onnx(&path).unwrap();
        assert_eq!(captured.bytes, payload);
        assert_eq!(captured.sha256, hex(&payload));
    }

    #[test]
    fn ordinary_symlink_is_followed_for_image_paths() {
        let dir = Temp::new();
        let target = dir.path("real.onnx");
        let link = dir.path("link.onnx");
        let payload = b"followed";
        fs::write(&target, payload).unwrap();
        symlink(&target, &link).unwrap();
        let captured = capture_image_onnx(&link).unwrap();
        assert_eq!(captured.bytes, payload);
        assert_eq!(captured.sha256, hex(payload));
    }

    #[test]
    fn nonregular_empty_and_oversize_sparse_are_refused() {
        let dir = Temp::new();
        let directory = dir.path("dir.onnx");
        fs::create_dir(&directory).unwrap();
        assert!(capture_image_onnx(&directory).is_err());

        let fifo = dir.path("fifo.onnx");
        assert!(
            rustix::fs::mknodat(
                rustix::fs::CWD,
                &fifo,
                rustix::fs::FileType::Fifo,
                rustix::fs::Mode::from_bits_truncate(0o600),
                0,
            )
            .is_ok()
        );
        assert!(capture_image_onnx(&fifo).is_err());

        let empty = dir.path("empty.onnx");
        fs::write(&empty, b"").unwrap();
        assert!(capture_image_onnx(&empty).is_err());

        let sparse = dir.path("sparse.onnx");
        let file = fs::File::create(&sparse).unwrap();
        file.set_len(MAX_ONNX_BYTES + 1).unwrap();
        assert!(capture_image_onnx(&sparse).is_err());
        assert_eq!(file.metadata().unwrap().len(), MAX_ONNX_BYTES + 1);
    }

    #[test]
    fn changed_bytes_change_the_captured_digest() {
        let dir = Temp::new();
        let path = dir.path("model.onnx");
        fs::write(&path, b"before").unwrap();
        let first = capture_image_onnx(&path).unwrap();
        let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.write_all(b"after!").unwrap();
        drop(file);
        let second = capture_image_onnx(Path::new(&path)).unwrap();
        assert_ne!(first.bytes, second.bytes);
        assert_ne!(first.sha256, second.sha256);
        assert_eq!(second.sha256, hex(&second.bytes));
        assert_ne!(second.sha256, hex(&first.bytes));
    }
}
