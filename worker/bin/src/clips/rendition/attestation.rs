//! `clip.playback-h264.json`, the attestation the backend reads before it
//! serves a rendition (`backend/app/features/clips/store.py`).

use crate::json::{Json, Serialiser};

use super::RenditionError;

/// The attested facts of a published rendition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attestation {
    pub rendition: String,
    pub rendition_sha256: String,
    pub source_sha256: String,
    pub pts_identical: bool,
    pub time_base: String,
    pub frames: u64,
    pub source_frames: u64,
}

impl Attestation {
    /// Sorted, compact, ASCII JSON plus one newline, as the Python writes it.
    pub fn to_bytes(&self) -> Result<Vec<u8>, RenditionError> {
        let model = Json::Object(vec![
            ("frames".to_owned(), Json::Int(i128::from(self.frames))),
            ("pts_identical".to_owned(), Json::Bool(self.pts_identical)),
            ("rendition".to_owned(), Json::Str(self.rendition.clone())),
            (
                "rendition_sha256".to_owned(),
                Json::Str(self.rendition_sha256.clone()),
            ),
            (
                "source_frames".to_owned(),
                Json::Int(i128::from(self.source_frames)),
            ),
            (
                "source_sha256".to_owned(),
                Json::Str(self.source_sha256.clone()),
            ),
            ("time_base".to_owned(), Json::Str(self.time_base.clone())),
        ]);
        let mut bytes = Serialiser::ModelSelection
            .canonical(&model)
            .map_err(|_| RenditionError::Mismatch)?
            .into_bytes();
        bytes.push(b'\n');
        Ok(bytes)
    }
}
