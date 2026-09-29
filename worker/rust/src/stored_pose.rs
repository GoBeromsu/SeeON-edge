//! Stored RGB preprocessing and ordered person boxes, not inference.
//! Numeric authority: OrtClipPoseRunner with NumPy 2.4.6 (weak Python scalars).

pub const NET_SIZE: usize = 640;
pub const TENSOR_SHAPE: [usize; 4] = [1, 3, NET_SIZE, NET_SIZE];
pub const TENSOR_VALUES: usize = 3 * NET_SIZE * NET_SIZE;
pub const OUTPUT_SHAPE: [usize; 3] = [1, 300, 57];
pub const OUTPUT_VALUES: usize = 300 * 57;
pub type PersonBox = [f64; 5];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoredPoseError {
    Dimensions,
    DimensionsOverflow,
    ImageLength,
    Threshold,
    OutputShape,
    NonFiniteOutput,
}

impl std::fmt::Display for StoredPoseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Dimensions => "RGB dimensions must be positive",
            Self::DimensionsOverflow => "RGB dimensions exceed addressable arithmetic",
            Self::ImageLength => "RGB byte length does not match dimensions",
            Self::Threshold => "threshold must be finite and in [0, 1]",
            Self::OutputShape => "output0 must have shape [1, 300, 57] and matching length",
            Self::NonFiniteOutput => "output0 must contain only finite float32 values",
        })
    }
}
impl std::error::Error for StoredPoseError {}

struct Layout {
    width: usize,
    height: usize,
    rgb_len: usize,
    scale: f64,
    resized_width: usize,
    resized_height: usize,
}

impl Layout {
    fn new(width: i64, height: i64) -> Result<Self, StoredPoseError> {
        if width <= 0 || height <= 0 {
            return Err(StoredPoseError::Dimensions);
        }
        let width = usize::try_from(width).map_err(|_| StoredPoseError::DimensionsOverflow)?;
        let height = usize::try_from(height).map_err(|_| StoredPoseError::DimensionsOverflow)?;
        let rgb_len = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(StoredPoseError::DimensionsOverflow)?;
        let scale = (NET_SIZE as f64 / width as f64).min(NET_SIZE as f64 / height as f64);
        Ok(Self {
            width,
            height,
            rgb_len,
            scale,
            // Python round uses ties to even, including a legitimate zero axis.
            resized_width: (width as f64 * scale).round_ties_even() as usize,
            resized_height: (height as f64 * scale).round_ties_even() as usize,
        })
    }
}

fn nearest_indices(
    source: usize,
    destination: usize,
) -> Result<[usize; NET_SIZE], StoredPoseError> {
    let mut indices = [0; NET_SIZE];
    if destination > NET_SIZE {
        return Err(StoredPoseError::DimensionsOverflow);
    }
    for (i, index) in indices[..destination].iter_mut().enumerate() {
        // np.arange(destination) * source is integer arithmetic BEFORE true_divide.
        let numerator = i
            .checked_mul(source)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(StoredPoseError::DimensionsOverflow)?;
        *index = ((numerator as f64 / destination as f64) as usize).min(source - 1);
    }
    Ok(indices)
}

/// One fixed NCHW allocation. Failed preprocessing leaves every previous bit intact.
/// Input is packed, row-major RGB8; arbitrary strides/dtypes are not this API's domain.
#[derive(Debug)]
pub struct StoredPoseTensor {
    values: Vec<f32>,
}

impl Default for StoredPoseTensor {
    fn default() -> Self {
        Self {
            values: vec![0.0; TENSOR_VALUES],
        }
    }
}

impl StoredPoseTensor {
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn preprocess(
        &mut self,
        rgb: &[u8],
        width: i64,
        height: i64,
    ) -> Result<&[f32], StoredPoseError> {
        let layout = Layout::new(width, height)?;
        if rgb.len() != layout.rgb_len {
            return Err(StoredPoseError::ImageLength);
        }
        let xs = nearest_indices(layout.width, layout.resized_width)?;
        let ys = nearest_indices(layout.height, layout.resized_height)?;
        // All fallible admission precedes mutation, including index arithmetic.
        self.values.fill(0.0);
        for (y, &source_y) in ys[..layout.resized_height].iter().enumerate() {
            for (x, &source_x) in xs[..layout.resized_width].iter().enumerate() {
                let pixel = (source_y * layout.width + source_x) * 3;
                let target = y * NET_SIZE + x;
                for channel in 0..3 {
                    self.values[channel * NET_SIZE * NET_SIZE + target] =
                        f32::from(rgb[pixel + channel]) / 255.0_f32;
                }
            }
        }
        Ok(&self.values)
    }
}

fn python_clip(coordinate: f32, scale: f32, upper: f64) -> f64 {
    let value = f64::from(coordinate / scale);
    // min(upper, value), then max(+0.0, inner): Python retains the FIRST tie.
    // In particular, a negative-zero coordinate becomes positive zero, not -0.0.
    // Finite raw coordinates may overflow division to infinity; clipping is canonical.
    let inner = if value < upper { value } else { upper };
    if inner > 0.0 { inner } else { 0.0 }
}

/// Raw output0 is already float32. All 17,100 values must be finite, even ignored
/// classes/keypoints. That fail-closed rule is separate from finite Python parity.
/// No sorting, NMS, keypoint interpretation, or score-range restriction is applied.
pub fn person_boxes(
    output: &[f32],
    shape: &[usize],
    width: i64,
    height: i64,
    threshold: f64,
) -> Result<Vec<PersonBox>, StoredPoseError> {
    let layout = Layout::new(width, height)?;
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(StoredPoseError::Threshold);
    }
    if shape != OUTPUT_SHAPE || output.len() != OUTPUT_VALUES {
        return Err(StoredPoseError::OutputShape);
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(StoredPoseError::NonFiniteOutput);
    }
    // NumPy 2.4.6 promotes the Python scalar INTO float32, not the row into f64.
    let scale = layout.scale as f32;
    let threshold = threshold as f32;
    let mut boxes = Vec::with_capacity(OUTPUT_SHAPE[1]);
    for row in output.chunks_exact(OUTPUT_SHAPE[2]) {
        if row[5] != 0.0 || row[4] < threshold {
            continue;
        }
        let x1 = python_clip(row[0], scale, layout.width as f64);
        let y1 = python_clip(row[1], scale, layout.height as f64);
        let x2 = python_clip(row[2], scale, layout.width as f64);
        let y2 = python_clip(row[3], scale, layout.height as f64);
        if x1 < x2 && y1 < y2 {
            boxes.push([x1, y1, x2, y2, f64::from(row[4])]);
        }
    }
    Ok(boxes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_preprocessing_is_transactional_and_reuse_keeps_the_allocation() {
        let mut tensor = StoredPoseTensor::default();
        tensor.preprocess(&[255, 127, 1], 1, 1).unwrap();
        let previous = tensor.as_slice().to_vec();
        let allocation = tensor.as_slice().as_ptr();
        for (bytes, width, height, error) in [
            (&[][..], 0, 1, StoredPoseError::Dimensions),
            (&[][..], 1, -1, StoredPoseError::Dimensions),
            (&[][..], i64::MAX, 2, StoredPoseError::DimensionsOverflow),
            (&[1, 2][..], 1, 1, StoredPoseError::ImageLength),
            (&[1, 2, 3, 4][..], 1, 1, StoredPoseError::ImageLength),
        ] {
            assert_eq!(tensor.preprocess(bytes, width, height), Err(error));
            assert_eq!(tensor.as_slice(), previous);
            assert_eq!(tensor.as_slice().as_ptr(), allocation);
        }
        // A positive thin input rounds to zero width: the WHOLE reused tensor clears.
        tensor.preprocess(&vec![255; 1280 * 3], 1, 1280).unwrap();
        assert!(tensor.as_slice().iter().all(|value| value.to_bits() == 0));
        assert_eq!(tensor.as_slice().as_ptr(), allocation);
    }
}
