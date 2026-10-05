//! Packed RGB8 to the current 1280-square bed input; no model or inference.
//! Numeric authority: seg_postprocess.letterbox_rgb with frozen NumPy 2.4.6.

pub const NET_SIZE: usize = 1280;
pub const TENSOR_SHAPE: [usize; 4] = [1, 3, NET_SIZE, NET_SIZE];
pub const TENSOR_VALUES: usize = 3 * NET_SIZE * NET_SIZE;
const PADDING: f32 = 114.0_f32 / 255.0_f32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BedInputError {
    Dimensions,
    DimensionsOverflow,
    ImageLength,
}

impl std::fmt::Display for BedInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Dimensions => "RGB dimensions must be positive",
            Self::DimensionsOverflow => "RGB dimensions exceed addressable arithmetic",
            Self::ImageLength => "RGB byte length does not match dimensions",
        })
    }
}
impl std::error::Error for BedInputError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub source_height: i64,
    pub source_width: i64,
    pub scale: f64,
    pub resized_height: usize,
    pub resized_width: usize,
    pub pad_top: usize,
    pub pad_left: usize,
}

struct Layout {
    width: usize,
    height: usize,
    rgb_len: usize,
    letterbox: Letterbox,
}

impl Layout {
    fn new(width: i64, height: i64) -> Result<Self, BedInputError> {
        if width <= 0 || height <= 0 {
            return Err(BedInputError::Dimensions);
        }
        let width = usize::try_from(width).map_err(|_| BedInputError::DimensionsOverflow)?;
        let height = usize::try_from(height).map_err(|_| BedInputError::DimensionsOverflow)?;
        let rgb_len = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(BedInputError::DimensionsOverflow)?;
        let scale = (NET_SIZE as f64 / width as f64).min(NET_SIZE as f64 / height as f64);
        // Python round, not round-away-from-zero and not a minimum of one pixel.
        let resized_width = (width as f64 * scale).round_ties_even() as usize;
        let resized_height = (height as f64 * scale).round_ties_even() as usize;
        if resized_width > NET_SIZE || resized_height > NET_SIZE {
            return Err(BedInputError::DimensionsOverflow);
        }
        Ok(Self {
            width,
            height,
            rgb_len,
            letterbox: Letterbox {
                source_height: height as i64,
                source_width: width as i64,
                scale,
                resized_height,
                resized_width,
                pad_top: ((NET_SIZE - resized_height) as f64 / 2.0 - 0.1).round_ties_even()
                    as usize,
                pad_left: ((NET_SIZE - resized_width) as f64 / 2.0 - 0.1).round_ties_even()
                    as usize,
            },
        })
    }
}

#[derive(Clone, Copy, Default)]
struct Sample {
    low: usize,
    high: usize,
    weight: f64,
}

fn axis_samples(source: usize, destination: usize) -> Result<[Sample; NET_SIZE], BedInputError> {
    let mut samples = [Sample::default(); NET_SIZE];
    for (i, sample) in samples[..destination].iter_mut().enumerate() {
        // Each grid operation is float32, including Python weak-scalar conversion.
        // Do not rewrite as a ratio, use f64 grids, or contract with mul_add.
        let center = i as f32 + 0.5_f32;
        let scaled = center * source as f32;
        let divided = scaled / destination as f32;
        let coordinate = (divided - 0.5_f32).clamp(0.0, (source - 1) as f32);
        let low = coordinate.floor() as usize;
        if low >= source {
            return Err(BedInputError::DimensionsOverflow);
        }
        *sample = Sample {
            low,
            high: (low + 1).min(source - 1),
            // NumPy float32 minus intp promotes BOTH operands to float64.
            weight: f64::from(coordinate) - low as f64,
        };
    }
    Ok(samples)
}

fn bilinear(rgb: &[u8], width: usize, x: Sample, y: Sample, channel: usize) -> f32 {
    let at = |row, column| f64::from(rgb[(row * width + column) * 3 + channel]);
    let top = at(y.low, x.low) * (1.0 - x.weight) + at(y.low, x.high) * x.weight;
    let bottom = at(y.high, x.low) * (1.0 - x.weight) + at(y.high, x.high) * x.weight;
    // Preserve the explicit NumPy operation order and the cast BEFORE uint8.
    (top * (1.0 - y.weight) + bottom * y.weight) as f32
}

/// One reusable float32 NCHW allocation, initially positive zeros.
/// Input must be exact packed row-major RGB8. Failures preserve every prior bit.
#[derive(Debug)]
pub struct BedInputTensor {
    values: Vec<f32>,
}

impl Default for BedInputTensor {
    fn default() -> Self {
        Self {
            values: vec![0.0; TENSOR_VALUES],
        }
    }
}

impl BedInputTensor {
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn preprocess(
        &mut self,
        rgb: &[u8],
        width: i64,
        height: i64,
    ) -> Result<(&[f32], Letterbox), BedInputError> {
        let layout = Layout::new(width, height)?;
        if rgb.len() != layout.rgb_len {
            return Err(BedInputError::ImageLength);
        }
        let letterbox = layout.letterbox;
        let xs = axis_samples(layout.width, letterbox.resized_width)?;
        let ys = axis_samples(layout.height, letterbox.resized_height)?;
        let same_size =
            layout.width == letterbox.resized_width && layout.height == letterbox.resized_height;
        // Admission and all index construction precede mutation. The checked
        // packed length bounds every RGB offset; resized axes bound every write.
        self.values.fill(PADDING);
        for (y, &sy) in ys[..letterbox.resized_height].iter().enumerate() {
            for (x, &sx) in xs[..letterbox.resized_width].iter().enumerate() {
                let target = (y + letterbox.pad_top) * NET_SIZE + x + letterbox.pad_left;
                for channel in 0..3 {
                    let resized = if same_size {
                        f32::from(rgb[(y * layout.width + x) * 3 + channel])
                    } else {
                        bilinear(rgb, layout.width, sx, sy, channel)
                    };
                    let byte = resized as u8;
                    self.values[channel * NET_SIZE * NET_SIZE + target] =
                        f32::from(byte) / 255.0_f32;
                }
            }
        }
        Ok((&self.values, letterbox))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_pixel_grids_clip_and_interpolation_truncates_after_float32() {
        let xs = axis_samples(2, 4).unwrap();
        let ys = axis_samples(1, 1).unwrap();
        assert_eq!((xs[0].low, xs[0].high, xs[0].weight), (0, 1, 0.0));
        assert_eq!((xs[1].low, xs[1].high, xs[1].weight), (0, 1, 0.25));
        assert_eq!((xs[3].low, xs[3].high, xs[3].weight), (1, 1, 0.0));
        let value = bilinear(&[0, 1, 254, 255, 254, 1], 2, xs[1], ys[0], 0);
        assert_eq!(value.to_bits(), 63.75_f32.to_bits());
        assert_eq!(value as u8, 63);
    }

    #[test]
    fn malformed_inputs_are_atomic_and_successful_reuse_resets_all_padding() {
        let mut tensor = BedInputTensor::default();
        tensor.preprocess(&[255, 127, 1], 1, 1).unwrap();
        let previous: Vec<u32> = tensor.as_slice().iter().map(|v| v.to_bits()).collect();
        let allocation = tensor.as_slice().as_ptr();
        for (bytes, width, height, error) in [
            (&[][..], 0, 1, BedInputError::Dimensions),
            (&[][..], 1, i64::MIN, BedInputError::Dimensions),
            (&[][..], i64::MAX, 1, BedInputError::DimensionsOverflow),
            (
                &[][..],
                i64::MAX,
                i64::MAX,
                BedInputError::DimensionsOverflow,
            ),
            (&[1, 2][..], 1, 1, BedInputError::ImageLength),
            (&[1, 2, 3, 4][..], 1, 1, BedInputError::ImageLength),
        ] {
            assert_eq!(tensor.preprocess(bytes, width, height), Err(error));
            assert!(
                tensor
                    .as_slice()
                    .iter()
                    .zip(&previous)
                    .all(|(a, b)| a.to_bits() == *b)
            );
            assert_eq!(tensor.as_slice().as_ptr(), allocation);
        }
        let (_, letterbox) = tensor.preprocess(&vec![255; 2560 * 3], 1, 2560).unwrap();
        assert_eq!(letterbox.resized_width, 0);
        assert_eq!(letterbox.pad_left, 640);
        assert_eq!(letterbox.scale.to_bits(), 0.5_f64.to_bits());
        assert!(
            tensor
                .as_slice()
                .iter()
                .all(|v| v.to_bits() == PADDING.to_bits())
        );
        assert_eq!(tensor.as_slice().as_ptr(), allocation);
        tensor.preprocess(&[0, 255, 254], 1, 1).unwrap();
        assert_eq!(tensor.as_slice()[0].to_bits(), 0);
        assert_eq!(
            tensor.as_slice()[NET_SIZE * NET_SIZE].to_bits(),
            1.0_f32.to_bits()
        );
        assert_eq!(tensor.as_slice().as_ptr(), allocation);
    }
}
