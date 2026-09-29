//! Test-only stored-pose transport. No SDK, inference, or runtime qualification.
//! Requires the parent-owned `pub mod stored_pose;` export in lib.rs.
//!
//! v1 binary protocol: all integers and IEEE-754 floats are little endian.
//! Request: `STPOSE01` (8 bytes), u32 call_count, then each call:
//! i64 width, i64 height, f64 threshold, u32 rank, rank*u64 output dimensions,
//! u32 RGB byte_count, u32 output value_count, packed RGB8, value_count*f32 output0.
//! Exact EOF is required. Rank <=4, 1..=8 calls, <=16 MiB total input.
//! No image-size limit beyond that transport budget is imposed by the library.
//!
//! Response: same magic and count; each call emits u8 status, 1,228,800*f32 NCHW,
//! u32 box_count, box_count*[f64;5]. Status: 0 success, 1 dimensions, 2 arithmetic
//! overflow, 3 RGB length, 4 threshold, 5 output shape/length, 6 nonfinite output.
//! Output conversion is preflighted before preprocessing. Every semantic failure
//! emits zero boxes and the unchanged previous tensor (initially positive zeros).
//! No error is a parity assertion. The finite differential owns parity claims.
//!
//! A single tensor is reused for all calls. Input is capped while reading; decoded
//! output0 storage is bounded by input size. The response is <=40 MiB, buffered
//! until the entire request is validated. Malformed/over-budget transport exits 2,
//! emits no stdout, and writes exactly `stored-pose-probe: rejected\n` to stderr.
//! Valid transport, including semantic errors, exits 0 with empty stderr.
//! The test caller owns a subprocess timeout and independent streaming I/O caps.
#![forbid(unsafe_code)]

use seeon_worker::stored_pose::{StoredPoseError, StoredPoseTensor, TENSOR_VALUES, person_boxes};
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"STPOSE01";
const MAX_CALLS: usize = 8;
const MAX_INPUT: usize = 16 * 1024 * 1024;
const MAX_OUTPUT: usize = 40 * 1024 * 1024;
type Result<T> = std::result::Result<T, ()>;

struct Cursor<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self.position.checked_add(count).ok_or(())?;
        let bytes = self.input.get(self.position..end).ok_or(())?;
        self.position = end;
        Ok(bytes)
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.take(N)?.try_into().map_err(|_| ())
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.bytes()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.bytes()?))
    }
}

fn status(error: StoredPoseError) -> u8 {
    match error {
        StoredPoseError::Dimensions => 1,
        StoredPoseError::DimensionsOverflow => 2,
        StoredPoseError::ImageLength => 3,
        StoredPoseError::Threshold => 4,
        StoredPoseError::OutputShape => 5,
        StoredPoseError::NonFiniteOutput => 6,
    }
}

fn call(cursor: &mut Cursor<'_>, tensor: &mut StoredPoseTensor, out: &mut Vec<u8>) -> Result<()> {
    let width = cursor.i64()?;
    let height = cursor.i64()?;
    let threshold = cursor.f64()?;
    let rank = cursor.u32()? as usize;
    if rank > 4 {
        return Err(());
    }
    let mut shape = [0; 4];
    for dimension in &mut shape[..rank] {
        *dimension = usize::try_from(u64::from_le_bytes(cursor.bytes()?)).map_err(|_| ())?;
    }
    let rgb_len = cursor.u32()? as usize;
    let value_count = cursor.u32()? as usize;
    let rgb = cursor.take(rgb_len)?;
    let raw = cursor.take(value_count.checked_mul(4).ok_or(())?)?;
    let output: Vec<f32> = raw
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect();
    let result = person_boxes(&output, &shape[..rank], width, height, threshold)
        .and_then(|boxes| tensor.preprocess(rgb, width, height).map(|_| boxes));
    let (code, boxes) = match result {
        Ok(boxes) => (0, boxes),
        Err(error) => (status(error), Vec::new()),
    };
    let additional = 1 + TENSOR_VALUES * 4 + 4 + boxes.len() * 5 * 8;
    if additional > MAX_OUTPUT || out.len() > MAX_OUTPUT - additional {
        return Err(());
    }
    out.push(code);
    for value in tensor.as_slice() {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&(boxes.len() as u32).to_le_bytes());
    for person in boxes {
        for value in person {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(())
}

fn run(input: &[u8]) -> Result<Vec<u8>> {
    if input.len() > MAX_INPUT {
        return Err(());
    }
    let mut cursor = Cursor { input, position: 0 };
    if cursor.take(MAGIC.len())? != MAGIC {
        return Err(());
    }
    let count = cursor.u32()? as usize;
    if count == 0 || count > MAX_CALLS {
        return Err(());
    }
    let mut out = Vec::with_capacity(12 + count * (5 + TENSOR_VALUES * 4 + 300 * 40));
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(count as u32).to_le_bytes());
    let mut tensor = StoredPoseTensor::default();
    for _ in 0..count {
        call(&mut cursor, &mut tensor, &mut out)?;
    }
    if cursor.position != input.len() {
        return Err(());
    }
    Ok(out)
}

fn main() {
    let result = (|| {
        let mut input = Vec::new();
        std::io::stdin()
            .lock()
            .take((MAX_INPUT + 1) as u64)
            .read_to_end(&mut input)
            .map_err(|_| ())?;
        let output = run(&input)?;
        std::io::stdout().lock().write_all(&output).map_err(|_| ())
    })();
    if result.is_err() {
        let _ = std::io::stderr()
            .lock()
            .write_all(b"stored-pose-probe: rejected\n");
        std::process::exit(2);
    }
}
