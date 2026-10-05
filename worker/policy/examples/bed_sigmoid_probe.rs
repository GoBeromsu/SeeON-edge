//! Test-only sigmoid transport: one f32 vector, never a model or decoder.
//! Requires the parent-owned `pub mod bed_sigmoid;` export in lib.rs.
//! The import path below is the whole API use. No crate-root reexport.
//!
//! v1 binary protocol: the count is little endian and every f32 is raw
//! little-endian IEEE-754 bits.
//! Request: `BDSIGM01` (8 bytes), u32 count, count float32 values.
//! Exact EOF is required. Count may be zero. Count is at most 1,048,576.
//! Total input, including the 12-byte header, is at most 4,194,316 bytes.
//! Every f32 bit pattern is data. NaN and infinity are not transport errors.
//!
//! Response: the same magic and count, then count float32 results.
//! The response is the same size as a valid request and is buffered until
//! every input bit has been accepted. Malformed or over-budget transport
//! exits 2, writes no stdout (no magic prefix), and writes exactly
//! `bed-sigmoid-probe: rejected\n` to stderr, never input or path details.
//! A well-formed vector, including nonfinite values, exits 0 with empty stderr.
//! The caller owns the wall-time bound and streaming output caps.
#![forbid(unsafe_code)]

use seeon_worker::bed_sigmoid::bed_mask_sigmoid;
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"BDSIGM01";
const MAX_COUNT: usize = 1_048_576;
const HEADER: usize = 12;
const MAX_INPUT: usize = HEADER + MAX_COUNT * 4;
const MAX_OUTPUT: usize = HEADER + MAX_COUNT * 4;
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
    if count > MAX_COUNT {
        return Err(());
    }
    let body = cursor.take(count.checked_mul(4).ok_or(())?)?;
    if cursor.position != input.len() {
        return Err(());
    }
    let total = body.len().checked_add(HEADER).ok_or(())?;
    if total > MAX_OUTPUT {
        return Err(());
    }
    let mut out = Vec::new();
    out.try_reserve(total).map_err(|_| ())?;
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(count as u32).to_le_bytes());
    for chunk in body.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.extend_from_slice(&bed_mask_sigmoid(value).to_le_bytes());
    }
    if out.len() != total {
        return Err(());
    }
    Ok(out)
}

fn main() {
    let result = (|| {
        let mut input = Vec::new();
        input.try_reserve(MAX_INPUT + 1).map_err(|_| ())?;
        std::io::stdin()
            .lock()
            .take((MAX_INPUT + 1) as u64)
            .read_to_end(&mut input)
            .map_err(|_| ())?;
        let output = run(&input)?;
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(&output).map_err(|_| ())?;
        stdout.flush().map_err(|_| ())
    })();
    if result.is_err() {
        let _ = std::io::stderr()
            .lock()
            .write_all(b"bed-sigmoid-probe: rejected\n");
        std::process::exit(2);
    }
}
