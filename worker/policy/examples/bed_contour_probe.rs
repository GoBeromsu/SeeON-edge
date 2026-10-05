//! Test-only bed-contour transport: one pure contour stage, never a model.
//! Requires the parent-owned `pub mod bed_contour;` export in lib.rs.
//! That module line is the whole export. Public API inside the module:
//! `largest_external_contour(mask: &[u8], width: i64, height: i64)`
//! `-> Result<Vec<[i64; 2]>, BedContourError>`
//! `BedContourError::{Dimensions, DimensionsOverflow, MaskLength, MaskValue,`
//! `AreaDomain, Allocation}`
//!
//! v1 binary protocol: every integer is little endian.
//! Request: `BEDCNT01` (8 bytes), u32 call_count, then each call:
//! i64 width, i64 height, u32 mask byte_count, byte_count mask bytes.
//! Exact EOF is required; 1..=4 calls; <=48 MiB total input, including headers.
//! The input ceiling only lets a parity test express the smallest mask outside
//! the core's proven shoelace domain. It is not a core geometry quota.
//! Transport lengths describe bytes present, not valid geometry.
//!
//! Response: same magic and count, then each call:
//! u8 status; ONLY on success, u32 vertex_count, then vertex_count times
//! i64 x, i64 y. An empty success is status 0 and count 0.
//! Status: 0 success, 1 dimensions, 2 addressable arithmetic overflow,
//! 3 mask length, 4 mask value, 5 proven shoelace domain, 6 allocation.
//! Semantic errors omit the count and vertices. Nothing is truncated.
//!
//! Each call is an independent pure function. The full request is validated
//! before any contour is computed, and the <=1 MiB response is buffered before
//! stdout is written. A response that would exceed that print budget rejects
//! the whole buffer. Malformed or over-budget transport exits 2, with empty
//! stdout and exactly `bed-contour-probe: rejected\n` on stderr, never
//! input or path details. Valid transport, including semantic errors, exits 0
//! with empty stderr. The caller additionally owns a wall-time bound and
//! streaming output caps.
#![forbid(unsafe_code)]

use seeon_worker::bed_contour::{BedContourError, largest_external_contour};
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"BEDCNT01";
const MAX_CALLS: usize = 4;
const MAX_INPUT: usize = 48 * 1024 * 1024;
const MAX_OUTPUT: usize = 1024 * 1024;
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
}

struct Call<'a> {
    width: i64,
    height: i64,
    mask: &'a [u8],
}

fn parse(input: &[u8]) -> Result<Vec<Call<'_>>> {
    if input.len() > MAX_INPUT {
        return Err(());
    }
    let mut cursor = Cursor { input, position: 0 };
    if cursor.take(MAGIC.len())? != MAGIC {
        return Err(());
    }
    let count = usize::try_from(cursor.u32()?).map_err(|_| ())?;
    if count == 0 || count > MAX_CALLS {
        return Err(());
    }
    let mut calls = Vec::new();
    calls.try_reserve(count).map_err(|_| ())?;
    for _ in 0..count {
        let width = cursor.i64()?;
        let height = cursor.i64()?;
        let mask_len = usize::try_from(cursor.u32()?).map_err(|_| ())?;
        calls.push(Call {
            width,
            height,
            mask: cursor.take(mask_len)?,
        });
    }
    if cursor.position != input.len() {
        return Err(());
    }
    Ok(calls)
}

fn status(error: BedContourError) -> u8 {
    match error {
        BedContourError::Dimensions => 1,
        BedContourError::DimensionsOverflow => 2,
        BedContourError::MaskLength => 3,
        BedContourError::MaskValue => 4,
        BedContourError::AreaDomain => 5,
        BedContourError::Allocation => 6,
    }
}

fn room(out: &[u8], extra: usize) -> Result<()> {
    match out.len().checked_add(extra) {
        Some(total) if total <= MAX_OUTPUT => Ok(()),
        _ => Err(()),
    }
}

fn write_success(out: &mut Vec<u8>, vertices: &[[i64; 2]]) -> Result<()> {
    let count = u32::try_from(vertices.len()).map_err(|_| ())?;
    let body = vertices.len().checked_mul(16).ok_or(())?;
    let extra = body.checked_add(5).ok_or(())?;
    room(out, extra)?;
    out.try_reserve(extra).map_err(|_| ())?;
    out.push(0);
    out.extend_from_slice(&count.to_le_bytes());
    for point in vertices {
        out.extend_from_slice(&point[0].to_le_bytes());
        out.extend_from_slice(&point[1].to_le_bytes());
    }
    Ok(())
}

fn write_status(out: &mut Vec<u8>, code: u8) -> Result<()> {
    room(out, 1)?;
    out.try_reserve(1).map_err(|_| ())?;
    out.push(code);
    Ok(())
}

fn run(input: &[u8]) -> Result<Vec<u8>> {
    let calls = parse(input)?;
    let mut out = Vec::new();
    let count = u32::try_from(calls.len()).map_err(|_| ())?;
    room(&out, 12)?;
    out.try_reserve(12).map_err(|_| ())?;
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&count.to_le_bytes());
    for call in calls {
        match largest_external_contour(call.mask, call.width, call.height) {
            Ok(vertices) => write_success(&mut out, &vertices)?,
            Err(error) => write_status(&mut out, status(error))?,
        }
    }
    if out.len() > MAX_OUTPUT {
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
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(&output).map_err(|_| ())?;
        stdout.flush().map_err(|_| ())
    })();
    if result.is_err() {
        let _ = std::io::stderr()
            .lock()
            .write_all(b"bed-contour-probe: rejected\n");
        std::process::exit(2);
    }
}
