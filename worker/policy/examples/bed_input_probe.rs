//! Test-only bed-input transport: conversion evidence, never model inference.
//! Requires the parent-owned `pub mod bed_input;` export in lib.rs.
//!
//! v1 binary protocol: all integers and IEEE-754 floats are little endian.
//! Request: `BEDINP01` (8 bytes), u32 call_count, then each call:
//! i64 width, i64 height, u32 RGB byte_count, byte_count packed RGB8 bytes.
//! Exact EOF is required; 1..=4 calls; <=16 MiB total input, including headers.
//! Transport lengths describe available bytes, not necessarily valid geometry.
//! The core imposes no image-size cap beyond checked addressable packed length.
//!
//! Response: same magic and count, then each call:
//! u8 status; ONLY on success, the following 40-byte Letterbox:
//! i64 source_height, i64 source_width, f64 scale,
//! u32 resized_height, u32 resized_width, u32 pad_top, u32 pad_left;
//! then, for EVERY status, exactly 4,915,200 float32 NCHW tensor values.
//! Status: 0 success, 1 dimensions, 2 addressable arithmetic overflow, 3 RGB length.
//! Float fields preserve exact bits. Semantic errors have no Letterbox and expose
//! the previous tensor unchanged (positive zeros before the first success).
//!
//! One real BedInputTensor is reused across all calls. The full request is
//! validated before preprocessing, and the <=80 MiB response is buffered before
//! writing stdout. Malformed/over-budget transport exits 2, with empty stdout and
//! exactly `bed-input-probe: rejected\n` on stderr, never input/path details.
//! Valid transport, including semantic errors, exits 0 with empty stderr.
//! The caller additionally owns a wall-time bound and streaming output caps.
#![forbid(unsafe_code)]

use seeon_worker::bed_input::{BedInputError, BedInputTensor, Letterbox, TENSOR_VALUES};
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"BEDINP01";
const MAX_CALLS: usize = 4;
const MAX_INPUT: usize = 16 * 1024 * 1024;
const MAX_OUTPUT: usize = 80 * 1024 * 1024;
const MAX_RECORD: usize = 1 + 40 + TENSOR_VALUES * 4;
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
    rgb: &'a [u8],
}

fn parse(input: &[u8]) -> Result<Vec<Call<'_>>> {
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
    let mut calls = Vec::with_capacity(count);
    for _ in 0..count {
        let width = cursor.i64()?;
        let height = cursor.i64()?;
        let rgb_len = usize::try_from(cursor.u32()?).map_err(|_| ())?;
        calls.push(Call {
            width,
            height,
            rgb: cursor.take(rgb_len)?,
        });
    }
    if cursor.position != input.len() {
        return Err(());
    }
    Ok(calls)
}

fn status(error: BedInputError) -> u8 {
    match error {
        BedInputError::Dimensions => 1,
        BedInputError::DimensionsOverflow => 2,
        BedInputError::ImageLength => 3,
    }
}

fn metadata(out: &mut Vec<u8>, letterbox: Letterbox) {
    out.extend_from_slice(&letterbox.source_height.to_le_bytes());
    out.extend_from_slice(&letterbox.source_width.to_le_bytes());
    out.extend_from_slice(&letterbox.scale.to_le_bytes());
    for dimension in [
        letterbox.resized_height,
        letterbox.resized_width,
        letterbox.pad_top,
        letterbox.pad_left,
    ] {
        // All four are bounded by the fixed 1280-square core geometry.
        out.extend_from_slice(&(dimension as u32).to_le_bytes());
    }
}

fn run(input: &[u8]) -> Result<Vec<u8>> {
    let calls = parse(input)?;
    let bound = calls
        .len()
        .checked_mul(MAX_RECORD)
        .and_then(|n| n.checked_add(12))
        .ok_or(())?;
    if bound > MAX_OUTPUT {
        return Err(());
    }
    let mut out = Vec::with_capacity(bound);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(calls.len() as u32).to_le_bytes());
    let mut tensor = BedInputTensor::default();
    for call in calls {
        match tensor.preprocess(call.rgb, call.width, call.height) {
            Ok((_, letterbox)) => {
                out.push(0);
                metadata(&mut out, letterbox);
            }
            Err(error) => out.push(status(error)),
        }
        for value in tensor.as_slice() {
            out.extend_from_slice(&value.to_le_bytes());
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
        std::io::stdout().lock().write_all(&output).map_err(|_| ())
    })();
    if result.is_err() {
        let _ = std::io::stderr()
            .lock()
            .write_all(b"bed-input-probe: rejected\n");
        std::process::exit(2);
    }
}
