//! Test-only BDPOLY01 transport, never a model or decoder.
//! Request: magic, u32LE call count1..4; each call is i64LE capacity,
//! u32LE vertex count0..4096, then signed i64LE x/y pairs. Exact EOF.
//! Response: magic/count; each call is u8 status, then (only for status0)
//! u32LE vertex count and i64LE pairs. Status1=Capacity,2=CoordinateDomain,
//! 3=Allocation. Input/output caps are1MiB, separate from core semantics.
//! Malformed/over-budget requests: exit2, empty stdout, static stderr.
#![forbid(unsafe_code)]

use seeon_worker::bed_polygon::{BedPolygonError, simplify_polygon};
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"BDPOLY01";
const MAX_BYTES: usize = 1024 * 1024;
const MAX_CALLS: usize = 4;
const MAX_POINTS: usize = 4096;
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

struct Call {
    capacity: i64,
    points: Vec<[i64; 2]>,
}

fn parse(input: &[u8]) -> Result<Vec<Call>> {
    if input.len() > MAX_BYTES {
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
        let capacity = cursor.i64()?;
        let count = usize::try_from(cursor.u32()?).map_err(|_| ())?;
        if count > MAX_POINTS {
            return Err(());
        }
        let mut points = Vec::new();
        points.try_reserve(count).map_err(|_| ())?;
        for _ in 0..count {
            points.push([cursor.i64()?, cursor.i64()?]);
        }
        calls.push(Call { capacity, points });
    }
    if cursor.position != input.len() {
        return Err(());
    }
    Ok(calls)
}

fn room(output: &mut Vec<u8>, extra: usize) -> Result<()> {
    if output.len().checked_add(extra).ok_or(())? > MAX_BYTES {
        return Err(());
    }
    output.try_reserve(extra).map_err(|_| ())
}

fn run(input: &[u8]) -> Result<Vec<u8>> {
    let calls = parse(input)?;
    let mut output = Vec::new();
    room(&mut output, 12)?;
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&u32::try_from(calls.len()).map_err(|_| ())?.to_le_bytes());
    for call in calls {
        match simplify_polygon(&call.points, call.capacity) {
            Ok(points) => {
                let extra = points
                    .len()
                    .checked_mul(16)
                    .and_then(|n| n.checked_add(5))
                    .ok_or(())?;
                room(&mut output, extra)?;
                output.push(0);
                output
                    .extend_from_slice(&u32::try_from(points.len()).map_err(|_| ())?.to_le_bytes());
                for point in points {
                    output.extend_from_slice(&point[0].to_le_bytes());
                    output.extend_from_slice(&point[1].to_le_bytes());
                }
            }
            Err(error) => {
                room(&mut output, 1)?;
                output.push(match error {
                    BedPolygonError::Capacity => 1,
                    BedPolygonError::CoordinateDomain => 2,
                    BedPolygonError::Allocation => 3,
                });
            }
        }
    }
    Ok(output)
}

fn main() {
    let result = (|| {
        let mut input = Vec::new();
        input.try_reserve(MAX_BYTES + 1).map_err(|_| ())?;
        std::io::stdin()
            .lock()
            .take((MAX_BYTES + 1) as u64)
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
            .write_all(b"bed-polygon-probe: rejected\n");
        std::process::exit(2);
    }
}
