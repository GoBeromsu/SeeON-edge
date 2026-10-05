//! Largest closed boundary of one binary mask. No simplification, sigmoid, or inference.
//! Numeric authority: `seg_postprocess.largest_external_contour` on frozen NumPy 2.4.6.
//!
//! Pixels are visited in row-major order. A foreground pixel may emit a top, right,
//! bottom, then left boundary edge, in that order. Destinations at one vertex are a
//! stack: the last inserted edge is followed first. Each walk starts at the earliest
//! inserted origin that still has a destination. Hash-map iteration never selects it.
//! Only a walk that returns to its own start is a contour, and the closing vertex is
//! not stored again. Holes and corner junctions remain whatever that walk produces.
//!
//! The champion is the maximum absolute shoelace; an equal value keeps the earlier
//! loop. Other loops are dropped as soon as they lose. The source expression is
//! `abs(dot(x, roll(y, -1)) - dot(y, roll(x, -1)))` on f64 products of the integer
//! coordinates, with no division by two. Coordinates are non-negative, so both dots
//! are sums of non-negative integers. Every such integer and every partial sum is
//! exact for any reduction order when `4 * (width * height)^2 <= 2^53`. A non-empty
//! mask outside that proven domain is refused before a contour is returned. That
//! refusal is not a geometry quota and not a claim of arbitrary-size equivalence.
//! An empty mask never evaluates the shoelace, so it returns no vertices even when
//! its dimensions are outside the domain.

use std::collections::HashMap;

/// Integers in `[-2^53, 2^53]` are exactly representable as f64.
const F64_EXACT_INTEGER_BITS: u32 = 53;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BedContourError {
    Dimensions,
    DimensionsOverflow,
    MaskLength,
    MaskValue,
    AreaDomain,
    Allocation,
}

impl std::fmt::Display for BedContourError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Dimensions => "mask dimensions must be non-negative",
            Self::DimensionsOverflow => "mask dimensions exceed addressable arithmetic",
            Self::MaskLength => "mask byte length does not match dimensions",
            Self::MaskValue => "mask bytes must be 0 or 1",
            Self::AreaDomain => "mask shoelace is outside the proven f64 domain",
            Self::Allocation => "mask contour storage could not be allocated",
        })
    }
}

impl std::error::Error for BedContourError {}

/// `mask` is packed row-major: index `y * width + x` is column `x`, row `y`.
/// Byte 0 is background and byte 1 is foreground. Vertices are `[x, y]`.
///
/// A valid mask with no foreground yields an empty vector, including zero width
/// or height. Every error leaves the caller with no contour at all.
pub fn largest_external_contour(
    mask: &[u8],
    width: i64,
    height: i64,
) -> Result<Vec<[i64; 2]>, BedContourError> {
    if width < 0 || height < 0 {
        return Err(BedContourError::Dimensions);
    }
    let width = usize::try_from(width).map_err(|_| BedContourError::DimensionsOverflow)?;
    let height = usize::try_from(height).map_err(|_| BedContourError::DimensionsOverflow)?;
    let pixels = width
        .checked_mul(height)
        .ok_or(BedContourError::DimensionsOverflow)?;
    if pixels != mask.len() {
        return Err(BedContourError::MaskLength);
    }
    let foreground = foreground_count(mask)?;
    if foreground == 0 {
        return Ok(Vec::new());
    }
    if !proven_shoelace_domain(pixels) {
        return Err(BedContourError::AreaDomain);
    }
    let mut edges = Origins::new();
    add_boundaries(&mut edges, mask, width, height)?;
    edges.select()
}

fn proven_shoelace_domain(pixels: usize) -> bool {
    let Ok(pixels) = u128::try_from(pixels) else {
        return false;
    };
    let Some(square) = pixels.checked_mul(pixels) else {
        return false;
    };
    let Some(scaled) = square.checked_mul(4) else {
        return false;
    };
    scaled <= 1u128 << F64_EXACT_INTEGER_BITS
}

fn foreground_count(mask: &[u8]) -> Result<usize, BedContourError> {
    let mut foreground = 0usize;
    for &pixel in mask {
        match pixel {
            0 => {}
            1 => {
                foreground = foreground
                    .checked_add(1)
                    .ok_or(BedContourError::DimensionsOverflow)?;
            }
            _ => return Err(BedContourError::MaskValue),
        }
    }
    Ok(foreground)
}

fn grid(x: usize, y: usize) -> Result<[i64; 2], BedContourError> {
    Ok([
        i64::try_from(x).map_err(|_| BedContourError::DimensionsOverflow)?,
        i64::try_from(y).map_err(|_| BedContourError::DimensionsOverflow)?,
    ])
}

fn boundary_open(
    mask: &[u8],
    on_border: bool,
    neighbor: Option<usize>,
) -> Result<bool, BedContourError> {
    if on_border {
        return Ok(true);
    }
    let index = neighbor.ok_or(BedContourError::DimensionsOverflow)?;
    Ok(*mask.get(index).ok_or(BedContourError::DimensionsOverflow)? == 0)
}

fn foreground_at(mask: &[u8], index: usize) -> Result<bool, BedContourError> {
    Ok(*mask.get(index).ok_or(BedContourError::DimensionsOverflow)? == 1)
}

/// Source `_area`: two f64 dots of integer products, then abs, no divide by two.
fn abs_shoelace(points: &[[i64; 2]]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut forward = 0.0_f64;
    let mut reverse = 0.0_f64;
    for (index, point) in points.iter().enumerate() {
        let next = points[(index + 1) % points.len()];
        forward += point[0] as f64 * next[1] as f64;
        reverse += point[1] as f64 * next[0] as f64;
    }
    (forward - reverse).abs()
}

/// Insertion-ordered origins. `at` is a point lookup only and is never iterated.
struct Origins {
    order: Vec<[i64; 2]>,
    dests: Vec<Vec<[i64; 2]>>,
    at: HashMap<[i64; 2], usize>,
    cursor: usize,
}

impl Origins {
    fn new() -> Self {
        Self {
            order: Vec::new(),
            dests: Vec::new(),
            at: HashMap::new(),
            cursor: 0,
        }
    }

    fn add(&mut self, start: [i64; 2], end: [i64; 2]) -> Result<(), BedContourError> {
        if let Some(slot) = self.at.get(&start).copied() {
            self.dests[slot]
                .try_reserve(1)
                .map_err(|_| BedContourError::Allocation)?;
            self.dests[slot].push(end);
            return Ok(());
        }
        let slot = self.order.len();
        self.order
            .try_reserve(1)
            .map_err(|_| BedContourError::Allocation)?;
        self.dests
            .try_reserve(1)
            .map_err(|_| BedContourError::Allocation)?;
        self.at
            .try_reserve(1)
            .map_err(|_| BedContourError::Allocation)?;
        let mut dests = Vec::new();
        dests
            .try_reserve(1)
            .map_err(|_| BedContourError::Allocation)?;
        dests.push(end);
        self.order.push(start);
        self.dests.push(dests);
        self.at.insert(start, slot);
        Ok(())
    }

    fn next_start(&mut self) -> Option<[i64; 2]> {
        while self.cursor < self.order.len() {
            let point = self.order[self.cursor];
            if self.at.contains_key(&point) {
                return Some(point);
            }
            self.cursor += 1;
        }
        None
    }

    fn pop_dest(&mut self, point: [i64; 2]) -> Option<[i64; 2]> {
        let slot = *self.at.get(&point)?;
        let next = self.dests[slot].pop();
        if self.dests[slot].is_empty() {
            self.at.remove(&point);
        }
        next
    }

    fn select(mut self) -> Result<Vec<[i64; 2]>, BedContourError> {
        let mut best: Option<(Vec<[i64; 2]>, f64)> = None;
        while let Some(start) = self.next_start() {
            let mut current = start;
            let mut path = Vec::new();
            loop {
                path.try_reserve(1)
                    .map_err(|_| BedContourError::Allocation)?;
                path.push(current);
                let Some(next) = self.pop_dest(current) else {
                    break;
                };
                current = next;
                if current == start {
                    let area = abs_shoelace(&path);
                    let replace = match &best {
                        Some((_, prior)) => area > *prior,
                        None => true,
                    };
                    if replace {
                        best = Some((path, area));
                    }
                    break;
                }
            }
        }
        Ok(best.map(|(path, _)| path).unwrap_or_default())
    }
}

fn add_boundaries(
    edges: &mut Origins,
    mask: &[u8],
    width: usize,
    height: usize,
) -> Result<(), BedContourError> {
    for y in 0..height {
        let row = y
            .checked_mul(width)
            .ok_or(BedContourError::DimensionsOverflow)?;
        for x in 0..width {
            let index = row
                .checked_add(x)
                .ok_or(BedContourError::DimensionsOverflow)?;
            if !foreground_at(mask, index)? {
                continue;
            }
            let x1 = x
                .checked_add(1)
                .ok_or(BedContourError::DimensionsOverflow)?;
            let y1 = y
                .checked_add(1)
                .ok_or(BedContourError::DimensionsOverflow)?;
            if boundary_open(mask, y == 0, index.checked_sub(width))? {
                edges.add(grid(x, y)?, grid(x1, y)?)?;
            }
            if boundary_open(mask, x1 == width, index.checked_add(1))? {
                edges.add(grid(x1, y)?, grid(x1, y1)?)?;
            }
            if boundary_open(mask, y1 == height, index.checked_add(width))? {
                edges.add(grid(x1, y1)?, grid(x, y1)?)?;
            }
            if boundary_open(mask, x == 0, index.checked_sub(1))? {
                edges.add(grid(x, y1)?, grid(x, y)?)?;
            }
        }
    }
    Ok(())
}
