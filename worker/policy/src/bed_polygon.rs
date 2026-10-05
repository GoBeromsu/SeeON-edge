//! Ordered closed-contour simplification, not inference or a decoder entrypoint.
//! Authority: seg_postprocess.simplify_polygon on frozen NumPy 2.4.6.
//! f64::hypot is a candidate pending actual target differential qualification.

const EXACT_COORDINATE: i64 = 1_i64 << 53;
type Result<T> = std::result::Result<T, BedPolygonError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BedPolygonError {
    Capacity,
    CoordinateDomain,
    Allocation,
}

impl std::fmt::Display for BedPolygonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Capacity => "polygon capacity must be positive",
            Self::CoordinateDomain => "polygon coordinate exceeds exact f64 integer domain",
            Self::Allocation => "polygon allocation failed",
        })
    }
}
impl std::error::Error for BedPolygonError {}

fn reserved<T>(capacity: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| BedPolygonError::Allocation)?;
    Ok(values)
}

fn furthest_pair(values: &[[f64; 2]]) -> (usize, usize) {
    let mut best = (0, 0);
    let mut distance = 0.0;
    // The full matrix's first row-major maximum, without allocating that matrix.
    for (i, first) in values.iter().enumerate() {
        for (j, second) in values.iter().enumerate() {
            let x = first[0] - second[0];
            let y = first[1] - second[1];
            let squared = x * x + y * y;
            if squared > distance {
                best = (i, j);
                distance = squared;
            }
        }
    }
    best
}

fn ring(count: usize, start: usize, end: usize) -> Result<Vec<usize>> {
    let length = if start <= end {
        end - start + 1
    } else {
        count - start + end + 1
    };
    let mut indices = reserved(length)?;
    if start <= end {
        indices.extend(start..=end);
    } else {
        indices.extend(start..count);
        indices.extend(0..=end);
    }
    Ok(indices)
}

fn reduce(
    values: &[[f64; 2]],
    chain: &[usize],
    epsilon: f64,
    output: &mut Vec<usize>,
    stack: &mut Vec<(usize, usize)>,
) -> Result<()> {
    output.clear();
    stack.clear();
    output
        .try_reserve(chain.len())
        .map_err(|_| BedPolygonError::Allocation)?;
    stack
        .try_reserve(chain.len())
        .map_err(|_| BedPolygonError::Allocation)?;
    if chain.len() <= 2 {
        output.extend_from_slice(chain);
        return Ok(());
    }
    stack.push((0, chain.len() - 1));
    while let Some((first, last)) = stack.pop() {
        if last - first <= 1 {
            output.push(chain[first]);
            continue;
        }
        let start = values[chain[first]];
        let end = values[chain[last]];
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let length = dx.hypot(dy);
        let distance = |index: usize| {
            let point = values[chain[index]];
            if length != 0.0 {
                let left = dx * (start[1] - point[1]);
                let right = (start[0] - point[0]) * dy;
                (left - right).abs()
            } else {
                (point[0] - start[0]).hypot(point[1] - start[1])
            }
        };
        let mut index = first + 1;
        let mut maximum = distance(index);
        for current in first + 2..last {
            let candidate = distance(current);
            if candidate > maximum {
                maximum = candidate;
                index = current;
            }
        }
        let threshold = if length != 0.0 {
            epsilon * length
        } else {
            epsilon
        };
        if maximum <= threshold {
            output.push(chain[first]);
        } else {
            // Right first on the stack means left first in the result. Each
            // leaf emits its start; the one final endpoint is appended below.
            stack.push((index, last));
            stack.push((first, index));
        }
    }
    output.push(chain[chain.len() - 1]);
    Ok(())
}

fn sample_indices(length: usize, count: usize) -> Result<Vec<usize>> {
    let mut indices = reserved(count)?;
    if count == 1 {
        indices.push(0);
        return Ok(indices);
    }
    // NumPy linspace performs divide then multiply, and replaces the final
    // endpoint before ties-even rounding. No integer-ratio shortcut.
    let step = (length - 1) as f64 / (count - 1) as f64;
    for index in 0..count - 1 {
        indices.push((index as f64 * step).round_ties_even() as usize);
    }
    indices.push(length - 1);
    Ok(indices)
}

pub fn simplify_polygon(points: &[[i64; 2]], max_points: i64) -> Result<Vec<[i64; 2]>> {
    if max_points <= 0 {
        return Err(BedPolygonError::Capacity);
    }
    if points.len() as u128 <= max_points as u128 {
        let mut copy = reserved(points.len())?;
        copy.extend_from_slice(points);
        return Ok(copy);
    }
    let capacity = usize::try_from(max_points).map_err(|_| BedPolygonError::Capacity)?;
    let mut values = reserved(points.len())?;
    for point in points {
        if point
            .iter()
            .any(|coordinate| !(-EXACT_COORDINATE..=EXACT_COORDINATE).contains(coordinate))
        {
            return Err(BedPolygonError::CoordinateDomain);
        }
        values.push([point[0] as f64, point[1] as f64]);
    }
    let (first, second) = furthest_pair(&values);
    let one = ring(values.len(), first, second)?;
    let two = ring(values.len(), second, first)?;
    let mut minimum = values[0];
    let mut maximum = values[0];
    for point in &values[1..] {
        for axis in 0..2 {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    let mut low = 0.0;
    let mut high = (maximum[0] - minimum[0]).max(maximum[1] - minimum[1]);
    let mut best = reserved(values.len())?;
    best.extend(0..values.len());
    let mut candidate = reserved(values.len())?;
    let (mut left, mut right, mut stack) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..32 {
        let midpoint = (low + high) / 2.0;
        reduce(&values, &one, midpoint, &mut left, &mut stack)?;
        reduce(&values, &two, midpoint, &mut right, &mut stack)?;
        candidate.clear();
        // Inclusive ring chains together have n+2 entries (or two when their
        // endpoints coincide); removing each last endpoint bounds this by n.
        candidate.extend_from_slice(&left[..left.len() - 1]);
        candidate.extend_from_slice(&right[..right.len() - 1]);
        if candidate.len() > capacity {
            low = midpoint;
        } else {
            std::mem::swap(&mut best, &mut candidate);
            high = midpoint;
        }
    }
    let mut output = reserved(best.len().min(capacity))?;
    if best.len() > capacity {
        for index in sample_indices(best.len(), capacity)? {
            output.push(points[best[index]]);
        }
    } else {
        for index in best {
            output.push(points[index]);
        }
    }
    // Slow-path coordinates were exactly representable: Python's final
    // round(float(coordinate)) recovers these same original integer values.
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_row_major_pair_wins_symmetric_ties() {
        let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert_eq!(furthest_pair(&points), (0, 2));
        assert_eq!(furthest_pair(&[[4.0, 7.0]; 4]), (0, 0));
    }

    #[test]
    fn zero_length_endpoints_preserve_inclusive_threshold() {
        let values = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]];
        let (mut output, mut stack) = (Vec::new(), Vec::new());
        reduce(&values, &[0, 1, 2], 1.0, &mut output, &mut stack).unwrap();
        assert_eq!(output, [0, 2]);
        reduce(&values, &[0, 1, 2], 0.5, &mut output, &mut stack).unwrap();
        assert_eq!(output, [0, 1, 2]);
    }

    #[test]
    fn sampling_uses_ties_even_and_exact_final_endpoint() {
        assert_eq!(sample_indices(6, 3).unwrap(), [0, 2, 5]);
        assert_eq!(sample_indices(8, 3).unwrap(), [0, 4, 7]);
        assert_eq!(sample_indices(37, 1).unwrap(), [0]);
    }

    #[test]
    fn numeric_domain_only_applies_when_simplifying() {
        let points = [[i64::MIN, i64::MAX], [0, 0]];
        assert_eq!(simplify_polygon(&points, 2).unwrap(), points);
        assert_eq!(
            simplify_polygon(&points, 1),
            Err(BedPolygonError::CoordinateDomain)
        );
        assert_eq!(simplify_polygon(&[], 0), Err(BedPolygonError::Capacity));
    }
}
