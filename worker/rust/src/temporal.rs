//! Mirrors PTS bucketing and TemporalProfile, not domain scheduling or policy.
use std::collections::BTreeMap;
use std::fmt;
use std::sync::LazyLock;

pub const CADENCE_NS: i64 = 66_666_667;
pub const DEFAULT_MAX_GAP_ROWS: i64 = 900;
const CURRENT_INGEST_FPS: f64 = 30.0;
const CURRENT_POSE_FPS: f64 = 15.0;
const CURRENT_BED_INTERVAL_FRAMES: u64 = 180;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResampledRow<T> {
    pub pts_ns: i64,
    pub value: Option<T>,
    pub valid: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtsGapTooLargeError {
    pub gap_rows: i128,
    pub max_gap_rows: i64,
}
impl fmt::Display for PtsGapTooLargeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PTS gap requires {} rows; limit is {}",
            self.gap_rows, self.max_gap_rows
        )
    }
}
impl std::error::Error for PtsGapTooLargeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtsResamplerConfigError;
impl fmt::Display for PtsResamplerConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("cadence_ns must be positive and max_gap_rows non-negative")
    }
}
impl std::error::Error for PtsResamplerConfigError {}

/// One epoch's cadence state. Retains no source values or unbounded history.
/// PTS inputs are signed 64-bit nanoseconds; intermediate arithmetic is widened.
#[derive(Debug)]
pub struct PtsResampler {
    cadence_ns: i64,
    max_gap_rows: i64,
    origin: Option<i64>,
    next_slot: i128,
    last_pts: Option<i64>,
}
impl Default for PtsResampler {
    fn default() -> Self {
        Self::new(CADENCE_NS, DEFAULT_MAX_GAP_ROWS).expect("valid source cadence")
    }
}
impl PtsResampler {
    pub fn new(cadence_ns: i64, max_gap_rows: i64) -> Result<Self, PtsResamplerConfigError> {
        if cadence_ns <= 0 || max_gap_rows < 0 {
            return Err(PtsResamplerConfigError);
        }
        Ok(Self {
            cadence_ns,
            max_gap_rows,
            origin: None,
            next_slot: 0,
            last_pts: None,
        })
    }

    /// Caller-owned epoch reset, including after rollback. push itself drops old PTS.
    /// The domain owner must also clear its own per-track histories on rollback.
    pub fn reset(&mut self) {
        self.origin = None;
        self.next_slot = 0;
        self.last_pts = None;
    }

    pub fn push<T>(
        &mut self,
        pts_ns: i64,
        value: T,
    ) -> Result<Vec<ResampledRow<T>>, PtsGapTooLargeError> {
        if self.last_pts.is_some_and(|last| pts_ns <= last) {
            return Ok(Vec::new());
        }
        // Python advances the high-water mark even for same-bucket rows and gap errors.
        self.last_pts = Some(pts_ns);
        let origin = match self.origin {
            Some(origin) => i128::from(origin),
            None => {
                self.origin = Some(pts_ns);
                self.next_slot = i128::from(pts_ns);
                self.next_slot
            }
        };
        let cadence = i128::from(self.cadence_ns);
        let slot = origin + ((i128::from(pts_ns) - origin) / cadence) * cadence;
        if slot < self.next_slot {
            return Ok(Vec::new());
        }
        let gap_rows = (slot - self.next_slot) / cadence;
        if gap_rows > i128::from(self.max_gap_rows) {
            return Err(PtsGapTooLargeError {
                gap_rows,
                max_gap_rows: self.max_gap_rows,
            });
        }
        let mut produced = Vec::new();
        let mut next_slot = self.next_slot;
        while next_slot < slot {
            // Emitted slots lie between the i64 origin and this i64 input PTS.
            produced.push(ResampledRow {
                pts_ns: next_slot as i64,
                value: None,
                valid: 0,
            });
            next_slot += cadence;
        }
        produced.push(ResampledRow {
            pts_ns: slot as i64,
            value: Some(value),
            valid: 1,
        });
        self.next_slot = slot + cadence;
        Ok(produced)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalProfileError(pub String);
impl fmt::Display for TemporalProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for TemporalProfileError {}

#[derive(Debug, Clone, PartialEq)]
pub struct TemporalProfile {
    ingest_fps: f64,
    pose_fps: f64,
    decision_hz: BTreeMap<String, f64>,
}
impl TemporalProfile {
    pub fn new(
        ingest_fps: f64,
        pose_fps: Option<f64>,
        decision_hz: Option<BTreeMap<String, f64>>,
    ) -> Result<Self, TemporalProfileError> {
        require_rate("ingest_fps", ingest_fps)?;
        let pose_fps = pose_fps.unwrap_or(ingest_fps);
        require_rate("pose_fps", pose_fps)?;
        let decision_hz = decision_hz.unwrap_or_else(|| {
            BTreeMap::from([(
                "bed".into(),
                CURRENT_INGEST_FPS / CURRENT_BED_INTERVAL_FRAMES as f64,
            )])
        });
        for (name, hz) in &decision_hz {
            if name.is_empty() {
                return Err(TemporalProfileError(
                    "decision_hz keys must be non-empty".into(),
                ));
            }
            require_rate(&format!("decision_hz[{name:?}]"), *hz)?;
        }
        Ok(Self {
            ingest_fps,
            pose_fps,
            decision_hz,
        })
    }

    pub fn ingest_fps(&self) -> f64 {
        self.ingest_fps
    }
    pub fn pose_fps(&self) -> f64 {
        self.pose_fps
    }
    pub fn decision_hz(&self) -> &BTreeMap<String, f64> {
        &self.decision_hz
    }
    pub fn target_fps(&self) -> f64 {
        self.ingest_fps
    }
    pub fn frame_interval_sec(&self) -> f64 {
        1.0 / self.ingest_fps
    }

    pub fn pose_interval_frames(&self, frame_stride: u64) -> Result<u64, TemporalProfileError> {
        if frame_stride == 0 {
            return Err(TemporalProfileError(
                "frame_stride must be an integer > 0".into(),
            ));
        }
        interval_frames(self.ingest_fps, self.pose_fps)?
            .checked_mul(frame_stride)
            .ok_or_else(|| TemporalProfileError("frame interval exceeds u64".into()))
    }

    pub fn decision_interval_frames(&self, domain: &str) -> Result<u64, TemporalProfileError> {
        let hz = self.decision_hz.get(domain).ok_or_else(|| {
            TemporalProfileError(format!("no decision_hz declared for {domain:?}"))
        })?;
        interval_frames(self.ingest_fps, *hz)
    }

    pub fn task_intervals(
        &self,
        frame_stride: u64,
    ) -> Result<BTreeMap<String, u64>, TemporalProfileError> {
        let mut intervals =
            BTreeMap::from([("pose".into(), self.pose_interval_frames(frame_stride)?)]);
        for domain in self.decision_hz.keys() {
            intervals.insert(domain.clone(), self.decision_interval_frames(domain)?);
        }
        Ok(intervals)
    }
}

fn require_rate(name: &str, value: f64) -> Result<(), TemporalProfileError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(TemporalProfileError(format!(
            "{name} must be a finite number > 0"
        )));
    }
    Ok(())
}

fn interval_frames(ingest_fps: f64, hz: f64) -> Result<u64, TemporalProfileError> {
    // Python round uses ties-to-even, unlike Rust's round. Never saturate a cast.
    let interval = (ingest_fps / hz).round_ties_even().max(1.0);
    if !interval.is_finite() || interval >= u64::MAX as f64 {
        return Err(TemporalProfileError("frame interval exceeds u64".into()));
    }
    Ok(interval as u64)
}

pub static CURRENT_TEMPORAL_PROFILE: LazyLock<TemporalProfile> = LazyLock::new(|| {
    TemporalProfile::new(CURRENT_INGEST_FPS, Some(CURRENT_POSE_FPS), None)
        .expect("valid source temporal profile")
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_duplicates_and_out_of_order_keep_first_bucket_row() {
        let mut sampler = PtsResampler::default();
        let mut rows = sampler.push(10, "a").unwrap();
        for pts in [10, 9, 10 + CADENCE_NS - 1] {
            assert!(sampler.push(pts, "dropped").unwrap().is_empty());
        }
        rows.extend(sampler.push(10 + 3 * CADENCE_NS, "b").unwrap());
        assert!(sampler.push(9, "old").unwrap().is_empty());
        assert_eq!(
            rows.into_iter()
                .map(|r| (r.pts_ns, r.value, r.valid))
                .collect::<Vec<_>>(),
            vec![
                (10, Some("a"), 1),
                (10 + CADENCE_NS, None, 0),
                (10 + 2 * CADENCE_NS, None, 0),
                (10 + 3 * CADENCE_NS, Some("b"), 1),
            ]
        );
    }

    #[test]
    fn gap_limit_is_inclusive_and_error_advances_only_high_water() {
        let mut sampler = PtsResampler::new(CADENCE_NS, 1).unwrap();
        sampler.push(-7, ()).unwrap();
        let rows = sampler.push(-7 + 2 * CADENCE_NS, ()).unwrap();
        assert_eq!(rows.iter().map(|r| r.valid).collect::<Vec<_>>(), vec![0, 1]);
        let rejected = -7 + 5 * CADENCE_NS;
        let expected = PtsGapTooLargeError {
            gap_rows: 2,
            max_gap_rows: 1,
        };
        assert_eq!(sampler.push(rejected, ()), Err(expected));
        assert!(sampler.push(rejected - CADENCE_NS, ()).unwrap().is_empty());
        assert!(sampler.push(rejected, ()).unwrap().is_empty());
        assert_eq!(sampler.push(rejected + 1, ()), Err(expected));
        sampler.reset();
        assert_eq!(sampler.push(5, ()).unwrap()[0].pts_ns, 5);
    }

    #[test]
    fn large_gaps_and_signed_timestamp_edges_do_not_wrap() {
        let mut sampler = PtsResampler::new(1, 0).unwrap();
        sampler.push(i64::MIN, ()).unwrap();
        assert_eq!(
            sampler.push(i64::MAX, ()),
            Err(PtsGapTooLargeError {
                gap_rows: i128::from(i64::MAX) - i128::from(i64::MIN) - 1,
                max_gap_rows: 0,
            })
        );
        sampler.reset();
        assert_eq!(sampler.push(i64::MAX, ()).unwrap()[0].pts_ns, i64::MAX);
        assert!(sampler.push(i64::MAX, ()).unwrap().is_empty());
        let mut contiguous = PtsResampler::new(CADENCE_NS, 0).unwrap();
        contiguous.push(-CADENCE_NS, ()).unwrap();
        assert_eq!(contiguous.push(0, ()).unwrap()[0].valid, 1);
        assert_eq!(contiguous.push(2 * CADENCE_NS, ()).unwrap_err().gap_rows, 1);
        for (cadence, gap) in [(0, 1), (-1, 1), (1, -1)] {
            assert!(PtsResampler::new(cadence, gap).is_err());
        }
    }

    #[test]
    fn profile_rounds_ties_even_before_stride_and_keeps_domain_overrides() {
        for (ingest, expected) in [(5.0, 2), (7.0, 4), (1.0, 1)] {
            let profile = TemporalProfile::new(
                ingest,
                Some(2.0),
                Some(BTreeMap::from([("bed".into(), 2.0), ("pose".into(), 2.0)])),
            )
            .unwrap();
            assert_eq!(profile.pose_interval_frames(3).unwrap(), expected * 3);
            assert_eq!(profile.decision_interval_frames("bed").unwrap(), expected);
            assert_eq!(profile.task_intervals(3).unwrap()["pose"], expected);
            assert_eq!(profile.frame_interval_sec(), 1.0 / ingest);
        }
    }

    #[test]
    fn profile_rejects_invalid_rates_names_stride_and_overflow() {
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(TemporalProfile::new(rate, None, None).is_err());
            assert!(TemporalProfile::new(30.0, Some(rate), None).is_err());
            assert!(
                TemporalProfile::new(30.0, None, Some(BTreeMap::from([("bed".into(), rate)])))
                    .is_err()
            );
        }
        assert!(
            TemporalProfile::new(30.0, None, Some(BTreeMap::from([("".into(), 1.0)]))).is_err()
        );
        let profile = TemporalProfile::new(30.0, Some(15.0), Some(BTreeMap::new())).unwrap();
        assert!(profile.decision_interval_frames("unknown").is_err());
        assert!(profile.pose_interval_frames(0).is_err());
        assert!(profile.pose_interval_frames(u64::MAX).is_err());
        let huge = TemporalProfile::new(f64::MAX, Some(f64::MIN_POSITIVE), None).unwrap();
        assert!(huge.pose_interval_frames(1).is_err());
    }
}
