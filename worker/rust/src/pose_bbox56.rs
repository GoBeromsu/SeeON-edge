//! COCO-17 pose-head features and a single track's bounded row storage.
use std::collections::VecDeque;

pub const COCO17_KEYPOINTS: usize = 17;
pub const COCO17_KEYPOINT_ORDER: [&str; COCO17_KEYPOINTS] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
];
pub const POSE_BBOX56_DIM: usize = 56;
pub const POSE_BBOX56_CONFIDENCE_GATE: f64 = 0.5;
// Vocabulary authority: contracts/model_selection.py, not a new preprocessing version.
pub const POSE_BBOX56_PREPROCESSING_IDENTITY: &str = "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1";
pub type Keypoint = [f64; 3];
pub type PoseBbox = [f64; 4];
pub type PoseBbox56Row = [f32; POSE_BBOX56_DIM];
pub const ZERO_ROW: PoseBbox56Row = [0.0; POSE_BBOX56_DIM];
// Storage bound from worker/domains/fall/classifier.py; not a scheduling decision.
pub const FALL_WINDOW_FRAMES: usize = 30;

/// Computes in float64 like Python, rounding only the final components to float32.
/// Slices admit malformed shapes for validation; nonnumeric values are excluded by type.
pub fn pose_bbox56_row<P: AsRef<[f64]>>(
    keypoints: &[P],
    bbox: Option<&[f64]>,
    frame_width: i64,
    frame_height: i64,
) -> PoseBbox56Row {
    let Some(bbox) = bbox else {
        return ZERO_ROW;
    };
    if frame_width <= 0
        || frame_height <= 0
        || keypoints.len() != COCO17_KEYPOINTS
        || bbox.len() != 4
    {
        return ZERO_ROW;
    }
    if bbox.iter().any(|v| !v.is_finite())
        || keypoints.iter().any(|point| {
            let point = point.as_ref();
            point.len() != 3 || point.iter().any(|v| !v.is_finite())
        })
    {
        return ZERO_ROW;
    }
    let max_x = (frame_width - 1) as f64;
    let max_y = (frame_height - 1) as f64;
    let x1 = bbox[0].clamp(0.0, max_x);
    let y1 = bbox[1].clamp(0.0, max_y);
    let x2 = bbox[2].clamp(0.0, max_x);
    let y2 = bbox[3].clamp(0.0, max_y);
    if x2 <= x1 || y2 <= y1 {
        return ZERO_ROW;
    }
    let width = frame_width as f64;
    let height = frame_height as f64;
    let mut row = ZERO_ROW;
    for (index, point) in keypoints.iter().enumerate() {
        let point = point.as_ref();
        if point[2] >= POSE_BBOX56_CONFIDENCE_GATE {
            row[index * 3] = (point[0].clamp(0.0, max_x) / width) as f32;
            row[index * 3 + 1] = (point[1].clamp(0.0, max_y) / height) as f32;
            row[index * 3 + 2] = point[2] as f32;
        }
    }
    row[51..55].copy_from_slice(&[
        (x1 / width) as f32,
        (y1 / height) as f32,
        (x2 / width) as f32,
        (y2 / height) as f32,
    ]);
    row[55] = 1.0;
    // Python struct.pack('f') raises OverflowError instead of returning infinity.
    if row.iter().any(|v| !v.is_finite()) {
        ZERO_ROW
    } else {
        row
    }
}

pub fn native_pose_bbox56_row<P: AsRef<[f64]>>(
    keypoints: &[P],
    bbox: Option<&[f64]>,
    frame_width: i64,
    frame_height: i64,
    box_source: &str,
) -> PoseBbox56Row {
    if box_source != "pose" {
        return ZERO_ROW;
    }
    pose_bbox56_row(keypoints, bbox, frame_width, frame_height)
}

#[derive(Debug)]
pub struct PoseBbox56Track<'a, Id> {
    pub track_id: Id,
    pub keypoints: &'a [Keypoint],
    pub bbox: Option<&'a [f64]>,
}

/// Stable ordering for comparable track ids, as in Python's sorted.
pub fn pose_bbox56_tracks<'a, Id: Ord>(
    tracks: impl IntoIterator<Item = PoseBbox56Track<'a, Id>>,
    frame_width: i64,
    frame_height: i64,
) -> Vec<(Id, PoseBbox56Row)> {
    let mut rows: Vec<_> = tracks
        .into_iter()
        .map(|track| {
            (
                track.track_id,
                pose_bbox56_row(track.keypoints, track.bbox, frame_width, frame_height),
            )
        })
        .collect();
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    rows
}

/// Per-track storage only: no track registry, TTL, model, stride, or policy ownership.
/// The domain caller clears this together with its resampler on a stream-epoch reset.
#[derive(Debug)]
pub struct PoseBbox56History {
    rows: VecDeque<PoseBbox56Row>,
    last_row: Option<PoseBbox56Row>,
}
impl Default for PoseBbox56History {
    fn default() -> Self {
        Self {
            rows: VecDeque::with_capacity(FALL_WINDOW_FRAMES),
            last_row: None,
        }
    }
}
impl PoseBbox56History {
    /// Missing/nonfinite observations coast; cadence gaps must pass Some(ZERO_ROW).
    /// Zero rows are valid numeric input and replace the previously held pose.
    pub fn push(&mut self, row: Option<PoseBbox56Row>) {
        if let Some(row) = row.filter(|row| row.iter().all(|v| v.is_finite())) {
            self.last_row = Some(row);
        }
        if self.rows.len() == FALL_WINDOW_FRAMES {
            self.rows.pop_front();
        }
        self.rows.push_back(self.last_row.unwrap_or(ZERO_ROW));
    }

    pub fn rows(&self) -> &VecDeque<PoseBbox56Row> {
        &self.rows
    }

    pub fn clear(&mut self) {
        self.rows.clear();
        self.last_row = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal::{CADENCE_NS, PtsResampler};

    #[test]
    fn threshold_clipping_and_tail_match_float32_feature_contract() {
        let mut points = [[20.0, 30.0, 0.9]; COCO17_KEYPOINTS];
        points[0] = [-10.0, 1000.0, 0.5];
        points[1] = [20.0, 30.0, 0.5 - f64::EPSILON];
        points[2] = [200.0, -10.0, 1.2];
        let row = pose_bbox56_row(&points, Some(&[-2.0, -3.0, 1000.0, 1000.0]), 100, 50);
        assert_eq!(&row[0..9], &[0.0, 0.98, 0.5, 0.0, 0.0, 0.0, 0.99, 0.0, 1.2]);
        assert_eq!(&row[51..56], &[0.0, 0.0, 0.99, 0.98, 1.0]);
        points.fill([20.0, 30.0, 0.49]);
        let row = pose_bbox56_row(&points, Some(&[0.0, 0.0, 50.0, 25.0]), 100, 50);
        assert_eq!(&row[..51], &[0.0; 51]);
        assert_eq!(&row[51..56], &[0.0, 0.0, 0.5, 0.5, 1.0]);
    }

    #[test]
    fn rounds_after_float64_division_and_preserves_signed_zero() {
        let points = [[1.00000006, -0.0, 0.5]; COCO17_KEYPOINTS];
        let row = pose_bbox56_row(&points, Some(&[-0.0, 0.0, 2.0, 2.0]), 3, 3);
        assert_eq!(row[0].to_bits(), 0x3eaa_aaab);
        assert_eq!(row[1].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(row[51].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn invalid_dimensions_shapes_and_boxes_zero_the_entire_row() {
        let points = [[20.0, 30.0, 0.9]; COCO17_KEYPOINTS];
        let bbox = [0.0, 0.0, 90.0, 40.0];
        for (width, height) in [(0, 50), (-1, 50), (100, 0), (100, -1), (1, 50), (100, 1)] {
            assert_eq!(
                pose_bbox56_row(&points, Some(&bbox), width, height),
                ZERO_ROW
            );
        }
        assert_eq!(
            pose_bbox56_row(&points[..16], Some(&bbox), 100, 50),
            ZERO_ROW
        );
        let short_points = [[1.0, 2.0]; COCO17_KEYPOINTS];
        assert_eq!(
            pose_bbox56_row(&short_points, Some(&bbox), 100, 50),
            ZERO_ROW
        );
        for bbox in [
            None,
            Some(&[0.0, 0.0, 1.0][..]),
            Some(&[4.0, 0.0, 3.0, 4.0][..]),
            Some(&[0.0, 4.0, 3.0, 4.0][..]),
            Some(&[-4.0, 0.0, -1.0, 4.0][..]),
        ] {
            assert_eq!(pose_bbox56_row(&points, bbox, 100, 50), ZERO_ROW);
        }
    }

    #[test]
    fn nonfinite_inputs_and_float32_overflow_never_escape() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for component in 0..3 {
                let mut points = [[20.0, 30.0, 0.1]; COCO17_KEYPOINTS];
                points[0][component] = bad;
                assert_eq!(
                    pose_bbox56_row(&points, Some(&[0.0, 0.0, 90.0, 40.0]), 100, 50),
                    ZERO_ROW
                );
            }
            for component in 0..4 {
                let mut bbox = [0.0, 0.0, 90.0, 40.0];
                bbox[component] = bad;
                assert_eq!(
                    pose_bbox56_row(&[[1.0, 1.0, 0.9]; 17], Some(&bbox), 100, 50),
                    ZERO_ROW
                );
            }
        }
        let points = [[20.0, 30.0, f64::MAX]; COCO17_KEYPOINTS];
        assert_eq!(
            pose_bbox56_row(&points, Some(&[0.0, 0.0, 90.0, 40.0]), 100, 50),
            ZERO_ROW
        );
    }

    #[test]
    fn native_authority_and_stable_track_order_are_preserved() {
        let points = [[20.0, 30.0, 0.9]; COCO17_KEYPOINTS];
        let bbox = [0.0, 0.0, 90.0, 40.0];
        let expected = pose_bbox56_row(&points, Some(&bbox), 100, 50);
        assert_eq!(
            native_pose_bbox56_row(&points, Some(&bbox), 100, 50, "pose"),
            expected
        );
        for source in ["person", "tracker", "", "Pose"] {
            assert_eq!(
                native_pose_bbox56_row(&points, Some(&bbox), 100, 50, source),
                ZERO_ROW
            );
        }
        let tracks =
            [(2, Some(&bbox[..])), (1, None), (1, Some(&bbox[..]))].map(|(track_id, bbox)| {
                PoseBbox56Track {
                    track_id,
                    keypoints: &points,
                    bbox,
                }
            });
        assert_eq!(
            pose_bbox56_tracks(tracks, 100, 50),
            vec![(1, ZERO_ROW), (1, expected), (2, expected)]
        );
    }

    #[test]
    fn history_bounds_coasting_and_explicit_gap_zeros_follow_classifier_storage() {
        let mut history = PoseBbox56History::default();
        history.push(None);
        assert_eq!(history.rows().back(), Some(&ZERO_ROW));
        for index in 0..35 {
            history.push(Some([index as f32; 56]));
        }
        assert_eq!(history.rows().len(), 30);
        assert_eq!(history.rows().front(), Some(&[5.0; 56]));
        history.push(Some([f32::NAN; 56]));
        history.push(None);
        assert_eq!(history.rows().back(), Some(&[34.0; 56]));
        history.push(Some(ZERO_ROW));
        history.push(None);
        assert_eq!(history.rows().back(), Some(&ZERO_ROW));
        assert_eq!(history.rows().len(), 30);
    }

    #[test]
    fn epoch_owner_resets_both_cadence_and_history_on_rollback() {
        let mut sampler = PtsResampler::default();
        let mut history = PoseBbox56History::default();
        for pts in [100, 100 + 3 * CADENCE_NS] {
            for row in sampler.push(pts, [0.5; 56]).unwrap() {
                history.push(Some(row.value.unwrap_or(ZERO_ROW)));
            }
        }
        assert_eq!(history.rows().len(), 4);
        assert_eq!(history.rows()[1], ZERO_ROW);
        assert_eq!(history.rows()[2], ZERO_ROW);
        assert!(sampler.push(5, [1.0; 56]).unwrap().is_empty());
        // Like FallDomainDecider._reset_on_pts_rollback: reset is the caller's decision.
        sampler.reset();
        history.clear();
        history.push(None);
        assert_eq!(history.rows().back(), Some(&ZERO_ROW));
        history.clear();
        for row in sampler.push(5, [1.0; 56]).unwrap() {
            history.push(Some(row.value.unwrap_or(ZERO_ROW)));
        }
        assert_eq!(history.rows().len(), 1);
        assert_eq!(history.rows().back(), Some(&[1.0; 56]));
    }
}
