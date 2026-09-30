//! Glue from one `PosePacket` to the `seeon-worker` fall inputs, ported from
//! `metadata.py` `association_pass`/`convert_frame` and the row building of
//! `FallDomainDecider.update` (`policy.py` L348-361). Row construction itself
//! is the domain crate's `pose_bbox56_tracks`.

use std::collections::BTreeMap;

use seeon_deepstream_native::{FrameIdentity, TrackedObject};
use seeon_worker::pose_bbox56::{
    COCO17_KEYPOINTS, Keypoint, PoseBbox56Row, PoseBbox56Track, pose_bbox56_tracks,
};

use crate::msg::PosePacket;

/// Python `_NET_SIZE`: the square nvinfer input the rows are expressed in.
const NET_SIZE: f64 = 640.0;
/// Python `_SCORE_MIN`: rows at or below this score are no candidates.
const SCORE_MIN: f32 = 0.05;
/// Python `_IOU_GATE`: pairs below this overlap never associate.
const IOU_GATE: f64 = 0.5;

type Box4 = [f64; 4];

/// One frame's fall inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub identity: FrameIdentity,
    pub width: i64,
    pub height: i64,
    /// Every tracked object, matched or not, in native order.
    pub live_track_ids: Vec<u64>,
    /// Rows of the tracks a pose row associated with.
    pub rows: BTreeMap<u64, PoseBbox56Row>,
    /// `None` when native marked the presentation time invalid.
    pub time_sec: Option<f64>,
    /// The publish sequence, Python `frame_index`.
    pub frame_index: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IngestRefusal {
    /// A zero source width or height leaves no letterbox scale.
    SourceSize,
    /// The publish sequence does not fit a frame index.
    Sequence,
}

pub fn ingest(packet: &PosePacket) -> Result<Frame, IngestRefusal> {
    let identity = packet.frame;
    if identity.source_width == 0 || identity.source_height == 0 {
        return Err(IngestRefusal::SourceSize);
    }
    let frame_index = i64::try_from(identity.sequence).map_err(|_| IngestRefusal::Sequence)?;
    let width = i64::from(identity.source_width);
    let height = i64::from(identity.source_height);
    let (frame_w, frame_h) = (
        f64::from(identity.source_width),
        f64::from(identity.source_height),
    );
    // NumPy float32 divided by a Python float stays float32 (NEP 50).
    let scale = (NET_SIZE / frame_w).min(NET_SIZE / frame_h) as f32;
    let candidates: Vec<(usize, Box4)> = packet
        .rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row[4] > SCORE_MIN)
        .map(|(index, row)| (index, std::array::from_fn(|i| f64::from(row[i] / scale))))
        .collect();
    let boxes: Vec<Box4> = packet.objects.iter().map(object_box).collect();
    let assigned = associate(&boxes, &candidates);
    let mut keypoints: Vec<(u64, Box4, Vec<Keypoint>)> = Vec::new();
    for ((object, object_box), candidate) in packet.objects.iter().zip(&boxes).zip(assigned) {
        let Some(candidate) = candidate else {
            continue;
        };
        let row = &packet.rows[candidates[candidate].0];
        let points = (0..COCO17_KEYPOINTS)
            .map(|point| {
                [
                    clamp(f64::from(row[6 + point * 3] / scale), frame_w),
                    clamp(f64::from(row[7 + point * 3] / scale), frame_h),
                    f64::from(row[8 + point * 3]),
                ]
            })
            .collect();
        keypoints.push((object.track_id, object_box.map(f64::trunc), points));
    }
    let tracks = keypoints
        .iter()
        .map(|(track_id, bbox, points)| PoseBbox56Track {
            track_id: *track_id,
            keypoints: points,
            bbox: Some(bbox.as_slice()),
        });
    let rows = pose_bbox56_tracks(tracks, width, height)
        .into_iter()
        .collect();
    let time_sec = (identity.pts_valid != 0).then(|| identity.pts_ns as f64 / 1e9);
    Ok(Frame {
        identity,
        width,
        height,
        live_track_ids: packet
            .objects
            .iter()
            .map(|object| object.track_id)
            .collect(),
        rows,
        time_sec,
        frame_index,
    })
}

fn object_box(object: &TrackedObject) -> Box4 {
    let (left, top) = (f64::from(object.left), f64::from(object.top));
    [
        left,
        top,
        left + f64::from(object.width),
        top + f64::from(object.height),
    ]
}

/// Greedy matching over all pairs by descending IoU, then object and
/// candidate position; the candidate index each object took, if any.
fn associate(objects: &[Box4], candidates: &[(usize, Box4)]) -> Vec<Option<usize>> {
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (object, object_box) in objects.iter().enumerate() {
        for (candidate, (_, candidate_box)) in candidates.iter().enumerate() {
            pairs.push((iou(object_box, candidate_box), object, candidate));
        }
    }
    pairs.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then(left.1.cmp(&right.1))
            .then(left.2.cmp(&right.2))
    });
    let mut assigned = vec![None; objects.len()];
    let mut used = vec![false; candidates.len()];
    for (score, object, candidate) in pairs {
        if score < IOU_GATE {
            break;
        }
        if assigned[object].is_some() || used[candidate] {
            continue;
        }
        assigned[object] = Some(candidate);
        used[candidate] = true;
    }
    assigned
}

/// Python `_iou`; a NaN overlap counts as none.
fn iou(a: &Box4, b: &Box4) -> f64 {
    let (left, top) = (pymax(a[0], b[0]), pymax(a[1], b[1]));
    let (right, bottom) = (pymin(a[2], b[2]), pymin(a[3], b[3]));
    let intersection = pymax(0.0, right - left) * pymax(0.0, bottom - top);
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
    let overlap = if union <= 0.0 {
        0.0
    } else {
        intersection / union
    };
    if overlap.is_nan() { 0.0 } else { overlap }
}

/// `int(max(0.0, min(float(limit), value)))`.
fn clamp(value: f64, limit: f64) -> f64 {
    pymax(0.0, pymin(limit, value)).trunc()
}

/// Python `max(a, b)`: `a` unless `b` compares greater.
fn pymax(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python `min(a, b)`: `a` unless `b` compares less.
fn pymin(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}
