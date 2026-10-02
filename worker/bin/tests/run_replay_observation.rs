//! Accepted-frame observation capture from the public ingest path.
//! Configured mux geometry is the matched object's converted pose, before
//! pose-bbox56 reduction. Capture is optional and does not change the fall inputs.

use seeon_deepstream_native::{FrameIdentity, MEDIA_POSE_COLUMNS, TrackedObject};
use seeon_ml_worker::msg::PosePacket;
use seeon_ml_worker::policy::ingest::{IngestRefusal, ObservedPose, ingest};
use seeon_worker::pose_bbox56::COCO17_KEYPOINTS;

const SOURCE_WIDTH: u32 = 1280;
const SOURCE_HEIGHT: u32 = 720;
/// Python `convert_frame` frame size for the proven unequal plane.
const MUX_WIDTH: u32 = 1280;
const MUX_HEIGHT: u32 = 720;

fn point(x: f32, y: f32, confidence: f32) -> [f32; 3] {
    [x, y, confidence]
}

fn row(
    box_xyxy: [f32; 4],
    score: f32,
    keypoints: [[f32; 3]; COCO17_KEYPOINTS],
) -> [f32; MEDIA_POSE_COLUMNS] {
    let mut values = [0.0; MEDIA_POSE_COLUMNS];
    values[..4].copy_from_slice(&box_xyxy);
    values[4] = score;
    for (index, point) in keypoints.iter().enumerate() {
        values[6 + index * 3..9 + index * 3].copy_from_slice(point);
    }
    values
}

fn uniform_points(xy: [f32; 2], confidence: f32) -> [[f32; 3]; COCO17_KEYPOINTS] {
    [point(xy[0], xy[1], confidence); COCO17_KEYPOINTS]
}

fn object(
    track_id: u64,
    left: f32,
    top: f32,
    width: f32,
    height: f32,
    confidence: f32,
) -> TrackedObject {
    TrackedObject {
        track_id,
        left,
        top,
        width,
        height,
        confidence,
    }
}

fn packet(
    sequence: u64,
    rows: Vec<[f32; MEDIA_POSE_COLUMNS]>,
    objects: Vec<TrackedObject>,
) -> PosePacket {
    PosePacket {
        frame: FrameIdentity {
            sequence,
            pts_ns: 2_000_000_000,
            pts_valid: 1,
            frame_number: i64::try_from(sequence).expect("sequence"),
            source_width: SOURCE_WIDTH,
            source_height: SOURCE_HEIGHT,
            analysis_width: SOURCE_WIDTH,
            analysis_height: SOURCE_HEIGHT,
            ..FrameIdentity::default()
        },
        tensor_present: true,
        rows,
        objects,
    }
}

fn crossed() -> PosePacket {
    // 1280x720 letterbox scale is 0.5. Rows are net coordinates, listed low
    // then high so a raw row zip cannot explain the native 9-then-2 match.
    // Feature keys still sort to 2 then 9. There is no equal-IoU tie.
    let low = row([100.25, 15.25, 140.25, 95.25], 0.375, {
        let mut points = uniform_points([105.0, 25.0], 0.75);
        points[3] = point(-4.0, 900.0, 0.25);
        points
    });
    let high = row(
        [5.125, 10.375, 55.125, 110.375],
        0.875,
        uniform_points([20.0, 30.0], 0.5),
    );
    packet(
        7,
        vec![low, high],
        vec![
            object(9, 10.25, 20.75, 100.0, 200.0, 0.125),
            object(2, 200.5, 30.5, 80.0, 160.0, 0.9375),
        ],
    )
}

fn expected_crossed() -> Vec<ObservedPose> {
    let high = [[40.0, 60.0, 0.5]; COCO17_KEYPOINTS];
    let mut low = [[210.0, 50.0, 0.75]; COCO17_KEYPOINTS];
    low[3] = [0.0, 720.0, 0.25];
    vec![
        ObservedPose {
            track_id: 9,
            bbox: [10.0, 20.0, 110.0, 220.0],
            confidence: 0.125,
            keypoints: high.to_vec(),
        },
        ObservedPose {
            track_id: 2,
            bbox: [200.0, 30.0, 280.0, 190.0],
            confidence: 0.9375,
            keypoints: low.to_vec(),
        },
    ]
}

#[test]
fn disabled_and_enabled_capture_return_the_same_frame() {
    let packet = crossed();
    let disabled = ingest(&packet, None).expect("disabled frame");
    let mut observed = vec![ObservedPose {
        track_id: 1,
        bbox: [1.0, 2.0, 3.0, 4.0],
        confidence: 0.5,
        keypoints: vec![[9.0, 9.0, 9.0]],
    }];
    let enabled = ingest(&packet, Some(&mut observed)).expect("enabled frame");
    assert_eq!(enabled, disabled);
    assert_eq!(observed, expected_crossed());
    assert_eq!(enabled.rows.keys().copied().collect::<Vec<_>>(), vec![2, 9]);
}

#[test]
fn confidence_comes_from_the_matched_object_not_the_row_score() {
    let packet = crossed();
    let mut observed = Vec::new();
    ingest(&packet, Some(&mut observed)).expect("frame");
    assert_eq!(observed, expected_crossed());
    assert_eq!(f64::from(packet.objects[0].confidence), 0.125);
    assert_eq!(f64::from(packet.rows[1][4]), 0.875);
    assert_eq!(
        observed[0].confidence,
        f64::from(packet.objects[0].confidence)
    );
    assert_ne!(observed[0].confidence, f64::from(packet.rows[1][4]));
    assert_eq!(observed[0].bbox, [10.0, 20.0, 110.0, 220.0]);
    assert_eq!(observed[1].keypoints[3], [0.0, 720.0, 0.25]);
}

#[test]
fn empty_and_unmatched_frames_replace_previous_observations() {
    let mut observed = expected_crossed();
    let empty = packet(8, Vec::new(), Vec::new());
    let frame = ingest(&empty, Some(&mut observed)).expect("empty frame");
    assert!(frame.rows.is_empty());
    assert!(frame.live_track_ids.is_empty());
    assert!(observed.is_empty());

    let unmatched = packet(
        9,
        vec![row(
            [400.0, 400.0, 420.0, 430.0],
            0.8,
            uniform_points([405.0, 410.0], 0.9),
        )],
        vec![object(4, 10.0, 20.0, 30.0, 40.0, 0.77)],
    );
    let frame = ingest(&unmatched, Some(&mut observed)).expect("unmatched frame");
    assert!(frame.rows.is_empty());
    assert_eq!(frame.live_track_ids, vec![4]);
    assert!(observed.is_empty());
}

#[test]
fn source_size_and_sequence_refusals_leave_prior_observations() {
    let prior = expected_crossed();
    let mut observed = prior.clone();
    let mut refused = crossed();
    refused.frame.source_width = 0;
    assert_eq!(
        ingest(&refused, Some(&mut observed)).expect_err("source size"),
        IngestRefusal::SourceSize
    );
    assert_eq!(observed, prior);

    let mut refused = crossed();
    refused.frame.source_height = 0;
    assert_eq!(
        ingest(&refused, Some(&mut observed)).expect_err("source height"),
        IngestRefusal::SourceSize
    );
    assert_eq!(observed, prior);

    let mut refused = crossed();
    refused.frame.sequence = u64::MAX;
    assert_eq!(
        ingest(&refused, Some(&mut observed)).expect_err("sequence"),
        IngestRefusal::Sequence
    );
    assert_eq!(observed, prior);
}

#[test]
fn unequal_decoded_planes_share_configured_mux_geometry() {
    // Parent's independent Python converter: mux 1280x720, decoded 640x360,
    // SDK box [200, 100, 400, 400] matches net box [100, 50, 200, 200] as ID 7.
    // The same net row against that decoded plane has IoU 0, so old source
    // scaling cannot produce this match. Keypoints follow the same scale.
    let mut points = uniform_points([120.0, 80.0], 0.75);
    // Both scale and clamp must use the mux plane, not the decoded extent.
    points[0] = [700.0, 400.0, 0.75];
    let row = row([100.0, 50.0, 200.0, 200.0], 0.8, points);
    let object = object(7, 200.0, 100.0, 200.0, 300.0, 0.625);
    let mut decoded = packet(7, vec![row], vec![object]);
    decoded.frame.source_width = 640;
    decoded.frame.source_height = 360;
    decoded.frame.analysis_width = MUX_WIDTH;
    decoded.frame.analysis_height = MUX_HEIGHT;
    let mut equal = packet(7, vec![row], vec![object]);
    equal.frame.source_width = MUX_WIDTH;
    equal.frame.source_height = MUX_HEIGHT;

    let mut decoded_observed = Vec::new();
    let mut equal_observed = Vec::new();
    let decoded_frame = ingest(&decoded, Some(&mut decoded_observed)).expect("decoded plane");
    let equal_frame = ingest(&equal, Some(&mut equal_observed)).expect("equal plane");

    let mut keypoints = [[240.0, 160.0, 0.75]; COCO17_KEYPOINTS];
    keypoints[0] = [1280.0, 720.0, 0.75];
    let expected = vec![ObservedPose {
        track_id: 7,
        bbox: [200.0, 100.0, 400.0, 400.0],
        confidence: 0.625,
        keypoints: keypoints.to_vec(),
    }];
    assert!(!decoded_observed.is_empty());
    assert_eq!(decoded_observed, expected);
    assert_eq!(equal_observed, expected);
    assert_eq!(decoded_frame.rows, equal_frame.rows);
    assert_eq!(decoded_frame.rows.len(), 1);
    assert_eq!(
        (decoded_frame.width, decoded_frame.height),
        (i64::from(MUX_WIDTH), i64::from(MUX_HEIGHT))
    );
    assert_eq!(
        (equal_frame.width, equal_frame.height),
        (i64::from(MUX_WIDTH), i64::from(MUX_HEIGHT))
    );
    assert_eq!(decoded_frame.identity, decoded.frame);
    assert_eq!(equal_frame.identity, equal.frame);
    assert_ne!(
        (
            decoded_frame.identity.source_width,
            decoded_frame.identity.source_height
        ),
        (
            equal_frame.identity.source_width,
            equal_frame.identity.source_height
        )
    );
}

#[test]
fn analysis_extent_refusals_leave_prior_observations() {
    let prior = expected_crossed();
    for (width, height) in [(0, MUX_HEIGHT), (MUX_WIDTH, 0)] {
        let mut observed = prior.clone();
        let mut refused = crossed();
        refused.frame.analysis_width = width;
        refused.frame.analysis_height = height;
        assert_eq!(
            ingest(&refused, Some(&mut observed)).expect_err("analysis extent"),
            IngestRefusal::SourceSize
        );
        assert_eq!(observed, prior);
    }
}
