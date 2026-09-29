//! Bed-exit and detection-window replay shared by the bed-policy and window
//! fixtures. Requests are BEDPROBE/WINDOWPROBE v1 rows the Python parity tests
//! built; replays drive the public `BedExitMonitor` and `DetectionWindow` and
//! render their answers in the probe's rows. Malformed request rows panic as
//! fixture defects; a refused construction, clock or window is `Err`.
#![allow(dead_code)]

use crate::policy::{self, Fields, MAX_ITEMS, bits, hex, optional};
use seeon_worker::bed_exit::*;
use seeon_worker::detection_window::{AwareDateTime, ClockRelation, DateTime, DetectionWindow};
use seeon_worker::episode::BusinessEvent;
use std::collections::BTreeSet;
use std::fmt::Write;
use std::path::Path;

pub const BED_HEADER: &str = "BEDPROBE\t1";
pub const WINDOW_HEADER: &str = "WINDOWPROBE\t1";

fn collect<'a, T>(fields: &mut Fields<'a>, read: fn(&mut Fields<'a>) -> T) -> Vec<T> {
    (0..fields.count()).map(|_| read(fields)).collect()
}

fn collect_floats(fields: &mut Fields<'_>) -> Vec<f64> {
    (0..fields.count()).map(|_| fields.float()).collect()
}

fn boolean(fields: &mut Fields<'_>) -> bool {
    match fields.next() {
        "0" => false,
        "1" => true,
        token => panic!("fixture boolean token {token:?}"),
    }
}

fn box_value(fields: &mut Fields<'_>) -> BoundingBox {
    let (x1, y1, x2, y2) = (
        fields.number(),
        fields.number(),
        fields.number(),
        fields.number(),
    );
    let confidence = fields.float();
    let polygon = match fields.next() {
        "-" => None,
        count => Some(
            (0..Fields::new(count).count())
                .map(|_| (fields.number(), fields.number()))
                .collect(),
        ),
    };
    BoundingBox {
        x1,
        y1,
        x2,
        y2,
        confidence,
        polygon,
    }
}

fn pose(fields: &mut Fields<'_>) -> BedPoseFeatures {
    BedPoseFeatures {
        track_id: fields.number(),
        bed_id: fields.optional_id(),
        torso_in_frac: fields.float(),
        lower_in_frac: fields.float(),
        keypoint_in_frac: fields.float(),
        hip_depth: fields.float(),
        torso_angle: fields.float(),
        centroid_displacement: fields.float(),
        hip_x_rel: fields.float(),
        hip_y_rel: fields.float(),
        observability: fields.float(),
        bed_polygon_valid: boolean(fields),
    }
}

/// `-` or S|E, hex civil ISO datetime and offset seconds (`-` = naive).
fn wall(fields: &mut Fields<'_>) -> Result<Option<AwareDateTime>, String> {
    let relation = match fields.next() {
        "-" => return Ok(None),
        "S" => ClockRelation::SameTargetTzinfo,
        "E" => ClockRelation::DifferentTzinfo,
        tag => panic!("fixture clock relation {tag:?}"),
    };
    let text = fields.text();
    let local: DateTime = text
        .parse()
        .map_err(|error| format!("civil datetime {text:?}: {error}"))?;
    let offset = match fields.next() {
        "-" => None,
        // Whole seconds only, like the probe: "0.5" is a transport refusal.
        seconds => Some(
            seconds
                .parse()
                .map_err(|error| format!("offset {seconds:?}: {error}"))?,
        ),
    };
    AwareDateTime::new(local, offset, relation)
        .map(Some)
        .map_err(|error| format!("clock: {error:?}"))
}

/// `-` or hex start, end and IANA key, read from the recorded zone tree only.
fn window(
    fields: &mut Fields<'_>,
    zoneinfo: Option<&Path>,
) -> Result<Option<DetectionWindow>, String> {
    let token = fields.next();
    if token == "-" {
        return Ok(None);
    }
    let start = Fields::new(token).text();
    let (end, zone) = (fields.text(), fields.text());
    let root = zoneinfo.expect("fixture window row needs the recorded zoneinfo tree");
    DetectionWindow::from_zoneinfo_dir(&start, &end, &zone, root)
        .map(Some)
        .map_err(|error| format!("window: {error:?}"))
}

fn construct(
    fields: &mut Fields<'_>,
    zoneinfo: Option<&Path>,
) -> Result<(BedExitMonitor, Vec<(u64, usize)>), String> {
    assert_eq!(fields.next(), "N", "construction row");
    let (camera_id, facility_id, boot, epoch) =
        (fields.text(), fields.text(), fields.text(), fields.text());
    let generation = fields.number();
    let mut config = BedExitConfig {
        camera_id,
        facility_id,
        min_containment: fields.float(),
        hold_frames: fields.number(),
        grace_frames: fields.number(),
        in_bed_dwell_sec: fields.float(),
        outside_dwell_sec: fields.float(),
        night_window: None,
    };
    let stale_after = fields.float();
    let episodes = fields.count();
    config.night_window = window(fields, zoneinfo)?;
    let watch: Vec<(u64, usize)> = (0..fields.count())
        .map(|_| (fields.number(), fields.number()))
        .collect();
    assert_eq!(
        watch.iter().collect::<BTreeSet<_>>().len(),
        watch.len(),
        "duplicate watch pair"
    );
    fields.end();
    let capacities = BedExitCapacities {
        tracks: MAX_ITEMS,
        observations: MAX_ITEMS,
        beds: MAX_ITEMS,
        pose_features: MAX_ITEMS,
        polygon_points: MAX_ITEMS,
        episodes,
    };
    let monitor = BedExitMonitor::new(config, boot, epoch, generation, capacities, stale_after)
        .map_err(|error| format!("monitor: {error:?}"))?;
    Ok((monitor, watch))
}

fn update(fields: &mut Fields<'_>) -> Result<(BedExitInput, BedExitClocks), String> {
    let (frame_index, time_sec) = (fields.number(), fields.optional_float());
    let clocks = BedExitClocks {
        observed_at: fields.float(),
        snapshot_at: fields.float(),
        wall_time: wall(fields)?,
    };
    let source = match fields.text().as_str() {
        "fresh" => BedRegionCacheState::Fresh,
        "cached" => BedRegionCacheState::Cached,
        "empty" => BedRegionCacheState::Empty,
        "expired" => BedRegionCacheState::Expired,
        other => panic!("fixture region source {other:?}"),
    };
    let input = BedExitInput {
        frame_index,
        time_sec,
        bed_region: BedRegionDebugSnapshot {
            source,
            empty_cycles: fields.number(),
        },
        person_boxes: collect(fields, box_value),
        bed_boxes: collect(fields, box_value),
        track_ids: collect(fields, Fields::optional_id),
        live_track_ids: collect(fields, Fields::number),
        bed_pose_features: collect(fields, pose),
        containments: collect(fields, collect_floats),
        prior_box_containments: collect(fields, |fields| PriorBoxContainments {
            previous_track_id: fields.number(),
            previous_box: box_value(fields),
            ratios: collect_floats(fields),
        }),
    };
    fields.end();
    Ok((input, clocks))
}

fn box_fields(value: &BoundingBox) -> String {
    let mut text = format!(
        "{}\t{}\t{}\t{}\t{}",
        value.x1,
        value.y1,
        value.x2,
        value.y2,
        bits(value.confidence)
    );
    match &value.polygon {
        None => text.push_str("\t-"),
        Some(points) => {
            write!(text, "\t{}", points.len()).expect("writing to String");
            for (x, y) in points {
                write!(text, "\t{x}\t{y}").expect("writing to String");
            }
        }
    }
    text
}

fn scoring(out: &mut String, tag: &str, score: BedExitScoring) {
    writeln!(
        out,
        "{tag}\t{}\t{}\t{}",
        bits(score.max_containment_observed),
        score.grace_positive_transitions,
        score.assignments_made
    )
    .expect("writing to String");
}

fn observe(
    out: &mut String,
    index: usize,
    monitor: &BedExitMonitor,
    events: &[BusinessEvent],
    score: Option<BedExitScoring>,
    watch: &[(u64, usize)],
) {
    writeln!(
        out,
        "O\t{index}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        monitor.track_id_switch_absorbed_total(),
        monitor.last_shadow_trace_count(),
        u8::from(monitor.last_update_complete()),
        u8::from(score.is_some()),
        events.len(),
        monitor.last_trace_snapshots().len(),
        monitor.assignments().len(),
        monitor.last_recovery_events().len(),
        monitor.last_lost_track_ids().len(),
        u8::from(monitor.last_debug_snapshot().is_some()),
        optional(
            monitor
                .last_episode_disposition()
                .map(|disposition| hex(disposition.as_str()))
        )
    )
    .expect("writing to String");
    policy::events(out, events);
    policy::traces(out, monitor.last_trace_snapshots());
    scoring(out, "G", monitor.scoring());
    if let Some(score) = score {
        scoring(out, "Q", score);
    }
    for (track, assignment) in monitor.assignments() {
        writeln!(
            out,
            "A\t{track}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            optional(assignment.bed_id),
            optional(assignment.candidate_bed_id),
            assignment.candidate_frames,
            bits(assignment.in_bed_dwell_sec),
            bits(assignment.outside_dwell_sec),
            u8::from(assignment.armed),
            optional(assignment.last_time_sec.map(bits)),
            assignment
                .last_box
                .as_ref()
                .map_or_else(|| "-".into(), |value| format!("1\t{}", box_fields(value)))
        )
        .expect("writing to String");
    }
    for event in monitor.last_recovery_events() {
        writeln!(out, "H\t{}\t{}", event.person_id, event.bed_id).expect("writing to String");
    }
    for track in monitor.last_lost_track_ids() {
        writeln!(out, "L\t{track}").expect("writing to String");
    }
    if let Some(debug) = monitor.last_debug_snapshot() {
        let region = debug.bed_region.map_or_else(
            || "-".into(),
            |region| format!("{}\t{}", hex(region.source.as_str()), region.empty_cycles),
        );
        writeln!(
            out,
            "D\t{}\t{}\t{}\t{region}\t{}\t{}\t{}\t{}",
            optional(debug.frame_index),
            u8::from(debug.stale),
            optional(debug.observation_age_sec.map(bits)),
            debug.person_boxes.len(),
            debug.bed_boxes.len(),
            debug.statuses.len(),
            debug.events.len()
        )
        .expect("writing to String");
        for value in &debug.person_boxes {
            writeln!(out, "P\t{}", box_fields(value)).expect("writing to String");
        }
        for value in &debug.bed_boxes {
            writeln!(out, "B\t{}", box_fields(value)).expect("writing to String");
        }
        for status in &debug.statuses {
            writeln!(
                out,
                "S\t{}\t{}\t{}\t{}",
                status.bed_id,
                hex(status.occupancy.as_str()),
                optional(status.person_id),
                box_fields(&status.box_value)
            )
            .expect("writing to String");
        }
        for event in &debug.events {
            writeln!(out, "J\t{}\t{}", event.person_id, event.bed_id).expect("writing to String");
        }
    }
    for &(track, bed) in watch {
        writeln!(
            out,
            "Z\t{track}\t{bed}\t{}",
            hex(monitor.episode_state(track, bed).as_str())
        )
        .expect("writing to String");
    }
}

fn failure(out: &mut String, index: usize, error: BedExitError) -> Vec<BusinessEvent> {
    let (kind, cause, phase, track, events) = match error {
        BedExitError::Rejected(cause) => ("rejected", cause, None, None, Vec::new()),
        BedExitError::Poisoned(cause) => ("poisoned", cause, None, None, Vec::new()),
        BedExitError::FatalPartialState {
            phase,
            track_id,
            cause,
            emitted_events,
        } => ("fatal", cause, Some(phase), track_id, emitted_events),
    };
    writeln!(
        out,
        "X\t{index}\t{kind}\t{}\t{}\t{}\t{}",
        hex(&format!("{cause:?}")),
        optional(phase.map(|phase| hex(&format!("{phase:?}")))),
        optional(track),
        events.len()
    )
    .expect("writing to String");
    events
}

/// BEDPROBE v1 replay: rows, and whether every call succeeded (no X row).
/// Like the probe, a domain failure is rendered and later calls still run.
pub fn replay_monitor(request: &[&str], zoneinfo: Option<&Path>) -> Result<(String, bool), String> {
    assert_eq!(request.first(), Some(&BED_HEADER), "request header");
    let (mut monitor, watch) = construct(&mut Fields::new(request[1]), zoneinfo)?;
    let mut out = format!("{BED_HEADER}\n");
    observe(&mut out, 0, &monitor, &[], None, &watch);
    let mut calls = 1;
    let mut success = true;
    for line in &request[2..] {
        let mut fields = Fields::new(line);
        let outcome = match fields.next() {
            "U" => {
                let (input, clocks) = update(&mut fields)?;
                monitor.update(&input, clocks)
            }
            "C" => {
                let frame = match fields.next() {
                    "-" => None,
                    token => Some(
                        token
                            .parse::<i64>()
                            .unwrap_or_else(|_| panic!("fixture frame {token:?}")),
                    ),
                };
                let now = fields.float();
                fields.end();
                monitor.coast(frame, now)
            }
            "W" => {
                let window = window(&mut fields, zoneinfo)?;
                fields.end();
                monitor
                    .update_night_window(window)
                    .map(|()| BedExitOutcome {
                        events: Vec::new(),
                        scoring_observation: None,
                    })
            }
            "R" => {
                let event = policy::event(&mut fields);
                fields.end();
                // Python's release returns nothing; only its later effects are compared.
                monitor.release_onset(&event.identity);
                Ok(BedExitOutcome {
                    events: Vec::new(),
                    scoring_observation: None,
                })
            }
            verb => panic!("unknown request verb {verb:?}"),
        };
        let (events, score) = match outcome {
            Ok(outcome) => (outcome.events, outcome.scoring_observation),
            Err(error) => {
                success = false;
                (failure(&mut out, calls, error), None)
            }
        };
        observe(&mut out, calls, &monitor, &events, score, &watch);
        calls += 1;
    }
    writeln!(out, "END\t{calls}").expect("writing to String");
    Ok((out, success))
}

/// BEDPROBE replay for parity: any X row is a refusal of the whole exchange.
pub fn replay(request: &[&str], zoneinfo: Option<&Path>) -> Result<String, String> {
    let (out, success) = replay_monitor(request, zoneinfo)?;
    if success {
        Ok(out)
    } else {
        let failures: Vec<&str> = out.lines().filter(|row| row.starts_with("X\t")).collect();
        Err(format!("domain failure rows {failures:?}"))
    }
}

/// WINDOWPROBE v1 replay: one `W index contains` row per window/clock pair.
pub fn replay_windows(request: &[&str], zoneinfo: &Path) -> Result<String, String> {
    assert_eq!(request.first(), Some(&WINDOW_HEADER), "request header");
    assert!(request.len() > 1, "window request without calls");
    let mut out = format!("{WINDOW_HEADER}\n");
    for (index, line) in request[1..].iter().enumerate() {
        let mut fields = Fields::new(line);
        assert_eq!(fields.next(), "W", "window row");
        let window = window(&mut fields, Some(zoneinfo))?.expect("window row names a window");
        let now = wall(&mut fields)?.expect("window row names a clock");
        fields.end();
        let contains = window
            .contains(now)
            .map_err(|error| format!("contains: {error:?}"))?;
        writeln!(out, "W\t{index}\t{}", u8::from(contains)).expect("writing to String");
    }
    writeln!(out, "END\t{}", request.len() - 1).expect("writing to String");
    Ok(out)
}
